#!/usr/bin/env python3
import argparse
import array
import gc
import json
import os
import random
import struct
import sys
import zlib

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

MAGIC = b"TUNGUZKA\x00"
VERSION = 2
ALIGN = 64

KERNEL_TERNARY_MMX = 0x01
KERNEL_Q4_3DNOW    = 0x02
KERNEL_Q8_3DNOW    = 0x03
KERNEL_BINARY_1BIT = 0x04

MODEL_FLAG_SHARED_CLASSIFIER   = 1 << 0
MODEL_FLAG_BITNET_SUBNORM      = 1 << 1
MODEL_FLAG_EMB_WCLS_Q8         = 1 << 2
MODEL_FLAG_HAS_SUBNORM_WEIGHTS = 1 << 3
MODEL_FLAG_RELU2               = 1 << 4
MODEL_FLAG_LAYER_MAJOR         = 1 << 5
MODEL_FLAG_ROPE_THETA_500K     = 1 << 6
MODEL_FLAG_LLAMA3_ROPE_SCALE   = 1 << 7

KERNEL_NAMES = {
    KERNEL_TERNARY_MMX: "1.58-bit Ternary (MMX LUT)",
    KERNEL_Q4_3DNOW:    "4-bit Symmetric (Q4)",
    KERNEL_Q8_3DNOW:    "8-bit Symmetric (Q8)",
    KERNEL_BINARY_1BIT: "1.0-bit Binary",
}


def compute_header_crc32(header_first_60b: bytes) -> int:
    assert len(header_first_60b) == 60
    return zlib.crc32(header_first_60b) & 0xFFFFFFFF


def pack_header_v2(
    kernel_type: int,
    emb_kernel_type: int,
    dim: int,
    hidden_dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int,
    vocab_size: int,
    max_seq_len: int,
    group_size: int,
    flags: int,
    tok_offset: int = 0,
    tok_size: int = 0,
    rms_norm_eps: float = 1e-5,
) -> bytes:
    body = struct.pack(
        "<9sBBB11If",
        MAGIC,
        VERSION,
        kernel_type,
        emb_kernel_type,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        max_seq_len,
        group_size,
        flags,
        tok_offset,
        tok_size,
        float(rms_norm_eps),
    )
    crc = compute_header_crc32(body)
    full = body + struct.pack("<I", crc)
    assert len(full) == 64
    return full


def inspect_mmx(path: str):
    if not os.path.exists(path):
        sys.exit(f"Error: file not found: {path}")
    fsize = os.path.getsize(path)
    with open(path, "rb") as f:
        hdr = f.read(64)
    if len(hdr) < 64 or hdr[:8] != b"TUNGUZKA":
        sys.exit(f"Error: {path} is not a valid Tunguzka .mmx file")

    (
        magic,
        version,
        ktype,
        emb_ktype,
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        vocab_size,
        max_seq_len,
        group_size,
        flags,
        tok_offset,
        tok_size,
        rms_eps,
        stored_crc,
    ) = struct.unpack("<9sBBB11IfI", hdr)

    if hdr[9] == 0 and hdr[10] == 1:
        version = 1
        ktype = hdr[11]
        emb_ktype = KERNEL_Q8_3DNOW if (flags & MODEL_FLAG_EMB_WCLS_Q8) else ktype
        rms_eps = 1e-5
        crc_status = "N/A (v1 legacy header)"
    else:
        calc_crc = compute_header_crc32(hdr[:60])
        if stored_crc == 0:
            crc_status = "0x00000000 (unchecked)"
        elif calc_crc == stored_crc:
            crc_status = f"0x{stored_crc:08X} (VALID)"
        else:
            crc_status = f"0x{stored_crc:08X} (MISMATCH! expected 0x{calc_crc:08X})"

    flag_list = []
    if flags & MODEL_FLAG_SHARED_CLASSIFIER:   flag_list.append("SHARED_CLS")
    if flags & MODEL_FLAG_BITNET_SUBNORM:      flag_list.append("BITNET_SUBNORM")
    if flags & MODEL_FLAG_EMB_WCLS_Q8:         flag_list.append("EMB_WCLS_Q8")
    if flags & MODEL_FLAG_HAS_SUBNORM_WEIGHTS: flag_list.append("SUBNORM_WEIGHTS")
    if flags & MODEL_FLAG_RELU2:               flag_list.append("RELU2")
    if flags & MODEL_FLAG_LAYER_MAJOR:         flag_list.append("LAYER_MAJOR")
    if flags & MODEL_FLAG_ROPE_THETA_500K:     flag_list.append("ROPE_THETA_500K")
    if flags & MODEL_FLAG_LLAMA3_ROPE_SCALE:   flag_list.append("LLAMA3_ROPE_SCALE")

    print(f"=== Tunguzka .mmx Cartridge Info: {os.path.basename(path)} ===")
    print(f"  File Size       : {fsize / 1024.0:.1f} KB ({fsize} bytes)")
    print(f"  Format Version  : v{version}")
    print(f"  Header CRC32    : {crc_status}")
    print(f"  Primary Kernel  : 0x{ktype:02x} ({KERNEL_NAMES.get(ktype, 'Unknown')})")
    print(f"  Emb/Cls Kernel  : 0x{emb_ktype:02x} ({KERNEL_NAMES.get(emb_ktype, 'Unknown')})")
    print(f"  Dimensions      : dim={dim}, hidden_dim={hidden_dim}, layers={n_layers}")
    print(f"  Attention Heads : n_heads={n_heads}, n_kv_heads={n_kv_heads} (head_dim={dim // max(n_heads, 1)})")
    print(f"  Vocab / Context : vocab_size={vocab_size}, max_seq_len={max_seq_len}, group_size={group_size}")
    print(f"  RMSNorm Epsilon : {rms_eps:.1e}")
    print(f"  Model Flags     : 0x{flags:02x} ({', '.join(flag_list) if flag_list else 'NONE'})")
    if tok_offset > 0 and tok_size > 0:
        print(f"  Embedded Vocab  : YES (offset={tok_offset}, size={tok_size / 1024.0:.1f} KB)")
    else:
        print(f"  Embedded Vocab  : NO (uses external tokenizer.bin)")


def pad_to_align(f, align=ALIGN):
    pos = f.tell()
    rem = pos % align
    if rem != 0:
        f.write(b"\x00" * (align - rem))


def quantize_binary_1bit(flat_f32, group_size: int, per_tensor_scale: bool = False):
    if HAS_NUMPY and isinstance(flat_f32, np.ndarray):
        flat_all = flat_f32.reshape(-1, group_size)
        num_g = flat_all.shape[0]
        g_scale = max(float(np.mean(np.abs(flat_all))), 1e-8) if per_tensor_scale else 0.0
        data_chunks = []
        scale_chunks = []
        step = 65536
        for start in range(0, num_g, step):
            chunk = flat_all[start : start + step].astype(np.float32, copy=False)
            if per_tensor_scale:
                sc = np.full((chunk.shape[0], 1), g_scale, dtype=np.float32)
            else:
                sc = np.maximum(np.mean(np.abs(chunk), axis=1, keepdims=True), 1e-8).astype(np.float32)
            bits = (chunk >= 0.0).astype(np.uint8).reshape(-1, 8)
            packed = np.zeros(bits.shape[0], dtype=np.uint8)
            for k in range(8):
                packed |= (bits[:, k] << k)
            data_chunks.append(packed.tobytes())
            scale_chunks.append(sc.reshape(-1).tobytes())
        return b"".join(data_chunks), b"".join(scale_chunks)

    n = len(flat_f32)
    num_groups = n // group_size
    scales = array.array("f")
    packed = bytearray(n // 8)
    p_idx = 0
    for g in range(num_groups):
        base = g * group_size
        s_abs = 0.0
        for i in range(group_size):
            s_abs += abs(flat_f32[base + i])
        scale = max(s_abs / group_size, 1e-8)
        scales.append(scale)
        for i in range(0, group_size, 8):
            byte_val = 0
            for k in range(8):
                if flat_f32[base + i + k] >= 0.0:
                    byte_val |= (1 << k)
            packed[p_idx] = byte_val
            p_idx += 1
    if sys.byteorder != "little":
        scales.byteswap()
    return bytes(packed), scales.tobytes()


def quantize_ternary(flat_f32, group_size: int, per_tensor_scale: bool = False):
    if HAS_NUMPY and isinstance(flat_f32, np.ndarray):
        flat_all = flat_f32.reshape(-1, group_size)
        num_g = flat_all.shape[0]
        gamma = max(float(np.mean(np.abs(flat_all))), 1e-5) if per_tensor_scale else 0.0
        data_chunks = []
        scale_chunks = []
        step = 65536
        for start in range(0, num_g, step):
            chunk = flat_all[start : start + step].astype(np.float32, copy=False)
            if per_tensor_scale:
                sc = np.full((chunk.shape[0], 1), gamma, dtype=np.float32)
            else:
                sc = np.maximum(np.mean(np.abs(chunk), axis=1, keepdims=True), 1e-8).astype(np.float32)
            q = np.clip(np.round(chunk / sc), -1.0, 1.0).astype(np.int8)
            codes = np.zeros_like(q, dtype=np.uint8)
            codes[q == 1] = 1
            codes[q == -1] = 2
            codes = codes.reshape(-1, 4)
            packed = (
                codes[:, 0]
                | (codes[:, 1] << 2)
                | (codes[:, 2] << 4)
                | (codes[:, 3] << 6)
            ).astype(np.uint8)
            data_chunks.append(packed.tobytes())
            scale_chunks.append(sc.reshape(-1).tobytes())
        return b"".join(data_chunks), b"".join(scale_chunks)

    n = len(flat_f32)
    num_groups = n // group_size
    scales = array.array("f")
    packed = bytearray(n // 4)
    p_idx = 0
    global_gamma = 0.0
    if per_tensor_scale and n > 0:
        s_all = 0.0
        for v in flat_f32:
            s_all += abs(v)
        global_gamma = max(s_all / n, 1e-5)
    for g in range(num_groups):
        base = g * group_size
        if per_tensor_scale:
            scale = global_gamma
        else:
            s_abs = 0.0
            for i in range(group_size):
                s_abs += abs(flat_f32[base + i])
            scale = max(s_abs / group_size, 1e-8)
        scales.append(scale)
        inv_s = 1.0 / scale
        for i in range(0, group_size, 4):
            byte_val = 0
            for k in range(4):
                v = round(flat_f32[base + i + k] * inv_s)
                if v >= 1:
                    code = 1
                elif v <= -1:
                    code = 2
                else:
                    code = 0
                byte_val |= (code << (k * 2))
            packed[p_idx] = byte_val
            p_idx += 1
    if sys.byteorder != "little":
        scales.byteswap()
    return bytes(packed), scales.tobytes()


def quantize_q8(flat_f32, group_size: int):
    if HAS_NUMPY and isinstance(flat_f32, np.ndarray):
        flat_all = flat_f32.reshape(-1, group_size)
        num_g = flat_all.shape[0]
        data_chunks = []
        scale_chunks = []
        step = 65536
        for start in range(0, num_g, step):
            chunk = flat_all[start : start + step].astype(np.float32, copy=False)
            max_val = np.max(np.abs(chunk), axis=1, keepdims=True)
            sc = (max_val / 127.0).astype(np.float32)
            inv_sc = np.where(sc == 0, 0.0, 1.0 / sc)
            q = np.clip(np.round(chunk * inv_sc), -127.0, 127.0).astype(np.int8)
            data_chunks.append(q.tobytes())
            scale_chunks.append(sc.reshape(-1).tobytes())
        return b"".join(data_chunks), b"".join(scale_chunks)

    n = len(flat_f32)
    num_groups = n // group_size
    scales = array.array("f")
    q_arr = array.array("b")
    for g in range(num_groups):
        base = g * group_size
        max_v = 0.0
        for i in range(group_size):
            av = abs(flat_f32[base + i])
            if av > max_v:
                max_v = av
        scale = max_v / 127.0
        scales.append(scale)
        inv_s = 0.0 if scale == 0.0 else (1.0 / scale)
        for i in range(group_size):
            v = round(flat_f32[base + i] * inv_s)
            if v > 127:
                v = 127
            elif v < -127:
                v = -127
            q_arr.append(v)
    if sys.byteorder != "little":
        scales.byteswap()
    return q_arr.tobytes(), scales.tobytes()


def quantize_q4(flat_f32, group_size: int):
    if HAS_NUMPY and isinstance(flat_f32, np.ndarray):
        flat_all = flat_f32.reshape(-1, group_size)
        num_g = flat_all.shape[0]
        data_chunks = []
        scale_chunks = []
        step = 65536
        for start in range(0, num_g, step):
            chunk = flat_all[start : start + step].astype(np.float32, copy=False)
            max_val = np.max(np.abs(chunk), axis=1, keepdims=True)
            sc = (max_val / 7.0).astype(np.float32)
            inv_sc = np.where(sc == 0, 0.0, 1.0 / sc)
            q = np.clip(np.round(chunk * inv_sc), -8.0, 7.0).astype(np.int8)
            u = (q + 8).astype(np.uint8).reshape(-1, 2)
            packed = (u[:, 0] | (u[:, 1] << 4)).astype(np.uint8)
            data_chunks.append(packed.tobytes())
            scale_chunks.append(sc.reshape(-1).tobytes())
        return b"".join(data_chunks), b"".join(scale_chunks)

    n = len(flat_f32)
    num_groups = n // group_size
    scales = array.array("f")
    packed = bytearray(n // 2)
    p_idx = 0
    for g in range(num_groups):
        base = g * group_size
        max_v = 0.0
        for i in range(group_size):
            av = abs(flat_f32[base + i])
            if av > max_v:
                max_v = av
        scale = max_v / 7.0
        scales.append(scale)
        inv_s = 0.0 if scale == 0.0 else (1.0 / scale)
        for i in range(0, group_size, 2):
            v0 = max(-8, min(7, round(flat_f32[base + i] * inv_s))) + 8
            v1 = max(-8, min(7, round(flat_f32[base + i + 1] * inv_s))) + 8
            packed[p_idx] = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
            p_idx += 1
    if sys.byteorder != "little":
        scales.byteswap()
    return bytes(packed), scales.tobytes()


def tensor_len(t):
    return t.size if (HAS_NUMPY and isinstance(t, np.ndarray)) else len(t)


def tensor_to_f32_bytes(t):
    if HAS_NUMPY and isinstance(t, np.ndarray):
        return t.astype(np.float32).tobytes()
    arr = array.array("f", t)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def write_quantized_tensor(f, tensor, kernel_type: int, group_size: int, per_tensor_scale: bool = False):
    n = tensor_len(tensor)
    if n % group_size != 0:
        raise ValueError(f"Tensor size {n} is not divisible by group_size {group_size}")
    if kernel_type == KERNEL_TERNARY_MMX:
        data_bytes, scale_bytes = quantize_ternary(tensor, group_size, per_tensor_scale)
    elif kernel_type == KERNEL_Q8_3DNOW:
        data_bytes, scale_bytes = quantize_q8(tensor, group_size)
    elif kernel_type == KERNEL_Q4_3DNOW:
        data_bytes, scale_bytes = quantize_q4(tensor, group_size)
    elif kernel_type == KERNEL_BINARY_1BIT:
        data_bytes, scale_bytes = quantize_binary_1bit(tensor, group_size, per_tensor_scale)
    else:
        raise ValueError(f"Unsupported kernel_type: {kernel_type}")

    pad_to_align(f)
    f.write(scale_bytes)
    pad_to_align(f)
    f.write(data_bytes)
    pad_to_align(f)


def read_f32_chunk(f, count: int):
    if HAS_NUMPY:
        return np.frombuffer(f.read(count * 4), dtype=np.float32)
    arr = array.array("f")
    arr.frombytes(f.read(count * 4))
    if sys.byteorder != "little":
        arr.byteswap()
    return arr


def permute_hf_rope(w_flat, n_heads: int, dim1: int, dim2: int):
    if not HAS_NUMPY:
        w_np = np.array(w_flat, dtype=np.float32)
    else:
        w_np = np.asarray(w_flat, dtype=np.float32)
    head_dim = dim1 // n_heads
    return (
        w_np.reshape(n_heads, 2, head_dim // 2, dim2)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )


def pad_rows_f32(tensor, orig_rows: int, new_rows: int, cols: int):
    if orig_rows == new_rows:
        return tensor
    if HAS_NUMPY and isinstance(tensor, np.ndarray):
        mat = tensor.reshape(orig_rows, cols)
        pad = np.zeros((new_rows - orig_rows, cols), dtype=np.float32)
        return np.vstack([mat, pad]).reshape(-1)
    out = array.array("f", tensor)
    out.extend([0.0] * ((new_rows - orig_rows) * cols))
    return out


def pad_cols_f32(tensor, rows: int, orig_cols: int, new_cols: int):
    if orig_cols == new_cols:
        return tensor
    if HAS_NUMPY and isinstance(tensor, np.ndarray):
        mat = tensor.reshape(rows, orig_cols)
        pad = np.zeros((rows, new_cols - orig_cols), dtype=np.float32)
        return np.hstack([mat, pad]).reshape(-1)
    out = array.array("f")
    pad_zeros = [0.0] * (new_cols - orig_cols)
    for r in range(rows):
        base = r * orig_cols
        out.extend(tensor[base : base + orig_cols])
        out.extend(pad_zeros)
    return out


def load_karpathy_bin(path: str, group_size: int = 32, permute_rope: bool = False):
    with open(path, "rb") as f:
        header = struct.unpack("<7i", f.read(28))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = header
        vocab_size = abs(vocab_size)
        kv_dim = (dim * n_kv_heads) // n_heads
        padded_hidden = ((hidden_dim + group_size - 1) // group_size) * group_size
        weights = {}
        weights["tok_embeddings"] = read_f32_chunk(f, vocab_size * dim)
        weights["rms_att"] = read_f32_chunk(f, n_layers * dim)
        for key, rows, cols in [
            ("wq", dim, dim),
            ("wk", kv_dim, dim),
            ("wv", kv_dim, dim),
            ("wo", dim, dim),
        ]:
            weights[key] = [read_f32_chunk(f, rows * cols) for _ in range(n_layers)]
        if permute_rope and HAS_NUMPY:
            weights["wq"] = [permute_hf_rope(t, n_heads, dim, dim) for t in weights["wq"]]
            weights["wk"] = [permute_hf_rope(t, n_kv_heads, kv_dim, dim) for t in weights["wk"]]
        weights["rms_ffn"] = read_f32_chunk(f, n_layers * dim)
        w1_list = [read_f32_chunk(f, hidden_dim * dim) for _ in range(n_layers)]
        w2_list = [read_f32_chunk(f, dim * hidden_dim) for _ in range(n_layers)]
        w3_list = [read_f32_chunk(f, hidden_dim * dim) for _ in range(n_layers)]
        if padded_hidden != hidden_dim:
            w1_list = [pad_rows_f32(t, hidden_dim, padded_hidden, dim) for t in w1_list]
            w3_list = [pad_rows_f32(t, hidden_dim, padded_hidden, dim) for t in w3_list]
            w2_list = [pad_cols_f32(t, dim, hidden_dim, padded_hidden) for t in w2_list]
        weights["w1"] = w1_list
        weights["w2"] = w2_list
        weights["w3"] = w3_list
        weights["rms_final"] = read_f32_chunk(f, dim)

        rem_bytes = os.path.getsize(path) - f.tell()
        expected_wcls = vocab_size * dim * 4
        expected_subnorms = n_layers * (dim + hidden_dim) * 4
        shared_classifier = True
        if rem_bytes >= expected_wcls + expected_subnorms:
            weights["rms_sub_att"] = read_f32_chunk(f, n_layers * dim)
            sub_ffn_raw = [read_f32_chunk(f, hidden_dim) for _ in range(n_layers)]
            if padded_hidden != hidden_dim:
                sub_ffn_raw = [pad_rows_f32(t, hidden_dim, padded_hidden, 1) for t in sub_ffn_raw]
            weights["rms_sub_ffn"] = np.concatenate(sub_ffn_raw) if HAS_NUMPY else [x for r in sub_ffn_raw for x in r]
            weights["wcls"] = read_f32_chunk(f, vocab_size * dim)
            shared_classifier = False
        elif rem_bytes >= expected_wcls:
            if rem_bytes > expected_wcls:
                f.seek(rem_bytes - expected_wcls, os.SEEK_CUR)
            weights["wcls"] = read_f32_chunk(f, vocab_size * dim)
            shared_classifier = False

    config = {
        "dim": dim,
        "hidden_dim": padded_hidden,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "vocab_size": vocab_size,
        "max_seq_len": min(seq_len, 512),
        "shared_classifier": shared_classifier,
        "hybrid_emb_q8": True,
        "is_bitnet": False,
        "use_relu2": False,
        "rms_norm_eps": 1e-5,
        "rope_theta_500k": vocab_size >= 100000,
        "llama3_rope_scale": False,
    }
    return config, weights


def _gpt2_bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def build_tokenizer_from_hf_json(tok_json_path: str, vocab_size: int):
    with open(tok_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_obj = data.get("model", {})
    vocab_raw = model_obj.get("vocab", {})

    dec_str = json.dumps(data.get("decoder", {})) + json.dumps(data.get("pre_tokenizer", {}))
    is_byte_level = "ByteLevel" in dec_str or len(vocab_raw) >= 49000
    u2b = _gpt2_bytes_to_unicode() if is_byte_level else None

    def piece_to_bytes(piece_str: str) -> bytes:
        if piece_str.startswith("<0x") and piece_str.endswith(">") and len(piece_str) == 6:
            return piece_str.encode("utf-8")
        if is_byte_level and u2b is not None:
            out_b = bytearray()
            for ch in piece_str:
                if ch in u2b:
                    out_b.append(u2b[ch])
                else:
                    out_b.extend(ch.encode("utf-8"))
            if bytes(out_b) == b"\x00":
                return b"<0x00>"
            return bytes(out_b)
        return piece_str.replace("\u2581", " ").replace("\u0120", " ").encode("utf-8")

    id_to_piece = {}
    id_to_score = {}

    if isinstance(vocab_raw, list):
        for idx, item in enumerate(vocab_raw):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                id_to_piece[idx] = str(item[0])
                id_to_score[idx] = float(item[1])
    elif isinstance(vocab_raw, dict):
        for piece_str, idx in vocab_raw.items():
            id_to_piece[int(idx)] = str(piece_str)
            id_to_score[int(idx)] = -float(int(idx) + 1)

    merges = model_obj.get("merges", [])
    if merges and isinstance(vocab_raw, dict):
        for rank, m in enumerate(merges):
            if isinstance(m, str):
                parts = m.split(" ", 1)
                merged = parts[0] + parts[1] if len(parts) == 2 else m
            elif isinstance(m, (list, tuple)) and len(m) >= 2:
                merged = str(m[0]) + str(m[1])
            else:
                continue
            mid = vocab_raw.get(merged)
            if mid is not None:
                id_to_score[int(mid)] = 1000000.0 - float(rank)

    for added in data.get("added_tokens", []):
        aid = int(added.get("id", -1))
        content = str(added.get("content", ""))
        if 0 <= aid < vocab_size:
            id_to_piece[aid] = content
            if added.get("special", False) or (content.startswith("<") and content.endswith(">")):
                id_to_score[aid] = -1e9

    tokens_bytes = []
    scores_list = []
    max_len = 1

    for i in range(vocab_size):
        p_str = id_to_piece.get(i, "")
        sc = id_to_score.get(i, -1e10)
        if p_str != "":
            if i >= 128000 or (p_str.startswith("<|") and p_str.endswith("|>")):
                b = p_str.encode("utf-8")
            else:
                b = piece_to_bytes(p_str)
        else:
            b = b""
            sc = -1e10
        if len(b) > max_len:
            max_len = len(b)
        tokens_bytes.append(b)
        scores_list.append(sc)

    out = bytearray(struct.pack("<I", max_len))
    for b, s in zip(tokens_bytes, scores_list):
        out.extend(struct.pack("<fI", float(s), len(b)))
        out.extend(b)
    return bytes(out)


def build_hf_tokenizer_bytes(model_dir: str, vocab_size: int):
    tok_bin_path = os.path.join(model_dir, "tokenizer.bin")
    if os.path.exists(tok_bin_path) and os.path.getsize(tok_bin_path) > 1024:
        with open(tok_bin_path, "rb") as f:
            return f.read()

    tok_json_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.exists(tok_json_path):
        try:
            return build_tokenizer_from_hf_json(tok_json_path, vocab_size)
        except Exception as e:
            print(f"[Warning] Failed to parse tokenizer.json ({e}), trying tokenizer.model...")

    tok_model_path = os.path.join(model_dir, "tokenizer.model")
    if os.path.exists(tok_model_path):
        try:
            from sentencepiece import SentencePieceProcessor
            sp = SentencePieceProcessor(model_file=tok_model_path)
            n_words = sp.vocab_size()
            tokens, scores = [], []
            max_len = 1
            for i in range(vocab_size):
                if i < n_words:
                    p = sp.id_to_piece(i)
                    s = float(sp.get_score(i))
                    if i == sp.bos_id() or i == sp.eos_id() or i == sp.unk_id():
                        b = p.encode("utf-8")
                    elif p.startswith("<0x") and p.endswith(">") and len(p) == 6:
                        b = p.encode("utf-8")
                    else:
                        b = p.replace("\u2581", " ").encode("utf-8")
                else:
                    b = b""
                    s = -1e10
                if len(b) > max_len:
                    max_len = len(b)
                tokens.append(b)
                scores.append(s)
            out = bytearray(struct.pack("<I", max_len))
            for b, s in zip(tokens, scores):
                out.extend(struct.pack("<fI", s, len(b)))
                out.extend(b)
            return bytes(out)
        except ImportError:
            pass
    return None


def load_hf_safetensors(repo_or_dir: str, group_size: int = 32):
    if not HAS_NUMPY:
        sys.exit("Error: numpy is required for --hf loading")
    try:
        from safetensors import safe_open
    except ImportError:
        sys.exit("Error: install safetensors (`pip install safetensors huggingface_hub`) for --hf")

    if os.path.isdir(repo_or_dir):
        model_dir = repo_or_dir
    else:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=repo_or_dir)

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    archs = " ".join(cfg.get("architectures", [])).lower()
    model_type = str(cfg.get("model_type", "")).lower()
    is_bitnet = ("bitnet" in archs) or ("bitnet" in model_type)

    hidden_act = str(cfg.get("hidden_act", cfg.get("activation_function", "silu"))).lower()
    use_relu2 = "relu2" in hidden_act or " squared" in hidden_act

    rms_eps = float(cfg.get("rms_norm_eps", cfg.get("layer_norm_eps", 1e-5)))
    rope_theta = float(cfg.get("rope_theta", 10000.0))
    rope_scaling = cfg.get("rope_scaling")
    llama3_rope_scale = False
    if isinstance(rope_scaling, dict):
        rtype = str(rope_scaling.get("rope_type", rope_scaling.get("type", ""))).lower()
        if rtype == "llama3":
            llama3_rope_scale = True

    dim = int(cfg["hidden_size"])
    hidden_dim = int(cfg["intermediate_size"])
    padded_hidden = ((hidden_dim + group_size - 1) // group_size) * group_size
    n_layers = int(cfg["num_hidden_layers"])
    n_heads = int(cfg["num_attention_heads"])
    n_kv_heads = int(cfg.get("num_key_value_heads", n_heads))
    vocab_size = int(cfg["vocab_size"])
    seq_len = int(cfg.get("max_position_embeddings", 2048))
    kv_dim = (dim * n_kv_heads) // n_heads

    embedded_tok = build_hf_tokenizer_bytes(model_dir, vocab_size)
    gc.collect()

    st_files = sorted([os.path.join(model_dir, fn) for fn in os.listdir(model_dir) if fn.endswith(".safetensors")])
    if not st_files:
        sys.exit(f"Error: no .safetensors files found in {model_dir}")
    handles = [safe_open(p, framework="np", device="cpu") for p in st_files]

    def get_t(name: str):
        for h in handles:
            if name in h.keys():
                return h.get_tensor(name).astype(np.float32).reshape(-1)
        return None

    w1_list = [get_t(f"model.layers.{l}.mlp.gate_proj.weight") for l in range(n_layers)]
    w2_list = [get_t(f"model.layers.{l}.mlp.down_proj.weight") for l in range(n_layers)]
    w3_list = [get_t(f"model.layers.{l}.mlp.up_proj.weight") for l in range(n_layers)]
    if padded_hidden != hidden_dim:
        w1_list = [pad_rows_f32(t, hidden_dim, padded_hidden, dim) for t in w1_list]
        w3_list = [pad_rows_f32(t, hidden_dim, padded_hidden, dim) for t in w3_list]
        w2_list = [pad_cols_f32(t, dim, hidden_dim, padded_hidden) for t in w2_list]

    weights = {
        "tok_embeddings": get_t("model.embed_tokens.weight"),
        "rms_att": np.concatenate([get_t(f"model.layers.{l}.input_layernorm.weight") for l in range(n_layers)]),
        "rms_ffn": np.concatenate([get_t(f"model.layers.{l}.post_attention_layernorm.weight") for l in range(n_layers)]),
        "rms_final": get_t("model.norm.weight"),
        "wq": [permute_hf_rope(get_t(f"model.layers.{l}.self_attn.q_proj.weight"), n_heads, dim, dim) for l in range(n_layers)],
        "wk": [permute_hf_rope(get_t(f"model.layers.{l}.self_attn.k_proj.weight"), n_kv_heads, kv_dim, dim) for l in range(n_layers)],
        "wv": [get_t(f"model.layers.{l}.self_attn.v_proj.weight") for l in range(n_layers)],
        "wo": [get_t(f"model.layers.{l}.self_attn.o_proj.weight") for l in range(n_layers)],
        "w1": w1_list,
        "w2": w2_list,
        "w3": w3_list,
    }

    sub_att0 = get_t("model.layers.0.self_attn.inner_attn_ln.weight")
    sub_ffn0 = get_t("model.layers.0.mlp.ffn_layernorm.weight")
    if sub_att0 is not None and sub_ffn0 is not None:
        weights["rms_sub_att"] = np.concatenate([get_t(f"model.layers.{l}.self_attn.inner_attn_ln.weight") for l in range(n_layers)])
        sub_ffn_list = [get_t(f"model.layers.{l}.mlp.ffn_layernorm.weight") for l in range(n_layers)]
        if padded_hidden != hidden_dim:
            sub_ffn_list = [pad_rows_f32(t, hidden_dim, padded_hidden, 1) for t in sub_ffn_list]
        weights["rms_sub_ffn"] = np.concatenate(sub_ffn_list)
        is_bitnet = True

    wcls = get_t("lm_head.weight")
    shared_classifier = wcls is None or bool(cfg.get("tie_word_embeddings", False))
    if not shared_classifier:
        weights["wcls"] = wcls

    config = {
        "dim": dim,
        "hidden_dim": padded_hidden,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "vocab_size": vocab_size,
        "max_seq_len": min(seq_len, 512),
        "shared_classifier": shared_classifier,
        "hybrid_emb_q8": True,
        "is_bitnet": is_bitnet,
        "use_relu2": use_relu2,
        "rms_norm_eps": rms_eps,
        "rope_theta_500k": (rope_theta >= 100000.0) or (vocab_size >= 100000),
        "llama3_rope_scale": llama3_rope_scale,
        "embedded_tok_bytes": embedded_tok,
    }
    return config, weights


def generate_dummy_model():
    rng = random.Random(42)
    dim = 64
    hidden_dim = 128
    n_layers = 2
    n_heads = 4
    n_kv_heads = 4
    vocab_size = 256
    seq_len = 128
    config = {
        "dim": dim,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "vocab_size": vocab_size,
        "max_seq_len": seq_len,
        "shared_classifier": True,
        "hybrid_emb_q8": False,
        "is_bitnet": False,
        "use_relu2": False,
        "rms_norm_eps": 1e-5,
        "rope_theta_500k": False,
        "llama3_rope_scale": False,
    }

    def rand_arr(count):
        return array.array("f", (rng.gauss(0.0, 0.05) for _ in range(count)))

    def ones_arr(count):
        return array.array("f", [1.0] * count)

    weights = {
        "tok_embeddings": rand_arr(vocab_size * dim),
        "rms_att": ones_arr(n_layers * dim),
        "wq": [rand_arr(dim * dim) for _ in range(n_layers)],
        "wk": [rand_arr(dim * dim) for _ in range(n_layers)],
        "wv": [rand_arr(dim * dim) for _ in range(n_layers)],
        "wo": [rand_arr(dim * dim) for _ in range(n_layers)],
        "rms_ffn": ones_arr(n_layers * dim),
        "w1": [rand_arr(hidden_dim * dim) for _ in range(n_layers)],
        "w2": [rand_arr(dim * hidden_dim) for _ in range(n_layers)],
        "w3": [rand_arr(hidden_dim * dim) for _ in range(n_layers)],
        "rms_final": ones_arr(dim),
    }
    return config, weights


def generate_dummy_tokenizer(path: str, vocab_size: int = 256):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", 1))
        for i in range(vocab_size):
            token_bytes = bytes([i])
            f.write(struct.pack("<fI", 0.0, len(token_bytes)))
            f.write(token_bytes)


def export_mmx(output_path: str, config: dict, weights: dict, kernel_type: int, group_size: int):
    flags = MODEL_FLAG_LAYER_MAJOR
    if config.get("shared_classifier", True):
        flags |= MODEL_FLAG_SHARED_CLASSIFIER

    use_q8_emb = config.get("hybrid_emb_q8", False) and kernel_type in (
        KERNEL_BINARY_1BIT,
        KERNEL_TERNARY_MMX,
        KERNEL_Q4_3DNOW,
    )
    if use_q8_emb:
        flags |= MODEL_FLAG_EMB_WCLS_Q8

    has_sub_w = ("rms_sub_att" in weights) and ("rms_sub_ffn" in weights)
    if has_sub_w:
        flags |= MODEL_FLAG_HAS_SUBNORM_WEIGHTS
    elif config.get("is_bitnet", False) or (use_q8_emb and kernel_type == KERNEL_TERNARY_MMX):
        flags |= MODEL_FLAG_BITNET_SUBNORM

    if config.get("use_relu2", False):
        flags |= MODEL_FLAG_RELU2
    if config.get("rope_theta_500k", False):
        flags |= MODEL_FLAG_ROPE_THETA_500K
    if config.get("llama3_rope_scale", False):
        flags |= MODEL_FLAG_LLAMA3_ROPE_SCALE

    per_tensor = bool(config.get("is_bitnet", False) or (use_q8_emb and kernel_type == KERNEL_TERNARY_MMX))
    emb_ktype = KERNEL_Q8_3DNOW if use_q8_emb else kernel_type
    rms_eps = float(config.get("rms_norm_eps", 1e-5))

    header = pack_header_v2(
        kernel_type=kernel_type,
        emb_kernel_type=emb_ktype,
        dim=config["dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        n_kv_heads=config["n_kv_heads"],
        vocab_size=config["vocab_size"],
        max_seq_len=config["max_seq_len"],
        group_size=group_size,
        flags=flags,
        tok_offset=0,
        tok_size=0,
        rms_norm_eps=rms_eps,
    )

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(tensor_to_f32_bytes(weights["rms_att"]))
        f.write(tensor_to_f32_bytes(weights["rms_ffn"]))
        f.write(tensor_to_f32_bytes(weights["rms_final"]))
        if has_sub_w:
            f.write(tensor_to_f32_bytes(weights["rms_sub_att"]))
            f.write(tensor_to_f32_bytes(weights["rms_sub_ffn"]))
        pad_to_align(f)

        write_quantized_tensor(f, weights["tok_embeddings"], emb_ktype, group_size, False)
        weights["tok_embeddings"] = None
        gc.collect()

        for l in range(config["n_layers"]):
            for key in ["wq", "wk", "wv", "wo", "w1", "w3", "w2"]:
                write_quantized_tensor(f, weights[key][l], kernel_type, group_size, per_tensor)
                weights[key][l] = None
            gc.collect()

        if not config["shared_classifier"]:
            write_quantized_tensor(f, weights["wcls"], emb_ktype, group_size, False)
            weights["wcls"] = None
            gc.collect()

        tok_bytes = config.get("embedded_tok_bytes")
        if tok_bytes:
            pad_to_align(f)
            tok_off = f.tell()
            tok_sz = len(tok_bytes)
            f.write(tok_bytes)
            updated_hdr = pack_header_v2(
                kernel_type=kernel_type,
                emb_kernel_type=emb_ktype,
                dim=config["dim"],
                hidden_dim=config["hidden_dim"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                n_kv_heads=config["n_kv_heads"],
                vocab_size=config["vocab_size"],
                max_seq_len=config["max_seq_len"],
                group_size=group_size,
                flags=flags,
                tok_offset=tok_off,
                tok_size=tok_sz,
                rms_norm_eps=rms_eps,
            )
            f.seek(0)
            f.write(updated_hdr)
            f.seek(0, os.SEEK_END)

    size_kb = os.path.getsize(output_path) / 1024.0
    backend_str = "numpy" if HAS_NUMPY else "pure-python"
    print(f"[OK] Saved {output_path} ({size_kb:.1f} KB) | v{VERSION} | kernel=0x{kernel_type:02x} | emb=0x{emb_ktype:02x} | flags=0x{flags:02x} | group={group_size} | engine={backend_str}")


def main():
    parser = argparse.ArgumentParser(description="Convert LLM / BitNet / LLaMA-3 checkpoints to Tunguzka .mmx v2 format")
    parser.add_argument("--info", type=str, help="Inspect metadata and verify CRC32 of an existing .mmx file")
    parser.add_argument("--input", "-i", type=str, help="Input Karpathy .bin checkpoint path")
    parser.add_argument("--hf", type=str, help="HuggingFace repo ID or local dir (BitNet / LLaMA / Vikhr / SmolLM)")
    parser.add_argument("--output", "-o", type=str, help="Output .mmx model path")
    parser.add_argument("--quant", "-q", choices=["binary", "ternary", "q8", "q4"], default="ternary")
    parser.add_argument("--group-size", "-g", type=int, default=32)
    parser.add_argument("--permute-rope", action="store_true", help="Permute HF wq/wk rows to interleaved RoPE")
    parser.add_argument("--dummy", action="store_true", help="Generate synthetic micro-model for testing")
    parser.add_argument("--dummy-tok", type=str, help="Output path for synthetic tokenizer.bin")
    args = parser.parse_args()

    if args.info:
        inspect_mmx(args.info)
        return

    if not args.output:
        sys.exit("Error: specify --output <model.mmx> (or use --info <model.mmx>)")

    q_map = {
        "binary": KERNEL_BINARY_1BIT,
        "ternary": KERNEL_TERNARY_MMX,
        "q8": KERNEL_Q8_3DNOW,
        "q4": KERNEL_Q4_3DNOW,
    }
    kernel_type = q_map[args.quant]

    if args.dummy:
        config, weights = generate_dummy_model()
        if args.dummy_tok:
            generate_dummy_tokenizer(args.dummy_tok, config["vocab_size"])
    elif args.hf:
        config, weights = load_hf_safetensors(args.hf, args.group_size)
    else:
        if not args.input:
            sys.exit("Error: specify --input <file.bin>, --hf <repo_or_dir>, --dummy, or --info <file.mmx>")
        config, weights = load_karpathy_bin(args.input, args.group_size, args.permute_rope)

    export_mmx(args.output, config, weights, kernel_type, args.group_size)


if __name__ == "__main__":
    main()