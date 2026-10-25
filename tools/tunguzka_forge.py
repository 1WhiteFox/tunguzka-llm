#!/usr/bin/env python3
import argparse
import os
import shutil
import struct
import sys
import urllib.request
import convert_to_mmx as conv

ALIGN = 64

BACKEND_MAP = {
    "scalar": 0,
    "p54c": 1,
    "mmx": 2,
    "3dnow": 3,
}

PRESET_MODELS = {
    "stories260k": {
        "model_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin",
        "tok_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin",
        "bin_file": "stories260K.bin",
        "tok_file": "tok512.bin",
    },
    "stories15m": {
        "model_url": "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "tok_url": "https://raw.githubusercontent.com/karpathy/llama2.c/master/tokenizer.bin",
        "bin_file": "stories15M.bin",
        "tok_file": "tokenizer.bin",
    },
}


def download_if_missing(url: str, dst_path: str):
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1024:
        print(f"[Forge] Cached file found: {dst_path} ({os.path.getsize(dst_path) / 1024.0:.1f} KB)")
        return
    print(f"[Forge] Downloading {url} -> {dst_path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Tunguzka-Forge/2.0"})
    with urllib.request.urlopen(req) as resp, open(dst_path, "wb") as out:
        shutil.copyfileobj(resp, out)
    print(f"[Forge] Downloaded {dst_path} ({os.path.getsize(dst_path) / 1024.0:.1f} KB)")


def build_runtime_cfg(
    model_file: str = "model.mmx",
    steps: int = 128,
    ctx_limit: int = 256,
    temp: float = 0.8,
    backend: str = "mmx",
    use_kv8: bool = False,
    force_stream: bool = False,
    lang: str = "ru",
    max_ram_kb: int = 0,
) -> bytes:
    mf_bytes = model_file.encode("ascii", errors="ignore")[:35]
    mf_padded = mf_bytes + b"\x00" * (36 - len(mf_bytes))
    cfg = struct.pack(
        "<8sIIfBBBBI36s",
        b"TZCFG\x00\x00\x00",
        steps,
        ctx_limit,
        temp,
        BACKEND_MAP.get(backend, 2),
        1 if use_kv8 else 0,
        1 if force_stream else 0,
        1 if lang == "ru" else 0,
        max_ram_kb,
        mf_padded,
    )
    assert len(cfg) == 64
    return cfg


def update_mmx_header_tok_and_crc(mmx_data: bytearray, tok_offset: int, tok_size: int):
    struct.pack_into("<II", mmx_data, 48, tok_offset, tok_size)
    version = mmx_data[9]
    if version >= 2:
        crc = conv.compute_header_crc32(bytes(mmx_data[:60]))
        struct.pack_into("<I", mmx_data, 60, crc)


def embed_tokenizer_into_mmx(mmx_path: str, tok_path: str, out_path: str):
    with open(mmx_path, "rb") as f:
        mmx_data = bytearray(f.read())
    with open(tok_path, "rb") as f:
        tok_data = f.read()
    if len(mmx_data) < 64 or mmx_data[:8] != b"TUNGUZKA":
        sys.exit("Error: invalid .mmx file")
    cur_tok_off = struct.unpack_from("<I", mmx_data, 48)[0]
    if cur_tok_off > 0:
        mmx_data = mmx_data[:cur_tok_off]
    rem = len(mmx_data) % ALIGN
    if rem != 0:
        mmx_data.extend(b"\x00" * (ALIGN - rem))
    tok_offset = len(mmx_data)
    tok_size = len(tok_data)
    update_mmx_header_tok_and_crc(mmx_data, tok_offset, tok_size)
    mmx_data.extend(tok_data)
    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(mmx_data)
    print(f"[Forge] Embedded tokenizer ({tok_size} B) into {out_path} ({len(mmx_data) / 1024.0:.1f} KB)")


def pack_standalone_exe(exe_path: str, mmx_path: str, tok_path: str, cfg_bytes: bytes, out_exe: str):
    with open(exe_path, "rb") as f:
        exe_data = bytearray(f.read())
    with open(mmx_path, "rb") as f:
        mmx_data = bytearray(f.read())
    if tok_path and os.path.exists(tok_path):
        with open(tok_path, "rb") as f:
            tok_data = f.read()
        cur_tok_off = struct.unpack_from("<I", mmx_data, 48)[0]
        if cur_tok_off == 0:
            rem = len(mmx_data) % ALIGN
            if rem != 0:
                mmx_data.extend(b"\x00" * (ALIGN - rem))
            tok_off = len(mmx_data)
            update_mmx_header_tok_and_crc(mmx_data, tok_off, len(tok_data))
            mmx_data.extend(tok_data)
    rem = len(exe_data) % ALIGN
    if rem != 0:
        exe_data.extend(b"\x00" * (ALIGN - rem))
    bundle_offset = len(exe_data)
    bundle_size = len(mmx_data)
    exe_data.extend(mmx_data)
    rem = len(exe_data) % ALIGN
    if rem != 0:
        exe_data.extend(b"\x00" * (ALIGN - rem))
    cfg_offset = len(exe_data)
    exe_data.extend(cfg_bytes)
    trailer = struct.pack("<III4s", bundle_offset, bundle_size, cfg_offset, b"TZPK")
    exe_data.extend(trailer)
    out_dir = os.path.dirname(os.path.abspath(out_exe))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_exe, "wb") as f:
        f.write(exe_data)
    try:
        os.chmod(out_exe, 0o755)
    except OSError:
        pass
    print(f"[Forge] Packed single-click executable: {out_exe} ({len(exe_data) / 1024.0:.1f} KB)")


def prepare_retro_dist(dist_dir: str, exe_path: str, mmx_path: str, tok_path: str, cfg_bytes: bytes):
    os.makedirs(dist_dir, exist_ok=True)
    dst_mmx = os.path.join(dist_dir, os.path.basename(mmx_path))
    if tok_path and os.path.exists(tok_path):
        embed_tokenizer_into_mmx(mmx_path, tok_path, dst_mmx)
    else:
        shutil.copy2(mmx_path, dst_mmx)
    exe_name = os.path.basename(exe_path)
    dst_exe = os.path.join(dist_dir, exe_name)
    shutil.copy2(exe_path, dst_exe)
    cfg_path = os.path.join(dist_dir, "TUNGUZKA.CFG")
    with open(cfg_path, "wb") as f:
        f.write(cfg_bytes)
    bat_bench = os.path.join(dist_dir, "BENCH_3DNOW.BAT")
    with open(bat_bench, "w", encoding="ascii") as f:
        f.write("@ECHO OFF\r\n")
        f.write("CLS\r\n")
        f.write("ECHO === Tunguzka-LLM AMD 3DNow! Benchmark (stories260K) ===\r\n")
        f.write(f'{exe_name} -m {os.path.basename(mmx_path)} -p "Once upon a time" -n 128 -b 3dnow\r\n')
        f.write(f'{exe_name} -m {os.path.basename(mmx_path)} -p "Once upon a time" -n 128 -b mmx\r\n')
        f.write("PAUSE\r\n")
    bat_tui = os.path.join(dist_dir, "RUN_TUI.BAT")
    with open(bat_tui, "w", encoding="ascii") as f:
        f.write("@ECHO OFF\r\n")
        f.write(f"{exe_name} -m {os.path.basename(mmx_path)} --tui --ru -b 3dnow\r\n")
    print(f"[Forge] Prepared retro distribution in '{dist_dir}/' (with BENCH_3DNOW.BAT & RUN_TUI.BAT)")


def main():
    parser = argparse.ArgumentParser(description="Tunguzka Forge: Fetch, Convert, Inspect & Bundle Retro LLM Packages")
    parser.add_argument("--info", type=str, help="Inspect metadata and verify CRC32 of a .mmx file")
    parser.add_argument("--fetch", choices=["stories260k", "stories15m"], help="Download & convert real TinyStories model")
    parser.add_argument("--hf", type=str, help="Convert HuggingFace repo or local dir directly into .mmx")
    parser.add_argument("--quant", "-q", choices=["binary", "ternary", "q8", "q4"], default="q8")
    parser.add_argument("--model", "-m", type=str, help="Input/Output .mmx model")
    parser.add_argument("--tokenizer", "-z", type=str, default="tokenizer.bin", help="Input tokenizer.bin")
    parser.add_argument("--exe", "-e", type=str, help="Base binary (e.g. tunguzka_3dnow.exe or ./tunguzka)")
    parser.add_argument("--out-exe", type=str, help="Output self-contained single-click executable")
    parser.add_argument("--embed-tok", type=str, help="Output .mmx cartridge with embedded tokenizer")
    parser.add_argument("--out-cfg", type=str, help="Output 64-byte TUNGUZKA.CFG file")
    parser.add_argument("--dist-dir", type=str, help="Output directory for Retro PC / 86Box test")
    parser.add_argument("--ctx", type=int, default=256, help="Default context limit")
    parser.add_argument("--steps", type=int, default=128, help="Default generation steps")
    parser.add_argument("--temp", type=float, default=0.8, help="Default sampling temperature")
    parser.add_argument("--backend", choices=["scalar", "p54c", "mmx", "3dnow"], default="3dnow")
    parser.add_argument("--kv8", action="store_true", help="Enable INT8 KV-Cache by default")
    parser.add_argument("--stream", action="store_true", help="Enable Disk Streaming by default")
    parser.add_argument("--lang", choices=["en", "ru"], default="ru", help="Default UI language")
    parser.add_argument("--max-ram-kb", type=int, default=0)
    args = parser.parse_args()

    if args.info:
        conv.inspect_mmx(args.info)
        return

    q_map = {
        "binary": conv.KERNEL_BINARY_1BIT,
        "ternary": conv.KERNEL_TERNARY_MMX,
        "q8": conv.KERNEL_Q8_3DNOW,
        "q4": conv.KERNEL_Q4_3DNOW,
    }
    mmx_path = args.model
    tok_path = args.tokenizer

    if args.fetch:
        preset = PRESET_MODELS[args.fetch]
        download_if_missing(preset["model_url"], preset["bin_file"])
        download_if_missing(preset["tok_url"], preset["tok_file"])
        tok_path = preset["tok_file"]
        if not mmx_path:
            mmx_path = f"{args.fetch}_{args.quant}.mmx"
        config, weights = conv.load_karpathy_bin(preset["bin_file"], 32)
        conv.export_mmx(mmx_path, config, weights, q_map[args.quant], 32)
        embed_tokenizer_into_mmx(mmx_path, tok_path, mmx_path)
    elif args.hf:
        if not mmx_path:
            mmx_path = "model_hf.mmx"
        config, weights = conv.load_hf_safetensors(args.hf, 32)
        conv.export_mmx(mmx_path, config, weights, q_map[args.quant], 32)

    if not mmx_path:
        sys.exit("Error: specify --model <file.mmx>, --fetch {stories260k,stories15m}, --hf <repo>, or --info <file.mmx>")

    cfg_bytes = build_runtime_cfg(
        model_file=os.path.basename(mmx_path),
        steps=args.steps,
        ctx_limit=args.ctx,
        temp=args.temp,
        backend=args.backend,
        use_kv8=args.kv8,
        force_stream=args.stream,
        lang=args.lang,
        max_ram_kb=args.max_ram_kb,
    )
    if args.out_cfg:
        with open(args.out_cfg, "wb") as f:
            f.write(cfg_bytes)
        print(f"[Forge] Saved 64-byte binary config: {args.out_cfg}")
    if args.embed_tok:
        embed_tokenizer_into_mmx(mmx_path, tok_path, args.embed_tok)
    if args.dist_dir:
        if not args.exe:
            sys.exit("Error: --exe is required when building --dist-dir")
        prepare_retro_dist(args.dist_dir, args.exe, mmx_path, tok_path, cfg_bytes)
    if args.out_exe:
        if not args.exe:
            sys.exit("Error: --exe is required when building --out-exe")
        pack_standalone_exe(args.exe, mmx_path, tok_path, cfg_bytes, args.out_exe)


if __name__ == "__main__":
    main()