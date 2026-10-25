#!/usr/bin/env python3
import re
import struct
import subprocess
import sys

INV_JCC = {
    0x40: 0x71,  # cmovo  -> jno
    0x41: 0x70,  # cmovno -> jo
    0x42: 0x73,  # cmovb  -> jae
    0x43: 0x72,  # cmovae -> jb
    0x44: 0x75,  # cmove  -> jne
    0x45: 0x74,  # cmovne -> je
    0x46: 0x77,  # cmovbe -> ja
    0x47: 0x76,  # cmova  -> jbe
    0x48: 0x79,  # cmovs  -> jns
    0x49: 0x78,  # cmovns -> js
    0x4A: 0x7B,  # cmovp  -> jnp
    0x4B: 0x7A,  # cmovnp -> jp
    0x4C: 0x7D,  # cmovl  -> jge
    0x4D: 0x7C,  # cmovge -> jl
    0x4E: 0x7F,  # cmovle -> jg
    0x4F: 0x7E,  # cmovg  -> jle
}


def parse_pe_text_section(data: bytes):
    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    num_sections = struct.unpack_from("<H", data, pe_off + 6)[0]
    opt_hdr_size = struct.unpack_from("<H", data, pe_off + 20)[0]
    image_base = struct.unpack_from("<I", data, pe_off + 24 + 28)[0]
    sec_table = pe_off + 24 + opt_hdr_size

    for i in range(num_sections):
        off = sec_table + i * 40
        name = data[off : off + 8].rstrip(b"\x00").decode("ascii", errors="ignore")
        vsize, vaddr, raw_size, raw_ptr = struct.unpack_from("<4I", data, off + 8)
        if name == ".text":
            return image_base, vaddr, vsize, raw_ptr, raw_size, off
    raise RuntimeError("Could not find .text section in PE binary")


def relocate_instruction(va: int, b_list: list, cave_cur_va: int) -> bytes:
    """Relocate a single x86 instruction from `va` to `cave_cur_va`."""
    has_66 = b_list[0] == 0x66
    idx = 1 if has_66 else 0

    # Check if this instruction is a CMOVcc (0F 40..4F)
    if len(b_list) >= idx + 3 and b_list[idx] == 0x0F and b_list[idx + 1] in INV_JCC:
        cc_byte = b_list[idx + 1]
        modrm = b_list[idx + 2]
        rest = bytes(b_list[idx + 3 :])
        inv_jcc = INV_JCC[cc_byte]
        mov_instr = (b"\x66" if has_66 else b"") + bytes([0x8B, modrm]) + rest
        return bytes([inv_jcc, len(mov_instr)]) + mov_instr

    op0 = b_list[0]
    length = len(b_list)

    # Short unconditional jump: EB rel8 -> E9 rel32
    if op0 == 0xEB and length == 2:
        rel8 = struct.unpack("<b", bytes([b_list[1]]))[0]
        target_va = va + 2 + rel8
        new_rel32 = target_va - (cave_cur_va + 5)
        return b"\xE9" + struct.pack("<i", new_rel32)

    # Short conditional jump: 70..7F rel8 -> 0F 80..8F rel32
    if 0x70 <= op0 <= 0x7F and length == 2:
        rel8 = struct.unpack("<b", bytes([b_list[1]]))[0]
        target_va = va + 2 + rel8
        new_rel32 = target_va - (cave_cur_va + 6)
        return bytes([0x0F, op0 + 0x10]) + struct.pack("<i", new_rel32)

    # Near call / jump: E8/E9 rel32
    if op0 in (0xE8, 0xE9) and length == 5:
        orig_rel32 = struct.unpack("<i", bytes(b_list[1:5]))[0]
        target_va = va + 5 + orig_rel32
        new_rel32 = target_va - (cave_cur_va + 5)
        return bytes([op0]) + struct.pack("<i", new_rel32)

    # Near conditional jump: 0F 80..8F rel32
    if length == 6 and op0 == 0x0F and 0x80 <= b_list[1] <= 0x8F:
        orig_rel32 = struct.unpack("<i", bytes(b_list[2:6]))[0]
        target_va = va + 6 + orig_rel32
        new_rel32 = target_va - (cave_cur_va + 6)
        return bytes([0x0F, b_list[1]]) + struct.pack("<i", new_rel32)

    # Non-relative i386 instruction: copy verbatim
    return bytes(b_list)


def disassemble_all(exe_path: str, objdump_cmd: str):
    out = subprocess.check_output([objdump_cmd, "-d", exe_path], text=True)
    insn_re = re.compile(r"^\s*([0-9a-fA-F]+):\s+((?:[0-9a-fA-F]{2}\s+)+)\s*(.*)$")
    target_re = re.compile(r"\b(?:j[a-z]+|call)\s+([0-9a-fA-F]+)\b")

    insns = []
    branch_targets = set()
    for line in out.splitlines():
        m = insn_re.match(line)
        if m:
            va = int(m.group(1), 16)
            raw_bytes = [int(x, 16) for x in m.group(2).split()]
            asm_str = m.group(3).strip()
            insns.append((va, raw_bytes, asm_str))
            tm = target_re.search(asm_str)
            if tm:
                branch_targets.add(int(tm.group(1), 16))
    return insns, branch_targets


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: sanitize_i586_pe.py <binary.exe> [objdump_cmd]")

    exe_path = sys.argv[1]
    objdump_cmd = sys.argv[2] if len(sys.argv) > 2 else "i686-w64-mingw32-objdump"

    insns, branch_targets = disassemble_all(exe_path, objdump_cmd)
    cmov_indices = [
        i for i, (_, _, asm_str) in enumerate(insns) if asm_str.startswith("cmov")
    ]

    if not cmov_indices:
        print(f"[OK] {exe_path}: 0 CMOV instructions found (100% pure Socket 7 i586!)")
        return

    with open(exe_path, "rb") as f:
        data = bytearray(f.read())

    image_base, vaddr, vsize, raw_ptr, raw_size, sec_hdr_off = parse_pe_text_section(data)
    text_start_va = image_base + vaddr

    # Locate embedded 512+ byte NOP cave in .text if present, else use .text tail
    text_slice = data[raw_ptr : raw_ptr + raw_size]
    nop_cave_idx = text_slice.find(b"\x90" * 512)
    if nop_cave_idx != -1:
        cave_file_off = raw_ptr + nop_cave_idx + 16
        cave_va = text_start_va + nop_cave_idx + 16
        cave_limit_off = raw_ptr + nop_cave_idx + 512
    else:
        cave_file_off = raw_ptr + vsize
        cave_va = text_start_va + vsize
        cave_limit_off = raw_ptr + raw_size

    patched_ranges = set()
    patched_count = 0

    for k in cmov_indices:
        if k in patched_ranges:
            continue

        start_idx = k
        end_idx = k
        total_len = len(insns[k][1])

        # Expand forward within the same contiguous block if < 5 bytes
        while total_len < 5 and end_idx + 1 < len(insns):
            cur_va, cur_b, cur_asm = insns[end_idx]
            if cur_asm.startswith("ret") or cur_asm.startswith("jmp"):
                break
            nxt_va, nxt_b, _ = insns[end_idx + 1]
            if nxt_va != cur_va + len(cur_b) or nxt_va in branch_targets:
                break
            end_idx += 1
            total_len += len(nxt_b)

        # Expand backward if still < 5 bytes
        while total_len < 5 and start_idx > 0:
            cur_va, _, _ = insns[start_idx]
            if cur_va in branch_targets:
                break
            prev_va, prev_b, prev_asm = insns[start_idx - 1]
            if prev_va + len(prev_b) != cur_va:
                break
            if prev_asm.startswith("ret") or prev_asm.startswith("jmp") or prev_asm.startswith("call"):
                break
            start_idx -= 1
            total_len += len(prev_b)

        if total_len < 5:
            raise RuntimeError(f"Could not form 5-byte detour window around {hex(insns[k][0])}")

        for idx in range(start_idx, end_idx + 1):
            patched_ranges.add(idx)

        win_start_va = insns[start_idx][0]
        win_end_va = insns[end_idx][0] + len(insns[end_idx][1])
        win_file_off = raw_ptr + (win_start_va - text_start_va)

        # Build relocated instruction sequence in the cave
        trampoline = bytearray()
        cur_cave_va = cave_va
        for idx in range(start_idx, end_idx + 1):
            iva, ibytes, _ = insns[idx]
            piece = relocate_instruction(iva, ibytes, cur_cave_va)
            trampoline.extend(piece)
            cur_cave_va += len(piece)

        # Append E9 rel32 jump back to win_end_va
        jmp_back_rel = win_end_va - (cur_cave_va + 5)
        trampoline.extend(b"\xE9" + struct.pack("<i", jmp_back_rel))

        if cave_file_off + len(trampoline) > cave_limit_off:
            raise RuntimeError("Trampoline cave capacity exceeded in .text")

        # Write 5-byte detour jump at win_start_va + NOP padding
        jmp_to_cave_rel = cave_va - (win_start_va + 5)
        detour_patch = b"\xE9" + struct.pack("<i", jmp_to_cave_rel) + b"\x90" * (total_len - 5)

        data[win_file_off : win_file_off + total_len] = detour_patch
        data[cave_file_off : cave_file_off + len(trampoline)] = trampoline

        cave_file_off += len(trampoline)
        cave_va += len(trampoline)
        patched_count += 1

    new_vsize = max(vsize, cave_va - text_start_va)
    if new_vsize > vsize:
        struct.pack_into("<I", data, sec_hdr_off + 8, new_vsize)

    with open(exe_path, "wb") as f:
        f.write(data)

    # Final verification pass via objdump
    post_insns, _ = disassemble_all(exe_path, objdump_cmd)
    rem_cmov = [asm for _, _, asm in post_insns if asm.startswith("cmov")]
    if rem_cmov:
        raise RuntimeError(f"Remaining CMOV instructions after patch: {rem_cmov}")

    print(f"[OK] {exe_path}: detour-patched {patched_count} CRT CMOV site(s) -> verified 0 CMOV remaining!")


if __name__ == "__main__":
    main()