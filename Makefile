.RECIPEPREFIX = >
CC         ?= gcc
WIN32_CC   ?= i686-w64-mingw32-gcc
WIN32_OBJD ?= i686-w64-mingw32-objdump
DOS_CC     ?= i586-pc-msdosdjgpp-gcc

SRCS       = src/ops_scalar.c src/kernels_mmx.c src/kernels_3dnow.c src/model.c src/tui.c src/main.c
CFLAGS_STD = -std=c99 -Wall -Wextra -Iinclude -fno-builtin

RETRO_P54C_FLAGS  = -march=pentium -mtune=pentium -mno-mmx -mno-3dnow -mno-sse -mno-sse2
RETRO_MMX_FLAGS   = -march=pentium-mmx -mtune=pentium-mmx -mmmx -mno-3dnow -mno-sse -mno-sse2
RETRO_3DNOW_FLAGS = -march=k6-2 -mtune=k6-2 -m3dnow -mno-sse -mno-sse2
WIN98_DEFS        = -D__USE_MINGW_ANSI_STDIO=0

.PHONY: all host host-scalar check-m32 x86-p54c x86-mmx k6-3dnow win98 win98-p54c win98-mmx win98-3dnow bundle k6-2-3dnow-260K dos test tui clean

all: host

host: $(SRCS)
> $(CC) $(CFLAGS_STD) -O3 -mmmx $(SRCS) -o tunguzka

host-scalar: $(SRCS)
> $(CC) $(CFLAGS_STD) -O2 $(SRCS) -o tunguzka

check-m32:
> @printf "#include <stdint.h>\nint main(void){return 0;}\n" | $(CC) -m32 -x c - -o /dev/null 2>/dev/null || \
  (echo "[SKIP/ERROR] 32-bit Linux headers not found for 'gcc -m32'. Run: sudo apt install gcc-multilib" && exit 1)

x86-p54c: check-m32 $(SRCS)
> $(CC) $(CFLAGS_STD) -m32 $(RETRO_P54C_FLAGS) -O3 $(SRCS) -o tunguzka_p54c
> @! objdump -d tunguzka_p54c | grep -E '\bcmov|\bfcomi|\bfucomi|\bmm[0-7]|\bxmm' || (echo "[ERROR] Forbidden opcode found!" && exit 1)
> @echo "[OK] tunguzka_p54c: 100% pure Pentium Classic P54C (zero MMX/CMOV/SSE)"

x86-mmx: check-m32 $(SRCS)
> $(CC) $(CFLAGS_STD) -m32 $(RETRO_MMX_FLAGS) -O3 $(SRCS) -o tunguzka_mmx
> @! objdump -d tunguzka_mmx | grep -E '\bcmov|\bfcomi|\bfucomi|\bxmm' || (echo "[ERROR] Forbidden i686+ opcode found!" && exit 1)
> @echo "[OK] tunguzka_mmx: 100% pure Pentium MMX (zero CMOV/SSE)"

k6-3dnow: check-m32 $(SRCS)
> $(CC) $(CFLAGS_STD) -m32 $(RETRO_3DNOW_FLAGS) -O3 $(SRCS) -o tunguzka_3dnow
> @! objdump -d tunguzka_3dnow | grep -E '\bcmov|\bfcomi|\bfucomi|\bxmm' || (echo "[ERROR] Forbidden i686+ opcode found!" && exit 1)
> @echo "[OK] tunguzka_3dnow: 100% pure AMD K6-2 3DNow! (zero CMOV/SSE)"

win98: win98-p54c win98-mmx win98-3dnow

win98-p54c: $(SRCS)
> $(WIN32_CC) $(CFLAGS_STD) $(WIN98_DEFS) $(RETRO_P54C_FLAGS) -O3 -s $(SRCS) -o tunguzka_p54c.exe
> @python3 tools/sanitize_i586_pe.py tunguzka_p54c.exe "$(WIN32_OBJD)"

win98-mmx: $(SRCS)
> $(WIN32_CC) $(CFLAGS_STD) $(WIN98_DEFS) $(RETRO_MMX_FLAGS) -O3 -s $(SRCS) -o tunguzka_mmx.exe
> @python3 tools/sanitize_i586_pe.py tunguzka_mmx.exe "$(WIN32_OBJD)"

win98-3dnow: $(SRCS)
> $(WIN32_CC) $(CFLAGS_STD) $(WIN98_DEFS) $(RETRO_3DNOW_FLAGS) -O3 -s $(SRCS) -o tunguzka_3dnow.exe
> @python3 tools/sanitize_i586_pe.py tunguzka_3dnow.exe "$(WIN32_OBJD)"

k6-2-3dnow-260K: host win98-3dnow
> python3 tools/tunguzka_forge.py --fetch stories260k -q q8 -m stories260k.mmx -e tunguzka_3dnow.exe --dist-dir dist_3dnow --out-exe dist_3dnow/tunguzka_k6_single.exe --backend 3dnow --lang ru --ctx 256 --steps 128
> @echo "=== Host Verification on stories260k.mmx (MMX vs Scalar, 128 tokens) ==="
> ./tunguzka -m stories260k.mmx -p "Once upon a time" -n 128 -b scalar
> ./tunguzka -m stories260k.mmx -p "Once upon a time" -n 128 -b mmx

bundle: host win98
> @test -f test_ternary.mmx || python3 tools/convert_to_mmx.py --dummy --dummy-tok test_dummy_tok.bin -o test_ternary.mmx -q ternary
> python3 tools/tunguzka_forge.py -m test_ternary.mmx -z test_dummy_tok.bin -e tunguzka_mmx.exe --out-exe tunguzka_standalone_win98.exe --lang ru --ctx 64
> python3 tools/tunguzka_forge.py -m test_ternary.mmx -z test_dummy_tok.bin -e ./tunguzka --out-exe tunguzka_standalone --lang ru --ctx 64

dos: $(SRCS)
> @command -v $(DOS_CC) >/dev/null 2>&1 || (echo "[SKIP/ERROR] DJGPP '$(DOS_CC)' not found. Use 'make win98' for Win98 MS-DOS Prompt." && exit 1)
> $(DOS_CC) $(CFLAGS_STD) $(RETRO_MMX_FLAGS) -O3 -s $(SRCS) -o tunguzka.exe

test: host
> python3 tools/convert_to_mmx.py --dummy --dummy-tok test_dummy_tok.bin -o test_ternary.mmx -q ternary
> python3 tools/convert_to_mmx.py --dummy -o test_q8.mmx -q q8
> python3 tools/convert_to_mmx.py --dummy -o test_q4.mmx -q q4
> python3 tools/convert_to_mmx.py --info test_ternary.mmx
> @echo "=== Testing Ternary (RAM + Stream + KV8 + Sliding Window) ==="
> ./tunguzka -m test_ternary.mmx -z test_dummy_tok.bin -n 16 -b mmx
> ./tunguzka -m test_ternary.mmx -z test_dummy_tok.bin -n 16 -b mmx --stream
> ./tunguzka -m test_ternary.mmx -z test_dummy_tok.bin -c 16 -n 24 -b mmx --kv8
> @echo "=== Testing Q8 ==="
> ./tunguzka -m test_q8.mmx -z test_dummy_tok.bin -n 16 -b mmx
> @echo "=== Testing Q4 ==="
> ./tunguzka -m test_q4.mmx -z test_dummy_tok.bin -n 16 -b mmx

tui: host
> @test -f test_ternary.mmx || python3 tools/convert_to_mmx.py --dummy --dummy-tok test_dummy_tok.bin -o test_ternary.mmx -q ternary
> ./tunguzka --tui --ru

clean:
> rm -f tunguzka tunguzka_p54c tunguzka_mmx tunguzka_3dnow tunguzka_standalone* tunguzka*.exe test_*.mmx test_dummy_tok.bin TUNGUZKA.CFG