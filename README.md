# Tunguzka-LLM (`RetroLLM-SIMD` v2.0) by staticfox15

> **Bare-metal C99 / x86 ASM LLM inference engine engineered specifically for classic Socket 7 and Super Socket 7 processors (Intel Pentium Classic, Pentium MMX, AMD K6-2 / K6-III 3DNow!) under MS-DOS, Windows 98, and Linux.**

[![Language](https://img.shields.io/badge/Language-C99%20%2F%20x86%20ASM-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Platform](https://img.shields.io/badge/Platform-i586%20%2F%20Socket%207-orange.svg)](https://en.wikipedia.org/wiki/Pentium_(original))
[![Supported OS](https://img.shields.io/badge/OS-MS--DOS%20(DJGPP)%20%7C%20Windows%2098%20%7C%20Linux-green.svg)](https://en.wikipedia.org/wiki/MS-DOS)
[![SIMD](https://img.shields.io/badge/SIMD-MMX%20%7C%203DNow!-purple.svg)](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data)
[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](LICENSE)

---

## What is Tunguzka-LLM?

**Tunguzka-LLM** is an ultra-compact, zero-dependency Transformer inference engine (supporting LLaMA, SmolLM, and BitNet b1.58 architectures) crafted from scratch specifically for late-1990s retro PC hardware.

The engine does not link against third-party math runtimes (including `libm`), completely avoids dynamic memory allocations during token generation (using a static pre-allocated `MemoryArena`), and eliminates every opcode newer than 1997 ($i686+$ opcodes such as `CMOV`, `FCOMI`, `SSE`). Thanks to **Layer-Major Disk Streaming**, it can run models whose physical footprints vastly exceed the machine's installed RAM capacity (operating on systems with just 16–32 MB of RAM).

```text
 ┌─Tunguzka-LLM v2.0 ─────────────────────────────────────────────────────────┐
 │ Backend: 3DNow![K6-2] | Mode: 1.58-bit | MEM: STREAM | T:0.7 K:40          │
 ├────────────────────────┬─────────────────────────┬─────────────────────────┤
 │ Arena: 1.2M/16M        │ [HDD: READ L04]         │ Speed: 1.4 tok/s        │
 │ KV(I8): 512K [42/256]  │ LRU: 58/64 (90%)        │ Latency: 714.28 ms      │
 ├────────────────────────┴─────────────────────────┴─────────────────────────┤
 │ You > Once upon a time in a retro computer lab...                          │
 │ Tunguzka > there was an old Socket 7 motherboard running code without SSE. │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ > Type your prompt here..._                                                │
 ├────────────────────────────────────────────────────────────────────────────┤
 │ Progress: [████████████░░░░░░░░] 64/128 tok | chk: 0.4129                  │
 └─ [1 Hlp] [2 Bck] [3 KV8] [4 Str] [5 Rst] [8 Lng] [10 Qut] ─────────────────┘
```

---

## Project Status, Disclaimer & Transparency

* **Experimental Status & No Warranty:** This project is developed purely as an exploratory hobby in my spare time. No guarantees regarding runtime stability, generation precision, or flawless operation across the vast variety of vintage motherboards, chipsets, and retro BIOSes can be provided. Everything is distributed strictly **"AS-IS"** — run it at your own risk.
* **Non-CS Background:** The author is an enthusiast from a non-engineering field (business informatics) rather than a veteran low-level systems engineer or compiler developer. I cannot promise much: development happens solely according to available energy, inspiration, and free time. Please do not expect 24/7 technical support, instant issue triage, or coverage of all exotic edge cases.
* **Roadmap & Updates:** A couple of focused maintenance updates are planned before the end of the year — targeting streaming stability, bottle-neck fixes, and minor optimizations.
* **AI Transparency:** The overall runtime architecture, the Layer-Major disk streaming concept, the `.mmx v2` cartridge layout, and platform engineering solutions were conceived and driven by the author. During code drafting, inline assembly syntax formatting, and edge-case C99 debugging, generative AI (LLMs) was actively and openly utilized as an interactive pair-programming companion.

---

## Key Architectural Highlights

### 1. Four Hardware Compute Backends (`-b`)
* **`3dnow` (AMD K6-2 / K6-III / early Athlon):**
  * Parallel 2-row SIMD matrix multiplication utilizing MMX registers and 3DNow! instructions.
  * Hardware cache line prefetching for 32-byte L1 lines (`prefetch 32`).
  * Direct vector integer-to-float conversions bypassing the slower x87 FPU (`pi2fd`).
  * Vector scaling and accumulation via `pfmul`, `pfadd`, and horizontal accumulation `pfacc`.
  * Hardware-accelerated reciprocal square root `RMSNorm` pipelined via `pfrsqrt`, `pfrsqit1`, and `pfrcpit2`.
  * Instant FPU/MMX register tag state clearance using zero-cycle `femms` without penalty cycles.
* **`mmx` (Intel Pentium MMX / P55C):**
  * Paired `pmaddwd` SIMD kernels balanced for the dual-issue superscalar (U/V) pipeline of the P55C.
  * 2 KB precomputed ternary lookup table (`g_ternary_lut`), guaranteed to fit entirely within L1D cache.
  * 4-bit nibble unpacking for Q4 weights using bitwise shifts and masks without extra memory lookups.
  * Row batching in 16-row tiles to amortize the overhead of the `emms` state-transition barrier.
* **`p54c` (Intel Pentium Classic, non-MMX CPUs):**
  * Dual-stream SWAR algorithms and integer lookup tables operating exclusively inside 32-bit General Purpose Registers (`EAX`–`EDX`).
* **`scalar`:**
  * Portable reference C99 backend for cross-platform debugging and numeric validation.

### 2. Cartridge File Format (`.mmx v2`)
* **Layer-Major Tensor Layout:** Within each transformer block, tensors are serialized strictly in the order of their forward-pass invocation (`wq -> wk -> wv -> wo -> w1 -> w3 -> w2`). This guarantees sequential disk reads and eliminates mechanical head thrashing on legacy IDE drives during streaming.
* **Integrity Guarantee:** A 64-byte binary header verified by an IEEE 802.3 CRC32 checksum, storing the exact normalization epsilon parameter $\varepsilon$, embedded tokenizer offsets, and activation flags (`SwiGLU` or $\text{ReLU}^2$).
* **Quantization Profiles:**
  * `binary`: 1.0-bit packed weights (8 weights per byte).
  * `ternary`: 1.58-bit BitNet b1.58 weights (2 bits per code, 4 codes per byte).
  * `q4`: 4-bit symmetric quantization with offset 8 (2 weights per byte).
  * `q8`: 8-bit symmetric signed integer quantization.
  * Hybrid mode retaining token embeddings and classifier projection matrices in Q8 to minimize perplexity degradation.

### 3. Memory & Caching Subsystems
* **Zero-Alloc Stack Sampling:** Combined Top-K and Nucleus (Top-P) candidate filtering performed entirely within fixed stack arrays without a single heap allocation (`malloc`).
* **Sliding Window KV-Cache:** When the sequence context limit is reached, the latter half of the cache shifts forward, while key vectors $K$ are analytically rotated by the phase angle $-\Delta\text{pos}$ using precomputed RoPE frequencies, preserving attention coherence.
* **INT8 KV-Cache (`--kv8`):** Decreases KV cache memory footprint by 75% compared to FP32 through dynamic per-head scaling factors.

### 4. MinGW CRT Opcode Sanitization (`tools/sanitize_i586_pe.py`)
Modern MinGW-w64 toolchains inject conditional move instructions (`CMOVcc`) into CRT startup code even when `-march=pentium` is specified. Tunguzka includes an automated disassembler and binary patcher: it locates an embedded NOP cave within the `.text` section (`__tz_i586_trampoline_cave`) and replaces unsupported `CMOV` instructions with safe i586 jump-and-branch trampolines, guaranteeing execution on authentic Socket 7 hardware.

---

## Quantization Matrix

| Profile | Weight Bitwidth | Packaging Density | Primary Target CPU | Primary Kernel Path |
| :--- | :---: | :---: | :---: | :--- |
| **`binary`** | 1.0-bit | 8 weights / byte | Pentium Classic (P54C) | `matmul_binary_p54c` |
| **`ternary`** | 1.58-bit | 4 weights / byte | Pentium MMX (P55C) | `matmul_ternary_mmx` |
| **`q4`** | 4.0-bit | 2 weights / byte | AMD K6-2 (3DNow!) | `matmul_q4_3dnow` |
| **`q8`** | 8.0-bit | 1 weight / byte | Universal / All x86 CPUs | `matmul_q8_3dnow` / `_mmx` |

---

## Building & Cross-Compilation

### Environment Requirements
* **Linux / WSL:** `gcc`, `make`. For 32-bit compilation verification: `gcc-multilib` package.
* **Windows 98 Target:** `i686-w64-mingw32-gcc` and `i686-w64-mingw32-objdump`.
* **MS-DOS Target:** DJGPP toolchain (`i586-pc-msdosdjgpp-gcc`).

### Primary Makefile Targets

```bash
# 1. Native host compilation and automated test suite run
make test

# 2. Launch interactive retro terminal UI
make tui

# 3. Compile clean PE binaries for Windows 98 (auto-patched, 0 CMOV opcodes)
make win98

# 4. Download TinyStories-260K, quantize to Q8, and assemble standalone bundle for AMD K6-2
make k6-2-3dnow-260K

# 5. Compile 32-bit DPMI executable for MS-DOS via DJGPP
make dos
```

---

## Model Conversion & Standalone Bundling

### Converting Weights from HuggingFace
Any compatible LLaMA, SmolLM, or BitNet model in SafeTensors format can be quantized directly into a `.mmx v2` cartridge:

```bash
# Quantize BitNet b1.58 from HuggingFace into a ternary cartridge
python3 tools/convert_to_mmx.py --hf 1bitLLM/bitnet_b1_58-large -o bitnet.mmx -q ternary

# Inspect cartridge metadata and verify header CRC32
python3 tools/convert_to_mmx.py --info bitnet.mmx
```

### Packaging Single-Click Executables
The `tools/tunguzka_forge.py` script merges the compiled executable, `.mmx` weights, tokenizer vocabulary, and a 64-byte binary runtime config with a `TZPK` trailer into a single portable binary:

```bash
python3 tools/tunguzka_forge.py \
    -m bitnet.mmx \
    -e tunguzka_3dnow.exe \
    --out-exe tunguzka_k6_bitnet.exe \
    --backend 3dnow \
    --kv8 \
    --ctx 256 \
    --steps 128 \
    --lang ru
```

---

## Interactive TUI Controls

The retro text user interface operates in standard 80x25 text mode (via VGA memory at `0xB800` in DOS, Win32 Console API, or ANSI terminal sequences):

* **`F1`**: Display help and keybindings reference.
* **`F2`**: Cycle active hardware compute backend (`3DNow!` -> `MMX` -> `P54C` -> `Scalar`).
* **`F3`**: Toggle KV cache quantization mode (INT8 / FP32 KV-Cache).
* **`F4`**: Toggle memory allocation mode (RAM-Resident / Layer-Major Disk Streaming).
* **`F5`**: Reset sequence context to position zero.
* **`F6`**: Cycle maximum sequence context window size.
* **`F7`**: Step sampling temperature preset (`0.0`, `0.5`, `0.8`, `1.0`).
* **`F8`**: Toggle interface language (RU/EN with native Unicode typography transliteration to CP866).
* **`F9`**: Configuration settings menu and local `.mmx` cartridge browser.
* **`PgUp` / `PgDn`**: Scroll conversation history buffer.
* **`ESC` / `F10`**: Abort active token generation or exit the application.

---

## License & Acknowledgments

* Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
* Conceptual inspiration: [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy.
* Grateful acknowledgment to the **86Box**, **PCem**, and **Vogons** retro computing communities for preserving exhaustive low-level hardware documentation for the Socket 7 platform.