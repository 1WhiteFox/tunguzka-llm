# Tunguzka-LLM (`RetroLLM-SIMD` v2.0)

**Tunguzka-LLM** — ультракомпактный C99/ASM движок инференса трансформеров (архитектуры LLaMA / SmolLM / BitNet b1.58), спроектированный специально для классических процессоров эпохи **Socket 7 / Super Socket 7** (**Intel Pentium Classic P54C**, **Intel Pentium MMX P55C**, **AMD K6-2 / K6-III 3DNow!**) под управлением **Windows 98**, **MS-DOS / FreeDOS** и **Linux**.

Движок не использует сторонних библиотек (даже `libm`), не содержит ни одной инструкции поколения `i686+` (`CMOV`, `FCOMI`, `SSE`), обходится без динамических аллокаций во время генерации токенов и способен запускать модели, размер которых превышает объём установленной оперативной памяти ПК, за счёт линейной послойной подкачки с диска (`Layer-Major Streaming`).

---

## Ключевые архитектурные особенности

* **4 аппаратных вычислительных бэкенда (`-b`):**
  * `3dnow` — **AMD K6-2 / K6-III / Athlon**: параллельная обработка по 2 строки матрицы в регистрах MMX/3DNow!, аппаратное упреждающее чтение кэша под 32-байтную линию L1 (`prefetch 32`), перевод `int32 -> float` без участия x87 (`pi2fd`), векторное слияние скейлов (`pfmul`, `pfadd`, `pfacc`), быстрый `RMSNorm` через `pfrsqrt + pfrsqit1 + pfrcpit2` и мгновенный сброс `femms`.
  * `mmx` — **Intel Pentium MMX (P55C)**: спаренные под U/V-конвейеры P55C ядра `pmaddwd`, 2-килобайтная таблица `g_ternary_lut` в L1D-кэше, распаковка `Q4` без LUT на битовых сдвигах и батчинг по 16 строк на один барьер `emms`.
  * `p54c` — **Intel Pentium Classic (P54C, без MMX)**: двухпоточные LUT/SWAR-ядра в 32-битных регистрах общего назначения.
  * `scalar` — эталонный переносимый движок на чистом C99.
* **Формат картриджей `.mmx v2`:**
  * **Линейная послойная укладка (`MODEL_FLAG_LAYER_MAJOR`):** внутри каждого слоя тензоры записаны строго в порядке вызова (`wq -> wk -> wv -> wo -> w1 -> w3 -> w2`), что устраняет холостые перемещения головки HDD при стриминге с диска.
  * **Контроль целостности (`CRC32`):** 64-байтный заголовок защищён контрольной суммой IEEE 802.3 CRC32 и хранит точный `rms_norm_eps` модели и тип активации (`SwiGLU` или `ReLU²`).
  * **4 профиля квантизации весов:** `binary` (1-bit), `ternary` (1.58-bit BitNet), `q4` (4-bit), `q8` (8-bit) + гибридный режим Q8 для эмбеддингов и классификатора.
* **Качество генерации и бесконечный диалог:**
  * **Zero-Alloc Top-K + Top-P (Nucleus) сэмплинг:** отбирает $K$ лучших кандидатов на стеке, вычисляя экспоненты `softmax` только для верхушки распределения.
  * **Скользящее окно контекста с фазовой коррекцией RoPE:** при заполнении контекста вторая половина `KV-Cache` (как `FP32`, так и `INT8`) сдвигается в начало с доворотом ключей $K$ на угол $-\Delta\text{pos}$.
* **Ретро-TUI интерфейс (CP866 / ANSI UTF-8):**
  * Двуязычный интерфейс (RU/EN), автоматическая транслитерация Юникод-типографики в CP866 без артефактов `?`, прокрутка истории чата (`PgUp`/`PgDn`) и панель настройки параметров генерации и выбора моделей по `F9`.

---

## Быстрый старт и сборка

```bash
# Сборка под хост (Linux / WSL) и прогон автотестов
make test

# Запуск интерактивного TUI-интерфейса
make tui

# Сборка чистых Socket 7 PE-бинарников для Windows 98 (с вычисткой CMOV из CRT)
make win98

# Скачивание и упаковка TinyStories-260K в автономный .exe для AMD K6-2
make k6-2-3dnow-260K

# Конвертация любой модели BitNet / LLaMA / SmolLM напрямую из HuggingFace (поддерживаются SafeTensors и tokenizer.json)
python3 tools/convert_to_mmx.py --hf 1bitLLM/bitnet_b1_58-large -o bitnet.mmx -q ternary

# Инспекция готового .mmx файла и проверка CRC32 заголовка
python3 tools/convert_to_mmx.py --info bitnet.mmx

# Сборка single-click .exe (вшивание модели, словаря и конфига внутрь исполняемого файла)
python3 tools/tunguzka_forge.py -m bitnet.mmx -e tunguzka_3dnow.exe --out-exe bitnet_k6.exe --backend 3dnow --kv8


tunguska-llm/
├── include/
│   ├── tunguzka.h           # Контракт данных, заголовок .mmx v2 (64 байта), быстрая математика и структуры
│   ├── mmx_kernels.h        # Интерфейс SIMD-ядер Intel MMX (P55C)
│   ├── 3dnow_kernels.h      # Интерфейс SIMD-ядер AMD 3DNow! (K6-2/K6-III)
│   └── tui.h                # Константы и структуры текстового интерфейса
├── src/
│   ├── ops_scalar.c         # CRC32, RMSNorm, Softmax, RoPE, эталонные скалярные и P54C ядра
│   ├── kernels_mmx.c        # SIMD-ядра MMX и безопасные целочисленные фоллбэки
│   ├── kernels_3dnow.c      # SIMD-ядра 3DNow! (PFRSQRT, PI2FD, PREFETCH 32)
│   ├── model.c              # Загрузчик .mmx v2, Arena, Layer-Major Stream, LRU (64 слота) и сдвиг RoPE KV-кэша
│   ├── tui.c                # Движок TUI (Win32 Console / DOS 0xB8000 / POSIX ANSI) и транслитерация CP866
│   └── main.c               # CLI, TUI-сессия, BPE-токенизатор и сэмплер Top-K / Top-P
├── tools/
│   ├── convert_to_mmx.py    # Универсальный конвертер SafeTensors / .bin / tokenizer.json -> .mmx v2 и инспектор --info
│   ├── tunguzka_forge.py    # Сборщик автономных .exe и ретро-дистрибутивов для 86Box / реального железа
│   └── sanitize_i586_pe.py  # Дизассемблерный верификатор и патчер i686 CMOV -> i586 для Win98
└── Makefile                 # Цели сборки под Linux, Pentium Classic, Pentium MMX, K6-2, Win98 и DOS