#include "../include/mmx_kernels.h"

int16_t g_ternary_lut[256][4]      TZ_ALIGNED(64);
int16_t g_binary_nibble_lut[16][4] TZ_ALIGNED(64);
static int g_luts_initialized = 0;

#define TZ_EMMS_BARRIER() \
    __asm__ volatile ("emms" : : : "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory")

void mmx_init_luts(void) {
    if (g_luts_initialized) return;
    for (int idx = 0; idx < 256; idx++) {
        for (int k = 0; k < 4; k++) {
            int code = (idx >> (k * 2)) & 0x03;
            g_ternary_lut[idx][k] = (code == 1) ? 1 : ((code == 2) ? -1 : 0);
        }
    }
    for (int n = 0; n < 16; n++) {
        for (int k = 0; k < 4; k++) {
            g_binary_nibble_lut[n][k] = ((n >> k) & 1) ? 1 : -1;
        }
    }
    p54c_init_luts();
    g_luts_initialized = 1;
}

int cpu_has_mmx(void) {
#if defined(__x86_64__) || defined(_M_X64)
    return 1;
#elif defined(__i386__) && (defined(__GNUC__) || defined(__clang__))
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1));
    return (edx & (1u << 23)) ? 1 : 0;
#else
    return 0;
#endif
}

static void fallback_binary_i16(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16,
                                const float * TZ_RESTRICT x_scales, const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows, cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 8;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 8;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int16_t *gx = xq16 + g * group_size;
            int32_t acc = 0;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t p = gb[b];
                for (uint32_t k = 0; k < 8; k++) acc += ((p >> k) & 1) ? (int32_t)gx[b * 8 + k] : -(int32_t)gx[b * 8 + k];
            }
            row_sum += (float)acc * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

static void fallback_ternary_i16(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16,
                                 const float * TZ_RESTRICT x_scales, const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows, cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 4;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 4;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int16_t *gx = xq16 + g * group_size;
            int32_t acc = 0;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                const int16_t *lut = g_ternary_lut[gb[b]];
                acc += (int32_t)lut[0] * gx[b * 4 + 0] + (int32_t)lut[1] * gx[b * 4 + 1]
                     + (int32_t)lut[2] * gx[b * 4 + 2] + (int32_t)lut[3] * gx[b * 4 + 3];
            }
            row_sum += (float)acc * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

static void fallback_q8_i16(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16,
                            const float * TZ_RESTRICT x_scales, const QuantizedTensor *w, uint32_t group_size) {
    const int8_t *w_data = (const int8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows, cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const int8_t *row_w = w_data + (size_t)r * cols;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const int8_t *gw = row_w + g * group_size;
            const int16_t *gx = xq16 + g * group_size;
            int32_t acc = 0;
            for (uint32_t i = 0; i < group_size; i++) acc += (int32_t)gw[i] * (int32_t)gx[i];
            row_sum += (float)acc * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

static void fallback_q4_i16(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16,
                            const float * TZ_RESTRICT x_scales, const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows, cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 2;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 2;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int16_t *gx = xq16 + g * group_size;
            int32_t acc = 0;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t packed = gb[b];
                acc += ((int32_t)(packed & 0x0F) - 8) * gx[b * 2 + 0]
                     + ((int32_t)(packed >> 4)   - 8) * gx[b * 2 + 1];
            }
            row_sum += (float)acc * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

#if (defined(__i386__) || defined(__x86_64__)) && defined(__MMX__)

static const uint64_t g_q4_mask_0f TZ_ALIGNED(8) = 0x0F0F0F0F0F0F0F0FULL;
static const uint64_t g_q4_bias_08 TZ_ALIGNED(8) = 0x0808080808080808ULL;

void unpack_xq_to_i16_mmx(const int8_t * TZ_RESTRICT xq,
                          int16_t * TZ_RESTRICT xq16,
                          uint32_t cols) {
    for (uint32_t i = 0; i < cols; i += 8) {
        const int8_t *src = xq + i;
        int16_t *dst = xq16 + i;
        __asm__ volatile (
            "movq (%[src]), %%mm2\n\t"
            "movq %%mm2, %%mm0\n\t"
            "movq %%mm2, %%mm1\n\t"
            "punpcklbw %%mm0, %%mm0\n\t"
            "punpckhbw %%mm1, %%mm1\n\t"
            "psraw $8, %%mm0\n\t"
            "psraw $8, %%mm1\n\t"
            "movq %%mm0, (%[dst])\n\t"
            "movq %%mm1, 8(%[dst])\n\t"
            : : [src] "r"(src), [dst] "r"(dst) : "mm0", "mm1", "mm2", "memory"
        );
    }
    TZ_EMMS_BARRIER();
}

int32_t dot_q8_i16_mmx(const int16_t * TZ_RESTRICT q16,
                       const int8_t  * TZ_RESTRICT k8,
                       uint32_t head_size) {
    int32_t sum = 0;
    uint32_t i = 0;
    __asm__ volatile ("pxor %%mm6, %%mm6\n\t" : : : "mm6");
    for (; i + 7 < head_size; i += 8) {
        const int16_t *qp = q16 + i;
        const int8_t  *kp = k8 + i;
        __asm__ volatile (
            "movq (%[k]), %%mm2\n\t"
            "movq %%mm2, %%mm3\n\t"
            "punpcklbw %%mm2, %%mm2\n\t"
            "punpckhbw %%mm3, %%mm3\n\t"
            "psraw $8, %%mm2\n\t"
            "psraw $8, %%mm3\n\t"
            "pmaddwd (%[q]), %%mm2\n\t"
            "pmaddwd 8(%[q]), %%mm3\n\t"
            "paddd %%mm2, %%mm6\n\t"
            "paddd %%mm3, %%mm6\n\t"
            : : [q] "r"(qp), [k] "r"(kp) : "mm2", "mm3", "mm6"
        );
    }
    __asm__ volatile (
        "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s]\n\t"
        : [s] "=r"(sum) : : "mm0", "mm6"
    );
    TZ_EMMS_BARRIER();
    for (; i < head_size; i++) sum += (int32_t)q16[i] * (int32_t)k8[i];
    return sum;
}

TZ_INLINE void lut_step_2rows_mmx(const int16_t *x_ptr,
                                  const int16_t *l0a, const int16_t *l0b,
                                  const int16_t *l1a, const int16_t *l1b) {
    __asm__ volatile (
        "movq (%[x]), %%mm0\n\t"
        "movq 8(%[x]), %%mm1\n\t"
        "movq (%[l0a]), %%mm2\n\t"
        "movq (%[l0b]), %%mm3\n\t"
        "pmaddwd %%mm0, %%mm2\n\t"
        "pmaddwd %%mm1, %%mm3\n\t"
        : : [x] "r"(x_ptr), [l0a] "r"(l0a), [l0b] "r"(l0b) : "mm0", "mm1", "mm2", "mm3"
    );
    __asm__ volatile (
        "movq (%[l1a]), %%mm4\n\t"
        "movq (%[l1b]), %%mm5\n\t"
        "pmaddwd %%mm0, %%mm4\n\t"
        "paddd %%mm2, %%mm6\n\t"
        "pmaddwd %%mm1, %%mm5\n\t"
        "paddd %%mm3, %%mm6\n\t"
        "paddd %%mm4, %%mm7\n\t"
        "paddd %%mm5, %%mm7\n\t"
        : : [l1a] "r"(l1a), [l1b] "r"(l1b) : "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"
    );
}

static void flush_tile_scales_fpu(float * TZ_RESTRICT out,
                                  const int32_t * TZ_RESTRICT acc_buf,
                                  const float * TZ_RESTRICT w_scales,
                                  const float * TZ_RESTRICT x_scales,
                                  uint32_t r_start, uint32_t r_count,
                                  uint32_t groups_per_row) {
    for (uint32_t tr = 0; tr < r_count; tr++) {
        uint32_t r = r_start + tr;
        const int32_t *row_acc = acc_buf + (size_t)tr * groups_per_row;
        const float   *ws      = w_scales + (size_t)r * groups_per_row;
        float row_sum = 0.0f;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            row_sum += (float)row_acc[g] * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

void matmul_binary_mmx(float * TZ_RESTRICT out,
                       const int16_t * TZ_RESTRICT xq16,
                       const float   * TZ_RESTRICT x_scales,
                       int32_t       * TZ_RESTRICT acc_buf,
                       const QuantizedTensor *w,
                       uint32_t group_size) {
    if (!acc_buf) { fallback_binary_i16(out, xq16, x_scales, w, group_size); return; }
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 8;
    for (uint32_t r_base = 0; r_base < rows; r_base += TUNGUZKA_MMX_TILE_R) {
        uint32_t tile_rows = rows - r_base;
        if (tile_rows > TUNGUZKA_MMX_TILE_R) tile_rows = TUNGUZKA_MMX_TILE_R;
        uint32_t tr = 0;
        for (; tr + 1 < tile_rows; tr += 2) {
            uint32_t r = r_base + tr;
            const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 8;
            const uint8_t *row1_bytes = row0_bytes + cols / 8;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            int32_t *acc_r1 = acc_r0 + groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const uint8_t *gb0 = row0_bytes + g * bytes_per_group;
                const uint8_t *gb1 = row1_bytes + g * bytes_per_group;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0, sum1;
                __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                for (uint32_t b = 0; b < bytes_per_group; b++) {
                    uint8_t p0 = gb0[b], p1 = gb1[b];
                    lut_step_2rows_mmx(gx16 + b * 8,
                                       g_binary_nibble_lut[p0 & 0x0F], g_binary_nibble_lut[p0 >> 4],
                                       g_binary_nibble_lut[p1 & 0x0F], g_binary_nibble_lut[p1 >> 4]);
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    "movq %%mm7, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm7\n\tmovd %%mm7, %[s1]\n\t"
                    : [s0] "=r"(sum0), [s1] "=r"(sum1) : : "mm0", "mm1", "mm6", "mm7"
                );
                acc_r0[g] = sum0;
                acc_r1[g] = sum1;
            }
        }
        for (; tr < tile_rows; tr++) {
            uint32_t r = r_base + tr;
            const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 8;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const uint8_t *gb0 = row0_bytes + g * bytes_per_group;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0;
                __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                for (uint32_t b = 0; b < bytes_per_group; b++) {
                    uint8_t p0 = gb0[b];
                    lut_step_2rows_mmx(gx16 + b * 8,
                                       g_binary_nibble_lut[p0 & 0x0F], g_binary_nibble_lut[p0 >> 4],
                                       g_binary_nibble_lut[0], g_binary_nibble_lut[0]);
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    : [s0] "=r"(sum0) : : "mm0", "mm6"
                );
                acc_r0[g] = sum0;
            }
        }
        TZ_EMMS_BARRIER();
        flush_tile_scales_fpu(out, acc_buf, w_scales, x_scales, r_base, tile_rows, groups_per_row);
    }
}

void matmul_ternary_mmx(float * TZ_RESTRICT out,
                        const int16_t * TZ_RESTRICT xq16,
                        const float   * TZ_RESTRICT x_scales,
                        int32_t       * TZ_RESTRICT acc_buf,
                        const QuantizedTensor *w,
                        uint32_t group_size) {
    if (!acc_buf) { fallback_ternary_i16(out, xq16, x_scales, w, group_size); return; }
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 4;
    for (uint32_t r_base = 0; r_base < rows; r_base += TUNGUZKA_MMX_TILE_R) {
        uint32_t tile_rows = rows - r_base;
        if (tile_rows > TUNGUZKA_MMX_TILE_R) tile_rows = TUNGUZKA_MMX_TILE_R;
        uint32_t tr = 0;
        for (; tr + 1 < tile_rows; tr += 2) {
            uint32_t r = r_base + tr;
            const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 4;
            const uint8_t *row1_bytes = row0_bytes + cols / 4;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            int32_t *acc_r1 = acc_r0 + groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const uint8_t *gb0 = row0_bytes + g * bytes_per_group;
                const uint8_t *gb1 = row1_bytes + g * bytes_per_group;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0, sum1;
                __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                if (bytes_per_group == 8) {
                    lut_step_2rows_mmx(gx16 +  0, g_ternary_lut[gb0[0]], g_ternary_lut[gb0[1]], g_ternary_lut[gb1[0]], g_ternary_lut[gb1[1]]);
                    lut_step_2rows_mmx(gx16 +  8, g_ternary_lut[gb0[2]], g_ternary_lut[gb0[3]], g_ternary_lut[gb1[2]], g_ternary_lut[gb1[3]]);
                    lut_step_2rows_mmx(gx16 + 16, g_ternary_lut[gb0[4]], g_ternary_lut[gb0[5]], g_ternary_lut[gb1[4]], g_ternary_lut[gb1[5]]);
                    lut_step_2rows_mmx(gx16 + 24, g_ternary_lut[gb0[6]], g_ternary_lut[gb0[7]], g_ternary_lut[gb1[6]], g_ternary_lut[gb1[7]]);
                } else {
                    for (uint32_t b = 0; b < bytes_per_group; b += 2) {
                        lut_step_2rows_mmx(gx16 + b * 4,
                                           g_ternary_lut[gb0[b + 0]], g_ternary_lut[gb0[b + 1]],
                                           g_ternary_lut[gb1[b + 0]], g_ternary_lut[gb1[b + 1]]);
                    }
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    "movq %%mm7, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm7\n\tmovd %%mm7, %[s1]\n\t"
                    : [s0] "=r"(sum0), [s1] "=r"(sum1) : : "mm0", "mm1", "mm6", "mm7"
                );
                acc_r0[g] = sum0;
                acc_r1[g] = sum1;
            }
        }
        for (; tr < tile_rows; tr++) {
            uint32_t r = r_base + tr;
            const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 4;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const uint8_t *gb0 = row0_bytes + g * bytes_per_group;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0;
                __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                for (uint32_t b = 0; b < bytes_per_group; b += 2) {
                    lut_step_2rows_mmx(gx16 + b * 4,
                                       g_ternary_lut[gb0[b + 0]], g_ternary_lut[gb0[b + 1]],
                                       g_ternary_lut[0], g_ternary_lut[0]);
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    : [s0] "=r"(sum0) : : "mm0", "mm6"
                );
                acc_r0[g] = sum0;
            }
        }
        TZ_EMMS_BARRIER();
        flush_tile_scales_fpu(out, acc_buf, w_scales, x_scales, r_base, tile_rows, groups_per_row);
    }
}

#define Q8_BLOCK8_2ROWS_ASM(X_OFF, W_OFF) \
    "movq " #X_OFF "(%[x]), %%mm0\n\t" \
    "movq " #X_OFF "+8(%[x]), %%mm1\n\t" \
    "movq " #W_OFF "(%[w0]), %%mm2\n\t" \
    "movq " #W_OFF "(%[w1]), %%mm4\n\t" \
    "movq %%mm2, %%mm3\n\t" \
    "movq %%mm4, %%mm5\n\t" \
    "punpcklbw %%mm2, %%mm2\n\t" \
    "punpckhbw %%mm3, %%mm3\n\t" \
    "punpcklbw %%mm4, %%mm4\n\t" \
    "punpckhbw %%mm5, %%mm5\n\t" \
    "psraw $8, %%mm2\n\t" \
    "psraw $8, %%mm3\n\t" \
    "psraw $8, %%mm4\n\t" \
    "psraw $8, %%mm5\n\t" \
    "pmaddwd %%mm0, %%mm2\n\t" \
    "pmaddwd %%mm1, %%mm3\n\t" \
    "pmaddwd %%mm0, %%mm4\n\t" \
    "paddd %%mm2, %%mm6\n\t" \
    "pmaddwd %%mm1, %%mm5\n\t" \
    "paddd %%mm3, %%mm6\n\t" \
    "paddd %%mm4, %%mm7\n\t" \
    "paddd %%mm5, %%mm7\n\t"

void matmul_q8_mmx(float * TZ_RESTRICT out,
                   const int16_t * TZ_RESTRICT xq16,
                   const float   * TZ_RESTRICT x_scales,
                   int32_t       * TZ_RESTRICT acc_buf,
                   const QuantizedTensor *w,
                   uint32_t group_size) {
    if (!acc_buf) { fallback_q8_i16(out, xq16, x_scales, w, group_size); return; }
    const int8_t *w_data = (const int8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    for (uint32_t r_base = 0; r_base < rows; r_base += TUNGUZKA_MMX_TILE_R) {
        uint32_t tile_rows = rows - r_base;
        if (tile_rows > TUNGUZKA_MMX_TILE_R) tile_rows = TUNGUZKA_MMX_TILE_R;
        uint32_t tr = 0;
        for (; tr + 1 < tile_rows; tr += 2) {
            uint32_t r = r_base + tr;
            const int8_t *row0_w = w_data + (size_t)r * cols;
            const int8_t *row1_w = row0_w + cols;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            int32_t *acc_r1 = acc_r0 + groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const int8_t  *gw0  = row0_w + g * group_size;
                const int8_t  *gw1  = row1_w + g * group_size;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0, sum1;
                if (group_size == 32) {
                    __asm__ volatile (
                        "pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t"
                        Q8_BLOCK8_2ROWS_ASM(0,  0)
                        Q8_BLOCK8_2ROWS_ASM(16, 8)
                        Q8_BLOCK8_2ROWS_ASM(32, 16)
                        Q8_BLOCK8_2ROWS_ASM(48, 24)
                        : : [x] "r"(gx16), [w0] "r"(gw0), [w1] "r"(gw1)
                        : "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"
                    );
                } else {
                    __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                    for (uint32_t i = 0; i < group_size; i += 8) {
                        const int16_t *x_ptr  = gx16 + i;
                        const int8_t  *w0_ptr = gw0 + i;
                        const int8_t  *w1_ptr = gw1 + i;
                        __asm__ volatile (
                            Q8_BLOCK8_2ROWS_ASM(0, 0)
                            : : [x] "r"(x_ptr), [w0] "r"(w0_ptr), [w1] "r"(w1_ptr)
                            : "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"
                        );
                    }
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    "movq %%mm7, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm7\n\tmovd %%mm7, %[s1]\n\t"
                    : [s0] "=r"(sum0), [s1] "=r"(sum1) : : "mm0", "mm1", "mm6", "mm7"
                );
                acc_r0[g] = sum0;
                acc_r1[g] = sum1;
            }
        }
        for (; tr < tile_rows; tr++) {
            uint32_t r = r_base + tr;
            const int8_t *row0_w = w_data + (size_t)r * cols;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const int8_t  *gw0  = row0_w + g * group_size;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0;
                __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                for (uint32_t i = 0; i < group_size; i += 8) {
                    const int16_t *x_ptr  = gx16 + i;
                    const int8_t  *w0_ptr = gw0 + i;
                    __asm__ volatile (
                        Q8_BLOCK8_2ROWS_ASM(0, 0)
                        : : [x] "r"(x_ptr), [w0] "r"(w0_ptr), [w1] "r"(w0_ptr)
                        : "mm0", "mm1", "mm2", "mm3", "mm4", "mm5", "mm6", "mm7"
                    );
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    : [s0] "=r"(sum0) : : "mm0", "mm6"
                );
                acc_r0[g] = sum0;
            }
        }
        TZ_EMMS_BARRIER();
        flush_tile_scales_fpu(out, acc_buf, w_scales, x_scales, r_base, tile_rows, groups_per_row);
    }
}

#define Q4_BLOCK8_2ROWS_ASM(X_OFF, W_OFF) \
    "movd " #W_OFF "(%[w0]), %%mm2\n\t" \
    "movd " #W_OFF "(%[w1]), %%mm0\n\t" \
    "movq %%mm2, %%mm3\n\t" \
    "movq %%mm0, %%mm1\n\t" \
    "psrlq $4, %%mm3\n\t" \
    "psrlq $4, %%mm1\n\t" \
    "punpcklbw %%mm3, %%mm2\n\t" \
    "punpcklbw %%mm1, %%mm0\n\t" \
    "pand %%mm4, %%mm2\n\t" \
    "pand %%mm4, %%mm0\n\t" \
    "psubb %%mm5, %%mm2\n\t" \
    "psubb %%mm5, %%mm0\n\t" \
    "movq %%mm2, %%mm3\n\t" \
    "movq %%mm0, %%mm1\n\t" \
    "punpcklbw %%mm2, %%mm2\n\t" \
    "punpckhbw %%mm3, %%mm3\n\t" \
    "punpcklbw %%mm0, %%mm0\n\t" \
    "punpckhbw %%mm1, %%mm1\n\t" \
    "psraw $8, %%mm2\n\t" \
    "psraw $8, %%mm3\n\t" \
    "psraw $8, %%mm0\n\t" \
    "psraw $8, %%mm1\n\t" \
    "pmaddwd " #X_OFF "(%[x]), %%mm2\n\t" \
    "pmaddwd " #X_OFF "+8(%[x]), %%mm3\n\t" \
    "pmaddwd " #X_OFF "(%[x]), %%mm0\n\t" \
    "paddd %%mm2, %%mm6\n\t" \
    "pmaddwd " #X_OFF "+8(%[x]), %%mm1\n\t" \
    "paddd %%mm3, %%mm6\n\t" \
    "paddd %%mm0, %%mm7\n\t" \
    "paddd %%mm1, %%mm7\n\t"

void matmul_q4_mmx(float * TZ_RESTRICT out,
                   const int16_t * TZ_RESTRICT xq16,
                   const float   * TZ_RESTRICT x_scales,
                   int32_t       * TZ_RESTRICT acc_buf,
                   const QuantizedTensor *w,
                   uint32_t group_size) {
    if (!acc_buf) { fallback_q4_i16(out, xq16, x_scales, w, group_size); return; }
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 2;
    for (uint32_t r_base = 0; r_base < rows; r_base += TUNGUZKA_MMX_TILE_R) {
        uint32_t tile_rows = rows - r_base;
        if (tile_rows > TUNGUZKA_MMX_TILE_R) tile_rows = TUNGUZKA_MMX_TILE_R;
        __asm__ volatile (
            "movq %[m0f], %%mm4\n\tmovq %[b08], %%mm5\n\t"
            : : [m0f] "m"(g_q4_mask_0f), [b08] "m"(g_q4_bias_08) : "mm4", "mm5"
        );
        uint32_t tr = 0;
        for (; tr + 1 < tile_rows; tr += 2) {
            uint32_t r = r_base + tr;
            const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 2;
            const uint8_t *row1_bytes = row0_bytes + cols / 2;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            int32_t *acc_r1 = acc_r0 + groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const uint8_t *gb0  = row0_bytes + g * bytes_per_group;
                const uint8_t *gb1  = row1_bytes + g * bytes_per_group;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0, sum1;
                if (bytes_per_group == 16) {
                    __asm__ volatile (
                        "pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t"
                        Q4_BLOCK8_2ROWS_ASM(0,  0)
                        Q4_BLOCK8_2ROWS_ASM(16, 4)
                        Q4_BLOCK8_2ROWS_ASM(32, 8)
                        Q4_BLOCK8_2ROWS_ASM(48, 12)
                        : : [x] "r"(gx16), [w0] "r"(gb0), [w1] "r"(gb1)
                        : "mm0", "mm1", "mm2", "mm3", "mm6", "mm7"
                    );
                } else {
                    __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                    for (uint32_t b = 0; b < bytes_per_group; b += 4) {
                        const int16_t *x_ptr  = gx16 + b * 2;
                        const uint8_t *w0_ptr = gb0 + b;
                        const uint8_t *w1_ptr = gb1 + b;
                        __asm__ volatile (
                            Q4_BLOCK8_2ROWS_ASM(0, 0)
                            : : [x] "r"(x_ptr), [w0] "r"(w0_ptr), [w1] "r"(w1_ptr)
                            : "mm0", "mm1", "mm2", "mm3", "mm6", "mm7"
                        );
                    }
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    "movq %%mm7, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm7\n\tmovd %%mm7, %[s1]\n\t"
                    : [s0] "=r"(sum0), [s1] "=r"(sum1) : : "mm0", "mm1", "mm6", "mm7"
                );
                acc_r0[g] = sum0;
                acc_r1[g] = sum1;
            }
        }
        for (; tr < tile_rows; tr++) {
            uint32_t r = r_base + tr;
            const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 2;
            int32_t *acc_r0 = acc_buf + (size_t)tr * groups_per_row;
            for (uint32_t g = 0; g < groups_per_row; g++) {
                const uint8_t *gb0  = row0_bytes + g * bytes_per_group;
                const int16_t *gx16 = xq16 + g * group_size;
                int32_t sum0;
                __asm__ volatile ("pxor %%mm6, %%mm6\n\tpxor %%mm7, %%mm7\n\t" : : : "mm6", "mm7");
                for (uint32_t b = 0; b < bytes_per_group; b += 4) {
                    const int16_t *x_ptr  = gx16 + b * 2;
                    const uint8_t *w0_ptr = gb0 + b;
                    __asm__ volatile (
                        Q4_BLOCK8_2ROWS_ASM(0, 0)
                        : : [x] "r"(x_ptr), [w0] "r"(w0_ptr), [w1] "r"(w0_ptr)
                        : "mm0", "mm1", "mm2", "mm3", "mm6", "mm7"
                    );
                }
                __asm__ volatile (
                    "movq %%mm6, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm6\n\tmovd %%mm6, %[s0]\n\t"
                    : [s0] "=r"(sum0) : : "mm0", "mm6"
                );
                acc_r0[g] = sum0;
            }
        }
        TZ_EMMS_BARRIER();
        flush_tile_scales_fpu(out, acc_buf, w_scales, x_scales, r_base, tile_rows, groups_per_row);
    }
}

#else

void unpack_xq_to_i16_mmx(const int8_t * TZ_RESTRICT xq,
                          int16_t * TZ_RESTRICT xq16,
                          uint32_t cols) {
    for (uint32_t i = 0; i < cols; i++) xq16[i] = (int16_t)xq[i];
}

int32_t dot_q8_i16_mmx(const int16_t * TZ_RESTRICT q16,
                       const int8_t  * TZ_RESTRICT k8,
                       uint32_t head_size) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < head_size; i++) sum += (int32_t)q16[i] * (int32_t)k8[i];
    return sum;
}

void matmul_binary_mmx(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                       int32_t * TZ_RESTRICT acc_buf, const QuantizedTensor *w, uint32_t group_size) {
    (void)acc_buf;
    fallback_binary_i16(out, xq16, x_scales, w, group_size);
}

void matmul_ternary_mmx(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                        int32_t * TZ_RESTRICT acc_buf, const QuantizedTensor *w, uint32_t group_size) {
    (void)acc_buf;
    fallback_ternary_i16(out, xq16, x_scales, w, group_size);
}

void matmul_q8_mmx(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                   int32_t * TZ_RESTRICT acc_buf, const QuantizedTensor *w, uint32_t group_size) {
    (void)acc_buf;
    fallback_q8_i16(out, xq16, x_scales, w, group_size);
}

void matmul_q4_mmx(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                   int32_t * TZ_RESTRICT acc_buf, const QuantizedTensor *w, uint32_t group_size) {
    (void)acc_buf;
    fallback_q4_i16(out, xq16, x_scales, w, group_size);
}

#endif