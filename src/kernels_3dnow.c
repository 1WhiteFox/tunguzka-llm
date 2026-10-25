#include "../include/3dnow_kernels.h"
#include "../include/mmx_kernels.h"

#define TZ_FEMMS_BARRIER() \
    __asm__ volatile ("femms" : : : "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory")

int cpu_has_3dnow(void) {
#if (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
    uint32_t eax, ebx, ecx, edx;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0x80000000u));
    if (eax < 0x80000001u) return 0;
    __asm__ volatile ("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(0x80000001u));
    return (edx & (1u << 31)) ? 1 : 0;
#else
    return 0;
#endif
}

#if (defined(__i386__) || defined(__x86_64__)) && defined(__3dnow__)

static const uint64_t g_q4_mask_0f_3dn TZ_ALIGNED(8) = 0x0F0F0F0F0F0F0F0FULL;
static const uint64_t g_q4_bias_08_3dn TZ_ALIGNED(8) = 0x0808080808080808ULL;

void rmsnorm_3dnow(float *out,
                   const float *x,
                   const float * TZ_RESTRICT weight,
                   uint32_t size,
                   float eps) {
    float ss = 0.0f;
    uint32_t j = 0;
    __asm__ volatile ("pxor %%mm0, %%mm0\n\t" : : : "mm0");
    for (; j + 3 < size; j += 4) {
        const float *xp = x + j;
        __asm__ volatile (
            "movq (%[xp]), %%mm1\n\t"
            "movq 8(%[xp]), %%mm2\n\t"
            "pfmul %%mm1, %%mm1\n\t"
            "pfmul %%mm2, %%mm2\n\t"
            "pfadd %%mm1, %%mm0\n\t"
            "pfadd %%mm2, %%mm0\n\t"
            : : [xp] "r"(xp) : "mm0", "mm1", "mm2"
        );
    }
    __asm__ volatile (
        "pfacc %%mm0, %%mm0\n\t"
        "movd %%mm0, %[ss]\n\t"
        "femms\n\t"
        : [ss] "=r"(ss) : : "mm0", "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory"
    );
    for (; j < size; j++) ss += x[j] * x[j];
    ss = ss / (float)size + eps;
    __asm__ volatile (
        "movd %[ss], %%mm0\n\t"
        "punpckldq %%mm0, %%mm0\n\t"
        "pfrsqrt %%mm0, %%mm1\n\t"
        "movq %%mm1, %%mm2\n\t"
        "pfmul %%mm1, %%mm1\n\t"
        "pfrsqit1 %%mm0, %%mm1\n\t"
        "pfrcpit2 %%mm2, %%mm1\n\t"
        : : [ss] "r"(ss) : "mm0", "mm1", "mm2"
    );
    j = 0;
    for (; j + 1 < size; j += 2) {
        const float *xp = x + j;
        const float *wp = weight + j;
        float *op = out + j;
        __asm__ volatile (
            "movq (%[xp]), %%mm3\n\t"
            "movq (%[wp]), %%mm4\n\t"
            "pfmul %%mm1, %%mm3\n\t"
            "pfmul %%mm4, %%mm3\n\t"
            "movq %%mm3, (%[op])\n\t"
            : : [xp] "r"(xp), [wp] "r"(wp), [op] "r"(op) : "mm1", "mm3", "mm4", "memory"
        );
    }
    float inv_sqrt;
    __asm__ volatile (
        "movd %%mm1, %[inv]\n\tfemms\n\t"
        : [inv] "=r"(inv_sqrt) : : "mm1", "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory"
    );
    for (; j < size; j++) out[j] = weight[j] * (inv_sqrt * x[j]);
}

float dot_f32_3dnow(const float * TZ_RESTRICT a,
                    const float * TZ_RESTRICT b,
                    uint32_t n) {
    float sum = 0.0f;
    uint32_t i = 0;
    __asm__ volatile ("pxor %%mm0, %%mm0\n\t" : : : "mm0");
    for (; i + 3 < n; i += 4) {
        const float *ap = a + i;
        const float *bp = b + i;
        __asm__ volatile (
            "movq (%[ap]), %%mm1\n\t"
            "movq 8(%[ap]), %%mm2\n\t"
            "pfmul (%[bp]), %%mm1\n\t"
            "pfmul 8(%[bp]), %%mm2\n\t"
            "pfadd %%mm1, %%mm0\n\t"
            "pfadd %%mm2, %%mm0\n\t"
            : : [ap] "r"(ap), [bp] "r"(bp) : "mm0", "mm1", "mm2"
        );
    }
    __asm__ volatile (
        "pfacc %%mm0, %%mm0\n\tmovd %%mm0, %[sum]\n\tfemms\n\t"
        : [sum] "=r"(sum) : : "mm0", "st", "st(1)", "st(2)", "st(3)", "st(4)", "st(5)", "st(6)", "st(7)", "memory"
    );
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void saxpy_f32_3dnow(float * TZ_RESTRICT y,
                     float a,
                     const float * TZ_RESTRICT x,
                     uint32_t n) {
    uint32_t i = 0;
    __asm__ volatile ("movd %[a], %%mm0\n\tpunpckldq %%mm0, %%mm0\n\t" : : [a] "r"(a) : "mm0");
    for (; i + 3 < n; i += 4) {
        float *yp = y + i;
        const float *xp = x + i;
        __asm__ volatile (
            "movq (%[xp]), %%mm1\n\t"
            "movq 8(%[xp]), %%mm2\n\t"
            "movq (%[yp]), %%mm3\n\t"
            "movq 8(%[yp]), %%mm4\n\t"
            "pfmul %%mm0, %%mm1\n\t"
            "pfmul %%mm0, %%mm2\n\t"
            "pfadd %%mm1, %%mm3\n\t"
            "pfadd %%mm2, %%mm4\n\t"
            "movq %%mm3, (%[yp])\n\t"
            "movq %%mm4, 8(%[yp])\n\t"
            : : [xp] "r"(xp), [yp] "r"(yp) : "mm0", "mm1", "mm2", "mm3", "mm4", "memory"
        );
    }
    TZ_FEMMS_BARRIER();
    for (; i < n; i++) y[i] += a * x[i];
}

#define THREEDNOW_ACCUM_2ROWS(ws0_ptr, ws1_ptr, xs_ptr) \
    __asm__ volatile ( \
        "movq %%mm5, %%mm0\n\tpsrlq $32, %%mm0\n\tpaddd %%mm0, %%mm5\n\t" \
        "movq %%mm6, %%mm1\n\tpsrlq $32, %%mm1\n\tpaddd %%mm1, %%mm6\n\t" \
        "punpckldq %%mm6, %%mm5\n\t" \
        "pi2fd %%mm5, %%mm5\n\t" \
        "movd (%[ws0]), %%mm1\n\tpunpckldq (%[ws1]), %%mm1\n\t" \
        "movd (%[xs]), %%mm2\n\tpunpckldq %%mm2, %%mm2\n\t" \
        "pfmul %%mm1, %%mm5\n\tpfmul %%mm2, %%mm5\n\tpfadd %%mm5, %%mm7\n\t" \
        : : [ws0] "r"(ws0_ptr), [ws1] "r"(ws1_ptr), [xs] "r"(xs_ptr) \
        : "mm0", "mm1", "mm2", "mm5", "mm6", "mm7" \
    )

void matmul_binary_3dnow(float * TZ_RESTRICT out,
                         const int16_t * TZ_RESTRICT xq16,
                         const float   * TZ_RESTRICT x_scales,
                         const QuantizedTensor *w,
                         uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 8;
    uint32_t r = 0;
    for (; r + 1 < rows; r += 2) {
        const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 8;
        const uint8_t *row1_bytes = row0_bytes + cols / 8;
        const float *ws0 = w_scales + (size_t)r * groups_per_row;
        const float *ws1 = ws0 + groups_per_row;
        float *out_pair = out + r;
        __asm__ volatile ("pxor %%mm7, %%mm7\n\t" : : : "mm7");
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb0  = row0_bytes + g * bytes_per_group;
            const uint8_t *gb1  = row1_bytes + g * bytes_per_group;
            const int16_t *gx16 = xq16 + g * group_size;
            __asm__ volatile (
                "prefetch 32(%[gb0])\n\tprefetch 32(%[gb1])\n\tpxor %%mm5, %%mm5\n\tpxor %%mm6, %%mm6\n\t"
                : : [gb0] "r"(gb0), [gb1] "r"(gb1) : "mm5", "mm6"
            );
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t p0 = gb0[b], p1 = gb1[b];
                const int16_t *l0a = g_binary_nibble_lut[p0 & 0x0F], *l0b = g_binary_nibble_lut[p0 >> 4];
                const int16_t *l1a = g_binary_nibble_lut[p1 & 0x0F], *l1b = g_binary_nibble_lut[p1 >> 4];
                const int16_t *x_ptr = gx16 + b * 8;
                __asm__ volatile (
                    "movq (%[x]), %%mm0\n\tmovq 8(%[x]), %%mm1\n\t"
                    "movq (%[l0a]), %%mm2\n\tmovq (%[l0b]), %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm5\n\tpaddd %%mm3, %%mm5\n\t"
                    : : [x] "r"(x_ptr), [l0a] "r"(l0a), [l0b] "r"(l0b) : "mm0", "mm1", "mm2", "mm3", "mm5"
                );
                __asm__ volatile (
                    "movq (%[l1a]), %%mm2\n\tmovq (%[l1b]), %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm6\n\tpaddd %%mm3, %%mm6\n\t"
                    : : [l1a] "r"(l1a), [l1b] "r"(l1b) : "mm0", "mm1", "mm2", "mm3", "mm6"
                );
            }
            THREEDNOW_ACCUM_2ROWS(ws0 + g, ws1 + g, x_scales + g);
        }
        __asm__ volatile ("movq %%mm7, (%[outp])\n\t" : : [outp] "r"(out_pair) : "mm7", "memory");
    }
    TZ_FEMMS_BARRIER();
    for (; r < rows; r++) {
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 8;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        float row_sum = 0.0f;
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

void matmul_ternary_3dnow(float * TZ_RESTRICT out,
                          const int16_t * TZ_RESTRICT xq16,
                          const float   * TZ_RESTRICT x_scales,
                          const QuantizedTensor *w,
                          uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 4;
    uint32_t r = 0;
    for (; r + 1 < rows; r += 2) {
        const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 4;
        const uint8_t *row1_bytes = row0_bytes + cols / 4;
        const float *ws0 = w_scales + (size_t)r * groups_per_row;
        const float *ws1 = ws0 + groups_per_row;
        float *out_pair = out + r;
        __asm__ volatile ("pxor %%mm7, %%mm7\n\t" : : : "mm7");
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb0  = row0_bytes + g * bytes_per_group;
            const uint8_t *gb1  = row1_bytes + g * bytes_per_group;
            const int16_t *gx16 = xq16 + g * group_size;
            __asm__ volatile (
                "prefetch 32(%[gb0])\n\tprefetch 32(%[gb1])\n\tpxor %%mm5, %%mm5\n\tpxor %%mm6, %%mm6\n\t"
                : : [gb0] "r"(gb0), [gb1] "r"(gb1) : "mm5", "mm6"
            );
            for (uint32_t b = 0; b < bytes_per_group; b += 2) {
                const int16_t *l0a = g_ternary_lut[gb0[b + 0]], *l0b = g_ternary_lut[gb0[b + 1]];
                const int16_t *l1a = g_ternary_lut[gb1[b + 0]], *l1b = g_ternary_lut[gb1[b + 1]];
                const int16_t *x_ptr = gx16 + b * 4;
                __asm__ volatile (
                    "movq (%[x]), %%mm0\n\tmovq 8(%[x]), %%mm1\n\t"
                    "movq (%[l0a]), %%mm2\n\tmovq (%[l0b]), %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm5\n\tpaddd %%mm3, %%mm5\n\t"
                    : : [x] "r"(x_ptr), [l0a] "r"(l0a), [l0b] "r"(l0b) : "mm0", "mm1", "mm2", "mm3", "mm5"
                );
                __asm__ volatile (
                    "movq (%[l1a]), %%mm2\n\tmovq (%[l1b]), %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm6\n\tpaddd %%mm3, %%mm6\n\t"
                    : : [l1a] "r"(l1a), [l1b] "r"(l1b) : "mm0", "mm1", "mm2", "mm3", "mm6"
                );
            }
            THREEDNOW_ACCUM_2ROWS(ws0 + g, ws1 + g, x_scales + g);
        }
        __asm__ volatile ("movq %%mm7, (%[outp])\n\t" : : [outp] "r"(out_pair) : "mm7", "memory");
    }
    TZ_FEMMS_BARRIER();
    for (; r < rows; r++) {
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 4;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        float row_sum = 0.0f;
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

void matmul_q8_3dnow(float * TZ_RESTRICT out,
                     const int16_t * TZ_RESTRICT xq16,
                     const float   * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w,
                     uint32_t group_size) {
    const int8_t *w_data = (const int8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t r = 0;
    for (; r + 1 < rows; r += 2) {
        const int8_t *row0_w = w_data + (size_t)r * cols;
        const int8_t *row1_w = row0_w + cols;
        const float *ws0 = w_scales + (size_t)r * groups_per_row;
        const float *ws1 = ws0 + groups_per_row;
        float *out_pair = out + r;
        __asm__ volatile ("pxor %%mm7, %%mm7\n\t" : : : "mm7");
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const int8_t  *gw0  = row0_w + g * group_size;
            const int8_t  *gw1  = row1_w + g * group_size;
            const int16_t *gx16 = xq16 + g * group_size;
            __asm__ volatile (
                "prefetch 32(%[gw0])\n\tprefetch 32(%[gw1])\n\tpxor %%mm5, %%mm5\n\tpxor %%mm6, %%mm6\n\t"
                : : [gw0] "r"(gw0), [gw1] "r"(gw1) : "mm5", "mm6"
            );
            for (uint32_t i = 0; i < group_size; i += 8) {
                const int8_t  *w0_ptr = gw0 + i, *w1_ptr = gw1 + i;
                const int16_t *x_ptr  = gx16 + i;
                __asm__ volatile (
                    "movq (%[x]), %%mm0\n\tmovq 8(%[x]), %%mm1\n\t"
                    "movq (%[w0]), %%mm2\n\tmovq %%mm2, %%mm3\n\t"
                    "punpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm5\n\tpaddd %%mm3, %%mm5\n\t"
                    "movq (%[w1]), %%mm2\n\tmovq %%mm2, %%mm3\n\t"
                    "punpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm6\n\tpaddd %%mm3, %%mm6\n\t"
                    : : [x] "r"(x_ptr), [w0] "r"(w0_ptr), [w1] "r"(w1_ptr)
                    : "mm0", "mm1", "mm2", "mm3", "mm5", "mm6"
                );
            }
            THREEDNOW_ACCUM_2ROWS(ws0 + g, ws1 + g, x_scales + g);
        }
        __asm__ volatile ("movq %%mm7, (%[outp])\n\t" : : [outp] "r"(out_pair) : "mm7", "memory");
    }
    TZ_FEMMS_BARRIER();
    for (; r < rows; r++) {
        const int8_t *row_w = w_data + (size_t)r * cols;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        float row_sum = 0.0f;
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

void matmul_q4_3dnow(float * TZ_RESTRICT out,
                     const int16_t * TZ_RESTRICT xq16,
                     const float   * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w,
                     uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 2;
    uint32_t r = 0;
    for (; r + 1 < rows; r += 2) {
        const uint8_t *row0_bytes = w_data + ((size_t)r * cols) / 2;
        const uint8_t *row1_bytes = row0_bytes + cols / 2;
        const float *ws0 = w_scales + (size_t)r * groups_per_row;
        const float *ws1 = ws0 + groups_per_row;
        float *out_pair = out + r;
        __asm__ volatile ("movq %[m0f], %%mm4\n\tpxor %%mm7, %%mm7\n\t" : : [m0f] "m"(g_q4_mask_0f_3dn) : "mm4", "mm7");
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb0  = row0_bytes + g * bytes_per_group;
            const uint8_t *gb1  = row1_bytes + g * bytes_per_group;
            const int16_t *gx16 = xq16 + g * group_size;
            __asm__ volatile (
                "prefetch 32(%[gb0])\n\tprefetch 32(%[gb1])\n\tpxor %%mm5, %%mm5\n\tpxor %%mm6, %%mm6\n\t"
                : : [gb0] "r"(gb0), [gb1] "r"(gb1) : "mm5", "mm6"
            );
            for (uint32_t b = 0; b < bytes_per_group; b += 4) {
                const uint8_t *w0_ptr = gb0 + b, *w1_ptr = gb1 + b;
                const int16_t *x_ptr  = gx16 + b * 2;
                __asm__ volatile (
                    "movq (%[x]), %%mm0\n\tmovq 8(%[x]), %%mm1\n\t"
                    "movd (%[w0]), %%mm2\n\tmovq %%mm2, %%mm3\n\tpsrlq $4, %%mm3\n\t"
                    "punpcklbw %%mm3, %%mm2\n\tpand %%mm4, %%mm2\n\tpsubb %[b08], %%mm2\n\t"
                    "movq %%mm2, %%mm3\n\tpunpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm5\n\tpaddd %%mm3, %%mm5\n\t"
                    "movd (%[w1]), %%mm2\n\tmovq %%mm2, %%mm3\n\tpsrlq $4, %%mm3\n\t"
                    "punpcklbw %%mm3, %%mm2\n\tpand %%mm4, %%mm2\n\tpsubb %[b08], %%mm2\n\t"
                    "movq %%mm2, %%mm3\n\tpunpcklbw %%mm2, %%mm2\n\tpunpckhbw %%mm3, %%mm3\n\t"
                    "psraw $8, %%mm2\n\tpsraw $8, %%mm3\n\t"
                    "pmaddwd %%mm0, %%mm2\n\tpmaddwd %%mm1, %%mm3\n\t"
                    "paddd %%mm2, %%mm6\n\tpaddd %%mm3, %%mm6\n\t"
                    : : [x] "r"(x_ptr), [w0] "r"(w0_ptr), [w1] "r"(w1_ptr), [b08] "m"(g_q4_bias_08_3dn)
                    : "mm0", "mm1", "mm2", "mm3", "mm5", "mm6"
                );
            }
            THREEDNOW_ACCUM_2ROWS(ws0 + g, ws1 + g, x_scales + g);
        }
        __asm__ volatile ("movq %%mm7, (%[outp])\n\t" : : [outp] "r"(out_pair) : "mm7", "memory");
    }
    TZ_FEMMS_BARRIER();
    for (; r < rows; r++) {
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 2;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        float row_sum = 0.0f;
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

#else

void rmsnorm_3dnow(float *out, const float *x, const float * TZ_RESTRICT weight, uint32_t size, float eps) {
    rmsnorm_scalar(out, x, weight, size, eps);
}

float dot_f32_3dnow(const float * TZ_RESTRICT a, const float * TZ_RESTRICT b, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

void saxpy_f32_3dnow(float * TZ_RESTRICT y, float a, const float * TZ_RESTRICT x, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) y[i] += a * x[i];
}

void matmul_binary_3dnow(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                         const QuantizedTensor *w, uint32_t group_size) {
    matmul_binary_mmx(out, xq16, x_scales, NULL, w, group_size);
}

void matmul_ternary_3dnow(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                          const QuantizedTensor *w, uint32_t group_size) {
    matmul_ternary_mmx(out, xq16, x_scales, NULL, w, group_size);
}

void matmul_q8_3dnow(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w, uint32_t group_size) {
    matmul_q8_mmx(out, xq16, x_scales, NULL, w, group_size);
}

void matmul_q4_3dnow(float * TZ_RESTRICT out, const int16_t * TZ_RESTRICT xq16, const float * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w, uint32_t group_size) {
    matmul_q4_mmx(out, xq16, x_scales, NULL, w, group_size);
}

#endif