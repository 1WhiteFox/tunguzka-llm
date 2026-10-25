#include "../include/tunguzka.h"
#include <string.h>

uint32_t tz_crc32(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint32_t)p[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = -(int32_t)(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

void rmsnorm_scalar(float *out, const float *x, const float * TZ_RESTRICT weight, uint32_t size, float eps) {
    float ss = 0.0f;
    for (uint32_t j = 0; j < size; j++) ss += x[j] * x[j];
    ss /= (float)size;
    ss += eps;
    ss = 1.0f / tz_sqrtf(ss);
    for (uint32_t j = 0; j < size; j++) out[j] = weight[j] * (ss * x[j]);
}

void softmax_scalar(float *x, uint32_t size) {
    if (size == 0) return;
    if (size == 1) {
        x[0] = 1.0f;
        return;
    }
    float max_val = x[0];
    for (uint32_t i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < size; i++) {
        float e = tz_expf(x[i] - max_val);
        x[i] = e;
        sum += e;
    }
    float inv_sum = (sum > 0.0f) ? (1.0f / sum) : (1.0f / (float)size);
    for (uint32_t i = 0; i < size; i++) x[i] *= inv_sum;
}

void rope_scalar(float *q, float *k, uint32_t dim, uint32_t kv_dim, uint32_t head_size, int pos) {
    for (uint32_t head_dim = 0; head_dim < head_size; head_dim += 2) {
        float freq = tz_expf(-(float)head_dim * (9.210340371976184f / (float)head_size));
        float val = (float)pos * freq;
        float fcr, fci;
        tz_sincosf(val, &fci, &fcr);
        for (uint32_t base = head_dim; base < dim; base += head_size) {
            float q0 = q[base];
            float q1 = q[base + 1];
            q[base]     = q0 * fcr - q1 * fci;
            q[base + 1] = q0 * fci + q1 * fcr;
            if (base < kv_dim) {
                float k0 = k[base];
                float k1 = k[base + 1];
                k[base]     = k0 * fcr - k1 * fci;
                k[base + 1] = k0 * fci + k1 * fcr;
            }
        }
    }
}

void rope_apply_precomputed(float * TZ_RESTRICT q, float * TZ_RESTRICT k,
                            const float * TZ_RESTRICT fcr, const float * TZ_RESTRICT fci,
                            uint32_t dim, uint32_t kv_dim, uint32_t head_size) {
    for (uint32_t base = 0; base < dim; base += head_size) {
        float *qh = q + base;
        if (k != NULL && base < kv_dim) {
            float *kh = k + base;
            for (uint32_t hd = 0, idx = 0; hd < head_size; hd += 2, idx++) {
                float cr = fcr[idx], ci = fci[idx];
                float q0 = qh[hd], q1 = qh[hd + 1];
                float k0 = kh[hd], k1 = kh[hd + 1];
                qh[hd]     = q0 * cr - q1 * ci;
                qh[hd + 1] = q0 * ci + q1 * cr;
                kh[hd]     = k0 * cr - k1 * ci;
                kh[hd + 1] = k0 * ci + k1 * cr;
            }
        } else {
            for (uint32_t hd = 0, idx = 0; hd < head_size; hd += 2, idx++) {
                float cr = fcr[idx], ci = fci[idx];
                float q0 = qh[hd], q1 = qh[hd + 1];
                qh[hd]     = q0 * cr - q1 * ci;
                qh[hd + 1] = q0 * ci + q1 * cr;
            }
        }
    }
}

void quantize_activations_q8(const float * TZ_RESTRICT x,
                             int8_t      * TZ_RESTRICT xq,
                             float       * TZ_RESTRICT x_scales,
                             uint32_t n, uint32_t group_size) {
    uint32_t num_groups = n / group_size;
    const uint32_t *x_u32 = (const uint32_t *)(const void *)x;
    for (uint32_t g = 0; g < num_groups; g++) {
        const float    *xg = x + g * group_size;
        const uint32_t *ug = x_u32 + g * group_size;
        int8_t         *qg = xq + g * group_size;
        uint32_t max_bits = 0;
        for (uint32_t i = 0; i < group_size; i++) {
            uint32_t b = ug[i] & 0x7FFFFFFFu;
            int32_t diff = (int32_t)(max_bits - b);
            max_bits -= (uint32_t)diff & (uint32_t)(diff >> 31);
        }
        TzFloatBits mb;
        mb.u = max_bits;
        float max_abs = mb.f;
        float scale = max_abs / 127.0f;
        float inv_scale = (max_bits == 0) ? 0.0f : (127.0f / max_abs);
        x_scales[g] = scale;
        for (uint32_t i = 0; i < group_size; i++) {
            int32_t iv = tz_round_nearest_i32(xg[i] * inv_scale);
            if (iv > 127)  iv = 127;
            if (iv < -127) iv = -127;
            qg[i] = (int8_t)iv;
        }
    }
}

void matmul_binary_scalar(float *out, const int8_t *xq, const float *x_scales,
                          const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 8;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        uint32_t row_group_base = r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 8;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int8_t *gx = xq + g * group_size;
            int32_t acc = 0;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t p = gb[b];
                uint32_t idx = b * 8;
                for (uint32_t k = 0; k < 8; k++) {
                    acc += ((p >> k) & 1) ? (int32_t)gx[idx + k] : -(int32_t)gx[idx + k];
                }
            }
            row_sum += (float)acc * w_scales[row_group_base + g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

void matmul_ternary_scalar(float *out, const int8_t *xq, const float *x_scales,
                           const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 4;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        uint32_t row_group_base = r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 4;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int8_t *gx = xq + g * group_size;
            int32_t acc = 0;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t packed = gb[b];
                uint32_t idx = b * 4;
                uint8_t c0 = packed & 0x03;
                uint8_t c1 = (packed >> 2) & 0x03;
                uint8_t c2 = (packed >> 4) & 0x03;
                uint8_t c3 = (packed >> 6) & 0x03;
                if (c0 == 1) acc += gx[idx + 0]; else if (c0 == 2) acc -= gx[idx + 0];
                if (c1 == 1) acc += gx[idx + 1]; else if (c1 == 2) acc -= gx[idx + 1];
                if (c2 == 1) acc += gx[idx + 2]; else if (c2 == 2) acc -= gx[idx + 2];
                if (c3 == 1) acc += gx[idx + 3]; else if (c3 == 2) acc -= gx[idx + 3];
            }
            row_sum += (float)acc * w_scales[row_group_base + g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

void matmul_q8_scalar(float *out, const int8_t *xq, const float *x_scales,
                      const QuantizedTensor *w, uint32_t group_size) {
    const int8_t *w_data = (const int8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        uint32_t row_group_base = r * groups_per_row;
        const int8_t *row_w = w_data + (size_t)r * cols;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const int8_t *gw = row_w + g * group_size;
            const int8_t *gx = xq + g * group_size;
            int32_t acc = 0;
            for (uint32_t i = 0; i < group_size; i++) acc += (int32_t)gw[i] * (int32_t)gx[i];
            row_sum += (float)acc * w_scales[row_group_base + g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

void matmul_q4_scalar(float *out, const int8_t *xq, const float *x_scales,
                      const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 2;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        uint32_t row_group_base = r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 2;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int8_t *gx = xq + g * group_size;
            int32_t acc = 0;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t packed = gb[b];
                int32_t w0 = (int32_t)(packed & 0x0F) - 8;
                int32_t w1 = (int32_t)(packed >> 4) - 8;
                acc += w0 * (int32_t)gx[b * 2 + 0] + w1 * (int32_t)gx[b * 2 + 1];
            }
            row_sum += (float)acc * w_scales[row_group_base + g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

static int8_t g_p54c_ternary_lut[256][4] TZ_ALIGNED(32);
static int8_t g_p54c_binary_lut[16][4]   TZ_ALIGNED(32);
static int g_p54c_inited = 0;

void p54c_init_luts(void) {
    if (g_p54c_inited) return;
    for (int i = 0; i < 256; i++) {
        for (int k = 0; k < 4; k++) {
            int c = (i >> (k * 2)) & 0x03;
            g_p54c_ternary_lut[i][k] = (int8_t)((c == 1) ? 1 : ((c == 2) ? -1 : 0));
        }
    }
    for (int n = 0; n < 16; n++) {
        for (int k = 0; k < 4; k++) {
            g_p54c_binary_lut[n][k] = (int8_t)(((n >> k) & 1) ? 1 : -1);
        }
    }
    g_p54c_inited = 1;
}

void matmul_binary_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                        const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 8;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 8;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int8_t  *gx = xq + g * group_size;
            int32_t acc0 = 0, acc1 = 0;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t p = gb[b];
                const int8_t *l_lo = g_p54c_binary_lut[p & 0x0F];
                const int8_t *l_hi = g_p54c_binary_lut[p >> 4];
                const int8_t *x_ptr = gx + b * 8;
                acc0 += (int32_t)l_lo[0] * x_ptr[0] + (int32_t)l_lo[1] * x_ptr[1]
                      + (int32_t)l_lo[2] * x_ptr[2] + (int32_t)l_lo[3] * x_ptr[3];
                acc1 += (int32_t)l_hi[0] * x_ptr[4] + (int32_t)l_hi[1] * x_ptr[5]
                      + (int32_t)l_hi[2] * x_ptr[6] + (int32_t)l_hi[3] * x_ptr[7];
            }
            row_sum += (float)(acc0 + acc1) * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

void matmul_ternary_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                         const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 4;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 4;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int8_t  *gx = xq + g * group_size;
            int32_t acc0 = 0, acc1 = 0;
            for (uint32_t b = 0; b < bytes_per_group; b += 2) {
                const int8_t *l0 = g_p54c_ternary_lut[gb[b + 0]];
                const int8_t *l1 = g_p54c_ternary_lut[gb[b + 1]];
                const int8_t *x_ptr = gx + b * 4;
                acc0 += (int32_t)l0[0] * x_ptr[0] + (int32_t)l0[1] * x_ptr[1]
                      + (int32_t)l0[2] * x_ptr[2] + (int32_t)l0[3] * x_ptr[3];
                acc1 += (int32_t)l1[0] * x_ptr[4] + (int32_t)l1[1] * x_ptr[5]
                      + (int32_t)l1[2] * x_ptr[6] + (int32_t)l1[3] * x_ptr[7];
            }
            row_sum += (float)(acc0 + acc1) * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

void matmul_q8_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                    const QuantizedTensor *w, uint32_t group_size) {
    const int8_t *w_data = (const int8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const int8_t *row_w = w_data + (size_t)r * cols;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const int8_t *gw = row_w + g * group_size;
            const int8_t *gx = xq + g * group_size;
            int32_t acc0 = 0, acc1 = 0;
            for (uint32_t i = 0; i < group_size; i += 8) {
                acc0 += (int32_t)gw[i + 0] * gx[i + 0] + (int32_t)gw[i + 1] * gx[i + 1]
                      + (int32_t)gw[i + 2] * gx[i + 2] + (int32_t)gw[i + 3] * gx[i + 3];
                acc1 += (int32_t)gw[i + 4] * gx[i + 4] + (int32_t)gw[i + 5] * gx[i + 5]
                      + (int32_t)gw[i + 6] * gx[i + 6] + (int32_t)gw[i + 7] * gx[i + 7];
            }
            row_sum += (float)(acc0 + acc1) * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}

void matmul_q4_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                    const QuantizedTensor *w, uint32_t group_size) {
    const uint8_t *w_data = (const uint8_t *)w->data;
    const float *w_scales = w->scales;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    uint32_t bytes_per_group = group_size / 2;
    for (uint32_t r = 0; r < rows; r++) {
        float row_sum = 0.0f;
        const float *ws = w_scales + (size_t)r * groups_per_row;
        const uint8_t *row_bytes = w_data + ((size_t)r * cols) / 2;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            const uint8_t *gb = row_bytes + g * bytes_per_group;
            const int8_t  *gx = xq + g * group_size;
            int32_t acc0 = 0, acc1 = 0;
            for (uint32_t b = 0; b < bytes_per_group; b += 2) {
                uint8_t p0 = gb[b + 0];
                uint8_t p1 = gb[b + 1];
                acc0 += ((int32_t)(p0 & 0x0F) - 8) * gx[b * 2 + 0]
                      + ((int32_t)(p0 >> 4)   - 8) * gx[b * 2 + 1];
                acc1 += ((int32_t)(p1 & 0x0F) - 8) * gx[b * 2 + 2]
                      + ((int32_t)(p1 >> 4)   - 8) * gx[b * 2 + 3];
            }
            row_sum += (float)(acc0 + acc1) * ws[g] * x_scales[g];
        }
        out[r] = row_sum;
    }
}