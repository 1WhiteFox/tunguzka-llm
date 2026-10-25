#ifndef MMX_KERNELS_H
#define MMX_KERNELS_H

#include "tunguzka.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int16_t g_ternary_lut[256][4]       TZ_ALIGNED(64);
extern int16_t g_binary_nibble_lut[16][4]  TZ_ALIGNED(64);

void mmx_init_luts(void);
int  cpu_has_mmx(void);

void unpack_xq_to_i16_mmx(const int8_t * TZ_RESTRICT xq,
                          int16_t * TZ_RESTRICT xq16,
                          uint32_t cols);

int32_t dot_q8_i16_mmx(const int16_t * TZ_RESTRICT q16,
                       const int8_t  * TZ_RESTRICT k8,
                       uint32_t head_size);

void matmul_binary_mmx(float * TZ_RESTRICT out,
                       const int16_t * TZ_RESTRICT xq16,
                       const float   * TZ_RESTRICT x_scales,
                       int32_t       * TZ_RESTRICT acc_buf,
                       const QuantizedTensor *w,
                       uint32_t group_size);

void matmul_ternary_mmx(float * TZ_RESTRICT out,
                        const int16_t * TZ_RESTRICT xq16,
                        const float   * TZ_RESTRICT x_scales,
                        int32_t       * TZ_RESTRICT acc_buf,
                        const QuantizedTensor *w,
                        uint32_t group_size);

void matmul_q8_mmx(float * TZ_RESTRICT out,
                   const int16_t * TZ_RESTRICT xq16,
                   const float   * TZ_RESTRICT x_scales,
                   int32_t       * TZ_RESTRICT acc_buf,
                   const QuantizedTensor *w,
                   uint32_t group_size);

void matmul_q4_mmx(float * TZ_RESTRICT out,
                   const int16_t * TZ_RESTRICT xq16,
                   const float   * TZ_RESTRICT x_scales,
                   int32_t       * TZ_RESTRICT acc_buf,
                   const QuantizedTensor *w,
                   uint32_t group_size);

#ifdef __cplusplus
}
#endif
#endif /* MMX_KERNELS_H */