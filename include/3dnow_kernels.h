#ifndef THREEDNOW_KERNELS_H
#define THREEDNOW_KERNELS_H

#include "tunguzka.h"

#ifdef __cplusplus
extern "C" {
#endif

int cpu_has_3dnow(void);

void rmsnorm_3dnow(float *out,
                   const float *x,
                   const float * TZ_RESTRICT weight,
                   uint32_t size,
                   float eps);

float dot_f32_3dnow(const float * TZ_RESTRICT a,
                    const float * TZ_RESTRICT b,
                    uint32_t n);

void saxpy_f32_3dnow(float * TZ_RESTRICT y,
                     float a,
                     const float * TZ_RESTRICT x,
                     uint32_t n);

void matmul_binary_3dnow(float * TZ_RESTRICT out,
                         const int16_t * TZ_RESTRICT xq16,
                         const float   * TZ_RESTRICT x_scales,
                         const QuantizedTensor *w,
                         uint32_t group_size);

void matmul_ternary_3dnow(float * TZ_RESTRICT out,
                          const int16_t * TZ_RESTRICT xq16,
                          const float   * TZ_RESTRICT x_scales,
                          const QuantizedTensor *w,
                          uint32_t group_size);

void matmul_q8_3dnow(float * TZ_RESTRICT out,
                     const int16_t * TZ_RESTRICT xq16,
                     const float   * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w,
                     uint32_t group_size);

void matmul_q4_3dnow(float * TZ_RESTRICT out,
                     const int16_t * TZ_RESTRICT xq16,
                     const float   * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w,
                     uint32_t group_size);

#ifdef __cplusplus
}
#endif
#endif /* THREEDNOW_KERNELS_H */