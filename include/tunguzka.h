#ifndef TUNGUZKA_H
#define TUNGUZKA_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TUNGUZKA_MAGIC          "TUNGUZKA"
#define TUNGUZKA_VERSION        2
#define TUNGUZKA_ALIGN          64
#define TUNGUZKA_GROUP_SIZE     32
#define TUNGUZKA_MAX_DIM        8192
#define TUNGUZKA_MMX_TILE_R     16
#define TUNGUZKA_EMB_CACHE_SIZE 64

#define MODEL_FLAG_SHARED_CLASSIFIER   (1u << 0)
#define MODEL_FLAG_BITNET_SUBNORM      (1u << 1)
#define MODEL_FLAG_EMB_WCLS_Q8         (1u << 2)
#define MODEL_FLAG_HAS_SUBNORM_WEIGHTS (1u << 3)
#define MODEL_FLAG_RELU2               (1u << 4)
#define MODEL_FLAG_LAYER_MAJOR         (1u << 5)
#define MODEL_FLAG_ROPE_THETA_500K     (1u << 6)
#define MODEL_FLAG_LLAMA3_ROPE_SCALE   (1u << 7)

#if defined(__GNUC__) || defined(__clang__)
    #define TZ_ALIGNED(x) __attribute__((aligned(x)))
    #define TZ_RESTRICT   __restrict__
    #define TZ_INLINE     static inline __attribute__((always_inline))
#else
    #define TZ_ALIGNED(x)
    #define TZ_RESTRICT
    #define TZ_INLINE     static __inline
#endif

typedef union {
    float    f;
    uint32_t u;
    int32_t  i;
} TzFloatBits;

TZ_INLINE int32_t tz_round_nearest_i32(float v) {
    TzFloatBits fb;
    fb.f = v + 12582912.0f;
    return (fb.i & 0x007FFFFF) - 0x00400000;
}

TZ_INLINE float tz_sqrtf(float x) {
    if (x <= 0.0f) return 0.0f;
#if defined(__i386__) || defined(__x86_64__)
    __asm__ ("fsqrt" : "+t"(x));
    return x;
#else
    TzFloatBits u;
    u.f = x;
    u.u = 0x5f3759dfu - (u.u >> 1);
    float y = u.f;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return x * y;
#endif
}

TZ_INLINE float tz_expf(float x) {
    if (x > 88.0f)  x = 88.0f;
    if (x < -87.3f) return 0.0f;

    TzFloatBits kd, res;
    kd.f = x * 1.4426950408889634f + 12582912.0f;
    int32_t k = (kd.i & 0x007FFFFF) - 0x00400000;
    float kf = (float)k;

    float r = (x - kf * 0.693145751953125f) - kf * 1.428606765330187e-6f;

    float p = 1.38888889e-3f;
    p = p * r + 8.33333333e-3f;
    p = p * r + 4.16666667e-2f;
    p = p * r + 1.66666667e-1f;
    p = p * r + 5.0f * 1e-1f;
    p = p * (r * r) + r + 1.0f;

    res.f = p;
    res.i += (k << 23);
    return res.f;
}

TZ_INLINE void tz_sincosf(float val, float * TZ_RESTRICT s_out, float * TZ_RESTRICT c_out) {
#if defined(__i386__) || defined(__x86_64__)
    float s, c;
    __asm__ ("fsincos" : "=t"(c), "=u"(s) : "0"(val));
    *s_out = s;
    *c_out = c;
#else
    float k = (float)tz_round_nearest_i32(val * 0.15915494309189535f);
    float x = (val - k * 6.283184051513672f) - k * 1.2556360731e-6f;
    float x2 = x * x;
    *s_out = x * (1.0f - x2 * (0.16666667f - x2 * (0.00833333f - x2 * 0.00019841f)));
    *c_out = 1.0f - x2 * (0.5f - x2 * (0.04166667f - x2 * (0.00138889f - x2 * 0.00002480f)));
#endif
}

typedef enum {
    KERNEL_TERNARY_MMX = 0x01,
    KERNEL_Q4_3DNOW    = 0x02,
    KERNEL_Q8_3DNOW    = 0x03,
    KERNEL_BINARY_1BIT = 0x04
} KernelType;

typedef enum {
    BACKEND_SCALAR = 0,
    BACKEND_P54C   = 1,
    BACKEND_MMX    = 2,
    BACKEND_3DNOW  = 3
} ComputeBackend;

#pragma pack(push, 1)
typedef struct {
    char     magic[9];          /* 0..8: "TUNGUZKA\0" */
    uint8_t  version;           /* 9: format version (2) */
    uint8_t  kernel_type;       /* 10: primary weight quantization */
    uint8_t  emb_kernel_type;   /* 11: embedding/classifier quantization */
    uint32_t dim;               /* 12..15 */
    uint32_t hidden_dim;        /* 16..19 */
    uint32_t n_layers;          /* 20..23 */
    uint32_t n_heads;           /* 24..27 */
    uint32_t n_kv_heads;        /* 28..31 */
    uint32_t vocab_size;        /* 32..35 */
    uint32_t max_seq_len;       /* 36..39 */
    uint32_t group_size;        /* 40..43 */
    uint32_t flags;             /* 44..47 */
    uint32_t tokenizer_offset;  /* 48..51 */
    uint32_t tokenizer_size;    /* 52..55 */
    float    rms_norm_eps;      /* 56..59 */
    uint32_t header_crc32;      /* 60..63: CRC32 of bytes 0..59 */
} TunguzkaHeader;

typedef struct {
    char     magic[8];
    uint32_t steps;
    uint32_t ctx_limit;
    float    temp;
    uint8_t  backend;
    uint8_t  use_kv8;
    uint8_t  force_stream;
    uint8_t  lang;
    uint32_t max_ram_kb;
    char     model_file[36];
} TunguzkaRuntimeCfg;

typedef struct {
    uint32_t bundle_offset;
    uint32_t bundle_size;
    uint32_t cfg_offset;
    char     magic[4];
} TunguzkaBundleTrailer;
#pragma pack(pop)

typedef char assert_header_64b[(sizeof(TunguzkaHeader) == 64) ? 1 : -1];
typedef char assert_rt_cfg_64b[(sizeof(TunguzkaRuntimeCfg) == 64) ? 1 : -1];
typedef char assert_trailer_16b[(sizeof(TunguzkaBundleTrailer) == 16) ? 1 : -1];

typedef struct {
    const void  *data;
    const float *scales;
    size_t       scales_offset;
    size_t       data_offset;
    uint32_t     rows;
    uint32_t     cols;
} QuantizedTensor;

typedef struct {
    uint8_t *raw_ptr;
    uint8_t *base_ptr;
    size_t   capacity;
    size_t   offset;
} MemoryArena;

typedef struct {
    QuantizedTensor token_embedding_table;
    const float *rms_att_weight;
    const float *rms_ffn_weight;
    const float *rms_final_weight;
    const float *rms_sub_att_weight;
    const float *rms_sub_ffn_weight;
    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;
    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;
    QuantizedTensor wcls;
} TransformerWeights;

typedef struct {
    float    *x;
    float    *xb;
    float    *xb2;
    float    *hb;
    float    *hb2;
    int8_t   *xq;
    int16_t  *xq16;
    float    *x_scales;
    int32_t  *acc_buf;
    float    *rope_inv_freq;
    float    *rope_cos;
    float    *rope_sin;
    float    *q;
    float    *k;
    float    *v;
    float    *att;
    float    *logits;
    float    *key_cache;
    float    *value_cache;
    int8_t   *key_cache_q8;
    int8_t   *value_cache_q8;
    float    *key_scales;
    float    *value_scales;
    uint32_t  stream_win_rows;
    float    *stream_scales_buf;
    uint8_t  *stream_data_buf;
    float    *emb_row_scales;
    uint8_t  *emb_row_data;
    int32_t  *emb_cache_tokens;
    uint32_t *emb_cache_lru;
    float    *emb_cache_vecs;
} RunState;

typedef struct {
    TunguzkaHeader     config;
    uint32_t           orig_max_seq_len;
    float              rms_eps;
    float              rope_theta;
    int                rope_scaled;
    TransformerWeights weights;
    RunState           state;
    MemoryArena        arena;
    void              *weights_mem;
    size_t             weights_size;
    size_t             base_file_offset;
    int                is_mmap;
    int                use_kv8;
    ComputeBackend     backend;
    int                stream_mode;
    FILE              *stream_fp;
    uint8_t           *pin_pool;
    size_t             pin_pool_size;
    uint32_t           pinned_tensors;
    uint32_t           total_tensors;
    uint32_t           emb_lru_clock;
    uint32_t           emb_cache_hits;
    uint32_t           emb_cache_misses;
    uint32_t           disk_reads;
    size_t             disk_bytes_read;
    int32_t            last_stream_layer;
} TunguzkaModel;

uint32_t tz_crc32(const void *data, size_t len);
int   arena_init(MemoryArena *arena, size_t capacity);
void *arena_alloc_aligned(MemoryArena *arena, size_t size, size_t alignment);
void  arena_free(MemoryArena *arena);

ComputeBackend tunguzka_detect_backend(void);
int    tunguzka_load(TunguzkaModel *model, const char *model_path, uint32_t override_max_seq, ComputeBackend backend);
int    tunguzka_load_ex(TunguzkaModel *model, const char *model_path, uint32_t override_max_seq, ComputeBackend backend, int use_kv8);
int    tunguzka_load_stream(TunguzkaModel *model, const char *model_path, uint32_t override_max_seq,
                            ComputeBackend backend, int use_kv8, int force_stream, size_t max_ram_bytes);
int    tunguzka_load_from_offset(TunguzkaModel *model, const char *file_path, size_t base_offset, size_t slice_size,
                                 uint32_t override_max_seq, ComputeBackend backend, int use_kv8, int force_stream, size_t max_ram_bytes);
int    tunguzka_shift_kv_cache(TunguzkaModel *model, int cur_pos, int shift);
float *tunguzka_forward(TunguzkaModel *model, int token, int pos);
void   tunguzka_free(TunguzkaModel *model);

void quantize_activations_q8(const float * TZ_RESTRICT x,
                             int8_t      * TZ_RESTRICT xq,
                             float       * TZ_RESTRICT x_scales,
                             uint32_t n, uint32_t group_size);
void prepare_activations(const float * TZ_RESTRICT x,
                         int8_t      * TZ_RESTRICT xq,
                         int16_t     * TZ_RESTRICT xq16,
                         float       * TZ_RESTRICT x_scales,
                         uint32_t n, uint32_t group_size,
                         ComputeBackend backend);
void matmul_prepared(float * TZ_RESTRICT out,
                     const int8_t  * TZ_RESTRICT xq,
                     const int16_t * TZ_RESTRICT xq16,
                     const float   * TZ_RESTRICT x_scales,
                     int32_t       * TZ_RESTRICT acc_buf,
                     const QuantizedTensor *w,
                     uint32_t group_size,
                     uint8_t kernel_type,
                     ComputeBackend backend);
void matmul_dispatch(float * TZ_RESTRICT out,
                     const float * TZ_RESTRICT x,
                     int8_t * TZ_RESTRICT xq,
                     float  * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w,
                     uint32_t group_size,
                     uint8_t kernel_type,
                     ComputeBackend backend);

void rmsnorm_scalar(float *out, const float *x, const float * TZ_RESTRICT weight, uint32_t size, float eps);
void softmax_scalar(float *x, uint32_t size);
void rope_scalar(float *q, float *k, uint32_t dim, uint32_t kv_dim, uint32_t head_size, int pos);
void rope_apply_precomputed(float * TZ_RESTRICT q, float * TZ_RESTRICT k,
                            const float * TZ_RESTRICT fcr, const float * TZ_RESTRICT fci,
                            uint32_t dim, uint32_t kv_dim, uint32_t head_size);
void matmul_binary_scalar(float *out, const int8_t *xq, const float *x_scales,
                          const QuantizedTensor *w, uint32_t group_size);
void matmul_ternary_scalar(float *out, const int8_t *xq, const float *x_scales,
                           const QuantizedTensor *w, uint32_t group_size);
void matmul_q8_scalar(float *out, const int8_t *xq, const float *x_scales,
                      const QuantizedTensor *w, uint32_t group_size);
void matmul_q4_scalar(float *out, const int8_t *xq, const float *x_scales,
                      const QuantizedTensor *w, uint32_t group_size);

void p54c_init_luts(void);
void matmul_binary_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                        const QuantizedTensor *w, uint32_t group_size);
void matmul_ternary_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                         const QuantizedTensor *w, uint32_t group_size);
void matmul_q8_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                    const QuantizedTensor *w, uint32_t group_size);
void matmul_q4_p54c(float * TZ_RESTRICT out, const int8_t * TZ_RESTRICT xq, const float * TZ_RESTRICT x_scales,
                    const QuantizedTensor *w, uint32_t group_size);

#ifdef __cplusplus
}
#endif

#endif /* TUNGUZKA_H */