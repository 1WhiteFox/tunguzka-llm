#include "../include/tunguzka.h"
#include "../include/mmx_kernels.h"
#include "../include/3dnow_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(TZ_USE_MMAP) && (defined(__unix__) || defined(__APPLE__))
    #define TZ_HAS_MMAP 1
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#else
    #define TZ_HAS_MMAP 0
#endif

ComputeBackend tunguzka_detect_backend(void) {
    if (cpu_has_3dnow()) return BACKEND_3DNOW;
    if (cpu_has_mmx())   return BACKEND_MMX;
    return BACKEND_P54C;
}

static size_t align_up(size_t val, size_t align) {
    size_t rem = val % align;
    return (rem == 0) ? val : (val + (align - rem));
}

static uint8_t get_emb_wcls_ktype(const TunguzkaHeader *c) {
    if (c->emb_kernel_type >= KERNEL_TERNARY_MMX && c->emb_kernel_type <= KERNEL_BINARY_1BIT) {
        return c->emb_kernel_type;
    }
    return (c->flags & MODEL_FLAG_EMB_WCLS_Q8) ? (uint8_t)KERNEL_Q8_3DNOW : c->kernel_type;
}

int arena_init(MemoryArena *arena, size_t capacity) {
    arena->raw_ptr = (uint8_t *)malloc(capacity + TUNGUZKA_ALIGN);
    if (!arena->raw_ptr) return -1;
    uintptr_t addr = (uintptr_t)arena->raw_ptr;
    uintptr_t aligned = (addr + (TUNGUZKA_ALIGN - 1)) & ~(uintptr_t)(TUNGUZKA_ALIGN - 1);
    arena->base_ptr = (uint8_t *)aligned;
    arena->capacity = capacity;
    arena->offset = 0;
    memset(arena->base_ptr, 0, capacity);
    return 0;
}

void *arena_alloc_aligned(MemoryArena *arena, size_t size, size_t alignment) {
    size_t aligned_off = align_up(arena->offset, alignment);
    if (aligned_off + size > arena->capacity) return NULL;
    void *ptr = arena->base_ptr + aligned_off;
    arena->offset = aligned_off + size;
    return ptr;
}

void arena_free(MemoryArena *arena) {
    if (arena->raw_ptr) {
        free(arena->raw_ptr);
        arena->raw_ptr = NULL;
        arena->base_ptr = NULL;
        arena->capacity = 0;
        arena->offset = 0;
    }
}

static size_t packed_data_bytes(uint32_t rows, uint32_t cols, uint8_t kernel_type) {
    size_t total = (size_t)rows * (size_t)cols;
    if (kernel_type == KERNEL_BINARY_1BIT) return total / 8;
    if (kernel_type == KERNEL_TERNARY_MMX) return total / 4;
    if (kernel_type == KERNEL_Q4_3DNOW)    return total / 2;
    return total;
}

static int bind_quantized_tensor(QuantizedTensor *qt, const uint8_t *base, size_t *offset,
                                 size_t file_size, uint32_t rows, uint32_t cols,
                                 uint32_t group_size, uint8_t kernel_type) {
    size_t num_groups = ((size_t)rows * (size_t)cols) / group_size;
    size_t scales_bytes = num_groups * sizeof(float);
    size_t data_bytes = packed_data_bytes(rows, cols, kernel_type);

    *offset = align_up(*offset, TUNGUZKA_ALIGN);
    if (*offset > file_size || scales_bytes > file_size - *offset) return -1;
    qt->scales_offset = *offset;
    qt->scales = (base != NULL) ? (const float *)(base + *offset) : NULL;
    *offset += scales_bytes;

    *offset = align_up(*offset, TUNGUZKA_ALIGN);
    if (*offset > file_size || data_bytes > file_size - *offset) return -1;
    qt->data_offset = *offset;
    qt->data = (base != NULL) ? (const void *)(base + *offset) : NULL;
    *offset += data_bytes;

    *offset = align_up(*offset, TUNGUZKA_ALIGN);
    qt->rows = rows;
    qt->cols = cols;
    return 0;
}

static void dequantize_token_row(float *out, const QuantizedTensor *emb, int token,
                                 uint32_t group_size, uint8_t kernel_type) {
    uint32_t cols = emb->cols;
    uint32_t groups_per_row = cols / group_size;
    const float *row_scales = emb->scales + (size_t)token * groups_per_row;

    if (kernel_type == KERNEL_BINARY_1BIT) {
        const uint8_t *row_data = (const uint8_t *)emb->data + ((size_t)token * cols) / 8;
        uint32_t bytes_per_group = group_size / 8;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            float s = row_scales[g];
            const uint8_t *gb = row_data + g * bytes_per_group;
            float *go = out + g * group_size;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t p = gb[b];
                for (uint32_t k = 0; k < 8; k++) {
                    go[b * 8 + k] = ((p >> k) & 1) ? s : -s;
                }
            }
        }
    } else if (kernel_type == KERNEL_TERNARY_MMX) {
        const uint8_t *row_data = (const uint8_t *)emb->data + ((size_t)token * cols) / 4;
        uint32_t bytes_per_group = group_size / 4;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            float s = row_scales[g];
            const uint8_t *gb = row_data + g * bytes_per_group;
            float *go = out + g * group_size;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t p = gb[b];
                for (uint32_t k = 0; k < 4; k++) {
                    uint8_t c = (p >> (k * 2)) & 0x03;
                    float v = (c == 1) ? 1.0f : ((c == 2) ? -1.0f : 0.0f);
                    go[b * 4 + k] = v * s;
                }
            }
        }
    } else if (kernel_type == KERNEL_Q8_3DNOW) {
        const int8_t *row_data = (const int8_t *)emb->data + (size_t)token * cols;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            float s = row_scales[g];
            const int8_t *gw = row_data + g * group_size;
            float *go = out + g * group_size;
            for (uint32_t i = 0; i < group_size; i++) go[i] = (float)gw[i] * s;
        }
    } else if (kernel_type == KERNEL_Q4_3DNOW) {
        const uint8_t *row_data = (const uint8_t *)emb->data + ((size_t)token * cols) / 2;
        uint32_t bytes_per_group = group_size / 2;
        for (uint32_t g = 0; g < groups_per_row; g++) {
            float s = row_scales[g];
            const uint8_t *gb = row_data + g * bytes_per_group;
            float *go = out + g * group_size;
            for (uint32_t b = 0; b < bytes_per_group; b++) {
                uint8_t p = gb[b];
                go[b * 2 + 0] = (float)((int32_t)(p & 0x0F) - 8) * s;
                go[b * 2 + 1] = (float)((int32_t)(p >> 4) - 8) * s;
            }
        }
    }
}

static void load_token_embedding(TunguzkaModel *model, int token) {
    TunguzkaHeader *c = &model->config;
    TransformerWeights *w = &model->weights;
    RunState *s = &model->state;
    uint32_t dim = c->dim;
    uint32_t group_size = c->group_size;
    uint8_t  emb_ktype = get_emb_wcls_ktype(c);

    if (w->token_embedding_table.data != NULL && w->token_embedding_table.scales != NULL) {
        dequantize_token_row(s->x, &w->token_embedding_table, token, group_size, emb_ktype);
        return;
    }

    uint32_t victim = 0;
    uint32_t min_lru = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < TUNGUZKA_EMB_CACHE_SIZE; i++) {
        if (s->emb_cache_tokens[i] == token) {
            model->emb_cache_hits++;
            s->emb_cache_lru[i] = ++model->emb_lru_clock;
            memcpy(s->x, s->emb_cache_vecs + (size_t)i * dim, dim * sizeof(float));
            return;
        }
        if (s->emb_cache_tokens[i] == -1) {
            victim = i;
            min_lru = 0;
        } else if (s->emb_cache_lru[i] < min_lru) {
            min_lru = s->emb_cache_lru[i];
            victim = i;
        }
    }

    model->emb_cache_misses++;
    uint32_t groups_per_row = dim / group_size;
    size_t row_scale_bytes = (size_t)groups_per_row * sizeof(float);
    size_t row_data_bytes  = packed_data_bytes(1, dim, emb_ktype);
    size_t s_pos = model->base_file_offset + w->token_embedding_table.scales_offset + (size_t)token * row_scale_bytes;
    size_t d_pos = model->base_file_offset + w->token_embedding_table.data_offset   + (size_t)token * row_data_bytes;

    if (fseek(model->stream_fp, (long)s_pos, SEEK_SET) == 0 &&
        fread(s->emb_row_scales, 1, row_scale_bytes, model->stream_fp) == row_scale_bytes &&
        fseek(model->stream_fp, (long)d_pos, SEEK_SET) == 0 &&
        fread(s->emb_row_data, 1, row_data_bytes, model->stream_fp) == row_data_bytes) {
        model->disk_reads += 2;
        model->disk_bytes_read += row_scale_bytes + row_data_bytes;

        QuantizedTensor tmp_row;
        tmp_row.scales        = s->emb_row_scales;
        tmp_row.data          = s->emb_row_data;
        tmp_row.scales_offset = s_pos;
        tmp_row.data_offset   = d_pos;
        tmp_row.rows          = 1;
        tmp_row.cols          = dim;

        float *cached_vec = s->emb_cache_vecs + (size_t)victim * dim;
        dequantize_token_row(cached_vec, &tmp_row, 0, group_size, emb_ktype);
        s->emb_cache_tokens[victim] = token;
        s->emb_cache_lru[victim]    = ++model->emb_lru_clock;
        memcpy(s->x, cached_vec, dim * sizeof(float));
    } else {
        memset(s->x, 0, dim * sizeof(float));
    }
}

void prepare_activations(const float * TZ_RESTRICT x,
                         int8_t      * TZ_RESTRICT xq,
                         int16_t     * TZ_RESTRICT xq16,
                         float       * TZ_RESTRICT x_scales,
                         uint32_t n, uint32_t group_size,
                         ComputeBackend backend) {
    quantize_activations_q8(x, xq, x_scales, n, group_size);
    if ((backend == BACKEND_MMX || backend == BACKEND_3DNOW) && xq16 != NULL) {
        unpack_xq_to_i16_mmx(xq, xq16, n);
    }
}

void matmul_prepared(float * TZ_RESTRICT out,
                     const int8_t  * TZ_RESTRICT xq,
                     const int16_t * TZ_RESTRICT xq16,
                     const float   * TZ_RESTRICT x_scales,
                     int32_t       * TZ_RESTRICT acc_buf,
                     const QuantizedTensor *w,
                     uint32_t group_size,
                     uint8_t kernel_type,
                     ComputeBackend backend) {
    if (backend == BACKEND_3DNOW && xq16 != NULL) {
        if (kernel_type == KERNEL_BINARY_1BIT)       matmul_binary_3dnow(out, xq16, x_scales, w, group_size);
        else if (kernel_type == KERNEL_TERNARY_MMX)  matmul_ternary_3dnow(out, xq16, x_scales, w, group_size);
        else if (kernel_type == KERNEL_Q8_3DNOW)     matmul_q8_3dnow(out, xq16, x_scales, w, group_size);
        else if (kernel_type == KERNEL_Q4_3DNOW)     matmul_q4_3dnow(out, xq16, x_scales, w, group_size);
    } else if (backend == BACKEND_MMX && xq16 != NULL && acc_buf != NULL) {
        if (kernel_type == KERNEL_BINARY_1BIT)       matmul_binary_mmx(out, xq16, x_scales, acc_buf, w, group_size);
        else if (kernel_type == KERNEL_TERNARY_MMX)  matmul_ternary_mmx(out, xq16, x_scales, acc_buf, w, group_size);
        else if (kernel_type == KERNEL_Q8_3DNOW)     matmul_q8_mmx(out, xq16, x_scales, acc_buf, w, group_size);
        else if (kernel_type == KERNEL_Q4_3DNOW)     matmul_q4_mmx(out, xq16, x_scales, acc_buf, w, group_size);
    } else if (backend == BACKEND_P54C) {
        if (kernel_type == KERNEL_BINARY_1BIT)       matmul_binary_p54c(out, xq, x_scales, w, group_size);
        else if (kernel_type == KERNEL_TERNARY_MMX)  matmul_ternary_p54c(out, xq, x_scales, w, group_size);
        else if (kernel_type == KERNEL_Q8_3DNOW)     matmul_q8_p54c(out, xq, x_scales, w, group_size);
        else if (kernel_type == KERNEL_Q4_3DNOW)     matmul_q4_p54c(out, xq, x_scales, w, group_size);
    } else {
        if (kernel_type == KERNEL_BINARY_1BIT)       matmul_binary_scalar(out, xq, x_scales, w, group_size);
        else if (kernel_type == KERNEL_TERNARY_MMX)  matmul_ternary_scalar(out, xq, x_scales, w, group_size);
        else if (kernel_type == KERNEL_Q8_3DNOW)     matmul_q8_scalar(out, xq, x_scales, w, group_size);
        else if (kernel_type == KERNEL_Q4_3DNOW)     matmul_q4_scalar(out, xq, x_scales, w, group_size);
    }
}

static void matmul_model_tensor_ex(TunguzkaModel *model,
                                   float * TZ_RESTRICT out,
                                   const QuantizedTensor *w,
                                   uint8_t ktype,
                                   int32_t layer_idx) {
    TunguzkaHeader *c = &model->config;
    RunState *s = &model->state;
    uint32_t group_size = c->group_size;

    if (w->data != NULL && w->scales != NULL) {
        matmul_prepared(out, s->xq, s->xq16, s->x_scales, s->acc_buf, w, group_size, ktype, model->backend);
        return;
    }

    model->last_stream_layer = layer_idx;
    uint32_t rows = w->rows;
    uint32_t cols = w->cols;
    uint32_t groups_per_row = cols / group_size;
    size_t row_scale_bytes  = (size_t)groups_per_row * sizeof(float);
    size_t row_data_bytes   = packed_data_bytes(1, cols, ktype);
    uint32_t win_rows       = s->stream_win_rows;

    for (uint32_t r_off = 0; r_off < rows; r_off += win_rows) {
        uint32_t chunk_r = rows - r_off;
        if (chunk_r > win_rows) chunk_r = win_rows;

        size_t s_bytes = (size_t)chunk_r * row_scale_bytes;
        size_t d_bytes = (size_t)chunk_r * row_data_bytes;
        size_t s_pos   = model->base_file_offset + w->scales_offset + (size_t)r_off * row_scale_bytes;
        size_t d_pos   = model->base_file_offset + w->data_offset   + (size_t)r_off * row_data_bytes;

        if (fseek(model->stream_fp, (long)s_pos, SEEK_SET) != 0 ||
            fread(s->stream_scales_buf, 1, s_bytes, model->stream_fp) != s_bytes ||
            fseek(model->stream_fp, (long)d_pos, SEEK_SET) != 0 ||
            fread(s->stream_data_buf, 1, d_bytes, model->stream_fp) != d_bytes) {
            memset(out + r_off, 0, chunk_r * sizeof(float));
            continue;
        }
        model->disk_reads += 2;
        model->disk_bytes_read += s_bytes + d_bytes;

        QuantizedTensor chunk_qt;
        chunk_qt.scales        = s->stream_scales_buf;
        chunk_qt.data          = s->stream_data_buf;
        chunk_qt.scales_offset = s_pos;
        chunk_qt.data_offset   = d_pos;
        chunk_qt.rows          = chunk_r;
        chunk_qt.cols          = cols;

        matmul_prepared(out + r_off, s->xq, s->xq16, s->x_scales, s->acc_buf,
                        &chunk_qt, group_size, ktype, model->backend);
    }
}

static void matmul_model_tensor(TunguzkaModel *model,
                                float * TZ_RESTRICT out,
                                const QuantizedTensor *w,
                                int32_t layer_idx) {
    matmul_model_tensor_ex(model, out, w, model->config.kernel_type, layer_idx);
}

void matmul_dispatch(float * TZ_RESTRICT out,
                     const float * TZ_RESTRICT x,
                     int8_t * TZ_RESTRICT xq,
                     float  * TZ_RESTRICT x_scales,
                     const QuantizedTensor *w,
                     uint32_t group_size,
                     uint8_t kernel_type,
                     ComputeBackend backend) {
    (void)backend;
    quantize_activations_q8(x, xq, x_scales, w->cols, group_size);
    matmul_prepared(out, xq, NULL, x_scales, NULL, w, group_size, kernel_type, BACKEND_P54C);
}

static void try_pin_tensor_ex(TunguzkaModel *model, uint8_t *pin_base, QuantizedTensor *qt,
                              uint8_t ktype, size_t *pool_off, size_t pool_cap) {
    uint32_t group_size = model->config.group_size;
    size_t s_bytes = ((size_t)qt->rows * qt->cols / group_size) * sizeof(float);
    size_t d_bytes = packed_data_bytes(qt->rows, qt->cols, ktype);

    size_t off1 = align_up(*pool_off, TUNGUZKA_ALIGN);
    size_t off2 = align_up(off1 + s_bytes, TUNGUZKA_ALIGN);
    size_t end_off = off2 + d_bytes;
    if (end_off > pool_cap) return;

    float   *s_dst = (float *)(pin_base + off1);
    uint8_t *d_dst = pin_base + off2;

    if (fseek(model->stream_fp, (long)(model->base_file_offset + qt->scales_offset), SEEK_SET) == 0 &&
        fread(s_dst, 1, s_bytes, model->stream_fp) == s_bytes &&
        fseek(model->stream_fp, (long)(model->base_file_offset + qt->data_offset), SEEK_SET) == 0 &&
        fread(d_dst, 1, d_bytes, model->stream_fp) == d_bytes) {
        qt->scales = s_dst;
        qt->data   = d_dst;
        *pool_off  = end_off;
        model->pinned_tensors++;
    }
}

int tunguzka_load(TunguzkaModel *model, const char *model_path, uint32_t override_max_seq, ComputeBackend backend) {
    return tunguzka_load_from_offset(model, model_path, 0, 0, override_max_seq, backend, 0, 0, 0);
}

int tunguzka_load_ex(TunguzkaModel *model, const char *model_path, uint32_t override_max_seq, ComputeBackend backend, int use_kv8) {
    return tunguzka_load_from_offset(model, model_path, 0, 0, override_max_seq, backend, use_kv8, 0, 0);
}

int tunguzka_load_stream(TunguzkaModel *model, const char *model_path, uint32_t override_max_seq,
                         ComputeBackend backend, int use_kv8, int force_stream, size_t max_ram_bytes) {
    return tunguzka_load_from_offset(model, model_path, 0, 0, override_max_seq, backend, use_kv8, force_stream, max_ram_bytes);
}

int tunguzka_load_from_offset(TunguzkaModel *model, const char *file_path, size_t base_offset, size_t slice_size,
                              uint32_t override_max_seq, ComputeBackend backend, int use_kv8, int force_stream, size_t max_ram_bytes) {
    memset(model, 0, sizeof(TunguzkaModel));
    model->use_kv8 = use_kv8;
    model->last_stream_layer = -1;
    model->base_file_offset = base_offset;

    mmx_init_luts();
    if (backend == BACKEND_3DNOW && !cpu_has_3dnow()) {
        backend = cpu_has_mmx() ? BACKEND_MMX : BACKEND_P54C;
    } else if (backend == BACKEND_MMX && !cpu_has_mmx()) {
        backend = BACKEND_P54C;
    }
    model->backend = backend;

    FILE *f = fopen(file_path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long fsz = ftell(f);
    if (fsz < (long)(base_offset + sizeof(TunguzkaHeader))) { fclose(f); return -1; }
    size_t total_slice = (slice_size > 0 && base_offset + slice_size <= (size_t)fsz)
                         ? slice_size
                         : ((size_t)fsz - base_offset);

    if (fseek(f, (long)base_offset, SEEK_SET) != 0) { fclose(f); return -1; }
    if (fread(&model->config, 1, sizeof(TunguzkaHeader), f) != sizeof(TunguzkaHeader)) {
        fclose(f);
        return -1;
    }

    const uint8_t *raw_hdr = (const uint8_t *)&model->config;
    int is_v1 = (raw_hdr[9] == 0 && raw_hdr[10] == 1);
    if (is_v1) {
        model->config.version         = 1;
        model->config.kernel_type     = raw_hdr[11];
        model->config.emb_kernel_type = (model->config.flags & MODEL_FLAG_EMB_WCLS_Q8)
                                        ? (uint8_t)KERNEL_Q8_3DNOW
                                        : model->config.kernel_type;
        model->config.rms_norm_eps    = 1e-5f;
        model->config.header_crc32    = 0;
    }

    if (memcmp(model->config.magic, "TUNGUZKA", 8) != 0 ||
        (model->config.version != 1 && model->config.version != TUNGUZKA_VERSION)) {
        fclose(f);
        return -1;
    }

    if (!is_v1 && model->config.header_crc32 != 0) {
        uint32_t calc_crc = tz_crc32(&model->config, 60);
        if (calc_crc != model->config.header_crc32) {
            fprintf(stderr, "[ERROR] Header CRC32 mismatch in %s (expected 0x%08X, got 0x%08X)\n",
                    file_path, model->config.header_crc32, calc_crc);
            fclose(f);
            return -1;
        }
    }

    model->rms_eps = (model->config.rms_norm_eps > 1e-9f && model->config.rms_norm_eps < 1e-1f)
                     ? model->config.rms_norm_eps
                     : 1e-5f;

    if ((model->config.flags & MODEL_FLAG_ROPE_THETA_500K) || model->config.vocab_size >= 100000) {
        model->rope_theta = 500000.0f;
    } else {
        model->rope_theta = 10000.0f;
    }
    model->rope_scaled = (model->config.flags & MODEL_FLAG_LLAMA3_ROPE_SCALE) != 0;

    if (model->config.tokenizer_offset > sizeof(TunguzkaHeader) &&
        model->config.tokenizer_offset < total_slice) {
        model->weights_size = model->config.tokenizer_offset;
    } else {
        model->weights_size = total_slice;
    }

    model->orig_max_seq_len = model->config.max_seq_len;
    if (override_max_seq > 0 && override_max_seq < model->config.max_seq_len) {
        model->config.max_seq_len = override_max_seq;
    }

    uint32_t dim        = model->config.dim;
    uint32_t hidden_dim = model->config.hidden_dim;
    uint32_t n_layers   = model->config.n_layers;
    uint32_t n_heads    = model->config.n_heads;
    uint32_t n_kv_heads = model->config.n_kv_heads;
    uint32_t vocab_size = model->config.vocab_size;
    uint32_t seq_len    = model->config.max_seq_len;
    uint32_t group_size = model->config.group_size;
    uint8_t  ktype      = model->config.kernel_type;
    uint8_t  emb_ktype  = get_emb_wcls_ktype(&model->config);
    int      has_sub_w  = (model->config.flags & MODEL_FLAG_HAS_SUBNORM_WEIGHTS) != 0;
    int      layer_maj  = (model->config.flags & MODEL_FLAG_LAYER_MAJOR) != 0 || (model->config.version >= 2);

    if (dim == 0 || dim > TUNGUZKA_MAX_DIM ||
        hidden_dim == 0 || hidden_dim > 32768 ||
        n_layers == 0 || n_layers > 128 ||
        vocab_size == 0 || vocab_size > 256000 ||
        seq_len == 0 || seq_len > 32768 ||
        group_size == 0 || (group_size % 8) != 0 ||
        n_heads == 0 || n_kv_heads == 0 || (n_heads % n_kv_heads) != 0 ||
        (dim % n_heads) != 0 || ((dim / n_heads) % 2) != 0 ||
        (dim % group_size) != 0 || (hidden_dim % group_size) != 0) {
        fclose(f);
        return -1;
    }

    uint32_t head_size    = dim / n_heads;
    uint32_t half_head    = head_size / 2;
    uint32_t kv_dim       = (dim * n_kv_heads) / n_heads;
    uint32_t max_dim      = (hidden_dim > dim) ? hidden_dim : dim;
    uint32_t max_groups   = max_dim / group_size;
    uint32_t max_x_scales = (n_heads > max_groups) ? n_heads : max_groups;

    if ((kv_dim % group_size) != 0) {
        fclose(f);
        return -1;
    }

    size_t kv_entries = (size_t)n_layers * seq_len * kv_dim;
    size_t kv_scale_entries = (size_t)n_layers * seq_len * n_kv_heads;
    size_t rms_floats = (size_t)2 * n_layers * dim + dim;
    if (has_sub_w) {
        rms_floats += (size_t)n_layers * dim + (size_t)n_layers * hidden_dim;
    }
    size_t rms_bytes = rms_floats * sizeof(float);

    size_t base_arena_bytes = 0;
    base_arena_bytes += 7 * (size_t)n_layers * sizeof(QuantizedTensor) + 1024;
    base_arena_bytes += (3 * (size_t)dim + 2 * (size_t)hidden_dim + dim + 2 * kv_dim + seq_len + vocab_size + 3 * half_head) * sizeof(float);
    base_arena_bytes += max_dim * sizeof(int8_t) + max_dim * sizeof(int16_t) + max_x_scales * sizeof(float);
    base_arena_bytes += (size_t)TUNGUZKA_MMX_TILE_R * max_groups * sizeof(int32_t);
    if (use_kv8) {
        base_arena_bytes += 2 * kv_entries * sizeof(int8_t) + 2 * kv_scale_entries * sizeof(float);
    } else {
        base_arena_bytes += 2 * kv_entries * sizeof(float);
    }
    base_arena_bytes += 2048;

    int want_stream = force_stream;
    if (!want_stream && max_ram_bytes > 0 && (model->weights_size + base_arena_bytes > max_ram_bytes)) {
        fprintf(stderr, "[WARNING] Model (%.1f KB) + Arena (%.1f KB) exceeds RAM limit (%.1f KB)! Switching to DISK STREAMING mode.\n",
                (double)((float)model->weights_size / 1024.0f),
                (double)((float)base_arena_bytes / 1024.0f),
                (double)((float)max_ram_bytes / 1024.0f));
        want_stream = 1;
    }

    if (!want_stream) {
#if TZ_HAS_MMAP
        if (base_offset == 0) {
            int fd = open(file_path, O_RDONLY);
            if (fd >= 0) {
                void *mapped = mmap(NULL, model->weights_size, PROT_READ, MAP_PRIVATE, fd, 0);
                close(fd);
                if (mapped != MAP_FAILED) {
                    model->weights_mem = mapped;
                    model->is_mmap = 1;
                }
            }
        }
#endif
        if (!model->weights_mem && fseek(f, (long)base_offset, SEEK_SET) == 0) {
            model->weights_mem = malloc(model->weights_size);
            if (model->weights_mem) {
                if (fread(model->weights_mem, 1, model->weights_size, f) != model->weights_size) {
                    free(model->weights_mem);
                    model->weights_mem = NULL;
                }
            }
        }
        if (!model->weights_mem) {
            fprintf(stderr, "[WARNING] Out of RAM loading full model (%.1f KB)! Falling back to DISK STREAMING mode.\n",
                    (double)((float)model->weights_size / 1024.0f));
            want_stream = 1;
        }
    }

    model->stream_mode = want_stream;
    uint32_t stream_win_rows = (max_ram_bytes > 0 && max_ram_bytes < 131072u) ? 32u : 128u;
    if (stream_win_rows > max_dim) stream_win_rows = max_dim;
    size_t stream_scales_bytes = 0;
    size_t stream_data_bytes   = 0;
    size_t emb_cache_bytes     = 0;

    if (model->stream_mode) {
        model->stream_fp = f;
        setvbuf(model->stream_fp, NULL, _IOFBF, 16384);
        stream_scales_bytes = (size_t)stream_win_rows * max_groups * sizeof(float);
        size_t d_ktype = packed_data_bytes(stream_win_rows, max_dim, ktype);
        size_t d_emb   = packed_data_bytes(stream_win_rows, max_dim, emb_ktype);
        stream_data_bytes   = (d_emb > d_ktype) ? d_emb : d_ktype;
        emb_cache_bytes     = rms_bytes
                            + (size_t)(dim / group_size) * sizeof(float)
                            + packed_data_bytes(1, dim, emb_ktype)
                            + (size_t)TUNGUZKA_EMB_CACHE_SIZE * (sizeof(int32_t) + sizeof(uint32_t) + dim * sizeof(float))
                            + 1024;
    } else {
        fclose(f);
        model->stream_fp = NULL;
    }

    size_t arena_bytes = base_arena_bytes + stream_scales_bytes + stream_data_bytes + emb_cache_bytes;
    if (arena_init(&model->arena, arena_bytes) != 0) {
        tunguzka_free(model);
        return -1;
    }

    TransformerWeights *w = &model->weights;
    w->wq = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->wk = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->wv = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->wo = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->w1 = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->w2 = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->w3 = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);

    RunState *s = &model->state;
    s->x             = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->xb            = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->xb2           = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->hb            = (float *)arena_alloc_aligned(&model->arena, hidden_dim * sizeof(float), TUNGUZKA_ALIGN);
    s->hb2           = (float *)arena_alloc_aligned(&model->arena, hidden_dim * sizeof(float), TUNGUZKA_ALIGN);
    s->xq            = (int8_t *)arena_alloc_aligned(&model->arena, max_dim * sizeof(int8_t), TUNGUZKA_ALIGN);
    s->xq16          = (int16_t *)arena_alloc_aligned(&model->arena, max_dim * sizeof(int16_t), TUNGUZKA_ALIGN);
    s->x_scales      = (float *)arena_alloc_aligned(&model->arena, max_x_scales * sizeof(float), TUNGUZKA_ALIGN);
    s->acc_buf       = (int32_t *)arena_alloc_aligned(&model->arena, (size_t)TUNGUZKA_MMX_TILE_R * max_groups * sizeof(int32_t), TUNGUZKA_ALIGN);
    s->rope_inv_freq = (float *)arena_alloc_aligned(&model->arena, half_head * sizeof(float), TUNGUZKA_ALIGN);
    s->rope_cos      = (float *)arena_alloc_aligned(&model->arena, half_head * sizeof(float), TUNGUZKA_ALIGN);
    s->rope_sin      = (float *)arena_alloc_aligned(&model->arena, half_head * sizeof(float), TUNGUZKA_ALIGN);
    s->q             = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->att           = (float *)arena_alloc_aligned(&model->arena, seq_len * sizeof(float), TUNGUZKA_ALIGN);
    s->logits        = (float *)arena_alloc_aligned(&model->arena, vocab_size * sizeof(float), TUNGUZKA_ALIGN);

    if (use_kv8) {
        s->k              = (float *)arena_alloc_aligned(&model->arena, kv_dim * sizeof(float), TUNGUZKA_ALIGN);
        s->v              = (float *)arena_alloc_aligned(&model->arena, kv_dim * sizeof(float), TUNGUZKA_ALIGN);
        s->key_cache_q8   = (int8_t *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(int8_t), TUNGUZKA_ALIGN);
        s->value_cache_q8 = (int8_t *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(int8_t), TUNGUZKA_ALIGN);
        s->key_scales     = (float *)arena_alloc_aligned(&model->arena, kv_scale_entries * sizeof(float), TUNGUZKA_ALIGN);
        s->value_scales   = (float *)arena_alloc_aligned(&model->arena, kv_scale_entries * sizeof(float), TUNGUZKA_ALIGN);
        if (!s->value_scales) { tunguzka_free(model); return -1; }
    } else {
        s->key_cache   = (float *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(float), TUNGUZKA_ALIGN);
        s->value_cache = (float *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(float), TUNGUZKA_ALIGN);
        if (!s->value_cache) { tunguzka_free(model); return -1; }
    }

    if (model->stream_mode) {
        s->stream_win_rows   = stream_win_rows;
        s->stream_scales_buf = (float *)arena_alloc_aligned(&model->arena, stream_scales_bytes, TUNGUZKA_ALIGN);
        s->stream_data_buf   = (uint8_t *)arena_alloc_aligned(&model->arena, stream_data_bytes, TUNGUZKA_ALIGN);
        s->emb_row_scales    = (float *)arena_alloc_aligned(&model->arena, (size_t)(dim / group_size) * sizeof(float), TUNGUZKA_ALIGN);
        s->emb_row_data      = (uint8_t *)arena_alloc_aligned(&model->arena, packed_data_bytes(1, dim, emb_ktype), TUNGUZKA_ALIGN);
        s->emb_cache_tokens  = (int32_t *)arena_alloc_aligned(&model->arena, TUNGUZKA_EMB_CACHE_SIZE * sizeof(int32_t), TUNGUZKA_ALIGN);
        s->emb_cache_lru     = (uint32_t *)arena_alloc_aligned(&model->arena, TUNGUZKA_EMB_CACHE_SIZE * sizeof(uint32_t), TUNGUZKA_ALIGN);
        s->emb_cache_vecs    = (float *)arena_alloc_aligned(&model->arena, (size_t)TUNGUZKA_EMB_CACHE_SIZE * dim * sizeof(float), TUNGUZKA_ALIGN);
        if (!s->emb_cache_vecs) { tunguzka_free(model); return -1; }
        for (uint32_t i = 0; i < TUNGUZKA_EMB_CACHE_SIZE; i++) {
            s->emb_cache_tokens[i] = -1;
            s->emb_cache_lru[i] = 0;
        }
    }

    float log_theta = (model->rope_theta > 100000.0f) ? 13.122363377404328f : 9.210340371976184f;
    float scale_factor = (dim <= 3072) ? 32.0f : 8.0f;
    for (uint32_t hd = 0, idx = 0; hd < head_size; hd += 2, idx++) {
        float freq = tz_expf(-(float)hd * (log_theta / (float)head_size));
        if (model->rope_scaled && freq > 0.0f) {
            float wavelen = 6.283185307179586f / freq;
            if (wavelen > 8192.0f) {
                freq /= scale_factor;
            } else if (wavelen > 2048.0f) {
                float smooth = (8192.0f / wavelen - 1.0f) * (1.0f / 3.0f);
                freq = (1.0f - smooth) * (freq / scale_factor) + smooth * freq;
            }
        }
        s->rope_inv_freq[idx] = freq;
    }

    const uint8_t *base = (const uint8_t *)model->weights_mem;
    size_t offset = sizeof(TunguzkaHeader);
    if (offset > model->weights_size || rms_bytes > model->weights_size - offset) {
        tunguzka_free(model);
        return -1;
    }

    const float *rms_ptr = NULL;
    if (model->stream_mode) {
        float *rms_buf = (float *)arena_alloc_aligned(&model->arena, rms_bytes, TUNGUZKA_ALIGN);
        if (!rms_buf ||
            fseek(model->stream_fp, (long)(base_offset + offset), SEEK_SET) != 0 ||
            fread(rms_buf, 1, rms_bytes, model->stream_fp) != rms_bytes) {
            tunguzka_free(model);
            return -1;
        }
        rms_ptr = rms_buf;
    } else {
        rms_ptr = (const float *)(base + offset);
    }

    w->rms_att_weight   = rms_ptr; rms_ptr += (size_t)n_layers * dim;
    w->rms_ffn_weight   = rms_ptr; rms_ptr += (size_t)n_layers * dim;
    w->rms_final_weight = rms_ptr; rms_ptr += (size_t)dim;
    if (has_sub_w) {
        w->rms_sub_att_weight = rms_ptr; rms_ptr += (size_t)n_layers * dim;
        w->rms_sub_ffn_weight = rms_ptr; rms_ptr += (size_t)n_layers * hidden_dim;
    }
    offset = align_up(offset + rms_bytes, TUNGUZKA_ALIGN);

    if (bind_quantized_tensor(&w->token_embedding_table, base, &offset, model->weights_size, vocab_size, dim, group_size, emb_ktype) != 0) {
        tunguzka_free(model);
        return -1;
    }

    if (layer_maj) {
        for (uint32_t l = 0; l < n_layers; l++) {
            if (bind_quantized_tensor(&w->wq[l], base, &offset, model->weights_size, dim, dim, group_size, ktype) != 0 ||
                bind_quantized_tensor(&w->wk[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype) != 0 ||
                bind_quantized_tensor(&w->wv[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype) != 0 ||
                bind_quantized_tensor(&w->wo[l], base, &offset, model->weights_size, dim, dim, group_size, ktype) != 0 ||
                bind_quantized_tensor(&w->w1[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype) != 0 ||
                bind_quantized_tensor(&w->w3[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype) != 0 ||
                bind_quantized_tensor(&w->w2[l], base, &offset, model->weights_size, dim, hidden_dim, group_size, ktype) != 0) {
                tunguzka_free(model);
                return -1;
            }
        }
    } else {
        for (uint32_t l = 0; l < n_layers; l++) if (bind_quantized_tensor(&w->wq[l], base, &offset, model->weights_size, dim, dim, group_size, ktype) != 0) { tunguzka_free(model); return -1; }
        for (uint32_t l = 0; l < n_layers; l++) if (bind_quantized_tensor(&w->wk[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype) != 0) { tunguzka_free(model); return -1; }
        for (uint32_t l = 0; l < n_layers; l++) if (bind_quantized_tensor(&w->wv[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype) != 0) { tunguzka_free(model); return -1; }
        for (uint32_t l = 0; l < n_layers; l++) if (bind_quantized_tensor(&w->wo[l], base, &offset, model->weights_size, dim, dim, group_size, ktype) != 0) { tunguzka_free(model); return -1; }
        for (uint32_t l = 0; l < n_layers; l++) if (bind_quantized_tensor(&w->w1[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype) != 0) { tunguzka_free(model); return -1; }
        for (uint32_t l = 0; l < n_layers; l++) if (bind_quantized_tensor(&w->w2[l], base, &offset, model->weights_size, dim, hidden_dim, group_size, ktype) != 0) { tunguzka_free(model); return -1; }
        for (uint32_t l = 0; l < n_layers; l++) if (bind_quantized_tensor(&w->w3[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype) != 0) { tunguzka_free(model); return -1; }
    }

    int shared_cls = (model->config.flags & MODEL_FLAG_SHARED_CLASSIFIER) != 0;
    model->total_tensors = 7 * n_layers + (shared_cls ? 1 : 2);

    if (shared_cls) {
        w->wcls = w->token_embedding_table;
    } else {
        if (bind_quantized_tensor(&w->wcls, base, &offset, model->weights_size, vocab_size, dim, group_size, emb_ktype) != 0) {
            tunguzka_free(model);
            return -1;
        }
    }

    if (model->stream_mode && max_ram_bytes > model->arena.offset + 1024) {
        size_t spare_budget = max_ram_bytes - model->arena.offset;
        model->pin_pool = (uint8_t *)malloc(spare_budget + TUNGUZKA_ALIGN);
        if (model->pin_pool) {
            uintptr_t raw_addr = (uintptr_t)model->pin_pool;
            uintptr_t aligned_addr = (raw_addr + (TUNGUZKA_ALIGN - 1)) & ~(uintptr_t)(TUNGUZKA_ALIGN - 1);
            uint8_t *pin_base = (uint8_t *)aligned_addr;
            size_t pool_off = 0;

            for (uint32_t l = 0; l < n_layers; l++) {
                try_pin_tensor_ex(model, pin_base, &w->wq[l], ktype, &pool_off, spare_budget);
                try_pin_tensor_ex(model, pin_base, &w->wk[l], ktype, &pool_off, spare_budget);
                try_pin_tensor_ex(model, pin_base, &w->wv[l], ktype, &pool_off, spare_budget);
                try_pin_tensor_ex(model, pin_base, &w->wo[l], ktype, &pool_off, spare_budget);
                try_pin_tensor_ex(model, pin_base, &w->w1[l], ktype, &pool_off, spare_budget);
                try_pin_tensor_ex(model, pin_base, &w->w3[l], ktype, &pool_off, spare_budget);
                try_pin_tensor_ex(model, pin_base, &w->w2[l], ktype, &pool_off, spare_budget);
            }
            try_pin_tensor_ex(model, pin_base, &w->token_embedding_table, emb_ktype, &pool_off, spare_budget);
            if (shared_cls) {
                w->wcls = w->token_embedding_table;
            } else {
                try_pin_tensor_ex(model, pin_base, &w->wcls, emb_ktype, &pool_off, spare_budget);
            }
            model->pin_pool_size = pool_off;
        }
    } else if (!model->stream_mode) {
        model->pinned_tensors = model->total_tensors;
    }

    return 0;
}

int tunguzka_reconfigure(TunguzkaModel *model, uint32_t new_max_seq, int new_use_kv8) {
    if (!model || model->stream_mode || !model->weights_mem) return -1;

    uint32_t target_seq = model->orig_max_seq_len;
    if (new_max_seq > 0 && new_max_seq < target_seq) {
        target_seq = new_max_seq;
    }
    if (target_seq == 0 || target_seq > 32768) return -1;

    uint32_t dim        = model->config.dim;
    uint32_t hidden_dim = model->config.hidden_dim;
    uint32_t n_layers   = model->config.n_layers;
    uint32_t n_heads    = model->config.n_heads;
    uint32_t n_kv_heads = model->config.n_kv_heads;
    uint32_t vocab_size = model->config.vocab_size;
    uint32_t group_size = model->config.group_size;
    uint8_t  ktype      = model->config.kernel_type;
    uint8_t  emb_ktype  = get_emb_wcls_ktype(&model->config);
    int      has_sub_w  = (model->config.flags & MODEL_FLAG_HAS_SUBNORM_WEIGHTS) != 0;
    int      layer_maj  = (model->config.flags & MODEL_FLAG_LAYER_MAJOR) != 0 || (model->config.version >= 2);

    uint32_t head_size    = dim / n_heads;
    uint32_t half_head    = head_size / 2;
    uint32_t kv_dim       = (dim * n_kv_heads) / n_heads;
    uint32_t max_dim      = (hidden_dim > dim) ? hidden_dim : dim;
    uint32_t max_groups   = max_dim / group_size;
    uint32_t max_x_scales = (n_heads > max_groups) ? n_heads : max_groups;

    size_t kv_entries = (size_t)n_layers * target_seq * kv_dim;
    size_t kv_scale_entries = (size_t)n_layers * target_seq * n_kv_heads;
    size_t rms_floats = (size_t)2 * n_layers * dim + dim;
    if (has_sub_w) {
        rms_floats += (size_t)n_layers * dim + (size_t)n_layers * hidden_dim;
    }
    size_t rms_bytes = rms_floats * sizeof(float);

    size_t base_arena_bytes = 0;
    base_arena_bytes += 7 * (size_t)n_layers * sizeof(QuantizedTensor) + 1024;
    base_arena_bytes += (3 * (size_t)dim + 2 * (size_t)hidden_dim + dim + 2 * kv_dim + target_seq + vocab_size + 3 * half_head) * sizeof(float);
    base_arena_bytes += max_dim * sizeof(int8_t) + max_dim * sizeof(int16_t) + max_x_scales * sizeof(float);
    base_arena_bytes += (size_t)TUNGUZKA_MMX_TILE_R * max_groups * sizeof(int32_t);
    if (new_use_kv8) {
        base_arena_bytes += 2 * kv_entries * sizeof(int8_t) + 2 * kv_scale_entries * sizeof(float);
    } else {
        base_arena_bytes += 2 * kv_entries * sizeof(float);
    }
    base_arena_bytes += 2048;

    MemoryArena new_arena;
    if (arena_init(&new_arena, base_arena_bytes) != 0) {
        return -1;
    }

    arena_free(&model->arena);
    model->arena = new_arena;
    model->config.max_seq_len = target_seq;
    model->use_kv8 = new_use_kv8 ? 1 : 0;
    memset(&model->state, 0, sizeof(RunState));

    TransformerWeights *w = &model->weights;
    w->wq = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->wk = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->wv = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->wo = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->w1 = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->w2 = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);
    w->w3 = (QuantizedTensor *)arena_alloc_aligned(&model->arena, n_layers * sizeof(QuantizedTensor), TUNGUZKA_ALIGN);

    RunState *s = &model->state;
    s->x             = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->xb            = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->xb2           = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->hb            = (float *)arena_alloc_aligned(&model->arena, hidden_dim * sizeof(float), TUNGUZKA_ALIGN);
    s->hb2           = (float *)arena_alloc_aligned(&model->arena, hidden_dim * sizeof(float), TUNGUZKA_ALIGN);
    s->xq            = (int8_t *)arena_alloc_aligned(&model->arena, max_dim * sizeof(int8_t), TUNGUZKA_ALIGN);
    s->xq16          = (int16_t *)arena_alloc_aligned(&model->arena, max_dim * sizeof(int16_t), TUNGUZKA_ALIGN);
    s->x_scales      = (float *)arena_alloc_aligned(&model->arena, max_x_scales * sizeof(float), TUNGUZKA_ALIGN);
    s->acc_buf       = (int32_t *)arena_alloc_aligned(&model->arena, (size_t)TUNGUZKA_MMX_TILE_R * max_groups * sizeof(int32_t), TUNGUZKA_ALIGN);
    s->rope_inv_freq = (float *)arena_alloc_aligned(&model->arena, half_head * sizeof(float), TUNGUZKA_ALIGN);
    s->rope_cos      = (float *)arena_alloc_aligned(&model->arena, half_head * sizeof(float), TUNGUZKA_ALIGN);
    s->rope_sin      = (float *)arena_alloc_aligned(&model->arena, half_head * sizeof(float), TUNGUZKA_ALIGN);
    s->q             = (float *)arena_alloc_aligned(&model->arena, dim * sizeof(float), TUNGUZKA_ALIGN);
    s->att           = (float *)arena_alloc_aligned(&model->arena, target_seq * sizeof(float), TUNGUZKA_ALIGN);
    s->logits        = (float *)arena_alloc_aligned(&model->arena, vocab_size * sizeof(float), TUNGUZKA_ALIGN);

    if (new_use_kv8) {
        s->k              = (float *)arena_alloc_aligned(&model->arena, kv_dim * sizeof(float), TUNGUZKA_ALIGN);
        s->v              = (float *)arena_alloc_aligned(&model->arena, kv_dim * sizeof(float), TUNGUZKA_ALIGN);
        s->key_cache_q8   = (int8_t *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(int8_t), TUNGUZKA_ALIGN);
        s->value_cache_q8 = (int8_t *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(int8_t), TUNGUZKA_ALIGN);
        s->key_scales     = (float *)arena_alloc_aligned(&model->arena, kv_scale_entries * sizeof(float), TUNGUZKA_ALIGN);
        s->value_scales   = (float *)arena_alloc_aligned(&model->arena, kv_scale_entries * sizeof(float), TUNGUZKA_ALIGN);
    } else {
        s->key_cache   = (float *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(float), TUNGUZKA_ALIGN);
        s->value_cache = (float *)arena_alloc_aligned(&model->arena, kv_entries * sizeof(float), TUNGUZKA_ALIGN);
    }

    float log_theta = (model->rope_theta > 100000.0f) ? 13.122363377404328f : 9.210340371976184f;
    float scale_factor = (dim <= 3072) ? 32.0f : 8.0f;
    for (uint32_t hd = 0, idx = 0; hd < head_size; hd += 2, idx++) {
        float freq = tz_expf(-(float)hd * (log_theta / (float)head_size));
        if (model->rope_scaled && freq > 0.0f) {
            float wavelen = 6.283185307179586f / freq;
            if (wavelen > 8192.0f) {
                freq /= scale_factor;
            } else if (wavelen > 2048.0f) {
                float smooth = (8192.0f / wavelen - 1.0f) * (1.0f / 3.0f);
                freq = (1.0f - smooth) * (freq / scale_factor) + smooth * freq;
            }
        }
        s->rope_inv_freq[idx] = freq;
    }

    const uint8_t *base = (const uint8_t *)model->weights_mem;
    size_t offset = sizeof(TunguzkaHeader);
    const float *rms_ptr = (const float *)(base + offset);

    w->rms_att_weight   = rms_ptr; rms_ptr += (size_t)n_layers * dim;
    w->rms_ffn_weight   = rms_ptr; rms_ptr += (size_t)n_layers * dim;
    w->rms_final_weight = rms_ptr; rms_ptr += (size_t)dim;
    if (has_sub_w) {
        w->rms_sub_att_weight = rms_ptr; rms_ptr += (size_t)n_layers * dim;
        w->rms_sub_ffn_weight = rms_ptr; rms_ptr += (size_t)n_layers * hidden_dim;
    }
    offset = align_up(offset + rms_bytes, TUNGUZKA_ALIGN);

    bind_quantized_tensor(&w->token_embedding_table, base, &offset, model->weights_size, vocab_size, dim, group_size, emb_ktype);
    if (layer_maj) {
        for (uint32_t l = 0; l < n_layers; l++) {
            bind_quantized_tensor(&w->wq[l], base, &offset, model->weights_size, dim, dim, group_size, ktype);
            bind_quantized_tensor(&w->wk[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype);
            bind_quantized_tensor(&w->wv[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype);
            bind_quantized_tensor(&w->wo[l], base, &offset, model->weights_size, dim, dim, group_size, ktype);
            bind_quantized_tensor(&w->w1[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype);
            bind_quantized_tensor(&w->w3[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype);
            bind_quantized_tensor(&w->w2[l], base, &offset, model->weights_size, dim, hidden_dim, group_size, ktype);
        }
    } else {
        for (uint32_t l = 0; l < n_layers; l++) bind_quantized_tensor(&w->wq[l], base, &offset, model->weights_size, dim, dim, group_size, ktype);
        for (uint32_t l = 0; l < n_layers; l++) bind_quantized_tensor(&w->wk[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype);
        for (uint32_t l = 0; l < n_layers; l++) bind_quantized_tensor(&w->wv[l], base, &offset, model->weights_size, kv_dim, dim, group_size, ktype);
        for (uint32_t l = 0; l < n_layers; l++) bind_quantized_tensor(&w->wo[l], base, &offset, model->weights_size, dim, dim, group_size, ktype);
        for (uint32_t l = 0; l < n_layers; l++) bind_quantized_tensor(&w->w1[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype);
        for (uint32_t l = 0; l < n_layers; l++) bind_quantized_tensor(&w->w2[l], base, &offset, model->weights_size, dim, hidden_dim, group_size, ktype);
        for (uint32_t l = 0; l < n_layers; l++) bind_quantized_tensor(&w->w3[l], base, &offset, model->weights_size, hidden_dim, dim, group_size, ktype);
    }

    if ((model->config.flags & MODEL_FLAG_SHARED_CLASSIFIER) != 0) {
        w->wcls = w->token_embedding_table;
    } else {
        bind_quantized_tensor(&w->wcls, base, &offset, model->weights_size, vocab_size, dim, group_size, emb_ktype);
    }

    return 0;
}

int tunguzka_shift_kv_cache(TunguzkaModel *model, int cur_pos, int shift) {
    TunguzkaHeader *c = &model->config;
    RunState *s = &model->state;
    if (shift <= 0 || shift >= cur_pos) return 0;

    int keep = cur_pos - shift;
    uint32_t dim        = c->dim;
    uint32_t n_layers   = c->n_layers;
    uint32_t n_heads    = c->n_heads;
    uint32_t n_kv_heads = c->n_kv_heads;
    uint32_t head_size  = dim / n_heads;
    uint32_t half_head  = head_size / 2;
    uint32_t kv_dim     = (dim * n_kv_heads) / n_heads;

    for (uint32_t idx = 0; idx < half_head; idx++) {
        float val = -(float)shift * s->rope_inv_freq[idx];
        tz_sincosf(val, &s->rope_sin[idx], &s->rope_cos[idx]);
    }

    for (uint32_t l = 0; l < n_layers; l++) {
        size_t loff = (size_t)l * c->max_seq_len * kv_dim;
        if (model->use_kv8) {
            size_t soff_base = (size_t)l * c->max_seq_len * n_kv_heads;
            memmove(s->key_cache_q8 + loff,
                    s->key_cache_q8 + loff + (size_t)shift * kv_dim,
                    (size_t)keep * kv_dim * sizeof(int8_t));
            memmove(s->value_cache_q8 + loff,
                    s->value_cache_q8 + loff + (size_t)shift * kv_dim,
                    (size_t)keep * kv_dim * sizeof(int8_t));
            memmove(s->key_scales + soff_base,
                    s->key_scales + soff_base + (size_t)shift * n_kv_heads,
                    (size_t)keep * n_kv_heads * sizeof(float));
            memmove(s->value_scales + soff_base,
                    s->value_scales + soff_base + (size_t)shift * n_kv_heads,
                    (size_t)keep * n_kv_heads * sizeof(float));

            for (int t = 0; t < keep; t++) {
                int8_t *k_q8 = s->key_cache_q8 + loff + (size_t)t * kv_dim;
                float  *ks   = s->key_scales + soff_base + (size_t)t * n_kv_heads;
                for (uint32_t h = 0; h < n_kv_heads; h++) {
                    float sc = ks[h];
                    for (uint32_t i = 0; i < head_size; i++) {
                        s->k[h * head_size + i] = (float)k_q8[h * head_size + i] * sc;
                    }
                }
                rope_apply_precomputed(s->k, NULL, s->rope_cos, s->rope_sin, kv_dim, 0, head_size);
                quantize_activations_q8(s->k, k_q8, ks, kv_dim, head_size);
            }
        } else {
            memmove(s->key_cache + loff,
                    s->key_cache + loff + (size_t)shift * kv_dim,
                    (size_t)keep * kv_dim * sizeof(float));
            memmove(s->value_cache + loff,
                    s->value_cache + loff + (size_t)shift * kv_dim,
                    (size_t)keep * kv_dim * sizeof(float));
            for (int t = 0; t < keep; t++) {
                float *k_vec = s->key_cache + loff + (size_t)t * kv_dim;
                rope_apply_precomputed(k_vec, NULL, s->rope_cos, s->rope_sin, kv_dim, 0, head_size);
            }
        }
    }
    return keep;
}

static void subnorm_unit_inplace(float *x, uint32_t size, float eps) {
    float ss = 0.0f;
    for (uint32_t j = 0; j < size; j++) ss += x[j] * x[j];
    ss = 1.0f / tz_sqrtf(ss / (float)size + eps);
    for (uint32_t j = 0; j < size; j++) x[j] *= ss;
}

float *tunguzka_forward_ex(TunguzkaModel *model, int token, int pos, int compute_logits) {
    TunguzkaHeader *c = &model->config;
    TransformerWeights *w = &model->weights;
    RunState *s = &model->state;

    if (token < 0 || (uint32_t)token >= c->vocab_size) token = 0;
    if (pos < 0 || (uint32_t)pos >= c->max_seq_len) pos = (int)c->max_seq_len - 1;

    uint32_t dim        = c->dim;
    uint32_t hidden_dim = c->hidden_dim;
    uint32_t n_layers   = c->n_layers;
    uint32_t n_heads    = c->n_heads;
    uint32_t n_kv_heads = c->n_kv_heads;
    uint32_t group_size = c->group_size;
    uint32_t head_size  = dim / n_heads;
    uint32_t half_head  = head_size / 2;
    uint32_t kv_dim     = (dim * n_kv_heads) / n_heads;
    uint32_t kv_mul     = n_heads / n_kv_heads;
    uint8_t  emb_ktype  = get_emb_wcls_ktype(c);
    int      has_sub_w  = (c->flags & MODEL_FLAG_HAS_SUBNORM_WEIGHTS) != 0;
    int      has_subnorm = (c->flags & MODEL_FLAG_BITNET_SUBNORM) != 0;
    int      use_relu2  = (c->flags & MODEL_FLAG_RELU2) != 0;
    float    eps        = model->rms_eps;

    load_token_embedding(model, token);

    for (uint32_t idx = 0; idx < half_head; idx++) {
        float val = (float)pos * s->rope_inv_freq[idx];
        tz_sincosf(val, &s->rope_sin[idx], &s->rope_cos[idx]);
    }

    float inv_sqrt_head = 1.0f / tz_sqrtf((float)head_size);

    for (uint32_t l = 0; l < n_layers; l++) {
        if (model->backend == BACKEND_3DNOW) {
            rmsnorm_3dnow(s->xb, s->x, w->rms_att_weight + l * dim, dim, eps);
        } else {
            rmsnorm_scalar(s->xb, s->x, w->rms_att_weight + l * dim, dim, eps);
        }

        size_t loff = (size_t)l * c->max_seq_len * kv_dim;
        if (!model->use_kv8) {
            s->k = s->key_cache + loff + (size_t)pos * kv_dim;
            s->v = s->value_cache + loff + (size_t)pos * kv_dim;
        }

        prepare_activations(s->xb, s->xq, s->xq16, s->x_scales, dim, group_size, model->backend);
        matmul_model_tensor(model, s->q, &w->wq[l], (int32_t)l);
        matmul_model_tensor(model, s->k, &w->wk[l], (int32_t)l);
        matmul_model_tensor(model, s->v, &w->wv[l], (int32_t)l);

        rope_apply_precomputed(s->q, s->k, s->rope_cos, s->rope_sin, dim, kv_dim, head_size);

        if (model->use_kv8) {
            size_t soff = ((size_t)l * c->max_seq_len + (size_t)pos) * n_kv_heads;
            int8_t *k_q8_dst = s->key_cache_q8 + loff + (size_t)pos * kv_dim;
            int8_t *v_q8_dst = s->value_cache_q8 + loff + (size_t)pos * kv_dim;
            quantize_activations_q8(s->k, k_q8_dst, s->key_scales + soff, kv_dim, head_size);
            quantize_activations_q8(s->v, v_q8_dst, s->value_scales + soff, kv_dim, head_size);
            quantize_activations_q8(s->q, s->xq, s->x_scales, dim, head_size);
            if (model->backend == BACKEND_MMX || model->backend == BACKEND_3DNOW) {
                unpack_xq_to_i16_mmx(s->xq, s->xq16, dim);
            }
        }

        float *att = s->att;
        for (uint32_t h = 0; h < n_heads; h++) {
            uint32_t kv_h = h / kv_mul;
            uint32_t kv_head_off = kv_h * head_size;
            float *xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));

            if (model->use_kv8) {
                const int16_t *q16 = s->xq16 + h * head_size;
                const int8_t  *q8  = s->xq + h * head_size;
                float q_scale = s->x_scales[h] * inv_sqrt_head;

                for (int t = 0; t <= pos; t++) {
                    const int8_t *k8 = s->key_cache_q8 + loff + (size_t)t * kv_dim + kv_head_off;
                    float k_scale = s->key_scales[((size_t)l * c->max_seq_len + (size_t)t) * n_kv_heads + kv_h];
                    int32_t iscore = 0;
                    if (model->backend == BACKEND_MMX || model->backend == BACKEND_3DNOW) {
                        iscore = dot_q8_i16_mmx(q16, k8, head_size);
                    } else {
                        for (uint32_t i = 0; i < head_size; i++) iscore += (int32_t)q8[i] * (int32_t)k8[i];
                    }
                    att[t] = (float)iscore * q_scale * k_scale;
                }

                softmax_scalar(att, (uint32_t)(pos + 1));

                for (int t = 0; t <= pos; t++) {
                    const int8_t *v8 = s->value_cache_q8 + loff + (size_t)t * kv_dim + kv_head_off;
                    float v_scale = s->value_scales[((size_t)l * c->max_seq_len + (size_t)t) * n_kv_heads + kv_h];
                    float a = att[t] * v_scale;
                    for (uint32_t i = 0; i < head_size; i++) {
                        xb[i] += a * (float)v8[i];
                    }
                }
            } else {
                float *q = s->q + h * head_size;
                for (int t = 0; t <= pos; t++) {
                    float *k = s->key_cache + loff + (size_t)t * kv_dim + kv_head_off;
                    float score = (model->backend == BACKEND_3DNOW) ? dot_f32_3dnow(q, k, head_size) : 0.0f;
                    if (model->backend != BACKEND_3DNOW) {
                        for (uint32_t i = 0; i < head_size; i++) score += q[i] * k[i];
                    }
                    att[t] = score * inv_sqrt_head;
                }

                softmax_scalar(att, (uint32_t)(pos + 1));

                for (int t = 0; t <= pos; t++) {
                    float *v = s->value_cache + loff + (size_t)t * kv_dim + kv_head_off;
                    float a = att[t];
                    if (model->backend == BACKEND_3DNOW) {
                        saxpy_f32_3dnow(xb, a, v, head_size);
                    } else {
                        for (uint32_t i = 0; i < head_size; i++) xb[i] += a * v[i];
                    }
                }
            }
        }

        if (has_sub_w) {
            if (model->backend == BACKEND_3DNOW) {
                rmsnorm_3dnow(s->xb, s->xb, w->rms_sub_att_weight + l * dim, dim, eps);
            } else {
                rmsnorm_scalar(s->xb, s->xb, w->rms_sub_att_weight + l * dim, dim, eps);
            }
        } else if (has_subnorm) {
            subnorm_unit_inplace(s->xb, dim, eps);
        }

        prepare_activations(s->xb, s->xq, s->xq16, s->x_scales, dim, group_size, model->backend);
        matmul_model_tensor(model, s->xb2, &w->wo[l], (int32_t)l);
        for (uint32_t i = 0; i < dim; i++) s->x[i] += s->xb2[i];

        if (model->backend == BACKEND_3DNOW) {
            rmsnorm_3dnow(s->xb, s->x, w->rms_ffn_weight + l * dim, dim, eps);
        } else {
            rmsnorm_scalar(s->xb, s->x, w->rms_ffn_weight + l * dim, dim, eps);
        }

        prepare_activations(s->xb, s->xq, s->xq16, s->x_scales, dim, group_size, model->backend);
        matmul_model_tensor(model, s->hb,  &w->w1[l], (int32_t)l);
        matmul_model_tensor(model, s->hb2, &w->w3[l], (int32_t)l);

        if (use_relu2) {
            for (uint32_t i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val = (val > 0.0f) ? (val * val) : 0.0f;
                s->hb[i] = val * s->hb2[i];
            }
        } else {
            for (uint32_t i = 0; i < hidden_dim; i++) {
                float val = s->hb[i];
                val *= (1.0f / (1.0f + tz_expf(-val)));
                val *= s->hb2[i];
                s->hb[i] = val;
            }
        }

        if (has_sub_w) {
            if (model->backend == BACKEND_3DNOW) {
                rmsnorm_3dnow(s->hb, s->hb, w->rms_sub_ffn_weight + l * hidden_dim, hidden_dim, eps);
            } else {
                rmsnorm_scalar(s->hb, s->hb, w->rms_sub_ffn_weight + l * hidden_dim, hidden_dim, eps);
            }
        } else if (has_subnorm) {
            subnorm_unit_inplace(s->hb, hidden_dim, eps);
        }

        prepare_activations(s->hb, s->xq, s->xq16, s->x_scales, hidden_dim, group_size, model->backend);
        matmul_model_tensor(model, s->xb, &w->w2[l], (int32_t)l);
        for (uint32_t i = 0; i < dim; i++) s->x[i] += s->xb[i];
    }

    if (!compute_logits) {
        return s->logits;
    }

    if (model->backend == BACKEND_3DNOW) {
        rmsnorm_3dnow(s->xb, s->x, w->rms_final_weight, dim, eps);
    } else {
        rmsnorm_scalar(s->xb, s->x, w->rms_final_weight, dim, eps);
    }

    prepare_activations(s->xb, s->xq, s->xq16, s->x_scales, dim, group_size, model->backend);
    matmul_model_tensor_ex(model, s->logits, &w->wcls, emb_ktype, (int32_t)n_layers);
    return s->logits;
}

float *tunguzka_forward(TunguzkaModel *model, int token, int pos) {
    return tunguzka_forward_ex(model, token, pos, 1);
}

void tunguzka_free(TunguzkaModel *model) {
    arena_free(&model->arena);
    if (model->pin_pool) {
        free(model->pin_pool);
        model->pin_pool = NULL;
    }
    if (model->stream_fp) {
        fclose(model->stream_fp);
        model->stream_fp = NULL;
    }
    if (model->weights_mem) {
#if TZ_HAS_MMAP
        if (model->is_mmap) munmap(model->weights_mem, model->weights_size);
        else free(model->weights_mem);
#else
        free(model->weights_mem);
#endif
        model->weights_mem = NULL;
    }
}