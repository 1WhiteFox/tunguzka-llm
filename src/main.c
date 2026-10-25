#ifndef _DEFAULT_SOURCE
#define _DEFAULT_SOURCE
#endif
#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif

#if defined(_WIN32) && !defined(__USE_MINGW_ANSI_STDIO)
#define __USE_MINGW_ANSI_STDIO 0
#endif

#include "../include/tunguzka.h"
#include "../include/mmx_kernels.h"
#include "../include/3dnow_kernels.h"
#include "../include/tui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#include <dirent.h>
#endif

#if defined(__DJGPP__)
unsigned _stklen = 65536U;
#endif

#if defined(_WIN32) && (defined(__GNUC__) || defined(__clang__))
__attribute__((used, noinline)) void __tz_i586_trampoline_cave(void) {
    __asm__ volatile (".fill 560, 1, 0x90\n\t");
}
#endif

typedef struct {
    char    **vocab;
    char     *vocab_pool;
    float    *vocab_scores;
    int32_t  *hash_table;
    uint32_t  hash_mask;
    uint32_t  vocab_size;
    uint32_t  max_token_length;
    int       is_byte_level;
    unsigned char byte_pieces[512];
} Tokenizer;

static uint32_t fnv1a_hash_str(const char *s) {
    uint32_t h = 2166136261u;
    while (*s) {
        h ^= (uint8_t)(*s++);
        h *= 16777619u;
    }
    return h;
}

static void tokenizer_free(Tokenizer *t) {
    if (t->hash_table) { free(t->hash_table); t->hash_table = NULL; }
    if (t->vocab_pool) { free(t->vocab_pool); t->vocab_pool = NULL; }
    if (t->vocab) { free(t->vocab); t->vocab = NULL; }
    if (t->vocab_scores) { free(t->vocab_scores); t->vocab_scores = NULL; }
}

static int tokenizer_init_slice(Tokenizer *t, const char *path, size_t offset, size_t slice_bytes, uint32_t vocab_size) {
    memset(t, 0, sizeof(Tokenizer));
    if (vocab_size == 0 || vocab_size > 256000) return -1;
    t->vocab_size = vocab_size;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    long fsize = ftell(f);
    if (fsize <= 0 || (size_t)fsize <= offset + 4) { fclose(f); return -1; }
    size_t avail = (slice_bytes > 0 && offset + slice_bytes <= (size_t)fsize) ? slice_bytes : ((size_t)fsize - offset);
    if (avail > 64u * 1024u * 1024u) { fclose(f); return -1; }
    if (fseek(f, (long)offset, SEEK_SET) != 0) { fclose(f); return -1; }

    uint32_t hcap = 1024u;
    while (hcap < vocab_size * 2u) hcap <<= 1;

    size_t pool_cap = avail + (size_t)vocab_size + 16;
    t->vocab = (char **)calloc(vocab_size, sizeof(char *));
    t->vocab_scores = (float *)calloc(vocab_size, sizeof(float));
    t->vocab_pool = (char *)malloc(pool_cap);
    t->hash_table = (int32_t *)malloc((size_t)hcap * sizeof(int32_t));
    if (!t->vocab || !t->vocab_scores || !t->vocab_pool || !t->hash_table) {
        fclose(f);
        tokenizer_free(t);
        return -1;
    }
    t->hash_mask = hcap - 1u;
    for (uint32_t i = 0; i < hcap; i++) t->hash_table[i] = -1;

    if (fread(&t->max_token_length, sizeof(uint32_t), 1, f) != 1 ||
        t->max_token_length == 0 || t->max_token_length > 1024) {
        fclose(f);
        tokenizer_free(t);
        return -1;
    }

    size_t pool_off = 0;
    int has_hex_bytes = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        uint32_t len = 0;
        if (fread(t->vocab_scores + i, sizeof(float), 1, f) != 1 ||
            fread(&len, sizeof(uint32_t), 1, f) != 1) {
            if (i >= 32000 && vocab_size <= 32064) {
                t->vocab_scores[i] = -1e10f;
                t->vocab[i] = "";
                continue;
            }
            fclose(f);
            tokenizer_free(t);
            return -1;
        }
        if (len > t->max_token_length || pool_off + len + 1 > pool_cap) {
            fclose(f);
            tokenizer_free(t);
            return -1;
        }
        char *dst = t->vocab_pool + pool_off;
        if (fread(dst, 1, len, f) != len) {
            fclose(f);
            tokenizer_free(t);
            return -1;
        }
        dst[len] = '\0';
        t->vocab[i] = dst;
        pool_off += len + 1;

        if (len == 6 && dst[0] == '<' && dst[1] == '0' && dst[2] == 'x' && dst[5] == '>') {
            has_hex_bytes = 1;
        }
        if (dst[0] != '\0') {
            uint32_t slot = fnv1a_hash_str(dst) & t->hash_mask;
            while (t->hash_table[slot] != -1) {
                if (strcmp(t->vocab[t->hash_table[slot]], dst) == 0) {
                    if (t->vocab_scores[i] > t->vocab_scores[t->hash_table[slot]]) {
                        t->hash_table[slot] = (int32_t)i;
                    }
                    goto next_tok;
                }
                slot = (slot + 1u) & t->hash_mask;
            }
            t->hash_table[slot] = (int32_t)i;
        }
    next_tok:;
    }
    fclose(f);
    t->is_byte_level = (vocab_size >= 49000 || (!has_hex_bytes && vocab_size > 512)) ? 1 : 0;
    return 0;
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static char *tokenizer_decode(Tokenizer *t, int prev_token, int token) {
    if (!t || token < 0 || (uint32_t)token >= t->vocab_size) return "";
    char *piece = t->vocab[token];
    if (t->vocab_size < 100000 && prev_token == 1 && piece[0] == ' ') piece++;
    if (piece[0] == '<' && piece[1] == '0' && (piece[2] == 'x' || piece[2] == 'X') &&
        piece[5] == '>' && piece[6] == '\0') {
        int hi = hex_nibble(piece[3]);
        int lo = hex_nibble(piece[4]);
        if (hi >= 0 && lo >= 0) {
            unsigned int byte_val = (unsigned int)((hi << 4) | lo);
            piece = (char *)t->byte_pieces + byte_val * 2;
        }
    }
    return piece;
}

static int utf8_expected_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead & 0xE0) == 0xC0) return 2;
    if ((lead & 0xF0) == 0xE0) return 3;
    if ((lead & 0xF8) == 0xF0) return 4;
    return 1;
}

static char g_cli_utf8_carry[4];
static int  g_cli_utf8_carry_len = 0;

static void safe_print(const char *piece) {
    if (!piece || piece[0] == '\0') return;
    if (piece[1] == '\0' && g_cli_utf8_carry_len == 0) {
        unsigned char b = (unsigned char)piece[0];
        if (b < 0x20 && b != '\n' && b != '\r' && b != '\t') return;
        if (b >= 0x7F && b < 0xC0) return;
    }

    char merged[512];
    int m_len = 0;
    if (g_cli_utf8_carry_len > 0) {
        memcpy(merged, g_cli_utf8_carry, (size_t)g_cli_utf8_carry_len);
        m_len = g_cli_utf8_carry_len;
        g_cli_utf8_carry_len = 0;
    }
    for (const char *p = piece; *p != '\0' && m_len < (int)sizeof(merged) - 2; p++) {
        merged[m_len++] = *p;
    }
    merged[m_len] = '\0';

    int check_back = (m_len < 3) ? m_len : 3;
    for (int back = 1; back <= check_back; back++) {
        int idx = m_len - back;
        unsigned char b = (unsigned char)merged[idx];
        if ((b & 0xC0) == 0x80) continue;
        if (b >= 0xC0) {
            int need = utf8_expected_len(b);
            if (back < need) {
                memcpy(g_cli_utf8_carry, merged + idx, (size_t)back);
                g_cli_utf8_carry_len = back;
                m_len = idx;
                merged[m_len] = '\0';
            }
        }
        break;
    }

    if (m_len > 0) {
        fwrite(merged, 1, (size_t)m_len, stdout);
        fflush(stdout);
    }
}

static int tok_lookup(const Tokenizer *t, const char *str) {
    if (!str || str[0] == '\0' || !t->hash_table) return -1;
    uint32_t slot = fnv1a_hash_str(str) & t->hash_mask;
    while (t->hash_table[slot] != -1) {
        int id = t->hash_table[slot];
        if (strcmp(t->vocab[id], str) == 0) return id;
        slot = (slot + 1u) & t->hash_mask;
    }
    return -1;
}

static int byte_category(unsigned char b) {
    if (b == '\n' || b == '\r') return 1;
    if (b == ' ' || b == '\t')  return 2;
    if (b >= '0' && b <= '9')   return 3;
    if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z') || b >= 0x80) return 4;
    return 5;
}

static int can_bpe_merge_bytelevel(const char *s1, const char *s2) {
    size_t l1 = strlen(s1), l2 = strlen(s2);
    if (l1 == 0 || l2 == 0) return 0;
    unsigned char f1 = (unsigned char)s1[0];
    unsigned char e1 = (unsigned char)s1[l1 - 1];
    unsigned char f2 = (unsigned char)s2[0];
    int c_e1 = byte_category(e1);
    int c_f2 = byte_category(f2);
    if (c_e1 == 1 || c_f2 == 1) return (c_e1 == 1 && c_f2 == 1);
    if (c_f2 == 2) return (c_e1 == 2);
    if (e1 == ' ' && l1 == 1 && (c_f2 == 4 || c_f2 == 5)) return 1;
    if (f1 == ' ' && l1 > 1) {
        int c_second = byte_category((unsigned char)s1[1]);
        return (c_second == c_f2 && c_f2 != 3);
    }
    if (c_e1 == 3 && c_f2 == 3) return (l1 + l2 <= 3);
    return (c_e1 == c_f2);
}

static void bpe_encode(Tokenizer *t, const char *text, int *tokens, int *n_tokens, int max_tokens, int add_bos) {
    *n_tokens = 0;
    if (!t || max_tokens < 1) return;
    if (add_bos) {
        if (t->vocab_size >= 100000) {
            tokens[(*n_tokens)++] = 128000;
        } else {
            tokens[(*n_tokens)++] = (t->vocab_size > 1) ? 1 : 0;
        }
    }
    if (!text || text[0] == '\0') return;

    size_t buf_cap = (size_t)t->max_token_length * 2 + 16;
    char *str_buffer = (char *)malloc(buf_cap);
    if (!str_buffer) return;

    int start_merge_idx = *n_tokens;
    if (!t->is_byte_level && t->vocab_size >= 512 && t->vocab_size < 100000 &&
        text[0] != ' ' && text[0] != '\n' && *n_tokens < max_tokens) {
        int sp_id = tok_lookup(t, " ");
        if (sp_id != -1) {
            tokens[(*n_tokens)++] = sp_id;
        }
    }

    for (const char *c = text; *c != '\0' && *n_tokens < max_tokens; ) {
        if (c[0] == '<') {
            const char *end = NULL;
            if (c[1] == '|') {
                end = strstr(c + 2, "|>");
                if (end) end += 1;
            } else {
                end = strchr(c + 1, '>');
            }
            if (end) {
                size_t tag_len = (size_t)(end - c + 1);
                if (tag_len < 48 && tag_len < buf_cap) {
                    memcpy(str_buffer, c, tag_len);
                    str_buffer[tag_len] = '\0';
                    int tag_id = tok_lookup(t, str_buffer);
                    if (tag_id != -1) {
                        tokens[(*n_tokens)++] = tag_id;
                        c += tag_len;
                        continue;
                    }
                }
            }
        }

        unsigned char b0 = (unsigned char)c[0];
        int char_len = 1;
        if (!t->is_byte_level) {
            if ((b0 & 0xE0) == 0xC0 && (c[1] & 0xC0) == 0x80) char_len = 2;
            else if ((b0 & 0xF0) == 0xE0 && (c[1] & 0xC0) == 0x80 && (c[2] & 0xC0) == 0x80) char_len = 3;
            else if ((b0 & 0xF8) == 0xF0 && (c[1] & 0xC0) == 0x80 && (c[2] & 0xC0) == 0x80 && (c[3] & 0xC0) == 0x80) char_len = 4;

            if (char_len > 1) {
                memcpy(str_buffer, c, (size_t)char_len);
                str_buffer[char_len] = '\0';
                int id = tok_lookup(t, str_buffer);
                if (id != -1) {
                    tokens[(*n_tokens)++] = id;
                    c += char_len;
                    continue;
                }
            }
        }

        for (int bi = 0; bi < char_len && *n_tokens < max_tokens; bi++) {
            unsigned char byte_c = (unsigned char)c[bi];
            str_buffer[0] = (char)byte_c;
            str_buffer[1] = '\0';
            int id = tok_lookup(t, str_buffer);
            if (id == -1) {
                char hex_buf[8];
                snprintf(hex_buf, sizeof(hex_buf), "<0x%02X>", byte_c);
                id = tok_lookup(t, hex_buf);
            }
            if (id == -1) {
                uint32_t fallback = (uint32_t)byte_c;
                id = (fallback < t->vocab_size) ? (int)fallback : 0;
            }
            tokens[(*n_tokens)++] = id;
        }
        c += char_len;
    }

    while (1) {
        float best_score = -1e10f;
        int best_id = -1, best_idx = -1;
        for (int i = start_merge_idx; i < (*n_tokens - 1); i++) {
            if (tokens[i] >= 128000 || tokens[i + 1] >= 128000) continue;

            const char *s1 = t->vocab[tokens[i]];
            const char *s2 = t->vocab[tokens[i + 1]];
            if (t->is_byte_level && !can_bpe_merge_bytelevel(s1, s2)) continue;

            size_t l1 = strlen(s1), l2 = strlen(s2);
            if (l1 + l2 + 1 > buf_cap) continue;
            memcpy(str_buffer, s1, l1);
            memcpy(str_buffer + l1, s2, l2 + 1);
            int id = tok_lookup(t, str_buffer);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    free(str_buffer);
}

static void push_recent_token(int *recent, int *recent_cnt, int cap, int token) {
    if (*recent_cnt < cap) {
        recent[(*recent_cnt)++] = token;
    } else {
        memmove(recent, recent + 1, (size_t)(cap - 1) * sizeof(int));
        recent[cap - 1] = token;
    }
}

static void apply_repetition_penalty(float *logits, uint32_t vocab_size, const int *recent, int recent_cnt, float penalty) {
    if (penalty <= 1.0f || recent_cnt <= 0) return;
    for (int i = 0; i < recent_cnt; i++) {
        int tok = recent[i];
        if (tok <= 2 || tok == 128000 || tok == 128001 || (tok >= 128006 && tok <= 128009) || (uint32_t)tok >= vocab_size) continue;
        int already = 0;
        for (int j = 0; j < i; j++) {
            if (recent[j] == tok) { already = 1; break; }
        }
        if (already) continue;
        if (logits[tok] > 0.0f) logits[tok] /= penalty;
        else logits[tok] *= penalty;
    }
}

static int sample_argmax(const float *logits, uint32_t n) {
    int max_i = 0;
    float max_p = logits[0];
    for (uint32_t i = 1; i < n; i++) {
        if (logits[i] > max_p) { max_i = (int)i; max_p = logits[i]; }
    }
    return max_i;
}

static uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    if (x == 0) x = 2463534242u;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x;
    return x;
}

#define TZ_MAX_TOP_K 64

static int sample_top_k_top_p(const float *logits, uint32_t n, float temp, int top_k, float top_p, uint32_t *rng) {
    if (temp <= 0.0f || top_k == 1 || n <= 1) {
        return sample_argmax(logits, n);
    }

    int k = top_k;
    if (k <= 0 || k > TZ_MAX_TOP_K) k = 40;
    if ((uint32_t)k > n) k = (int)n;

    int   cand_id[TZ_MAX_TOP_K];
    float cand_val[TZ_MAX_TOP_K];
    int   count = 0;

    for (uint32_t i = 0; i < n; i++) {
        float v = logits[i];
        if (count == k && v <= cand_val[k - 1]) continue;
        int pos = count;
        if (count < k) count++;
        else pos = k - 1;
        while (pos > 0 && v > cand_val[pos - 1]) {
            cand_val[pos] = cand_val[pos - 1];
            cand_id[pos]  = cand_id[pos - 1];
            pos--;
        }
        cand_val[pos] = v;
        cand_id[pos]  = (int)i;
    }

    float inv_temp = 1.0f / temp;
    float max_l = cand_val[0] * inv_temp;
    float sum = 0.0f;
    for (int i = 0; i < count; i++) {
        float e = tz_expf(cand_val[i] * inv_temp - max_l);
        cand_val[i] = e;
        sum += e;
    }
    if (sum <= 0.0f) return cand_id[0];
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < count; i++) cand_val[i] *= inv_sum;

    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        int cutoff = count;
        for (int i = 0; i < count; i++) {
            cum += cand_val[i];
            if (cum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        if (cutoff < count && cum > 0.0f) {
            count = cutoff;
            float renorm = 1.0f / cum;
            for (int i = 0; i < count; i++) cand_val[i] *= renorm;
        }
    }

    float coin = (float)(xorshift32(rng) >> 8) * (1.0f / 16777216.0f);
    float cdf = 0.0f;
    for (int i = 0; i < count; i++) {
        cdf += cand_val[i];
        if (coin < cdf) return cand_id[i];
    }
    return cand_id[count - 1];
}

static int check_stop_sequence(char *tail_window, size_t window_cap, const char *piece, int check_qa) {
    if (!piece || piece[0] == '\0') return 0;
    size_t tlen = strlen(tail_window);
    size_t plen = strlen(piece);
    if (plen >= window_cap) {
        piece += (plen - (window_cap - 1));
        plen = strlen(piece);
        tail_window[0] = '\0';
        tlen = 0;
    } else if (tlen + plen >= window_cap) {
        size_t keep = window_cap / 2;
        if (keep > tlen) keep = tlen;
        memmove(tail_window, tail_window + tlen - keep, keep);
        tail_window[keep] = '\0';
    }
    strncat(tail_window, piece, window_cap - strlen(tail_window) - 1);

    if (strstr(tail_window, "<|eot_id|>") ||
        strstr(tail_window, "<|start_header_id|>") ||
        strstr(tail_window, "<|im_end|>") ||
        strstr(tail_window, "<|endoftext|>") ||
        strstr(tail_window, "</s>")) {
        return 1;
    }
    if (check_qa && (strstr(tail_window, "\nQuestion:") || strstr(tail_window, "\nQ:") || strstr(tail_window, "\nUser:"))) {
        return 1;
    }
    return 0;
}

static float get_time_ms(void) {
    return (float)clock() * (1000.0f / (float)CLOCKS_PER_SEC);
}

static const char *backend_name(ComputeBackend b) {
    if (b == BACKEND_3DNOW) return "3DNow!";
    if (b == BACKEND_MMX)   return "MMX";
    if (b == BACKEND_P54C)  return "P54C";
    return "Scalar";
}

static const char *backend_cli_name(ComputeBackend b) {
    if (b == BACKEND_3DNOW) return "3dnow";
    if (b == BACKEND_MMX)   return "mmx";
    if (b == BACKEND_P54C)  return "p54c";
    return "scalar";
}

static const char *kernel_name(uint8_t k) {
    if (k == KERNEL_BINARY_1BIT) return "1-bit";
    if (k == KERNEL_TERNARY_MMX) return "1.58-bit";
    if (k == KERNEL_Q4_3DNOW)    return "Q4";
    if (k == KERNEL_Q8_3DNOW)    return "Q8";
    return "Unknown";
}

static ComputeBackend cycle_valid_backend(ComputeBackend cur) {
    for (int step = 1; step <= 4; step++) {
        ComputeBackend next = (ComputeBackend)((cur + step) % 4);
        if (next == BACKEND_3DNOW && !cpu_has_3dnow()) continue;
        if (next == BACKEND_MMX && !cpu_has_mmx()) continue;
        return next;
    }
    return BACKEND_SCALAR;
}

#define CHAT_HISTORY_CAP 64
#define CHAT_VIEW_LINES  14
#define CHAT_COLS        76
#define MAX_MODELS       16
#define MODAL_CFG_ITEMS  4

static char g_chat_history[CHAT_HISTORY_CAP][256];
static int  g_chat_count  = 0;
static int  g_chat_scroll = 0;

static char g_utf8_carry[4];
static int  g_utf8_carry_len = 0;

static char g_model_list[MAX_MODELS][32];
static int  g_model_count = 0;

static void scan_mmx_models(void) {
    g_model_count = 0;
#if defined(_WIN32)
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA("*.mmx", &fd);
    if (h != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) && g_model_count < MAX_MODELS) {
                strncpy(g_model_list[g_model_count], fd.cFileName, 31);
                g_model_list[g_model_count][31] = '\0';
                g_model_count++;
            }
        } while (FindNextFileA(h, &fd) && g_model_count < MAX_MODELS);
        FindClose(h);
    }
#else
    DIR *d = opendir(".");
    if (d) {
        struct dirent *de;
        while ((de = readdir(d)) != NULL && g_model_count < MAX_MODELS) {
            size_t len = strlen(de->d_name);
            if (len > 4 && strcmp(de->d_name + len - 4, ".mmx") == 0) {
                strncpy(g_model_list[g_model_count], de->d_name, 31);
                g_model_list[g_model_count][31] = '\0';
                g_model_count++;
            }
        }
        closedir(d);
    }
#endif
}

static void chat_clear(void) {
    g_chat_count     = 0;
    g_chat_scroll    = 0;
    g_utf8_carry_len = 0;
    for (int i = 0; i < CHAT_HISTORY_CAP; i++) {
        g_chat_history[i][0] = '\0';
    }
}

static void chat_push_line(const char *line) {
    g_chat_scroll = 0;
    if (g_chat_count < CHAT_HISTORY_CAP) {
        strncpy(g_chat_history[g_chat_count], line, sizeof(g_chat_history[0]) - 1);
        g_chat_history[g_chat_count][sizeof(g_chat_history[0]) - 1] = '\0';
        g_chat_count++;
    } else {
        for (int i = 0; i < CHAT_HISTORY_CAP - 1; i++) {
            memcpy(g_chat_history[i], g_chat_history[i + 1], sizeof(g_chat_history[0]));
        }
        strncpy(g_chat_history[CHAT_HISTORY_CAP - 1], line, sizeof(g_chat_history[0]) - 1);
        g_chat_history[CHAT_HISTORY_CAP - 1][sizeof(g_chat_history[0]) - 1] = '\0';
    }
}

static void chat_append_piece(const char *piece, int *resp_cols) {
    if (!piece || piece[0] == '\0') return;
    g_chat_scroll = 0;

    char merged_buf[264];
    int m_len = 0;
    if (g_utf8_carry_len > 0) {
        memcpy(merged_buf, g_utf8_carry, (size_t)g_utf8_carry_len);
        m_len = g_utf8_carry_len;
        g_utf8_carry_len = 0;
    }
    for (const char *p = piece; *p != '\0' && m_len < (int)sizeof(merged_buf) - 2; p++) {
        merged_buf[m_len++] = *p;
    }
    merged_buf[m_len] = '\0';

    int check_back = (m_len < 3) ? m_len : 3;
    for (int back = 1; back <= check_back; back++) {
        int idx = m_len - back;
        unsigned char b = (unsigned char)merged_buf[idx];
        if ((b & 0xC0) == 0x80) continue;
        if (b >= 0xC0) {
            int need = utf8_expected_len(b);
            if (back < need) {
                memcpy(g_utf8_carry, merged_buf + idx, (size_t)back);
                g_utf8_carry_len = back;
                m_len = idx;
                merged_buf[m_len] = '\0';
            }
        }
        break;
    }

    char chunk[256];
    int c_idx = 0;
    for (int i = 0; i < m_len; i++) {
        unsigned char b = (unsigned char)merged_buf[i];
        if (b == '\r') continue;
        if (b == '\n') {
            if (c_idx == 0 && *resp_cols <= 12) continue;
            if (c_idx > 0) {
                chunk[c_idx] = '\0';
                int pcols = tui_utf8_cols(chunk);
                if (*resp_cols + pcols >= CHAT_COLS) {
                    char wrap_line[264];
                    snprintf(wrap_line, sizeof(wrap_line), "  %s", chunk);
                    chat_push_line(wrap_line);
                } else if (g_chat_count > 0) {
                    strncat(g_chat_history[g_chat_count - 1], chunk,
                            sizeof(g_chat_history[0]) - strlen(g_chat_history[g_chat_count - 1]) - 1);
                }
                c_idx = 0;
            }
            chat_push_line("  ");
            *resp_cols = 2;
            continue;
        }
        if (b < 0x20 && b != '\t') continue;
        if (c_idx < (int)sizeof(chunk) - 2) {
            chunk[c_idx++] = (char)b;
        }
    }
    if (c_idx > 0) {
        chunk[c_idx] = '\0';
        int pcols = tui_utf8_cols(chunk);
        if (*resp_cols + pcols >= CHAT_COLS) {
            char wrap_line[264];
            snprintf(wrap_line, sizeof(wrap_line), "  %s", chunk);
            chat_push_line(wrap_line);
            *resp_cols = tui_utf8_cols(wrap_line);
        } else if (g_chat_count > 0) {
            strncat(g_chat_history[g_chat_count - 1], chunk,
                    sizeof(g_chat_history[0]) - strlen(g_chat_history[g_chat_count - 1]) - 1);
            *resp_cols += pcols;
        }
    }
}

static int save_runtime_cfg_file(const TunguzkaModel *model, const char *model_path,
                                 int steps, float temp, TuiLang lang, size_t max_ram_bytes) {
    TunguzkaRuntimeCfg cfg;
    memset(&cfg, 0, sizeof(cfg));
    memcpy(cfg.magic, "TZCFG\0\0\0", 8);
    cfg.steps        = (uint32_t)steps;
    cfg.ctx_limit    = model->config.max_seq_len;
    cfg.temp         = temp;
    cfg.backend      = (uint8_t)model->backend;
    cfg.use_kv8      = (uint8_t)(model->use_kv8 ? 1 : 0);
    cfg.force_stream = (uint8_t)(model->stream_mode ? 1 : 0);
    cfg.lang         = (uint8_t)(lang == TUI_LANG_RU ? 1 : 0);
    cfg.max_ram_kb   = (uint32_t)(max_ram_bytes / 1024u);
    if (model_path && model_path[0] != '\0') {
        strncpy(cfg.model_file, model_path, sizeof(cfg.model_file) - 1);
    }
    FILE *fp = fopen("TUNGUZKA.CFG", "wb");
    if (!fp) return -1;
    size_t wr = fwrite(&cfg, 1, sizeof(cfg), fp);
    fclose(fp);
    return (wr == sizeof(cfg)) ? 0 : -1;
}

static void tui_render_screen(TuiEngine *tui, TunguzkaModel *model, const char *input_buf,
                              int pos, float temp, int top_k, float top_p, float rep_pen,
                              int tok_cur, int tok_total, float tps, float ms_per_tok, float step0_chk,
                              int modal_open, int modal_sel) {
    tui_clear(tui, TUI_ATTR_NORMAL);
    tui_draw_box(tui, 0, 0, 80, 25, TUI_ATTR_BORDER, tui_str(tui, STR_SYS_TITLE));

    tui_printf(tui, 2, 1, TUI_ATTR_NORMAL, "%s:%s[%s] | %s:%s | %s | T:%.1f K:%d",
               tui_str(tui, STR_BACKEND), backend_name(model->backend),
               (model->backend == BACKEND_MMX ? "P55C" : (model->backend == BACKEND_3DNOW ? "K6-2" : "i586")),
               tui_str(tui, STR_MODE), kernel_name(model->config.kernel_type),
               model->stream_mode ? tui_str(tui, STR_MEM_STREAM) : tui_str(tui, STR_MEM_RAM),
               (double)temp, top_k);

    tui->backbuffer[2][0].ch = 0xC3;
    for (int i = 1; i < 79; i++) tui->backbuffer[2][i].ch = 0xC4;
    tui->backbuffer[2][79].ch = 0xB4;
    tui->backbuffer[2][25].ch = 0xC2;
    tui->backbuffer[2][52].ch = 0xC2;

    float arena_kb = (float)model->arena.offset / 1024.0f;
    float arena_max_kb = (float)model->arena.capacity / 1024.0f;
    if (arena_max_kb >= 10000.0f) {
        tui_printf(tui, 2, 3, TUI_ATTR_NORMAL, "%s: %.1fM/%.0fM",
                   tui_str(tui, STR_ARENA), (double)(arena_kb / 1024.0f), (double)(arena_max_kb / 1024.0f));
    } else {
        tui_printf(tui, 2, 3, TUI_ATTR_NORMAL, "%s: %.1fK/%.0fK",
                   tui_str(tui, STR_ARENA), (double)arena_kb, (double)arena_max_kb);
    }

    uint32_t kv_dim = (model->config.dim * model->config.n_kv_heads) / model->config.n_heads;
    float kv_kb = (float)((size_t)model->config.n_layers * model->config.max_seq_len * kv_dim * (model->use_kv8 ? 1 : 4)) / 1024.0f;
    if (kv_kb >= 10000.0f) {
        tui_printf(tui, 2, 4, TUI_ATTR_NORMAL, "%s(%s):%.0fM[%d/%u]",
                   tui_str(tui, STR_KV_CACHE), model->use_kv8 ? "I8" : "F32",
                   (double)(kv_kb / 1024.0f), pos, model->config.max_seq_len);
    } else {
        tui_printf(tui, 2, 4, TUI_ATTR_NORMAL, "%s(%s):%.1fK[%d/%u]",
                   tui_str(tui, STR_KV_CACHE), model->use_kv8 ? "I8" : "F32",
                   (double)kv_kb, pos, model->config.max_seq_len);
    }

    tui->backbuffer[3][25].ch = 0xB3; tui->backbuffer[3][25].attr = TUI_ATTR_BORDER;
    tui->backbuffer[4][25].ch = 0xB3; tui->backbuffer[4][25].attr = TUI_ATTR_BORDER;
    tui->backbuffer[3][52].ch = 0xB3; tui->backbuffer[3][52].attr = TUI_ATTR_BORDER;
    tui->backbuffer[4][52].ch = 0xB3; tui->backbuffer[4][52].attr = TUI_ATTR_BORDER;

    if (model->stream_mode) {
        if (model->last_stream_layer >= 0) {
            tui_printf(tui, 27, 3, TUI_ATTR_WARN, "[HDD: READ L%02d]", model->last_stream_layer);
        } else {
            tui_print(tui, 27, 3, TUI_ATTR_NORMAL, tui_str(tui, STR_HDD_IDLE));
        }
        uint32_t total_emb = model->emb_cache_hits + model->emb_cache_misses;
        float hit_pct = (total_emb > 0) ? ((float)model->emb_cache_hits * 100.0f / (float)total_emb) : 0.0f;
        tui_printf(tui, 27, 4, TUI_ATTR_NORMAL, "LRU: %u/%u (%.0f%%)", model->emb_cache_hits, total_emb, (double)hit_pct);
    } else {
        tui_print(tui, 27, 3, TUI_ATTR_NORMAL, tui_str(tui, STR_HDD_IDLE));
        tui_printf(tui, 27, 4, TUI_ATTR_NORMAL, "%s: %u/%u", tui_str(tui, STR_PINNED), model->pinned_tensors, model->total_tensors);
    }

    tui_printf(tui, 54, 3, TUI_ATTR_WARN, "%s: %.1f tok/s", tui_str(tui, STR_SPEED), (double)tps);
    tui_printf(tui, 54, 4, TUI_ATTR_NORMAL, "%s: %.2f ms", tui_str(tui, STR_LATENCY), (double)ms_per_tok);

    tui->backbuffer[5][0].ch = 0xC3;
    for (int i = 1; i < 79; i++) tui->backbuffer[5][i].ch = 0xC4;
    tui->backbuffer[5][79].ch = 0xB4;
    tui->backbuffer[5][25].ch = 0xC1;
    tui->backbuffer[5][52].ch = 0xC1;

    int start_line = 0;
    if (g_chat_count > CHAT_VIEW_LINES) {
        start_line = g_chat_count - CHAT_VIEW_LINES - g_chat_scroll;
        if (start_line < 0) start_line = 0;
    }
    int view_cnt = g_chat_count - start_line;
    if (view_cnt > CHAT_VIEW_LINES) view_cnt = CHAT_VIEW_LINES;

    for (int i = 0; i < view_cnt; i++) {
        const char *line = g_chat_history[start_line + i];
        uint8_t attr = TUI_ATTR_NORMAL;
        if (strncmp(line, "You >", 5) == 0 || strncmp(line, "Вы >", 5) == 0) {
            attr = TUI_ATTR_USER;
        }
        tui_print(tui, 2, 6 + i, attr, line);
    }

    if (g_chat_scroll > 0) {
        tui_printf(tui, 62, 5, TUI_ATTR_WARN, "[PgUp/Dn: -%d]", g_chat_scroll);
    }

    if (modal_open) {
        int mw = 48;
        int total_items = MODAL_CFG_ITEMS + (g_model_count > 0 ? g_model_count : 1);
        int mh = total_items + 5;
        if (mh > 14) mh = 14;
        int mx = (TUI_COLS - mw) / 2;
        int my = 6;
        for (int ry = my; ry < my + mh; ry++) {
            for (int rx = mx; rx < mx + mw; rx++) {
                tui->backbuffer[ry][rx].ch = ' ';
                tui->backbuffer[ry][rx].attr = TUI_ATTR_NORMAL;
            }
        }
        tui_draw_box(tui, mx, my, mw, mh, TUI_ATTR_WARN, tui_str(tui, STR_SELECT_MODEL));

        uint8_t a0 = (modal_sel == 0) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
        uint8_t a1 = (modal_sel == 1) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
        uint8_t a2 = (modal_sel == 2) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
        uint8_t a3 = (modal_sel == 3) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_WARN;
        tui_printf(tui, mx + 2, my + 1, a0, "%s Max Gen Steps : < %d tok >", (modal_sel == 0) ? ">" : " ", tok_total);
        tui_printf(tui, mx + 2, my + 2, a1, "%s Top-K / Top-P : < K=%d P=%.2f >", (modal_sel == 1) ? ">" : " ", top_k, (double)top_p);
        tui_printf(tui, mx + 2, my + 3, a2, "%s Rep. Penalty  : < %.2f >", (modal_sel == 2) ? ">" : " ", (double)rep_pen);
        tui_printf(tui, mx + 2, my + 4, a3, "%s [Save Settings to TUNGUZKA.CFG]", (modal_sel == 3) ? ">" : " ");

        for (int i = 1; i < mw - 1; i++) {
            tui->backbuffer[my + 5][mx + i].ch = 0xC4;
            tui->backbuffer[my + 5][mx + i].attr = TUI_ATTR_BORDER;
        }

        int max_mod_rows = mh - 7;
        if (g_model_count == 0) {
            uint8_t am = (modal_sel == MODAL_CFG_ITEMS) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_WARN;
            tui_print(tui, mx + 2, my + 6, am, "No *.mmx files found in current dir");
        } else {
            int mod_sel = modal_sel - MODAL_CFG_ITEMS;
            int m_start = 0;
            if (mod_sel >= max_mod_rows) m_start = mod_sel - max_mod_rows + 1;
            for (int m = 0; m < max_mod_rows && (m_start + m) < g_model_count; m++) {
                int idx = m_start + m;
                int item_idx = MODAL_CFG_ITEMS + idx;
                uint8_t mattr = (item_idx == modal_sel) ? TUI_ATTR_HOTKEY_LBL : TUI_ATTR_NORMAL;
                tui_printf(tui, mx + 2, my + 6 + m, mattr, "%s [Model] %-32s",
                           (item_idx == modal_sel) ? ">" : " ", g_model_list[idx]);
            }
        }
    }

    tui->backbuffer[20][0].ch = 0xC3;
    for (int i = 1; i < 79; i++) tui->backbuffer[20][i].ch = 0xC4;
    tui->backbuffer[20][79].ch = 0xB4;
    tui_printf(tui, 2, 21, TUI_ATTR_USER, "> %s_", input_buf);

    tui->backbuffer[22][0].ch = 0xC3;
    for (int i = 1; i < 79; i++) tui->backbuffer[22][i].ch = 0xC4;
    tui->backbuffer[22][79].ch = 0xB4;
    tui_printf(tui, 2, 23, TUI_ATTR_NORMAL, "%s: [", tui_str(tui, STR_GEN_PROG));
    tui_draw_bar(tui, 15, 23, 20, tok_cur, tok_total > 0 ? tok_total : 1, TUI_ATTR_TITLE, TUI_ATTR_BORDER);
    tui_printf(tui, 36, 23, TUI_ATTR_NORMAL, "] %d/%d tok | chk: %.4f", tok_cur, tok_total, (double)step0_chk);

    for (int i = 0; i < 80; i++) {
        tui->backbuffer[24][i].ch = ' ';
        tui->backbuffer[24][i].attr = TUI_ATTR_BAR;
    }
    const TuiStringId fkeys[] = {
        STR_F1_HELP, STR_F2_BACKEND, STR_F3_KV8, STR_F4_STREAM,
        STR_F5_RESET, STR_F6_CTX, STR_F7_TEMP, STR_F8_LANG, STR_F9_MODELS, STR_F10_QUIT
    };
    int fx = 1;
    for (size_t k = 0; k < sizeof(fkeys) / sizeof(fkeys[0]); k++) {
        const char *label = tui_str(tui, fkeys[k]);
        tui_print(tui, fx, 24, TUI_ATTR_HOTKEY_NUM, label);
        fx += tui_utf8_cols(label) + 2;
    }
    tui_flip(tui);
}

static void reload_tokenizer_for_model(TunguzkaModel *model, const char *active_path, size_t base_offset,
                                       const char *fallback_tok_path, Tokenizer *tok, int *has_tok) {
    if (*has_tok) {
        tokenizer_free(tok);
        *has_tok = 0;
    }
    if (model->config.tokenizer_offset > 0 && model->config.tokenizer_size > 0) {
        *has_tok = (tokenizer_init_slice(tok, active_path,
                                         base_offset + model->config.tokenizer_offset,
                                         model->config.tokenizer_size,
                                         model->config.vocab_size) == 0);
    }
    if (!(*has_tok)) {
        *has_tok = (tokenizer_init_slice(tok, fallback_tok_path, 0, 0, model->config.vocab_size) == 0);
    }
}

static int calc_tui_steps_for_ctx(uint32_t max_seq_len, int requested_steps) {
    int target = requested_steps;
    if (target > (int)max_seq_len - 16) {
        target = (int)max_seq_len / 2;
    }
    if (target < 16) target = 16;
    return target;
}

static void adjust_modal_setting(int modal_sel, int dir, uint32_t max_seq_len,
                                 int *steps, int *top_k, float *top_p, float *rep_pen) {
    if (modal_sel == 0) {
        int step_opts[] = { 32, 64, 128, 256, 384 };
        int n_opts = 5;
        int cur_i = 0;
        for (int i = 0; i < n_opts; i++) {
            if (*steps >= step_opts[i]) cur_i = i;
        }
        cur_i = (cur_i + dir + n_opts) % n_opts;
        *steps = calc_tui_steps_for_ctx(max_seq_len, step_opts[cur_i]);
    } else if (modal_sel == 1) {
        int   k_opts[] = { 1, 20, 40, 64 };
        float p_opts[] = { 1.0f, 0.85f, 0.90f, 0.95f };
        int   n_opts = 4;
        int   cur_i = 2;
        for (int i = 0; i < n_opts; i++) {
            if (*top_k == k_opts[i]) { cur_i = i; break; }
        }
        cur_i = (cur_i + dir + n_opts) % n_opts;
        *top_k = k_opts[cur_i];
        *top_p = p_opts[cur_i];
    } else if (modal_sel == 2) {
        float r_opts[] = { 1.00f, 1.10f, 1.15f, 1.25f };
        int   n_opts = 4;
        int   cur_i = 2;
        for (int i = 0; i < n_opts; i++) {
            if (*rep_pen >= r_opts[i] - 0.02f && *rep_pen <= r_opts[i] + 0.02f) { cur_i = i; break; }
        }
        cur_i = (cur_i + dir + n_opts) % n_opts;
        *rep_pen = r_opts[cur_i];
    }
}

static void run_tui_session(TunguzkaModel *model, const char *init_model_path, size_t base_offset, size_t slice_size,
                            const char *tok_path, Tokenizer *tokenizer, int *has_tok,
                            int steps, float temp, int top_k, float top_p, float rep_pen,
                            size_t max_ram_bytes, TuiLang init_lang) {
    TuiEngine tui;
    tui_init(&tui, init_lang);
    steps = calc_tui_steps_for_ctx(model->config.max_seq_len, steps);

    char active_model_path[64];
    strncpy(active_model_path, init_model_path, sizeof(active_model_path) - 1);
    active_model_path[sizeof(active_model_path) - 1] = '\0';

    char input_buf[256] = {0};
    int  input_len = 0;
    uint32_t seed = 42u;
    int *prompt_tokens = (int *)malloc(32768 * sizeof(int));
    if (!prompt_tokens) {
        tui_shutdown(&tui);
        return;
    }

    int recent_tokens[64];
    int recent_cnt = 0;

    chat_push_line(tui_str(&tui, STR_WELCOME));
    chat_push_line("");

    float last_tps   = 0.0f;
    float last_ms    = 0.0f;
    float last_chk   = 0.0f;
    int   last_gen   = 0;
    int   pos        = 0;
    int   modal_open = 0;
    int   modal_sel  = 0;

    tui_render_screen(&tui, model, input_buf, pos, temp, top_k, top_p, rep_pen,
                      last_gen, steps, last_tps, last_ms, last_chk, modal_open, modal_sel);

    int running = 1;
    while (running) {
        int key = tui_poll_key();
        if (key == -1) {
#if defined(_WIN32)
            Sleep(15);
#else
            usleep(15000);
#endif
            continue;
        }

        if (modal_open) {
            int max_items = MODAL_CFG_ITEMS + g_model_count;
            if (key == TUI_KEY_ESC || key == TUI_KEY_F9) {
                modal_open = 0;
            } else if (key == TUI_KEY_UP && modal_sel > 0) {
                modal_sel--;
            } else if (key == TUI_KEY_DOWN && modal_sel + 1 < max_items) {
                modal_sel++;
            } else if (key == TUI_KEY_LEFT && modal_sel < 3) {
                adjust_modal_setting(modal_sel, -1, model->config.max_seq_len, &steps, &top_k, &top_p, &rep_pen);
            } else if (key == TUI_KEY_RIGHT && modal_sel < 3) {
                adjust_modal_setting(modal_sel, +1, model->config.max_seq_len, &steps, &top_k, &top_p, &rep_pen);
            } else if (key == '\r' || key == '\n') {
                if (modal_sel < 3) {
                    adjust_modal_setting(modal_sel, +1, model->config.max_seq_len, &steps, &top_k, &top_p, &rep_pen);
                } else if (modal_sel == 3) {
                    if (save_runtime_cfg_file(model, active_model_path, steps, temp, tui.lang, max_ram_bytes) == 0) {
                        chat_push_line("[OK] Saved runtime settings to TUNGUZKA.CFG");
                    } else {
                        chat_push_line("[Error] Failed to write TUNGUZKA.CFG");
                    }
                    modal_open = 0;
                } else if (g_model_count > 0) {
                    int m_idx = modal_sel - MODAL_CFG_ITEMS;
                    int cur_kv8 = model->use_kv8;
                    int cur_stream = model->stream_mode;
                    ComputeBackend cur_b = model->backend;
                    tunguzka_free(model);
                    if (tunguzka_load_from_offset(model, g_model_list[m_idx], 0, 0, 256,
                                                  cur_b, cur_kv8, cur_stream, max_ram_bytes) == 0) {
                        strncpy(active_model_path, g_model_list[m_idx], sizeof(active_model_path) - 1);
                        base_offset = 0;
                        slice_size = 0;
                        reload_tokenizer_for_model(model, active_model_path, 0, tok_path, tokenizer, has_tok);
                        steps = calc_tui_steps_for_ctx(model->config.max_seq_len, steps);
                        pos = 0;
                        recent_cnt = 0;
                        last_gen = 0;
                        chat_push_line(active_model_path);
                    } else {
                        tunguzka_load_from_offset(model, active_model_path, base_offset, slice_size, 256,
                                                  cur_b, cur_kv8, cur_stream, max_ram_bytes);
                    }
                    modal_open = 0;
                }
            }
            tui_render_screen(&tui, model, input_buf, pos, temp, top_k, top_p, rep_pen,
                              last_gen, steps, last_tps, last_ms, last_chk, modal_open, modal_sel);
            continue;
        }

        if (key == TUI_KEY_ESC || key == TUI_KEY_F10) {
            running = 0;
            break;
        } else if (key == TUI_KEY_PGUP) {
            if (g_chat_count > CHAT_VIEW_LINES) {
                int max_sc = g_chat_count - CHAT_VIEW_LINES;
                g_chat_scroll += 4;
                if (g_chat_scroll > max_sc) g_chat_scroll = max_sc;
            }
        } else if (key == TUI_KEY_PGDN) {
            g_chat_scroll -= 4;
            if (g_chat_scroll < 0) g_chat_scroll = 0;
        } else if (key == TUI_KEY_F1) {
            chat_push_line(tui_str(&tui, STR_HELP_INFO));
        } else if (key == TUI_KEY_F2) {
            model->backend = cycle_valid_backend(model->backend);
        } else if (key == TUI_KEY_F3) {
            int new_kv8 = !model->use_kv8;
            int cur_stream = model->stream_mode;
            uint32_t cur_ctx = model->config.max_seq_len;
            ComputeBackend cur_b = model->backend;
            tunguzka_free(model);
            tunguzka_load_from_offset(model, active_model_path, base_offset, slice_size, cur_ctx, cur_b, new_kv8, cur_stream, max_ram_bytes);
            pos = 0;
            recent_cnt = 0;
        } else if (key == TUI_KEY_F4) {
            int new_stream = !model->stream_mode;
            int cur_kv8 = model->use_kv8;
            uint32_t cur_ctx = model->config.max_seq_len;
            ComputeBackend cur_b = model->backend;
            tunguzka_free(model);
            tunguzka_load_from_offset(model, active_model_path, base_offset, slice_size, cur_ctx, cur_b, cur_kv8, new_stream, max_ram_bytes);
            pos = 0;
            recent_cnt = 0;
        } else if (key == TUI_KEY_F5) {
            pos = 0;
            recent_cnt = 0;
            chat_push_line(tui_str(&tui, STR_CTX_RESET));
        } else if (key == TUI_KEY_F6) {
            uint32_t max_hw_ctx = model->orig_max_seq_len;
            uint32_t next_ctx = model->config.max_seq_len * 2;
            if (next_ctx > max_hw_ctx) next_ctx = 64;
            if (next_ctx > max_hw_ctx) next_ctx = max_hw_ctx;
            int cur_kv8 = model->use_kv8;
            int cur_stream = model->stream_mode;
            ComputeBackend cur_b = model->backend;
            tunguzka_free(model);
            tunguzka_load_from_offset(model, active_model_path, base_offset, slice_size, next_ctx, cur_b, cur_kv8, cur_stream, max_ram_bytes);
            steps = calc_tui_steps_for_ctx(model->config.max_seq_len, (int)model->config.max_seq_len / 2);
            pos = 0;
            recent_cnt = 0;
        } else if (key == TUI_KEY_F7) {
            if (temp < 0.1f) temp = 0.5f;
            else if (temp < 0.6f) temp = 0.8f;
            else if (temp < 0.9f) temp = 1.0f;
            else temp = 0.0f;
        } else if (key == TUI_KEY_F8) {
            tui_toggle_lang(&tui);
            chat_clear();
            chat_push_line(tui_str(&tui, STR_WELCOME));
            chat_push_line("");
        } else if (key == TUI_KEY_F9) {
            scan_mmx_models();
            modal_sel = 0;
            modal_open = 1;
        } else if (key == '\b' || key == 127) {
            if (input_len > 0) {
                while (input_len > 0 && ((unsigned char)input_buf[input_len - 1] & 0xC0) == 0x80) {
                    input_buf[--input_len] = '\0';
                }
                if (input_len > 0) {
                    input_buf[--input_len] = '\0';
                }
            }
        } else if (key == '\r' || key == '\n') {
            if (input_len == 0) continue;

            char line_prompt[300];
            snprintf(line_prompt, sizeof(line_prompt), "%s > %s", tui_str(&tui, STR_YOU), input_buf);
            chat_push_line(line_prompt);

            char formatted_prompt[512];
            int is_qa_wrapped = 0;
            if (model->config.vocab_size >= 100000 && input_buf[0] != '!') {
                snprintf(formatted_prompt, sizeof(formatted_prompt),
                         "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", input_buf);
                is_qa_wrapped = 1;
            } else if (input_buf[0] == '?') {
                snprintf(formatted_prompt, sizeof(formatted_prompt),
                         "%sQuestion: %s\nAnswer:", (pos == 0) ? "" : "\n", input_buf + 1);
                is_qa_wrapped = 1;
            } else {
                const char *raw_in = (input_buf[0] == '!') ? (input_buf + 1) : input_buf;
                snprintf(formatted_prompt, sizeof(formatted_prompt), "%s", raw_in);
            }

            int max_prompt_cap = (int)model->config.max_seq_len - 16;
            if (max_prompt_cap < 16) max_prompt_cap = 16;
            int num_p = 0;
            if (*has_tok) {
                bpe_encode(tokenizer, formatted_prompt, prompt_tokens, &num_p, max_prompt_cap, (pos == 0));
            } else {
                prompt_tokens[0] = 1;
                num_p = 1;
            }
            if (num_p <= 0) {
                prompt_tokens[0] = 1;
                num_p = 1;
            }

            if (pos > 0 && pos + num_p + 16 >= (int)model->config.max_seq_len) {
                int needed = (pos + num_p + steps) - (int)model->config.max_seq_len;
                int shift = (pos / 2 > needed) ? (pos / 2) : needed;
                if (shift > pos) shift = pos;
                if (shift < 8) shift = pos;
                pos = tunguzka_shift_kv_cache(model, pos, shift);
                chat_push_line(tui_str(&tui, STR_CTX_SHIFTED));
            }

            input_buf[0] = '\0';
            input_len = 0;
            g_utf8_carry_len = 0;

            char resp_line[300];
            snprintf(resp_line, sizeof(resp_line), "%s > ", tui_str(&tui, STR_ASSISTANT));
            int resp_cols = tui_utf8_cols(resp_line);
            chat_push_line(resp_line);

            int token = prompt_tokens[0];
            int next = 0;
            float t_start = get_time_ms();
            float step0_chk = 0.0f;
            int generated = 0;
            char tail_window[128] = {0};

            for (int p_idx = 0; generated < steps; p_idx++) {
                if (pos >= (int)model->config.max_seq_len) {
                    int shift = (int)model->config.max_seq_len / 2;
                    if (shift < 16) shift = pos;
                    pos = tunguzka_shift_kv_cache(model, pos, shift);
                }

                float *logits = tunguzka_forward(model, token, pos);
                pos++;

                if (p_idx == 0) {
                    for (uint32_t i = 0; i < model->config.vocab_size; i++) {
                        step0_chk += logits[i] * (float)((int)(i % 7) - 3);
                    }
                }

                if (token < 128000) {
                    push_recent_token(recent_tokens, &recent_cnt, 64, token);
                }

                if (p_idx < num_p - 1) {
                    token = prompt_tokens[p_idx + 1];
                    continue;
                }

                apply_repetition_penalty(logits, model->config.vocab_size, recent_tokens, recent_cnt, rep_pen);
                next = sample_top_k_top_p(logits, model->config.vocab_size, temp, top_k, top_p, &seed);

                if (next == 1 || next == 2 || (next >= 128001 && next <= 128009)) {
                    int stop_tok = (model->config.vocab_size >= 100000) ? 128009 : 2;
                    if (pos < (int)model->config.max_seq_len) {
                        tunguzka_forward(model, stop_tok, pos);
                        pos++;
                    }
                    break;
                }
                generated++;

                char *piece = (*has_tok) ? tokenizer_decode(tokenizer, token, next) : NULL;
                char fallback_tok[16];
                if (!piece && !(*has_tok)) {
                    snprintf(fallback_tok, sizeof(fallback_tok), "[%d]", next);
                    piece = fallback_tok;
                }
                token = next;

                if (check_stop_sequence(tail_window, sizeof(tail_window), piece, is_qa_wrapped)) {
                    int stop_tok = (model->config.vocab_size >= 100000) ? 128009 : 2;
                    if (pos < (int)model->config.max_seq_len) {
                        tunguzka_forward(model, stop_tok, pos);
                        pos++;
                    }
                    break;
                }

                chat_append_piece(piece, &resp_cols);
                float dt = get_time_ms() - t_start;
                last_tps = (dt > 0.0f) ? ((float)(p_idx + 1) * 1000.0f / dt) : 0.0f;
                last_ms  = (dt > 0.0f) ? (dt / (float)(p_idx + 1)) : 0.0f;
                last_gen = generated;
                last_chk = step0_chk;

                tui_render_screen(&tui, model, input_buf, pos, temp, top_k, top_p, rep_pen,
                                  last_gen, steps, last_tps, last_ms, last_chk, modal_open, modal_sel);

                int abort_key = tui_poll_key();
                if (abort_key == TUI_KEY_ESC) {
                    break;
                }
            }
            model->last_stream_layer = -1;
            chat_push_line("");
        } else if (key >= 32 && key < 256 && input_len < (int)sizeof(input_buf) - 5) {
            input_buf[input_len++] = (char)key;
            input_buf[input_len] = '\0';
        }
        tui_render_screen(&tui, model, input_buf, pos, temp, top_k, top_p, rep_pen,
                          last_gen, steps, last_tps, last_ms, last_chk, modal_open, modal_sel);
    }

    free(prompt_tokens);
    tui_shutdown(&tui);
}

static int probe_self_bundle(const char *exe_path, size_t *out_mod_off, size_t *out_mod_sz, TunguzkaRuntimeCfg *out_cfg) {
    FILE *f = fopen(exe_path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long fsz = ftell(f);
    if (fsz < (long)(sizeof(TunguzkaBundleTrailer) + sizeof(TunguzkaHeader))) { fclose(f); return 0; }
    if (fseek(f, fsz - (long)sizeof(TunguzkaBundleTrailer), SEEK_SET) != 0) { fclose(f); return 0; }
    TunguzkaBundleTrailer tr;
    if (fread(&tr, 1, sizeof(tr), f) != sizeof(tr) || memcmp(tr.magic, "TZPK", 4) != 0) {
        fclose(f);
        return 0;
    }
    if (tr.bundle_offset + tr.bundle_size > (uint32_t)fsz) {
        fclose(f);
        return 0;
    }
    if (tr.cfg_offset > 0 && tr.cfg_offset + sizeof(TunguzkaRuntimeCfg) <= (uint32_t)fsz) {
        if (fseek(f, (long)tr.cfg_offset, SEEK_SET) == 0) {
            if (fread(out_cfg, 1, sizeof(TunguzkaRuntimeCfg), f) != sizeof(TunguzkaRuntimeCfg)) {
                memset(out_cfg, 0, sizeof(TunguzkaRuntimeCfg));
            }
        }
    }
    fclose(f);
    *out_mod_off = (size_t)tr.bundle_offset;
    *out_mod_sz  = (size_t)tr.bundle_size;
    return 1;
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *tok_path = "tokenizer.bin";
    const char *prompt = "";
    int steps = 0;
    uint32_t ctx_override = 0;
    float temp = 0.5f;
    int temp_set = 0;
    int top_k = 40;
    float top_p = 0.90f;
    float rep_penalty = 1.15f;
    uint32_t seed = 42u;
    int use_kv8 = 0;
    int tui_mode = 0;
    int force_stream = 0;
    int debug_tokens = 0;
    size_t max_ram_bytes = 0;
    size_t bundle_offset = 0;
    size_t bundle_size = 0;
    TuiLang default_lang = TUI_LANG_EN;
    ComputeBackend requested_backend = tunguzka_detect_backend();

    char cfg_model_path[64] = {0};
    TunguzkaRuntimeCfg rt_cfg;
    memset(&rt_cfg, 0, sizeof(rt_cfg));

    FILE *cfg_fp = fopen("TUNGUZKA.CFG", "rb");
    if (cfg_fp) {
        if (fread(&rt_cfg, 1, sizeof(rt_cfg), cfg_fp) == sizeof(rt_cfg) &&
            memcmp(rt_cfg.magic, "TZCFG", 5) == 0) {
            if (rt_cfg.steps > 0) steps = (int)rt_cfg.steps;
            if (rt_cfg.ctx_limit > 0) ctx_override = rt_cfg.ctx_limit;
            temp = rt_cfg.temp;
            temp_set = 1;
            if (rt_cfg.backend <= 3) requested_backend = (ComputeBackend)rt_cfg.backend;
            use_kv8 = rt_cfg.use_kv8 ? 1 : 0;
            force_stream = rt_cfg.force_stream ? 1 : 0;
            default_lang = rt_cfg.lang ? TUI_LANG_RU : TUI_LANG_EN;
            if (rt_cfg.max_ram_kb > 0) max_ram_bytes = (size_t)rt_cfg.max_ram_kb * 1024u;
            if (rt_cfg.model_file[0] != '\0') {
                strncpy(cfg_model_path, rt_cfg.model_file, sizeof(cfg_model_path) - 1);
                model_path = cfg_model_path;
            }
        }
        fclose(cfg_fp);
    }

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) tok_path = argv[++i];
        else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) ctx_override = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) { temp = (float)atof(argv[++i]); temp_set = 1; }
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--rep-penalty") == 0 && i + 1 < argc) rep_penalty = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = (uint32_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--kv8") == 0) use_kv8 = 1;
        else if (strcmp(argv[i], "--tui") == 0 || strcmp(argv[i], "--repl") == 0) tui_mode = 1;
        else if (strcmp(argv[i], "--ru") == 0) default_lang = TUI_LANG_RU;
        else if (strcmp(argv[i], "--debug-tokens") == 0) debug_tokens = 1;
        else if (strcmp(argv[i], "--stream") == 0 || strcmp(argv[i], "-s") == 0) force_stream = 1;
        else if (strcmp(argv[i], "--max-ram") == 0 && i + 1 < argc) {
            max_ram_bytes = (size_t)atoi(argv[++i]) * 1024u * 1024u;
        } else if (strcmp(argv[i], "--max-ram-kb") == 0 && i + 1 < argc) {
            max_ram_bytes = (size_t)atoi(argv[++i]) * 1024u;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            const char *b = argv[++i];
            if (strcmp(b, "mmx") == 0) requested_backend = BACKEND_MMX;
            else if (strcmp(b, "3dnow") == 0) requested_backend = BACKEND_3DNOW;
            else if (strcmp(b, "p54c") == 0) requested_backend = BACKEND_P54C;
            else requested_backend = BACKEND_SCALAR;
        }
    }

    if (!model_path && argv[0] && probe_self_bundle(argv[0], &bundle_offset, &bundle_size, &rt_cfg)) {
        model_path = argv[0];
        tui_mode = 1;
        if (memcmp(rt_cfg.magic, "TZCFG", 5) == 0) {
            if (rt_cfg.steps > 0) steps = (int)rt_cfg.steps;
            if (rt_cfg.ctx_limit > 0) ctx_override = rt_cfg.ctx_limit;
            temp = rt_cfg.temp;
            temp_set = 1;
            if (rt_cfg.backend <= 3) requested_backend = (ComputeBackend)rt_cfg.backend;
            use_kv8 = rt_cfg.use_kv8 ? 1 : 0;
            force_stream = rt_cfg.force_stream ? 1 : 0;
            default_lang = rt_cfg.lang ? TUI_LANG_RU : TUI_LANG_EN;
            if (rt_cfg.max_ram_kb > 0) max_ram_bytes = (size_t)rt_cfg.max_ram_kb * 1024u;
        }
    }

    if (!model_path) {
        const char *candidates[] = { "vikhr_1b_q4.mmx", "bitnet.mmx", "stories260k.mmx", "stories15m.mmx", "model.mmx", "test_ternary.mmx", "test_q4.mmx", "test_q8.mmx" };
        for (size_t c = 0; c < sizeof(candidates) / sizeof(candidates[0]); c++) {
            FILE *probe = fopen(candidates[c], "rb");
            if (probe) {
                fclose(probe);
                model_path = candidates[c];
                break;
            }
        }
        if (model_path) tui_mode = 1;
    }

    if (!model_path) {
        fprintf(stderr, "Usage: %s [-m model.mmx] [-z tokenizer.bin] [-c ctx_limit] [-n steps] [-t temp] [-k top_k] [--top-p P] [--debug-tokens] [--tui] [--stream] [-b mmx|3dnow|p54c|scalar]\n", argv[0]);
        return 1;
    }

    if (!tui_mode && !temp_set) temp = 0.0f;
    if (steps <= 0) steps = tui_mode ? 128 : 32;
    if (ctx_override == 0) {
        ctx_override = tui_mode ? 256u : ((prompt[0] != '\0') ? (uint32_t)(steps + 128) : (uint32_t)steps);
    }

    TunguzkaModel model;
    if (tunguzka_load_from_offset(&model, model_path, bundle_offset, bundle_size, ctx_override,
                                  requested_backend, use_kv8, force_stream, max_ram_bytes) != 0) {
        fprintf(stderr, "Error: failed to load model %s\n", model_path);
        return 1;
    }

    Tokenizer tokenizer;
    int has_tok = 0;
    reload_tokenizer_for_model(&model, model_path, bundle_offset, tok_path, &tokenizer, &has_tok);
    if (!has_tok && strcmp(tok_path, "tokenizer.bin") == 0) {
        reload_tokenizer_for_model(&model, model_path, bundle_offset, "test_dummy_tok.bin", &tokenizer, &has_tok);
    }

    if (tui_mode) {
        run_tui_session(&model, model_path, bundle_offset, bundle_size, tok_path,
                        &tokenizer, &has_tok, steps, temp, top_k, top_p, rep_penalty,
                        max_ram_bytes, default_lang);
        if (has_tok) tokenizer_free(&tokenizer);
        tunguzka_free(&model);
        return 0;
    }

    fprintf(stderr, "[Tunguzka-LLM] dim=%u | layers=%u | heads=%u | vocab=%u | group=%u | kernel=0x%02x | backend=%s | kv=%s | mode=%s | arena=%.2f KB\n",
            model.config.dim, model.config.n_layers, model.config.n_heads,
            model.config.vocab_size, model.config.group_size, model.config.kernel_type,
            backend_cli_name(model.backend), model.use_kv8 ? "int8" : "fp32",
            model.stream_mode ? "STREAM" : "RAM",
            (double)((float)model.arena.offset / 1024.0f));

    char cli_prompt_buf[1024];
    const char *effective_prompt = prompt;
    int is_chat_wrapped = 0;
    if (model.config.vocab_size >= 100000 && prompt[0] != '\0' && prompt[0] != '<' && prompt[0] != '!') {
        snprintf(cli_prompt_buf, sizeof(cli_prompt_buf),
                 "<|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", prompt);
        effective_prompt = cli_prompt_buf;
        is_chat_wrapped = 1;
    } else if (prompt[0] == '!') {
        effective_prompt = prompt + 1;
    }

    int max_prompt_tokens = (int)model.config.max_seq_len;
    int *prompt_tokens = (int *)malloc((size_t)max_prompt_tokens * sizeof(int));
    if (!prompt_tokens) {
        if (has_tok) tokenizer_free(&tokenizer);
        tunguzka_free(&model);
        return 1;
    }

    int num_prompt_tokens = 0;
    if (has_tok) {
        bpe_encode(&tokenizer, effective_prompt, prompt_tokens, &num_prompt_tokens, max_prompt_tokens, 1);
    } else {
        prompt_tokens[0] = 1;
        num_prompt_tokens = 1;
    }

    if (debug_tokens && has_tok) {
        fprintf(stderr, "[Debug-Tokens] encoded %d prompt token(s) (byte_level=%d):\n  ", num_prompt_tokens, tokenizer.is_byte_level);
        for (int i = 0; i < num_prompt_tokens; i++) {
            fprintf(stderr, "%d:\"%s\" ", prompt_tokens[i], tokenizer.vocab[prompt_tokens[i]]);
        }
        fprintf(stderr, "\n");
    }

    int token = prompt_tokens[0];
    int next = 0;
    int pos = 0;
    int total_forwards = 0;
    int total_steps = is_chat_wrapped ? (num_prompt_tokens - 1 + steps) : steps;
    int recent_tokens[64];
    int recent_cnt = 0;
    char tail_window[128] = {0};
    g_cli_utf8_carry_len = 0;
    float first_logit_checksum = 0.0f;
    float t_start = get_time_ms();

    for (int step = 0; step < total_steps; step++) {
        if (pos >= (int)model.config.max_seq_len) {
            int shift = (int)model.config.max_seq_len / 2;
            if (shift < 8) shift = pos;
            pos = tunguzka_shift_kv_cache(&model, pos, shift);
        }

        float *logits = tunguzka_forward(&model, token, pos);
        pos++;
        total_forwards++;

        if (step == 0) {
            for (uint32_t i = 0; i < model.config.vocab_size; i++) {
                first_logit_checksum += logits[i] * (float)((int)(i % 7) - 3);
            }
        }

        if (token < 128000) {
            push_recent_token(recent_tokens, &recent_cnt, 64, token);
        }

        if (step < num_prompt_tokens - 1) {
            next = prompt_tokens[step + 1];
            if (has_tok && !is_chat_wrapped && next != 1 && next != 2) {
                safe_print(tokenizer_decode(&tokenizer, token, next));
            }
            token = next;
            continue;
        }

        if (temp > 0.0f) {
            apply_repetition_penalty(logits, model.config.vocab_size, recent_tokens, recent_cnt, rep_penalty);
        }
        next = sample_top_k_top_p(logits, model.config.vocab_size, temp, top_k, top_p, &seed);
        if (next == 1 || next == 2 || (next >= 128001 && next <= 128009)) {
            break;
        }

        if (has_tok) {
            char *piece = tokenizer_decode(&tokenizer, token, next);
            if (check_stop_sequence(tail_window, sizeof(tail_window), piece, is_chat_wrapped)) {
                break;
            }
            safe_print(piece);
        } else {
            printf("[%d]", next);
        }
        token = next;
    }
    printf("\n");

    float elapsed_ms = get_time_ms() - t_start;
    float tok_s = (elapsed_ms > 0.0f) ? ((float)total_forwards * 1000.0f / elapsed_ms) : 0.0f;
    fprintf(stderr, "[Stats] %d tokens in %.2f ms (%.2f tok/s) | step0_checksum=%.6f\n",
            total_forwards, (double)elapsed_ms, (double)tok_s, (double)first_logit_checksum);

    if (model.stream_mode) {
        uint32_t total_emb = model.emb_cache_hits + model.emb_cache_misses;
        float hit_pct = (total_emb > 0) ? ((float)model.emb_cache_hits * 100.0f / (float)total_emb) : 0.0f;
        fprintf(stderr, "[Stream] disk_reads=%u (%.1f KB) | emb_lru=%u/%u hits (%.0f%%) | pinned=%u/%u tensors (%.1f KB)\n",
                model.disk_reads,
                (double)((float)model.disk_bytes_read / 1024.0f),
                model.emb_cache_hits, total_emb, (double)hit_pct,
                model.pinned_tensors, model.total_tensors,
                (double)((float)model.pin_pool_size / 1024.0f));
    }

    free(prompt_tokens);
    if (has_tok) tokenizer_free(&tokenizer);
    tunguzka_free(&model);
    return 0;
}