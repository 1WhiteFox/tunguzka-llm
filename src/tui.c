#include "../include/tui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>

#if defined(_WIN32)
    #include <windows.h>
    #include <conio.h>
    static HANDLE g_hStdout = INVALID_HANDLE_VALUE;
#elif defined(__DJGPP__)
    #include <conio.h>
    #include <sys/farptr.h>
    #include <go32.h>
    #define DOS_SCREEN_SEG 0xB800
#else
    #include <termios.h>
    #include <unistd.h>
    #include <sys/select.h>
    static struct termios g_orig_termios;
    static int g_termios_saved = 0;
    static int g_is_utf8 = 0;
#endif

static const char * const g_strings[2][STR_TOTAL_COUNT] = {
    {
        "Tunguzka-LLM v2.0",
        "Backend",
        "Mode",
        "RAM RESIDENT",
        "DISK STREAM",
        "Arena",
        "KV",
        "[HDD: IDLE]",
        "[HDD: READ]",
        "Pinned",
        "Speed",
        "Latency",
        "You",
        "Tunguzka",
        "Generating",
        "1Help",
        "2Back",
        "3KV8",
        "4Strm",
        "5Rst",
        "6Ctx",
        "7Tmp",
        "8Lng",
        "9Cfg",
        "10Quit",
        "Welcome to Tunguzka-LLM! Type a prompt and press Enter. (PgUp/PgDn: scroll)",
        "[Help] F2:Back F3:KV8 F4:Stream F5:Reset F6:Ctx F7:Temp F8:Lang F9:Cfg/Models",
        "[Context reset to pos=0]",
        "[Sliding Window: KV-cache shifted with RoPE phase correction]",
        "Settings & Models (Arrows / Enter)"
    },
    {
        "Тунгузка-LLM v2.0",
        "Бэкенд",
        "Режим",
        "ОЗУ: РЕЗИДЕНТ",
        "ОЗУ: СТРИМ",
        "Арена",
        "КВ",
        "[ДИСК: ПОКОЙ]",
        "[ДИСК: ЧТЕНИЕ]",
        "Закреп",
        "Скорость",
        "Задержка",
        "Вы",
        "Тунгузка",
        "Генерация",
        "1Пом",
        "2Бэк",
        "3KV8",
        "4Стр",
        "5Сбр",
        "6Кон",
        "7Тмп",
        "8Язк",
        "9Наст",
        "10Вых",
        "Добро пожаловать! Введите запрос и нажмите Enter. (PgUp/PgDn: прокрутка)",
        "[Справка] F2:Бэк F3:KV8 F4:Стрим F5:Сброс F6:Конт F7:Темп F8:Язк F9:Настр/Мод",
        "[Контекст сброшен на pos=0]",
        "[Скользящее окно: KV-кэш сдвинут с фазовой коррекцией RoPE]",
        "Настройки и модели (Стрелки / Enter)"
    }
};

static int decode_utf8_cp(const unsigned char *s, uint32_t *out_cp) {
    if (*s < 0x80) {
        *out_cp = *s;
        return 1;
    } else if ((*s & 0xE0) == 0xC0) {
        if ((s[1] & 0xC0) == 0x80) {
            *out_cp = ((uint32_t)(s[0] & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
            return 2;
        }
        *out_cp = *s;
        return 1;
    } else if ((*s & 0xF0) == 0xE0) {
        if ((s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
            *out_cp = ((uint32_t)(s[0] & 0x0F) << 12) |
                      ((uint32_t)(s[1] & 0x3F) << 6)  |
                      (uint32_t)(s[2] & 0x3F);
            return 3;
        }
        *out_cp = *s;
        return 1;
    } else if ((*s & 0xF8) == 0xF0) {
        if ((s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 && (s[3] & 0xC0) == 0x80) {
            *out_cp = ((uint32_t)(s[0] & 0x07) << 18) |
                      ((uint32_t)(s[1] & 0x3F) << 12) |
                      ((uint32_t)(s[2] & 0x3F) << 6)  |
                      (uint32_t)(s[3] & 0x3F);
            return 4;
        }
        *out_cp = *s;
        return 1;
    }
    *out_cp = *s;
    return 1;
}

int tui_utf8_cols(const char *s) {
    if (!s) return 0;
    int count = 0;
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        uint32_t cp = 0;
        int len = decode_utf8_cp(p, &cp);
        if (cp == 0x2026) count += 3; /* '...' ellipsis */
        else count += 1;
        p += len;
    }
    return count;
}

static uint8_t utf8_to_cp866(uint32_t cp) {
    if (cp < 0x80) return (uint8_t)cp;
    if (cp >= 0x0410 && cp <= 0x043F) return (uint8_t)(0x80 + (cp - 0x0410));
    if (cp >= 0x0440 && cp <= 0x044F) return (uint8_t)(0xE0 + (cp - 0x0440));
    if (cp == 0x0401) return 0xF0; 
    if (cp == 0x0451) return 0xF1; 
    if (cp == 0x2116) return 0xFC; 
    if (cp == 0x00A0 || cp == 0x202F || cp == 0x2009) return ' ';
    if (cp == 0x2018 || cp == 0x2019 || cp == 0x201B || cp == 0x00B4 || cp == 0x02BC) return '\'';
    if (cp == 0x201C || cp == 0x201D || cp == 0x201E || cp == 0x201F || cp == 0x00AB || cp == 0x00BB) return '"';
    if (cp == 0x2010 || cp == 0x2011 || cp == 0x2012 || cp == 0x2013 || cp == 0x2014 || cp == 0x2015 || cp == 0x2212) return '-';
    if (cp == 0x2022 || cp == 0x2023 || cp == 0x25AA || cp == 0x25CF) return '*';
    return '?';
}

const char *tui_str(TuiEngine *tui, TuiStringId id) {
    if (id < 0 || id >= STR_TOTAL_COUNT) return "";
    return g_strings[tui->lang][id];
}

void tui_toggle_lang(TuiEngine *tui) {
    tui->lang = (tui->lang == TUI_LANG_EN) ? TUI_LANG_RU : TUI_LANG_EN;
}

void tui_init(TuiEngine *tui, TuiLang default_lang) {
    memset(tui, 0, sizeof(TuiEngine));
    tui->lang = default_lang;
#if defined(_WIN32)
    g_hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cci;
    cci.dwSize = 25;
    cci.bVisible = FALSE;
    SetConsoleCursorInfo(g_hStdout, &cci);
#elif defined(__DJGPP__)
    _setcursortype(_NOCURSOR);
#else
    if (!g_termios_saved) {
        tcgetattr(STDIN_FILENO, &g_orig_termios);
        g_termios_saved = 1;
    }
    struct termios raw = g_orig_termios;
    raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    const char *lang = getenv("LC_ALL");
    if (!lang) lang = getenv("LC_CTYPE");
    if (!lang) lang = getenv("LANG");
    if (lang && (strstr(lang, "UTF-8") || strstr(lang, "utf-8") || strstr(lang, "UTF8") || strstr(lang, "utf8"))) {
        g_is_utf8 = 1;
    } else {
        g_is_utf8 = 0;
    }
    fputs("\033[?1049h\033[?25l\033[2J", stdout);
    fflush(stdout);
#endif
    tui_clear(tui, TUI_ATTR_NORMAL);
}

void tui_shutdown(TuiEngine *tui) {
    (void)tui;
#if defined(_WIN32)
    if (g_hStdout != INVALID_HANDLE_VALUE) {
        CONSOLE_CURSOR_INFO cci;
        cci.dwSize = 25;
        cci.bVisible = TRUE;
        SetConsoleCursorInfo(g_hStdout, &cci);
    }
#elif defined(__DJGPP__)
    _setcursortype(_NORMALCURSOR);
#else
    if (g_termios_saved) {
        tcsetattr(STDIN_FILENO, TCSAFLUSH, &g_orig_termios);
        g_termios_saved = 0;
    }
    fputs("\033[0m\033[?1049l\033[?25h", stdout);
    fflush(stdout);
#endif
}

void tui_clear(TuiEngine *tui, uint8_t attr) {
    for (int y = 0; y < TUI_ROWS; y++) {
        for (int x = 0; x < TUI_COLS; x++) {
            tui->backbuffer[y][x].ch = ' ';
            tui->backbuffer[y][x].attr = attr;
        }
    }
}

void tui_print(TuiEngine *tui, int x, int y, uint8_t attr, const char *str) {
    if (y < 0 || y >= TUI_ROWS || !str) return;
    int cur_x = x;
    const unsigned char *s = (const unsigned char *)str;
    while (*s && cur_x < TUI_COLS) {
        uint32_t cp = 0;
        int len = decode_utf8_cp(s, &cp);
        if (cp == 0x2026) {
            for (int d = 0; d < 3 && cur_x < TUI_COLS; d++) {
                if (cur_x >= 0) {
                    tui->backbuffer[y][cur_x].ch = '.';
                    tui->backbuffer[y][cur_x].attr = attr;
                }
                cur_x++;
            }
        } else {
            if (cur_x >= 0) {
                tui->backbuffer[y][cur_x].ch = utf8_to_cp866(cp);
                tui->backbuffer[y][cur_x].attr = attr;
            }
            cur_x++;
        }
        s += len;
    }
}

void tui_printf(TuiEngine *tui, int x, int y, uint8_t attr, const char *fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    tui_print(tui, x, y, attr, buf);
}

void tui_draw_box(TuiEngine *tui, int x, int y, int w, int h, uint8_t attr, const char *title) {
    if (x < 0 || y < 0 || x + w > TUI_COLS || y + h > TUI_ROWS || w < 2 || h < 2) return;
    tui->backbuffer[y][x].ch = 0xDA;
    tui->backbuffer[y][x].attr = attr;
    tui->backbuffer[y][x + w - 1].ch = 0xBF;
    tui->backbuffer[y][x + w - 1].attr = attr;
    tui->backbuffer[y + h - 1][x].ch = 0xC0;
    tui->backbuffer[y + h - 1][x].attr = attr;
    tui->backbuffer[y + h - 1][x + w - 1].ch = 0xD9;
    tui->backbuffer[y + h - 1][x + w - 1].attr = attr;
    for (int i = 1; i < w - 1; i++) {
        tui->backbuffer[y][x + i].ch = 0xC4;
        tui->backbuffer[y][x + i].attr = attr;
        tui->backbuffer[y + h - 1][x + i].ch = 0xC4;
        tui->backbuffer[y + h - 1][x + i].attr = attr;
    }
    for (int j = 1; j < h - 1; j++) {
        tui->backbuffer[y + j][x].ch = 0xB3;
        tui->backbuffer[y + j][x].attr = attr;
        tui->backbuffer[y + j][x + w - 1].ch = 0xB3;
        tui->backbuffer[y + j][x + w - 1].attr = attr;
    }
    if (title && title[0] != '\0') {
        int char_count = tui_utf8_cols(title);
        if (char_count + 4 < w) {
            int tx = x + 2;
            tui->backbuffer[y][tx - 1].ch = ' ';
            tui_print(tui, tx, y, TUI_ATTR_TITLE, title);
            tui->backbuffer[y][tx + char_count].ch = ' ';
        }
    }
}

void tui_draw_bar(TuiEngine *tui, int x, int y, int width, int current, int max, uint8_t fill_attr, uint8_t empty_attr) {
    if (width <= 0 || max <= 0) return;
    int filled = (current * width) / max;
    if (filled > width) filled = width;
    for (int i = 0; i < width; i++) {
        if (i < filled) {
            tui->backbuffer[y][x + i].ch = 0xDB;
            tui->backbuffer[y][x + i].attr = fill_attr;
        } else {
            tui->backbuffer[y][x + i].ch = 0xB0;
            tui->backbuffer[y][x + i].attr = empty_attr;
        }
    }
}

#if !defined(_WIN32) && !defined(__DJGPP__)
static void tui_print_cell_utf8(uint8_t ch) {
    if (ch < 0x80) {
        putchar(ch);
        return;
    }
    switch (ch) {
        case 0xB0: fputs("\xE2\x96\x91", stdout); return;
        case 0xB1: fputs("\xE2\x96\x92", stdout); return;
        case 0xB2: fputs("\xE2\x96\x93", stdout); return;
        case 0xB3: fputs("\xE2\x94\x82", stdout); return;
        case 0xB4: fputs("\xE2\x94\xA4", stdout); return;
        case 0xBF: fputs("\xE2\x94\x90", stdout); return;
        case 0xC0: fputs("\xE2\x94\x94", stdout); return;
        case 0xC1: fputs("\xE2\x94\xB4", stdout); return;
        case 0xC2: fputs("\xE2\x94\xAC", stdout); return;
        case 0xC3: fputs("\xE2\x94\x9C", stdout); return;
        case 0xC4: fputs("\xE2\x94\x80", stdout); return;
        case 0xC5: fputs("\xE2\x94\xBC", stdout); return;
        case 0xD9: fputs("\xE2\x94\x98", stdout); return;
        case 0xDA: fputs("\xE2\x94\x8C", stdout); return;
        case 0xDB: fputs("\xE2\x96\x88", stdout); return;
        case 0xDC: fputs("\xE2\x96\x84", stdout); return;
        case 0xDD: fputs("\xE2\x96\x8C", stdout); return;
        case 0xDE: fputs("\xE2\x96\x90", stdout); return;
        case 0xDF: fputs("\xE2\x96\x80", stdout); return;
        case 0xFC: fputs("\xE2\x84\x96", stdout); return;
        default: break;
    }
    if (ch >= 0x80 && ch <= 0x9F) {
        putchar(0xD0);
        putchar(0x90 + (ch - 0x80));
    } else if (ch >= 0xA0 && ch <= 0xAF) {
        putchar(0xD0);
        putchar(0xB0 + (ch - 0xA0));
    } else if (ch >= 0xE0 && ch <= 0xEF) {
        putchar(0xD1);
        putchar(0x80 + (ch - 0xE0));
    } else if (ch == 0xF0) {
        putchar(0xD0);
        putchar(0x81);
    } else if (ch == 0xF1) {
        putchar(0xD1);
        putchar(0x91);
    } else {
        putchar('?');
    }
}
#endif

void tui_flip(TuiEngine *tui) {
#if defined(_WIN32)
    CHAR_INFO win_buf[TUI_ROWS * TUI_COLS];
    COORD buf_size = { TUI_COLS, TUI_ROWS };
    COORD buf_coord = { 0, 0 };
    SMALL_RECT write_region = { 0, 0, TUI_COLS - 1, TUI_ROWS - 1 };
    int changed = 0;
    for (int y = 0; y < TUI_ROWS; y++) {
        for (int x = 0; x < TUI_COLS; x++) {
            if (tui->backbuffer[y][x].ch != tui->frontbuffer[y][x].ch ||
                tui->backbuffer[y][x].attr != tui->frontbuffer[y][x].attr) {
                changed = 1;
            }
            int idx = y * TUI_COLS + x;
            win_buf[idx].Char.AsciiChar = (char)tui->backbuffer[y][x].ch;
            win_buf[idx].Attributes = tui->backbuffer[y][x].attr;
            tui->frontbuffer[y][x] = tui->backbuffer[y][x];
        }
    }
    if (changed && g_hStdout != INVALID_HANDLE_VALUE) {
        WriteConsoleOutputA(g_hStdout, win_buf, buf_size, buf_coord, &write_region);
    }
#elif defined(__DJGPP__)
    for (int y = 0; y < TUI_ROWS; y++) {
        for (int x = 0; x < TUI_COLS; x++) {
            if (tui->backbuffer[y][x].ch != tui->frontbuffer[y][x].ch ||
                tui->backbuffer[y][x].attr != tui->frontbuffer[y][x].attr) {
                int off = (y * TUI_COLS + x) * 2;
                _farpokeb(DOS_SCREEN_SEG, off, tui->backbuffer[y][x].ch);
                _farpokeb(DOS_SCREEN_SEG, off + 1, tui->backbuffer[y][x].attr);
                tui->frontbuffer[y][x] = tui->backbuffer[y][x];
            }
        }
    }
#else
    for (int y = 0; y < TUI_ROWS; y++) {
        for (int x = 0; x < TUI_COLS; x++) {
            if (tui->backbuffer[y][x].ch != tui->frontbuffer[y][x].ch ||
                tui->backbuffer[y][x].attr != tui->frontbuffer[y][x].attr) {
                uint8_t attr = tui->backbuffer[y][x].attr;
                uint8_t fg = attr & 0x0F;
                uint8_t bg = (attr >> 4) & 0x07;
                printf("\033[%d;%dH\033[%d;%dm", y + 1, x + 1,
                       30 + (fg % 8) + (fg >= 8 ? 60 : 0),
                       40 + bg);
                if (g_is_utf8) {
                    tui_print_cell_utf8(tui->backbuffer[y][x].ch);
                } else {
                    putchar(tui->backbuffer[y][x].ch);
                }
                tui->frontbuffer[y][x] = tui->backbuffer[y][x];
            }
        }
    }
    fflush(stdout);
#endif
}

int tui_poll_key(void) {
#if defined(_WIN32) || defined(__DJGPP__)
    if (_kbhit()) {
        int c = _getch();
        if (c == 0 || c == 0xE0) {
            return 0x100 | _getch();
        }
        return c;
    }
    return -1;
#else
    struct timeval tv = { 0, 0 };
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    if (select(STDIN_FILENO + 1, &fds, NULL, NULL, &tv) > 0) {
        unsigned char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1) {
            if (ch == 27) {
                struct timeval esc_tv = { 0, 25000 };
                fd_set esc_fds;
                FD_ZERO(&esc_fds);
                FD_SET(STDIN_FILENO, &esc_fds);
                if (select(STDIN_FILENO + 1, &esc_fds, NULL, NULL, &esc_tv) > 0) {
                    unsigned char seq[8];
                    int n = (int)read(STDIN_FILENO, seq, sizeof(seq) - 1);
                    if (n > 0) {
                        seq[n] = '\0';
                        if (seq[0] == 'O') {
                            if (seq[1] == 'P') return TUI_KEY_F1;
                            if (seq[1] == 'Q') return TUI_KEY_F2;
                            if (seq[1] == 'R') return TUI_KEY_F3;
                            if (seq[1] == 'S') return TUI_KEY_F4;
                        } else if (seq[0] == '[') {
                            if (seq[1] == 'A') return TUI_KEY_UP;
                            if (seq[1] == 'B') return TUI_KEY_DOWN;
                            if (seq[1] == 'C') return TUI_KEY_RIGHT;
                            if (seq[1] == 'D') return TUI_KEY_LEFT;
                            if (strncmp((char *)seq + 1, "5~", 2) == 0) return TUI_KEY_PGUP;
                            if (strncmp((char *)seq + 1, "6~", 2) == 0) return TUI_KEY_PGDN;
                            if (seq[1] == '[') {
                                if (seq[2] == 'A') return TUI_KEY_F1;
                                if (seq[2] == 'B') return TUI_KEY_F2;
                                if (seq[2] == 'C') return TUI_KEY_F3;
                                if (seq[2] == 'D') return TUI_KEY_F4;
                                if (seq[2] == 'E') return TUI_KEY_F5;
                            } else if (strncmp((char *)seq + 1, "11~", 3) == 0) {
                                return TUI_KEY_F1;
                            } else if (strncmp((char *)seq + 1, "12~", 3) == 0) {
                                return TUI_KEY_F2;
                            } else if (strncmp((char *)seq + 1, "13~", 3) == 0) {
                                return TUI_KEY_F3;
                            } else if (strncmp((char *)seq + 1, "14~", 3) == 0) {
                                return TUI_KEY_F4;
                            } else if (strncmp((char *)seq + 1, "15~", 3) == 0) {
                                return TUI_KEY_F5;
                            } else if (strncmp((char *)seq + 1, "17~", 3) == 0) {
                                return TUI_KEY_F6;
                            } else if (strncmp((char *)seq + 1, "18~", 3) == 0) {
                                return TUI_KEY_F7;
                            } else if (strncmp((char *)seq + 1, "19~", 3) == 0) {
                                return TUI_KEY_F8;
                            } else if (strncmp((char *)seq + 1, "20~", 3) == 0) {
                                return TUI_KEY_F9;
                            } else if (strncmp((char *)seq + 1, "21~", 3) == 0) {
                                return TUI_KEY_F10;
                            }
                        }
                    }
                }
            }
            return (int)ch;
        }
    }
    return -1;
#endif
}