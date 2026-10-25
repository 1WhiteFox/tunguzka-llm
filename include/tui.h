#ifndef TUNGUZKA_TUI_H
#define TUNGUZKA_TUI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TUI_ROWS 25
#define TUI_COLS 80

#define TUI_COLOR_BLACK         0x00
#define TUI_COLOR_DARK_GREEN    0x02
#define TUI_COLOR_LIGHT_GREEN   0x0A
#define TUI_COLOR_BROWN_AMBER   0x06
#define TUI_COLOR_YELLOW        0x0E
#define TUI_COLOR_WHITE         0x0F

#define TUI_ATTR_NORMAL         ((TUI_COLOR_BLACK << 4) | TUI_COLOR_LIGHT_GREEN)
#define TUI_ATTR_TEXT           TUI_ATTR_NORMAL
#define TUI_ATTR_BORDER         ((TUI_COLOR_BLACK << 4) | TUI_COLOR_DARK_GREEN)
#define TUI_ATTR_TITLE          ((TUI_COLOR_BLACK << 4) | TUI_COLOR_LIGHT_GREEN)
#define TUI_ATTR_USER           ((TUI_COLOR_BLACK << 4) | TUI_COLOR_WHITE)
#define TUI_ATTR_WARN           ((TUI_COLOR_BLACK << 4) | TUI_COLOR_YELLOW)
#define TUI_ATTR_BAR            ((TUI_COLOR_DARK_GREEN << 4) | TUI_COLOR_BLACK)
#define TUI_ATTR_HOTKEY_NUM     ((TUI_COLOR_DARK_GREEN << 4) | TUI_COLOR_YELLOW)
#define TUI_ATTR_HOTKEY_LBL     ((TUI_COLOR_DARK_GREEN << 4) | TUI_COLOR_WHITE)

#define TUI_KEY_ESC             27
#define TUI_KEY_F1              0x13B
#define TUI_KEY_F2              0x13C
#define TUI_KEY_F3              0x13D
#define TUI_KEY_F4              0x13E
#define TUI_KEY_F5              0x13F
#define TUI_KEY_F6              0x140
#define TUI_KEY_F7              0x141
#define TUI_KEY_F8              0x142
#define TUI_KEY_F9              0x143
#define TUI_KEY_F10             0x144
#define TUI_KEY_UP              0x148
#define TUI_KEY_PGUP            0x149
#define TUI_KEY_LEFT            0x14B
#define TUI_KEY_RIGHT           0x14D
#define TUI_KEY_DOWN            0x150
#define TUI_KEY_PGDN            0x151

typedef enum {
    TUI_LANG_EN = 0,
    TUI_LANG_RU = 1
} TuiLang;

typedef enum {
    STR_SYS_TITLE = 0,
    STR_BACKEND,
    STR_MODE,
    STR_MEM_RAM,
    STR_MEM_STREAM,
    STR_ARENA,
    STR_KV_CACHE,
    STR_HDD_IDLE,
    STR_HDD_READ,
    STR_PINNED,
    STR_SPEED,
    STR_LATENCY,
    STR_YOU,
    STR_ASSISTANT,
    STR_GEN_PROG,
    STR_F1_HELP,
    STR_F2_BACKEND,
    STR_F3_KV8,
    STR_F4_STREAM,
    STR_F5_RESET,
    STR_F6_CTX,
    STR_F7_TEMP,
    STR_F8_LANG,
    STR_F9_MODELS,
    STR_F10_QUIT,
    STR_WELCOME,
    STR_HELP_INFO,
    STR_CTX_RESET,
    STR_CTX_SHIFTED,
    STR_SELECT_MODEL,
    STR_TOTAL_COUNT
} TuiStringId;

#pragma pack(push, 1)
typedef struct {
    uint8_t ch;
    uint8_t attr;
} TuiCell;
#pragma pack(pop)

typedef struct {
    TuiCell backbuffer[TUI_ROWS][TUI_COLS];
    TuiCell frontbuffer[TUI_ROWS][TUI_COLS];
    TuiLang lang;
    int     cursor_x;
    int     cursor_y;
    int     cursor_visible;
} TuiEngine;

void tui_init(TuiEngine *tui, TuiLang default_lang);
void tui_shutdown(TuiEngine *tui);
void tui_clear(TuiEngine *tui, uint8_t attr);
void tui_draw_box(TuiEngine *tui, int x, int y, int w, int h, uint8_t attr, const char *title);
void tui_print(TuiEngine *tui, int x, int y, uint8_t attr, const char *str);
void tui_printf(TuiEngine *tui, int x, int y, uint8_t attr, const char *fmt, ...);
void tui_draw_bar(TuiEngine *tui, int x, int y, int width, int current, int max, uint8_t fill_attr, uint8_t empty_attr);
void tui_flip(TuiEngine *tui);
int  tui_poll_key(void);
void tui_toggle_lang(TuiEngine *tui);
const char *tui_str(TuiEngine *tui, TuiStringId id);
int  tui_utf8_cols(const char *s);

#ifdef __cplusplus
}
#endif
#endif /* TUNGUZKA_TUI_H */