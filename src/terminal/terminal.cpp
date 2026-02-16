#include "terminal.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <array>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#include <termios.h>
#endif

namespace ascii {

namespace {
    struct Color256Lookup {
        std::array<uint8_t, 256> r{};
        std::array<uint8_t, 256> g{};
        std::array<uint8_t, 256> b{};
        bool initialized = false;
        
        Color256Lookup() {
            const uint8_t palette[16][3] = {
                {0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
                {0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {192, 192, 192},
                {128, 128, 128}, {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
                {0, 0, 255}, {255, 0, 255}, {0, 255, 255}, {255, 255, 255}
            };
            
            for (int i = 0; i < 16; ++i) {
                r[i] = palette[i][0];
                g[i] = palette[i][1];
                b[i] = palette[i][2];
            }
            
            for (int i = 16; i < 232; ++i) {
                int idx = i - 16;
                int rv = idx / 36;
                int gv = (idx % 36) / 6;
                int bv = idx % 6;
                r[i] = rv ? static_cast<uint8_t>(55 + rv * 40) : 0;
                g[i] = gv ? static_cast<uint8_t>(55 + gv * 40) : 0;
                b[i] = bv ? static_cast<uint8_t>(55 + bv * 40) : 0;
            }
            
            for (int i = 232; i < 256; ++i) {
                uint8_t gray = static_cast<uint8_t>(8 + (i - 232) * 10);
                r[i] = g[i] = b[i] = gray;
            }
            
            initialized = true;
        }
    };
    
    static Color256Lookup color256_lookup;
    
    inline uint8_t color_distance_fast(uint8_t r1, uint8_t g1, uint8_t b1, uint8_t r2, uint8_t g2, uint8_t b2) {
        int dr = r1 - r2;
        int dg = g1 - g2;
        int db = b1 - b2;
        return static_cast<uint8_t>((dr*dr + dg*dg + db*db) >> 8);
    }
}

Terminal::Terminal() {
    info_ = get_info();
}

Terminal::~Terminal() {
    if (in_alt_screen_) exit_alt_screen();
    if (cursor_hidden_) show_cursor();
    reset_colors();
    flush();
}

TerminalInfo Terminal::get_info() const {
    TerminalInfo info;
    
#ifdef _WIN32
    HANDLE h_console = GetStdHandle(STD_OUTPUT_HANDLE);
    if (h_console != INVALID_HANDLE_VALUE) {
        CONSOLE_SCREEN_BUFFER_INFO csbi;
        if (GetConsoleScreenBufferInfo(h_console, &csbi)) {
            info.cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
            info.rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;
        }
    }
#else
    winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) {
        info.cols = ws.ws_col;
        info.rows = ws.ws_row;
    }
#endif
    
    if (info.cols <= 0) info.cols = 80;
    if (info.rows <= 0) info.rows = 24;
    
    info.color_mode = detect_color_mode();
    
    const char* term = std::getenv("TERM");
    if (term) {
        std::string t(term);
        info.supports_utf8 = (t.find("xterm") != std::string::npos ||
                              t.find("screen") != std::string::npos ||
                              t.find("tmux") != std::string::npos ||
                              t.find("rxvt") != std::string::npos ||
                              t.find("alacritty") != std::string::npos ||
                              t.find("kitty") != std::string::npos);
    }
#ifdef _WIN32
    info.supports_utf8 = true;
#endif
    
    info.supports_box_drawing = info.supports_utf8;
    
    return info;
}

Size Terminal::get_size() const {
    return {info_.cols, info_.rows};
}

ColorMode Terminal::detect_color_mode() const {
    const char* colorterm = std::getenv("COLORTERM");
    if (colorterm) {
        std::string ct(colorterm);
        if (ct == "truecolor" || ct == "24bit") {
            return ColorMode::Truecolor;
        }
    }
    
    const char* term = std::getenv("TERM");
    if (term) {
        std::string t(term);
        if (t.find("256color") != std::string::npos) {
            return ColorMode::Ansi256;
        }
    }
    
    return ColorMode::Ansi16;
}

void Terminal::enter_alt_screen() {
    if (in_alt_screen_) return;
    printf("\033[?1049h");
    in_alt_screen_ = true;
    flush();
}

void Terminal::exit_alt_screen() {
    if (!in_alt_screen_) return;
    printf("\033[?1049l");
    in_alt_screen_ = false;
    flush();
}

void Terminal::hide_cursor() {
    if (cursor_hidden_) return;
    printf("\033[?25l");
    cursor_hidden_ = true;
    flush();
}

void Terminal::show_cursor() {
    if (!cursor_hidden_) return;
    printf("\033[?25h");
    cursor_hidden_ = false;
    flush();
}

void Terminal::clear_screen() {
    printf("\033[2J");
    move_cursor_home();
    flush();
}

void Terminal::move_cursor(int row, int col) {
    printf("\033[%d;%dH", row + 1, col + 1);
}

void Terminal::move_cursor_home() {
    printf("\033[H");
}

void Terminal::set_foreground(ColorMode mode, uint8_t r, uint8_t g, uint8_t b) {
    printf("%s", color_code(mode, r, g, b, true).c_str());
}

void Terminal::set_background(ColorMode mode, uint8_t r, uint8_t g, uint8_t b) {
    printf("%s", color_code(mode, r, g, b, false).c_str());
}

void Terminal::reset_colors() {
    printf("\033[0m");
}

void Terminal::write(const std::string& s) {
    printf("%s", s.c_str());
}

void Terminal::flush() {
    fflush(stdout);
}

std::string Terminal::color_code(ColorMode mode, uint8_t r, uint8_t g, uint8_t b, bool fg) {
    char buf[32];
    switch (mode) {
        case ColorMode::None:
            return "";
        case ColorMode::Ansi16: {
            int idx = rgb_to_16(r, g, b);
            int base;
            int color_idx;
            if (idx < 8) {
                base = fg ? 30 : 40;
                color_idx = idx;
            } else {
                base = fg ? 90 : 100;
                color_idx = idx - 8;
            }
            snprintf(buf, sizeof(buf), "\033[%dm", base + color_idx);
            return buf;
        }
        case ColorMode::Ansi256: {
            int idx = rgb_to_256(r, g, b);
            snprintf(buf, sizeof(buf), fg ? "\033[38;5;%dm" : "\033[48;5;%dm", idx);
            return buf;
        }
        case ColorMode::Truecolor:
        case ColorMode::BlockArt:
            snprintf(buf, sizeof(buf), fg ? "\033[38;2;%d;%d;%dm" : "\033[48;2;%d;%d;%dm", r, g, b);
            return buf;
    }
    return "";
}

uint8_t Terminal::rgb_to_256(uint8_t r, uint8_t g, uint8_t b) {
    uint8_t best_idx = 0;
    uint32_t best_dist = UINT32_MAX;
    
    for (int i = 0; i < 256; ++i) {
        int dr = r - color256_lookup.r[i];
        int dg = g - color256_lookup.g[i];
        int db = b - color256_lookup.b[i];
        uint32_t dist = dr*dr + dg*dg + db*db;
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<uint8_t>(i);
        }
    }
    
    return best_idx;
}

uint8_t Terminal::rgb_to_16(uint8_t r, uint8_t g, uint8_t b) {
    static const uint8_t palette[16][3] = {
        {0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
        {0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {192, 192, 192},
        {128, 128, 128}, {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
        {0, 0, 255}, {255, 0, 255}, {0, 255, 255}, {255, 255, 255}
    };
    
    uint8_t best_idx = 0;
    uint32_t best_dist = UINT32_MAX;
    
    for (int i = 0; i < 16; ++i) {
        int dr = r - palette[i][0];
        int dg = g - palette[i][1];
        int db = b - palette[i][2];
        uint32_t dist = dr*dr + dg*dg + db*db;
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<uint8_t>(i);
        }
    }
    
    return best_idx;
}

}
