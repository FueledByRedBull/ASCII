#pragma once

#include "core/types.hpp"
#include <string>

namespace ascii {

enum class ColorMode {
    None,
    Ansi16,
    Ansi256,
    Truecolor,
    BlockArt
};

struct TerminalInfo {
    int cols = 80;
    int rows = 24;
    ColorMode color_mode = ColorMode::Truecolor;
    bool supports_utf8 = true;
    bool supports_box_drawing = true;
};

class Terminal {
public:
    Terminal();
    ~Terminal();
    Terminal(const Terminal&) = delete;
    Terminal& operator=(const Terminal&) = delete;
    
    TerminalInfo get_info() const;
    Size get_size() const;
    ColorMode detect_color_mode() const;
    
    void enter_alt_screen();
    void exit_alt_screen();
    void hide_cursor();
    void show_cursor();
    void clear_screen();
    void move_cursor(int row, int col);
    void move_cursor_home();
    
    void set_foreground(ColorMode mode, uint8_t r, uint8_t g, uint8_t b);
    void set_background(ColorMode mode, uint8_t r, uint8_t g, uint8_t b);
    void reset_colors();
    
    void write(const std::string& s);
    void flush();
    
    static std::string color_code(ColorMode mode, uint8_t r, uint8_t g, uint8_t b, bool fg);
    static uint8_t rgb_to_256(uint8_t r, uint8_t g, uint8_t b);
    static uint8_t rgb_to_16(uint8_t r, uint8_t g, uint8_t b);
    
private:
    TerminalInfo info_;
    bool in_alt_screen_ = false;
    bool cursor_hidden_ = false;
    bool output_written_ = false;
#ifdef _WIN32
    void* native_output_handle_ = nullptr;
    uint32_t original_output_mode_ = 0;
    unsigned int original_output_codepage_ = 0;
    bool output_mode_changed_ = false;
    bool output_codepage_changed_ = false;
#endif
};

}
