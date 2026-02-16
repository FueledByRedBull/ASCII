#pragma once

#include "core/types.hpp"
#include "terminal/terminal.hpp"
#include "mapping/color_mapper.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace ascii {

struct ASCIICell {
    uint32_t codepoint = ' ';
    uint8_t fg_r = 255, fg_g = 255, fg_b = 255;
    uint8_t bg_r = 0, bg_g = 0, bg_b = 0;
};

class TerminalRenderer {
public:
    TerminalRenderer(Terminal& term, ColorMode color_mode);
    
    void set_grid_size(int cols, int rows);
    void set_color_mode(ColorMode mode) { color_mode_ = mode; }
    void render(const std::vector<ASCIICell>& cells);
    
private:
    Terminal& term_;
    ColorMode color_mode_;
    int cols_ = 80;
    int rows_ = 24;
    std::vector<ASCIICell> prev_buffer_;
    std::vector<char> out_buffer_;
    std::string write_buffer_;

    void append_utf8(uint32_t cp);
    void append_cursor_move(int row, int col);
    void append_string(const std::string& s);
};

}
