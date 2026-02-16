#include "terminal_renderer.hpp"
#include <sstream>
#include <cstring>

namespace ascii {

TerminalRenderer::TerminalRenderer(Terminal& term, ColorMode color_mode)
    : term_(term), color_mode_(color_mode) {}

void TerminalRenderer::set_grid_size(int cols, int rows) {
    cols_ = cols;
    rows_ = rows;
    prev_buffer_.assign(static_cast<size_t>(cols) * rows, ASCIICell{});
}

void TerminalRenderer::render(const std::vector<ASCIICell>& cells) {
    std::ostringstream out;
    int cursor_x = -1;
    int cursor_y = -1;
    
    int y = 0;
    while (y < rows_) {
        int x = 0;
        while (x < cols_) {
            int idx = y * cols_ + x;
            if (idx >= static_cast<int>(cells.size())) break;
            
            const ASCIICell& cell = cells[idx];
            const ASCIICell& prev = prev_buffer_.empty() ? ASCIICell{} : prev_buffer_[idx];
            
            bool changed = cell.codepoint != prev.codepoint ||
                          cell.fg_r != prev.fg_r || cell.fg_g != prev.fg_g || cell.fg_b != prev.fg_b ||
                          cell.bg_r != prev.bg_r || cell.bg_g != prev.bg_g || cell.bg_b != prev.bg_b;
            
            if (!changed) {
                x++;
                continue;
            }

            const int target_x = x + 1;
            const int target_y = y + 1;
            if (cursor_x != target_x || cursor_y != target_y) {
                char cursor_buf[32];
                snprintf(cursor_buf, sizeof(cursor_buf), "\033[%d;%dH", target_y, target_x);
                out << cursor_buf;
                cursor_x = target_x;
                cursor_y = target_y;
            }
            
            if (color_mode_ != ColorMode::None) {
                out << Terminal::color_code(color_mode_, cell.fg_r, cell.fg_g, cell.fg_b, true);
                if (color_mode_ == ColorMode::BlockArt) {
                    out << Terminal::color_code(color_mode_, cell.bg_r, cell.bg_g, cell.bg_b, false);
                }
            }
            
            int run_end = x;
            uint8_t run_r = cell.fg_r, run_g = cell.fg_g, run_b = cell.fg_b;
            uint8_t run_bg_r = cell.bg_r, run_bg_g = cell.bg_g, run_bg_b = cell.bg_b;
            
            for (int rx = x; rx < cols_; ++rx) {
                int ridx = y * cols_ + rx;
                if (ridx >= static_cast<int>(cells.size())) break;
                
                const ASCIICell& rcell = cells[ridx];
                const ASCIICell& rprev = prev_buffer_.empty() ? ASCIICell{} : prev_buffer_[ridx];
                
                bool rchanged = rcell.codepoint != rprev.codepoint ||
                               rcell.fg_r != rprev.fg_r || rcell.fg_g != rprev.fg_g || rcell.fg_b != rprev.fg_b ||
                               rcell.bg_r != rprev.bg_r || rcell.bg_g != rprev.bg_g || rcell.bg_b != rprev.bg_b;
                
                if (!rchanged) break;
                
                bool color_matches = rcell.fg_r == run_r && rcell.fg_g == run_g && rcell.fg_b == run_b;
                bool bg_matches = rcell.bg_r == run_bg_r && rcell.bg_g == run_bg_g && rcell.bg_b == run_bg_b;
                if (!color_matches || (color_mode_ == ColorMode::BlockArt && !bg_matches)) break;
                
                run_end = rx;
            }
            
            for (int rx = x; rx <= run_end; ++rx) {
                int ridx = y * cols_ + rx;
                if (ridx < static_cast<int>(cells.size())) {
                    out << codepoint_to_utf8(cells[ridx].codepoint);
                }
            }

            cursor_x = run_end + 2;
            cursor_y = target_y;
            
            x = run_end + 1;
        }
        y++;
    }
    
    term_.write(out.str());
    term_.flush();
    
    prev_buffer_ = cells;
}

std::string TerminalRenderer::codepoint_to_utf8(uint32_t cp) const {
    std::string result;
    if (cp < 0x80) {
        result += static_cast<char>(cp);
    } else if (cp < 0x800) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

}
