#include "terminal_renderer.hpp"
#include <charconv>
#include <system_error>
#include <limits>
#include <stdexcept>

namespace ascii {

TerminalRenderer::TerminalRenderer(Terminal& term, ColorMode color_mode)
    : term_(term), color_mode_(color_mode) {}

void TerminalRenderer::set_color_mode(ColorMode mode) {
    if (mode == color_mode_) return;
    color_mode_ = mode;
    prev_buffer_.clear();
    force_color_reset_ = true;
}

void TerminalRenderer::set_grid_size(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || cols > std::numeric_limits<int>::max() / rows) {
        throw std::invalid_argument("Terminal grid dimensions are invalid");
    }
    cols_ = cols;
    rows_ = rows;
    prev_buffer_.clear();
    const size_t reserve_bytes = static_cast<size_t>(cols) * static_cast<size_t>(rows) * 8;
    if (out_buffer_.capacity() < reserve_bytes) {
        out_buffer_.reserve(reserve_bytes);
    }
}

void TerminalRenderer::render(const std::vector<ASCIICell>& cells) {
    write_buffer_ = render_to_string(cells);
    if (!write_buffer_.empty()) {
        term_.write(write_buffer_);
    }
    term_.flush();
}

std::string TerminalRenderer::render_to_string(const std::vector<ASCIICell>& cells) {
    out_buffer_.clear();
    if (force_color_reset_) {
        append_string("\033[0m");
        force_color_reset_ = false;
    }
    int cursor_x = -1;
    int cursor_y = -1;
    const auto changed = [&](size_t idx) {
        if (idx >= prev_buffer_.size()) return true;
        const auto& cell = cells[idx];
        const auto& prev = prev_buffer_[idx];
        return cell.codepoint != prev.codepoint ||
               cell.fg_r != prev.fg_r || cell.fg_g != prev.fg_g || cell.fg_b != prev.fg_b ||
               cell.bg_r != prev.bg_r || cell.bg_g != prev.bg_g || cell.bg_b != prev.bg_b;
    };
    
    int y = 0;
    while (y < rows_) {
        int x = 0;
        while (x < cols_) {
            int idx = y * cols_ + x;
            if (static_cast<size_t>(idx) >= cells.size()) break;
            
            const ASCIICell& cell = cells[idx];
            if (!changed(static_cast<size_t>(idx))) {
                x++;
                continue;
            }

            const int target_x = x + 1;
            const int target_y = y + 1;
            if (cursor_x != target_x || cursor_y != target_y) {
                append_cursor_move(target_y, target_x);
                cursor_x = target_x;
                cursor_y = target_y;
            }
            
            if (color_mode_ != ColorMode::None) {
                append_string(Terminal::color_code(color_mode_, cell.fg_r, cell.fg_g, cell.fg_b, true));
                if (color_mode_ == ColorMode::BlockArt) {
                    append_string(Terminal::color_code(color_mode_, cell.bg_r, cell.bg_g, cell.bg_b, false));
                }
            }
            
            int run_end = x;
            uint8_t run_r = cell.fg_r, run_g = cell.fg_g, run_b = cell.fg_b;
            uint8_t run_bg_r = cell.bg_r, run_bg_g = cell.bg_g, run_bg_b = cell.bg_b;
            
            for (int rx = x; rx < cols_; ++rx) {
                int ridx = y * cols_ + rx;
                if (static_cast<size_t>(ridx) >= cells.size()) break;
                
                const ASCIICell& rcell = cells[ridx];
                if (!changed(static_cast<size_t>(ridx))) break;
                
                bool color_matches = rcell.fg_r == run_r && rcell.fg_g == run_g && rcell.fg_b == run_b;
                bool bg_matches = rcell.bg_r == run_bg_r && rcell.bg_g == run_bg_g && rcell.bg_b == run_bg_b;
                if (!color_matches || (color_mode_ == ColorMode::BlockArt && !bg_matches)) break;
                
                run_end = rx;
            }
            
            for (int rx = x; rx <= run_end; ++rx) {
                int ridx = y * cols_ + rx;
                if (static_cast<size_t>(ridx) < cells.size()) {
                    append_utf8(cells[ridx].codepoint);
                }
            }

            cursor_x = run_end + 2;
            cursor_y = target_y;
            
            x = run_end + 1;
        }
        y++;
    }

    prev_buffer_ = cells;
    return std::string(out_buffer_.begin(), out_buffer_.end());
}

void TerminalRenderer::append_string(const std::string& s) {
    out_buffer_.insert(out_buffer_.end(), s.begin(), s.end());
}

void TerminalRenderer::append_cursor_move(int row, int col) {
    out_buffer_.push_back('\033');
    out_buffer_.push_back('[');

    char tmp[16];
    auto row_res = std::to_chars(tmp, tmp + sizeof(tmp), row);
    if (row_res.ec == std::errc()) {
        out_buffer_.insert(out_buffer_.end(), tmp, row_res.ptr);
    } else {
        out_buffer_.push_back('1');
    }

    out_buffer_.push_back(';');

    auto col_res = std::to_chars(tmp, tmp + sizeof(tmp), col);
    if (col_res.ec == std::errc()) {
        out_buffer_.insert(out_buffer_.end(), tmp, col_res.ptr);
    } else {
        out_buffer_.push_back('1');
    }

    out_buffer_.push_back('H');
}

void TerminalRenderer::append_utf8(uint32_t cp) {
    if (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F) ||
        (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
        cp = 0xFFFD;
    }
    if (cp < 0x80) {
        out_buffer_.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out_buffer_.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out_buffer_.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out_buffer_.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out_buffer_.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out_buffer_.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out_buffer_.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out_buffer_.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out_buffer_.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out_buffer_.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

}
