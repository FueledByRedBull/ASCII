#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "../src/render/terminal_renderer.hpp"

using namespace ascii;

namespace {

ASCIICell make_cell(uint32_t cp, uint8_t r, uint8_t g, uint8_t b, uint8_t br = 0, uint8_t bg = 0, uint8_t bb = 0) {
    ASCIICell cell;
    cell.codepoint = cp;
    cell.fg_r = r;
    cell.fg_g = g;
    cell.fg_b = b;
    cell.bg_r = br;
    cell.bg_g = bg;
    cell.bg_b = bb;
    return cell;
}

void test_glyph_and_foreground_changes_emit() {
    Terminal terminal;
    TerminalRenderer renderer(terminal, ColorMode::Truecolor);
    renderer.set_grid_size(1, 1);

    std::vector<ASCIICell> cells{make_cell('A', 255, 0, 0)};
    std::string first = renderer.render_to_string(cells);
    assert(first.find('A') != std::string::npos);
    assert(first.find("\033[38;2;255;0;0m") != std::string::npos);

    std::string unchanged = renderer.render_to_string(cells);
    assert(unchanged.empty());

    cells[0] = make_cell('A', 0, 255, 0);
    std::string fg_changed = renderer.render_to_string(cells);
    assert(fg_changed.find('A') != std::string::npos);
    assert(fg_changed.find("\033[38;2;0;255;0m") != std::string::npos);

    cells[0] = make_cell('B', 0, 255, 0);
    std::string glyph_changed = renderer.render_to_string(cells);
    assert(glyph_changed.find('B') != std::string::npos);
}

void test_blockart_background_change_emits_background_code() {
    Terminal terminal;
    TerminalRenderer renderer(terminal, ColorMode::BlockArt);
    renderer.set_grid_size(1, 1);

    std::vector<ASCIICell> cells{make_cell(0x2580, 255, 255, 255, 0, 0, 0)};
    std::string first = renderer.render_to_string(cells);
    assert(first.find("\033[48;2;0;0;0m") != std::string::npos);

    cells[0] = make_cell(0x2580, 255, 255, 255, 10, 20, 30);
    std::string bg_changed = renderer.render_to_string(cells);
    assert(bg_changed.find("\033[48;2;10;20;30m") != std::string::npos);
}

}  // namespace

int main() {
    test_glyph_and_foreground_changes_emit();
    test_blockart_background_change_emits_background_code();
    std::cout << "Terminal renderer tests passed\n";
    return 0;
}
