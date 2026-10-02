#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

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

void test_color_mode_change_forces_reset_and_repaint() {
    Terminal terminal;
    TerminalRenderer renderer(terminal, ColorMode::BlockArt);
    renderer.set_grid_size(1, 1);
    std::vector<ASCIICell> cells{make_cell(0x2580, 255, 0, 0, 0, 0, 255)};
    assert(!renderer.render_to_string(cells).empty());
    assert(renderer.render_to_string(cells).empty());

    renderer.set_color_mode(ColorMode::None);
    std::string none = renderer.render_to_string(cells);
    assert(none.find("\033[0m") != std::string::npos);
    assert(none.find("\033[48;2;") == std::string::npos);
    assert(none.find("\xE2\x96\x80") != std::string::npos);

    renderer.set_color_mode(ColorMode::Truecolor);
    std::string truecolor = renderer.render_to_string(cells);
    assert(truecolor.find("\033[0m") != std::string::npos);
    assert(truecolor.find("\033[38;2;255;0;0m") != std::string::npos);
}

void test_blank_cells_repaint_and_short_frames_extend() {
    Terminal terminal;
    TerminalRenderer renderer(terminal, ColorMode::None);
    renderer.set_grid_size(2, 1);
    const ASCIICell blank;
    assert(renderer.render_to_string({blank}).find(' ') != std::string::npos);
    auto extended = renderer.render_to_string({blank, make_cell('B', 255, 255, 255)});
    assert(extended.find("\033[1;2HB") != std::string::npos);
    assert(renderer.render_to_string({blank, make_cell('B', 255, 255, 255)}).empty());

    renderer.set_color_mode(ColorMode::Truecolor);
    const auto repaint = renderer.render_to_string({blank, blank});
    assert(repaint.find("  ") != std::string::npos);
    renderer.set_grid_size(1, 1);
    assert(renderer.render_to_string({blank}).find(' ') != std::string::npos);
}

void test_nonprinting_codepoints_render_as_replacement_glyphs() {
    Terminal terminal;
    TerminalRenderer renderer(terminal, ColorMode::None);
    renderer.set_grid_size(1, 1);
    for (uint32_t codepoint : {0u, 9u, 10u, 13u, 27u, 127u, 128u, 159u, 0xD800u, 0x110000u}) {
        const auto output = renderer.render_to_string({make_cell(codepoint, 255, 255, 255)});
        assert(output == "\033[1;1H\xEF\xBF\xBD");
    }
}

void test_console_state_is_restored() {
#ifdef _WIN32
    const HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD original_mode = 0;
    if (!GetConsoleMode(handle, &original_mode)) return;
    const auto original_codepage = GetConsoleOutputCP();
    {
        Terminal terminal;
        DWORD active_mode = 0;
        assert(GetConsoleMode(handle, &active_mode));
        assert((active_mode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0);
        assert(GetConsoleOutputCP() == CP_UTF8);
    }
    DWORD restored_mode = 0;
    assert(GetConsoleMode(handle, &restored_mode));
    assert(restored_mode == original_mode);
    assert(GetConsoleOutputCP() == original_codepage);
#endif
}

}  // namespace

int main() {
    test_glyph_and_foreground_changes_emit();
    test_blockart_background_change_emits_background_code();
    test_color_mode_change_forces_reset_and_repaint();
    test_blank_cells_repaint_and_short_frames_extend();
    test_nonprinting_codepoints_render_as_replacement_glyphs();
    test_console_state_is_restored();
    std::cout << "Terminal renderer tests passed\n";
    return 0;
}
