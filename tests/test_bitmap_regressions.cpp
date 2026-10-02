#include "render/bitmap_renderer.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ascii;

namespace {
int failures = 0;
void check(const char* label, bool passed) {
    std::cout << (passed ? "PASS " : "FAIL ") << label << '\n';
    failures += !passed;
}
double decode(uint8_t value) {
    const double encoded = value / 255.0;
    return encoded <= 0.04045 ? encoded / 12.92 : std::pow((encoded + 0.055) / 1.055, 2.4);
}
int reference_blend(uint8_t foreground, uint8_t background, uint8_t alpha) {
    const double a = alpha / 255.0;
    const double linear = decode(background) * (1.0 - a) + decode(foreground) * a;
    return static_cast<int>(std::lround(255.0 * (linear <= 0.0031308 ? 12.92 * linear :
                                              1.055 * std::pow(linear, 1.0 / 2.4) - 0.055)));
}
bool same_pixels(const FrameBuffer& a, const FrameBuffer& b) {
    return a.width() == b.width() && a.height() == b.height() &&
           std::equal(a.data(), a.data() + a.byte_size(), b.data());
}
}

int main() {
    FontLoader loader;
    if (loader.load_system_fallback(16).failure()) {
        std::cerr << "A system monospace font is required for bitmap regressions\n";
        return 2;
    }
    GlyphCache cache;
    if (!cache.initialize(&loader, {' ', '@', 0x2588}, 8, 16)) return 2;
    BitmapRenderer renderer;
    renderer.set_cache(&cache);
    const auto* glyph = cache.get_bitmap('@');
    const auto plain = renderer.render(std::vector<uint32_t>{'@'}, 1, 1);
    int plain_error = 0;
    int partial_pixels = 0;
    for (int i = 0; i < 128; ++i) {
        const uint8_t alpha = glyph->pixels[i];
        partial_pixels += alpha > 0 && alpha < 255;
        const auto pixel = plain.get_pixel(i % 8, i / 8);
        const int expected = reference_blend(255, 0, alpha);
        plain_error = std::max(plain_error, std::abs(static_cast<int>(pixel.r) - expected));
    }
    check("font fixture includes antialiased coverage", partial_pixels > 0);
    check("plain white glyph blends coverage in linear light", plain_error <= 1);

    ASCIICell cell;
    cell.codepoint = '@';
    const auto white = renderer.render(std::vector<ASCIICell>{cell}, 1, 1);
    check("plain and colored white glyph overloads agree", same_pixels(plain, white));
    cell.fg_r = 230; cell.fg_g = 140; cell.fg_b = 60;
    cell.bg_r = 20; cell.bg_g = 50; cell.bg_b = 100;
    const auto colored = renderer.render(std::vector<ASCIICell>{cell}, 1, 1);
    int colored_error = 0;
    bool opaque_output = true;
    for (int i = 0; i < 128; ++i) {
        const auto pixel = colored.get_pixel(i % 8, i / 8);
        const auto alpha = glyph->pixels[i];
        colored_error = std::max({colored_error,
            std::abs(static_cast<int>(pixel.r) - reference_blend(cell.fg_r, cell.bg_r, alpha)),
            std::abs(static_cast<int>(pixel.g) - reference_blend(cell.fg_g, cell.bg_g, alpha)),
            std::abs(static_cast<int>(pixel.b) - reference_blend(cell.fg_b, cell.bg_b, alpha))});
        opaque_output &= pixel.a == 255;
    }
    check("colored glyph blends foreground and background in linear light", colored_error <= 1);
    check("bitmap compositing produces opaque output", opaque_output);
    cell.codepoint = ' ';
    const auto background = renderer.render(std::vector<ASCIICell>{cell}, 1, 1).get_pixel(0,0);
    check("zero coverage preserves exact background bytes", background.r == 20 && background.g == 50 && background.b == 100);
    cell.codepoint = 0x2588;
    const auto foreground = renderer.render(std::vector<ASCIICell>{cell}, 1, 1).get_pixel(0,0);
    check("full coverage preserves exact foreground bytes", foreground.r == 230 && foreground.g == 140 && foreground.b == 60);

    renderer.set_cell_size(4, 8);
    const auto clipped = renderer.render(std::vector<uint32_t>{0x2588, ' '}, 2, 1);
    bool no_spill = true;
    for (int y = 0; y < 8; ++y)
        for (int x = 4; x < 8; ++x) no_spill &= clipped.get_pixel(x,y).r == 0;
    check("oversized cached glyph is clipped to its own cell", no_spill);

    for (const auto [width, height] : {std::pair{0,16}, {-1,16}, {8,0}, {8,-1},
                                      {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()}}) {
        bool rejected = false;
        try { renderer.set_cell_size(width, height); }
        catch (const std::invalid_argument&) { rejected = true; }
        catch (const std::length_error&) { rejected = true; }
        check("invalid cell geometry is rejected before rendering", rejected);
    }
    renderer.set_cell_size(8,16);
    check("empty plain grid stays empty", renderer.render(std::vector<uint32_t>{}, 0, 0).empty());
    check("empty colored grid stays empty", renderer.render(std::vector<ASCIICell>{}, 0, 0).empty());
    check("zero-area grid ignores unused large extent",
          renderer.render(std::vector<uint32_t>{}, std::numeric_limits<int>::max(), 0).empty());
    for (int overload = 0; overload < 2; ++overload) {
        bool rejected = false;
        try {
            if (overload == 0) renderer.render(std::vector<uint32_t>{}, -1, 1);
            else renderer.render(std::vector<ASCIICell>{}, 1, -1);
        } catch (const std::invalid_argument&) { rejected = true; }
        check("negative grid geometry is rejected", rejected);
        rejected = false;
        try {
            if (overload == 0) renderer.render(std::vector<uint32_t>{}, std::numeric_limits<int>::max(), 1);
            else renderer.render(std::vector<ASCIICell>{}, 10000, 10000);
        } catch (const std::length_error&) { rejected = true; }
        check("excessive grid geometry is rejected before allocation", rejected);
    }
    std::cout << "Maximum plain error: " << plain_error << "; maximum colored error: " << colored_error
              << "; failed expectations: " << failures << '\n';
    return failures != 0;
}
