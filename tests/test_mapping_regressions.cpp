#include "core/frame_composer.hpp"
#include "glyph/char_sets.hpp"
#include "render/bitmap_renderer.hpp"

#include <array>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <string>

using namespace ascii;

namespace {

int failures = 0;

void check(bool condition, const std::string& description) {
    std::cout << (condition ? "PASS " : "FAIL ") << description << '\n';
    if (!condition) ++failures;
}

BlockRenderer::CellData quadrants(unsigned mask) {
    BlockRenderer::CellData data;
    const float tl = (mask & 1) ? 1.0f : 0.0f;
    const float tr = (mask & 2) ? 1.0f : 0.0f;
    const float bl = (mask & 4) ? 1.0f : 0.0f;
    const float br = (mask & 8) ? 1.0f : 0.0f;
    data.top_left_lum = data.top_left_r = data.top_left_g = data.top_left_b = tl;
    data.top_right_lum = data.top_right_r = data.top_right_g = data.top_right_b = tr;
    data.bottom_left_lum = data.bottom_left_r = data.bottom_left_g = data.bottom_left_b = bl;
    data.bottom_right_lum = data.bottom_right_r = data.bottom_right_g = data.bottom_right_b = br;
    data.mean_luminance = data.mean_r = data.mean_g = data.mean_b = (tl + tr + bl + br) * .25f;
    return data;
}

ASCIICell to_ascii(const BlockCell& block) {
    ASCIICell cell;
    cell.codepoint = block.codepoint;
    cell.fg_r = block.fg_r; cell.fg_g = block.fg_g; cell.fg_b = block.fg_b;
    cell.bg_r = block.bg_r; cell.bg_g = block.bg_g; cell.bg_b = block.bg_b;
    return cell;
}

bool matches_quadrants(const FrameBuffer& frame, unsigned mask) {
    for (int y = 0; y < frame.height(); ++y) {
        for (int x = 0; x < frame.width(); ++x) {
            const unsigned quadrant = (y >= frame.height() / 2 ? 2 : 0) + (x >= frame.width() / 2 ? 1 : 0);
            const uint8_t expected = (mask & (1u << quadrant)) ? 255 : 0;
            const Color pixel = frame.get_pixel(x, y);
            if (pixel.r != expected || pixel.g != expected || pixel.b != expected) return false;
        }
    }
    return true;
}

Pipeline::Result make_frame(unsigned mask) {
    Pipeline::Result frame;
    frame.grid_cols = frame.grid_rows = 1;
    frame.luminance = FloatImage(8, 16);
    frame.color_buffer = FrameBuffer(8, 16);
    frame.cell_stats.resize(1);
    const auto data = quadrants(mask);
    auto& stats = frame.cell_stats[0];
    stats.mean_luminance = data.mean_luminance;
    stats.mean_r = stats.mean_g = stats.mean_b = data.mean_luminance;
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 8; ++x) {
            const unsigned quadrant = (y >= 8 ? 2 : 0) + (x >= 4 ? 1 : 0);
            const uint8_t value = (mask & (1u << quadrant)) ? 255 : 0;
            frame.luminance.set(x, y, value / 255.0f);
            frame.color_buffer.set_pixel(x, y, Color(value, value, value));
        }
    }
    return frame;
}

void test_contours(FontLoader& loader) {
    for (const std::string set : {"traditional", "blocks", "line-art"}) {
        GlyphCache cache;
        const auto requested = CharSet::get_set(set);
        check(cache.initialize(&loader, requested, 8, 16), "cache initialize " + set);
        const auto selectable = cache.get_by_brightness();
        check(std::set<uint32_t>(selectable.begin(), selectable.end()) ==
              std::set<uint32_t>(requested.begin(), requested.end()), "selectable set unchanged " + set);
        BitmapRenderer raster;
        raster.set_cache(&cache);
        for (const uint32_t cp : std::array<uint32_t, 13>{'-', '|', '/', '\\', '+',
                                                        '.', ':', '=', '*', '#', '%', '@', 'O'}) {
            ASCIICell cell;
            cell.codepoint = cp;
            const auto image = raster.render(std::vector<ASCIICell>{cell}, 1, 1);
            bool has_ink = false;
            for (int y = 0; y < image.height(); ++y) {
                for (int x = 0; x < image.width(); ++x) has_ink |= image.get_pixel(x, y).r != 0;
            }
            check(has_ink, "contour/debug raster " + set + " glyph=" + static_cast<char>(cp));
        }
    }
}

void test_quadrants(FontLoader& loader) {
    GlyphCache cache;
    check(cache.initialize(&loader, CharSet::get_set("basic"), 8, 16), "quadrant cache initialize");
    BitmapRenderer raster;
    raster.set_cache(&cache);
    BlockRenderer renderer;
    for (unsigned mask = 0; mask < 16; ++mask) {
        const auto block = renderer.render_cell(quadrants(mask));
        const auto image = raster.render(std::vector<ASCIICell>{to_ascii(block)}, 1, 1);
        check(matches_quadrants(image, mask), "quadrant mask=" + std::to_string(mask) +
              " codepoint=" + std::to_string(block.codepoint));
    }
    const std::array<uint32_t, 16> glyphs = {
        0x0020, 0x2598, 0x259D, 0x2580, 0x2596, 0x258C, 0x259E, 0x259B,
        0x2597, 0x259A, 0x2590, 0x259C, 0x2584, 0x2599, 0x259F, 0x2588
    };
    check(cache.initialize(&loader, CharSet::get_set("basic"), 7, 15), "odd-sized quadrant cache initialize");
    raster.set_cell_size(7, 15);
    for (unsigned mask = 0; mask < glyphs.size(); ++mask) {
        ASCIICell cell;
        cell.codepoint = glyphs[mask];
        check(matches_quadrants(raster.render({cell}, 1, 1), mask),
              "procedural quadrant coverage mask=" + std::to_string(mask));
    }

    auto data = quadrants(3);
    data.top_left_r = data.top_right_r = 1.0f;
    data.top_left_g = data.top_right_g = data.top_left_b = data.top_right_b = 0.0f;
    data.bottom_left_g = data.bottom_right_g = 1.0f;
    data.bottom_left_r = data.bottom_right_r = data.bottom_left_b = data.bottom_right_b = 0.0f;
    data.top_left_lum = data.top_right_lum = data.bottom_left_lum = data.bottom_right_lum = .5f;
    const auto image = raster.render({to_ascii(renderer.render_cell(data))}, 1, 1);
    bool chromatic_split = true;
    for (int y = 0; y < image.height(); ++y) {
        for (int x = 0; x < image.width(); ++x) {
            const auto pixel = image.get_pixel(x, y);
            chromatic_split &= pixel.r == (y < image.height() / 2 ? 255 : 0) &&
                               pixel.g == (y < image.height() / 2 ? 0 : 255) && pixel.b == 0;
        }
    }
    check(chromatic_split, "block selection preserves a chromatic split with equal luminance");
}

void test_temporal(FontLoader& loader) {
    GlyphCache cache;
    check(cache.initialize(&loader, CharSet::get_set("basic"), 8, 16), "temporal cache initialize");
    BitmapRenderer raster;
    raster.set_cache(&cache);
    Config config;
    config.edge.contours_enabled = false;
    TemporalSmoother smoother;
    CharSelector selector;
    ColorMapper mapper(ColorMode::BlockArt);
    BilateralGrid bilateral;
    MotionEstimator motion;
    BlockRenderer blocks;
    FrameComposer composer;
    FrameComposer::Context context{smoother, selector, mapper, bilateral, motion, blocks,
                                   config, ColorMode::BlockArt, .1f};
    auto output = composer.compose(make_frame(3), context);
    check(matches_quadrants(raster.render(output.cells, 1, 1), 3), "temporal initial top-half raster");
    for (int frame = 0; frame < 20; ++frame) output = composer.compose(make_frame(12), context);
    check(matches_quadrants(raster.render(output.cells, 1, 1), 12),
          "temporal converges to bottom-half raster after 20 frames, codepoint=" +
          std::to_string(output.cells[0].codepoint));
}

void test_adaptive_contour_precedence(FontLoader& loader) {
    GlyphCache cache;
    check(cache.initialize(&loader, {' '}, 8, 16), "adaptive cache initialize");
    Config config;
    config.grid.quad_tree_adaptive = true;
    TemporalSmoother smoother;
    CharSelector selector;
    selector.set_cache(&cache);
    ColorMapper mapper(ColorMode::Truecolor);
    BilateralGrid bilateral;
    MotionEstimator motion;
    BlockRenderer blocks;
    FrameComposer composer;
    FrameComposer::Context context{smoother, selector, mapper, bilateral, motion, blocks,
                                   config, ColorMode::Truecolor, .1f};
    auto frame = make_frame(3);
    auto& stats = frame.cell_stats[0];
    stats.adaptive_level = 2;
    auto output = composer.compose(frame, context);
    check(output.cells[0].codepoint == ' ' && output.block_cells.empty(),
          "adaptive non-blockart honors requested selectable glyphs");
    check(output.cells[0].bg_r == 0 && output.cells[0].bg_g == 0 && output.cells[0].bg_b == 0,
          "adaptive non-blockart preserves black background");
    const auto foreground = output.cells[0];
    stats.has_contour = true;
    stats.contour_strength = 1.0f;
    for (const uint32_t cp : std::array<uint32_t, 5>{'-', '|', '/', '\\', '+'}) {
        stats.contour_codepoint = cp;
        output = composer.compose(frame, context);
        check(output.cells[0].codepoint == cp, "adaptive contour overrides previous glyph " +
              std::to_string(cp));
        check(output.cells[0].fg_r == foreground.fg_r && output.cells[0].fg_g == foreground.fg_g &&
              output.cells[0].fg_b == foreground.fg_b, "contour preserves normal foreground color");
    }
    config.edge.contours_enabled = false;
    for (int i = 0; i < 20; ++i) output = composer.compose(frame, context);
    check(output.cells[0].codepoint == ' ', "disabled contours return to requested glyph set");
}

void test_palette_symmetry() {
    BlockRenderer renderer;
    std::vector<BlockCell> input(12);
    for (size_t i = 0; i < input.size(); ++i) {
        auto& cell = input[i];
        cell.codepoint = 0x2588;
        cell.fg_r = cell.bg_r = i % 3 == 0 ? 255 : 0;
        cell.fg_g = cell.bg_g = i % 3 == 1 ? 255 : 0;
        cell.fg_b = cell.bg_b = i % 3 == 2 ? 255 : 0;
    }
    auto first = input;
    auto second = input;
    renderer.spectral_quantize_frame(first, 2, 64, 8);
    renderer.spectral_quantize_frame(second, 2, 64, 8);
    std::set<std::array<uint8_t, 3>> colors;
    bool deterministic = true;
    for (size_t i = 0; i < first.size(); ++i) {
        const auto& a = first[i];
        const auto& b = second[i];
        colors.insert({a.fg_r, a.fg_g, a.fg_b});
        colors.insert({a.bg_r, a.bg_g, a.bg_b});
        deterministic &= a.codepoint == b.codepoint && a.fg_r == b.fg_r && a.fg_g == b.fg_g &&
                         a.fg_b == b.fg_b && a.bg_r == b.bg_r && a.bg_g == b.bg_g && a.bg_b == b.bg_b;
    }
    check(colors.size() <= 2, "symmetric primaries respect requested palette size");
    check(deterministic, "palette reduction is deterministic");
    std::fill(input.begin(), input.end(), input[0]);
    renderer.spectral_quantize_frame(input, 8, 64, 8);
    check(std::all_of(input.begin(), input.end(), [](const BlockCell& c) {
        return c.fg_r == 255 && c.bg_r == 255 && c.fg_g == 0 && c.bg_g == 0 &&
               c.fg_b == 0 && c.bg_b == 0;
    }), "solid color does not acquire empty black palette entries");
}

void test_small_weight_pruning(FontLoader& loader) {
    GlyphCache cache;
    check(cache.initialize(&loader, {' ', 0x2588}, 8, 16), "small-weight cache initialize");
    CharSelector::Config config;
    config.loss_weights = {.0001f, 0.0f, 0.0f, 0.0f, 0.0f};
    config.transition_penalty = 0.0f;
    CharSelector selector(config);
    selector.set_cache(&cache);
    CellStats stats;
    stats.mean_luminance = .9f;
    stats.adaptive_level = 1;
    const auto selected = selector.select_unified(stats, 0);
    check(selected.codepoint == 0x2588 && std::abs(selected.loss - .00001f) < 1e-8f,
          "pruning retains minimum true loss for tiny positive weights");
    config.loss_weights = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    selector.set_config(config);
    check(selector.select_unified(stats, 0).codepoint == ' ', "zero-loss tie retains first eligible glyph");
}

void test_nearest_orientation() {
    CharSelector selector;
    constexpr float pi = 3.14159265358979323846f;
    const std::array<std::pair<float, uint32_t>, 8> cases = {{
        {-.01f, '|'}, {.01f, '|'}, {-.5f * pi - .01f, '-'}, {-.5f * pi + .01f, '-'},
        {-.25f * pi - .01f, '/'}, {-.25f * pi + .01f, '/'},
        {.25f * pi - .01f, '\\'}, {.25f * pi + .01f, '\\'}
    }};
    for (const auto& [angle, glyph] : cases) {
        for (const float turns : {-4.0f * pi, 0.0f, 4.0f * pi}) {
            check(selector.select_edge_simple(angle + turns).codepoint == glyph,
                  "simple orientation chooses nearest tangent with wrapping " + std::to_string(angle + turns));
        }
    }
}

void test_strict_utf8() {
    const std::array<std::string, 8> invalid = {
        "\xC0\xAF", "\xC1\xBF", "\xE0\x80\xAF", "\xF0\x80\x80\xAF",
        "\xF4\x90\x80\x80", "\xF5\x80\x80\x80", "\xED\xA0\x80", "\x80\xBF"
    };
    for (size_t i = 0; i < invalid.size(); ++i) {
        check(CharSet::to_codepoints(invalid[i]).empty(), "reject invalid UTF-8 sequence " + std::to_string(i));
        check(CharSet::to_codepoints("A" + invalid[i] + "Z") == std::vector<uint32_t>{'A', 'Z'},
              "recover ASCII around invalid UTF-8 sequence " + std::to_string(i));
    }
    check(CharSet::to_codepoints("\xC2\x80\xDF\xBF\xE0\xA0\x80\xEF\xBF\xBF\xF0\x90\x80\x80\xF4\x8F\xBF\xBF") ==
          std::vector<uint32_t>{0x80, 0x7FF, 0x800, 0xFFFF, 0x10000, 0x10FFFF},
          "UTF-8 scalar boundaries remain valid");
}

void test_configured_dither_clamp() {
    Ditherer::Config config;
    config.error_clamp = .5f;
    Ditherer dither(config);
    dither.begin_frame(3, 2);
    dither.distribute_error(0, 0, 1, .5f, -.5f, .25f);
    float r = .5f, g = .5f, b = .5f;
    dither.apply_dithering(1, 0, r, g, b);
    check(std::abs(r - .71875f) < 1e-6f && std::abs(g - .28125f) < 1e-6f &&
          std::abs(b - .609375f) < 1e-6f, "configured dither clamp is not capped by a hidden .12 limit");
    for (int i = 0; i < 4; ++i) dither.distribute_error(0, 0, 1, .5f, -.5f, 0.0f);
    r = g = b = .5f;
    dither.apply_dithering(1, 0, r, g, b);
    check(r == 1.0f && g == 0.0f, "accumulated dither error uses configured bounds");
    config.error_clamp = .05f;
    dither.set_config(config);
    dither.begin_frame(3, 2);
    for (int i = 0; i < 8; ++i) dither.distribute_error(0, 0, 1, .5f, -.5f, 0.0f);
    r = g = b = .5f;
    dither.apply_dithering(1, 0, r, g, b);
    check(std::abs(r - .55f) < 1e-6f && std::abs(g - .45f) < 1e-6f,
          "small configured clamp also bounds accumulated error");
}

void test_dither_boundaries() {
    DitherBuffer buffer(3, 2);
    buffer.add_error(-1, 0, .1f, .1f, .1f);
    buffer.add_error(3, 1, .1f, .1f, .1f);
    check(buffer.get_error_r(-1, 0) == 0.0f && buffer.get_error_g(3, 1) == 0.0f,
          "dither ignores diffusion outside the grid");
    buffer.add_error(1, 0, .1f, -.1f, .05f);
    bool rejected = false;
    try {
        buffer.resize(-1, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "negative dither dimensions are rejected");
    if (rejected) {
        check(buffer.width() == 3 && buffer.height() == 2 && buffer.get_error_r(1, 0) == .1f,
              "rejected dither resize preserves its previous grid");
    }
}

void test_block_api_edges() {
    BlockRenderer::Config config;
    config.color_quantization_levels = 1;
    BlockRenderer renderer(config);
    const auto white = renderer.render_cell(quadrants(15));
    check(white.fg_r == 255 && white.bg_r == 255 && white.fg_g == 255 && white.bg_g == 255 &&
          white.fg_b == 255 && white.bg_b == 255, "one quantization level normalizes to black-white quantization");
    renderer.set_grid_size(2, 1);
    std::vector<BlockCell> cells(2);
    cells[0].codepoint = 'A';
    cells[1].codepoint = 'B';
    const auto previous = cells;
    cells[1].codepoint = 'C';
    const auto delta = renderer.render_to_ansi(cells, ColorMode::Truecolor, &previous);
    check(delta.find("\033[1C") != std::string::npos && delta.find(' ') == std::string::npos &&
          delta.ends_with("C\033[0m\n"), "block ANSI delta advances over unchanged cell without erasing it");
    check(renderer.render_to_ansi(cells, ColorMode::None, &previous) == "AC\n",
          "plain block output emits complete glyph rows without ANSI controls");
    renderer.set_grid_size(1, 1);
    for (uint32_t codepoint : {0u, 9u, 10u, 13u, 27u, 127u, 128u, 159u, 0xD800u, 0x110000u}) {
        cells.resize(1);
        cells[0].codepoint = codepoint;
        check(renderer.render_to_ansi(cells, ColorMode::None) == "\xEF\xBF\xBD\n",
              "block output replaces nonprinting codepoint " + std::to_string(codepoint));
    }
}

}  // namespace

int main() {
    FontLoader loader;
    const auto loaded = loader.load_system_fallback(16.0f);
    if (!loaded.success()) {
        std::cerr << "Unable to load installed system font: " << loaded.message << '\n';
        return 2;
    }
    test_contours(loader);
    test_quadrants(loader);
    test_temporal(loader);
    test_adaptive_contour_precedence(loader);
    test_palette_symmetry();
    test_small_weight_pruning(loader);
    test_nearest_orientation();
    test_strict_utf8();
    test_configured_dither_clamp();
    test_dither_boundaries();
    test_block_api_edges();
    std::cout << "FAILURES=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
