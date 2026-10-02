#include "mapping/char_selector.hpp"
#include "glyph/char_sets.hpp"
#include "core/pipeline.hpp"
#include <cmath>
#include <iostream>
#include <filesystem>

namespace {
uint8_t encode_coverage(uint8_t coverage) {
    const double linear = coverage / 255.0;
    const double encoded = linear <= 0.0031308 ? 12.92 * linear :
        1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
    return static_cast<uint8_t>(std::lround(255.0 * encoded));
}

ascii::FrameBuffer reference_cell(const ascii::GlyphCache& cache, uint32_t codepoint, int grid_size = 1) {
    ascii::FrameBuffer image(8 * grid_size, 16 * grid_size);
    const auto* bitmap = cache.get_bitmap(codepoint);
    const int origin_x = (grid_size / 2) * 8;
    const int origin_y = (grid_size / 2) * 16;
    for (int y = 0; y < 16; ++y)
        for (int x = 0; x < 8; ++x) {
            const auto value = encode_coverage(bitmap->pixels[y * 8 + x]);
            image.set_pixel(origin_x + x, origin_y + y, ascii::Color(value,value,value));
        }
    return image;
}

int orientation_references(ascii::FontLoader& font, int variant) {
    using namespace ascii;
    Pipeline::Config pc;
    pc.target_cols = pc.target_rows = 1;
    pc.scale_mode = "stretch";
    pc.contours_enabled = false;
    pc.multi_scale = variant != 1;
    pc.blur_sigma = variant == 1 ? 0.1f : 1.0f;
    if (variant == 2) {
        pc.blur_sigma = 0.4f;
        pc.scale_sigma_0 = 0.5f;
        pc.scale_sigma_1 = 1.8f;
    }
    Pipeline pipeline(pc);
    GlyphCache cache;
    if (!cache.initialize(&font, CharSet::get_set("basic"), 8, 16, pipeline.edge_config())) return 1;
    CharSelector selector;
    selector.set_cache(&cache);
    int failures = 0;
    for (auto codepoint : cache.get_by_brightness()) {
        const auto result = pipeline.process(reference_cell(cache, codepoint));
        const auto selected = selector.select_unified(result.cell_stats[0], 0).codepoint;
        if (selected != codepoint) {
            std::cerr << "Exact glyph reference selected " << selected << " instead of " << codepoint << '\n';
            ++failures;
        }
    }
    pc.target_cols = pc.target_rows = 3;
    pipeline.set_config(pc);
    for (uint32_t codepoint : {'-', '|', '/', '\\'}) {
        const auto result = pipeline.process(reference_cell(cache, codepoint, 3));
        const auto selected = selector.select_unified(result.cell_stats[4], 0).codepoint;
        if (selected != codepoint) {
            std::cerr << "Line direction in multicell reference selected " << selected << " instead of " << codepoint << '\n';
            ++failures;
        }
    }
    return failures;
}
}

int main() {
    using namespace ascii;
    FontLoader font;
    if (!font.load_system_fallback(16).success()) return 2;
    GlyphCache cache;
    if (!cache.initialize(&font, CharSet::get_set("traditional"), 8, 16)) return 2;
    CharSelector::Config config;
    config.use_unified_loss = false;
    config.use_simple_orientation = true;
    config.use_orientation_matching = false;
    CharSelector selector(config);
    selector.set_cache(&cache);
    TemporalSmoother smoother;
    smoother.initialize(1, 1);
    int failures = 0;
    for (bool edge : {false, true}) {
        CellStats stats;
        stats.mean_luminance = 0.5f;
        stats.is_edge_cell = edge;
        stats.cell_orientation = 0.0f;
        const auto chosen = selector.select(stats, smoother, 0);
        const float retained_loss = selector.compute_loss_for_glyph(stats, chosen.codepoint);
        if (std::abs(chosen.loss - retained_loss) > 1e-6f) {
            std::cerr << "Selected and retained glyph use different losses in simple mode\n";
            ++failures;
        }
    }
    GlyphCache no_edges;
    if (!no_edges.initialize(&font, {' ', 0x2588}, 8, 16)) return 2;
    config.use_simple_orientation = false;
    config.use_orientation_matching = true;
    selector.set_config(config);
    selector.set_cache(&no_edges);
    for (float luminance : {0.1f, 0.9f}) {
        CellStats stats;
        stats.is_edge_cell = true;
        stats.mean_luminance = luminance;
        const auto chosen = selector.select(stats, smoother, 0);
        const auto expected = luminance < 0.5f ? ' ' : 0x2588;
        if (chosen.codepoint != expected ||
            std::abs(chosen.loss - 0.1f) > 1e-6f ||
            std::abs(chosen.loss - selector.compute_loss_for_glyph(stats, chosen.codepoint)) > 1e-6f) {
            std::cerr << "A cache without edge glyphs must select and score the actual cell luminance\n";
            ++failures;
        }
    }
    FontLoader unloaded;
    GlyphCache invalid_cache;
    if (invalid_cache.initialize(&unloaded, {'A'}, 8, 16)) {
        std::cerr << "Glyph cache accepted an unloaded font\n";
        ++failures;
    }
    GlyphCache limited;
    if (!limited.initialize(&font, {' ', 0x10FFFF}, 8, 16) || limited.get_stats(0x10FFFF)) {
        std::cerr << "Unavailable codepoint became a selectable replacement-box glyph\n";
        ++failures;
    }
    const std::filesystem::path font_path(FontLoader::find_system_monospace_font());
    const auto parent_relative = font_path.parent_path() / ".." /
        font_path.parent_path().filename() / font_path.filename();
    if (!font.load(parent_relative.string(), 16).success()) {
        std::cerr << "Explicit parent-relative font path was rejected\n";
        ++failures;
    }
    for (int variant = 0; variant < 3; ++variant) failures += orientation_references(font, variant);
    std::cout << "Selector regression failures: " << failures << '\n';
    return failures ? 1 : 0;
}
