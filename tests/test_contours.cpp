#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../src/cli/args.hpp"
#include "../src/core/config.hpp"
#include "../src/core/contour_extractor.hpp"
#include "../src/core/frame_composer.hpp"

using namespace ascii;

#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    try { \
        test_##name(); \
        std::cout << "PASSED\n"; \
    } catch (...) { \
        std::cout << "FAILED\n"; \
        failures++; \
    } \
} while (0)

int failures = 0;

static FloatImage make_step(int w, int h, const std::string& mode) {
    FloatImage img(w, h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = 0.0f;
            if (mode == "vertical") {
                v = x >= w / 2 ? 1.0f : 0.0f;
            } else if (mode == "horizontal") {
                v = y >= h / 2 ? 1.0f : 0.0f;
            } else if (mode == "rising") {
                v = y <= (h - 1 - x) ? 1.0f : 0.0f;
            } else if (mode == "falling") {
                v = y >= x ? 1.0f : 0.0f;
            } else if (mode == "cross") {
                v = (std::abs(x - w / 2) <= 1 || std::abs(y - h / 2) <= 1) ? 1.0f : 0.0f;
            }
            img.set(x, y, v);
        }
    }
    return img;
}

static CellStats extract_one_cell(const FloatImage& img) {
    ContourExtractor::Config cfg;
    cfg.min_occupancy = 0.02f;
    cfg.min_pixels = 2;
    ContourExtractor extractor(cfg);
    std::vector<CellStats> cells(1);
    EdgeData no_gate;
    extractor.apply(img, no_gate, img.width(), img.height(), 1, 1, cells);
    return cells[0];
}

TEST(vertical_split_produces_pipe) {
    CellStats stats = extract_one_cell(make_step(32, 32, "vertical"));
    assert(stats.has_contour);
    assert(stats.contour_codepoint == '|');
}

TEST(horizontal_split_produces_dash) {
    CellStats stats = extract_one_cell(make_step(32, 32, "horizontal"));
    assert(stats.has_contour);
    assert(stats.contour_codepoint == '-');
}

TEST(rising_diagonal_produces_slash) {
    CellStats stats = extract_one_cell(make_step(32, 32, "rising"));
    assert(stats.has_contour);
    assert(stats.contour_codepoint == '/');
}

TEST(falling_diagonal_produces_backslash) {
    CellStats stats = extract_one_cell(make_step(32, 32, "falling"));
    assert(stats.has_contour);
    assert(stats.contour_codepoint == '\\');
}

TEST(cross_produces_plus) {
    CellStats stats = extract_one_cell(make_step(32, 32, "cross"));
    assert(stats.has_contour);
    assert(stats.contour_codepoint == '+');
}

TEST(inverted_polarity_preserves_line_orientation) {
    for (const std::string mode : {"vertical", "horizontal", "rising", "falling"}) {
        FloatImage normal = make_step(32, 32, mode);
        FloatImage inverted(32, 32, 0.0f);
        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                inverted.set(x, y, 1.0f - normal.get(x, y));
            }
        }
        const CellStats a = extract_one_cell(normal);
        const CellStats b = extract_one_cell(inverted);
        assert(a.has_contour && b.has_contour);
        assert(a.contour_codepoint == b.contour_codepoint);
    }
}

TEST(low_occupancy_edge_produces_no_contour) {
    FloatImage img(32, 32, 0.0f);
    img.set(16, 16, 1.0f);
    ContourExtractor::Config cfg;
    cfg.min_occupancy = 0.20f;
    cfg.min_pixels = 8;
    ContourExtractor extractor(cfg);
    std::vector<CellStats> cells(1);
    EdgeData no_gate;
    extractor.apply(img, no_gate, 32, 32, 1, 1, cells);
    assert(!cells[0].has_contour);
}

TEST(tangent_conversion_keeps_vertical_split_unrotated) {
    CellStats stats = extract_one_cell(make_step(32, 32, "vertical"));
    assert(stats.has_contour);
    assert(stats.contour_codepoint != '-');
    assert(stats.contour_codepoint == '|');
}

TEST(composer_contour_overrides_selection) {
    Pipeline::Result result;
    result.grid_cols = 1;
    result.grid_rows = 1;
    result.luminance = FloatImage(8, 8, 0.5f);
    result.color_buffer = FrameBuffer(8, 8, Color(255, 255, 255));
    result.cell_stats.resize(1);
    result.cell_stats[0].mean_luminance = 0.5f;
    result.cell_stats[0].mean_r = 1.0f;
    result.cell_stats[0].mean_g = 1.0f;
    result.cell_stats[0].mean_b = 1.0f;
    result.cell_stats[0].has_contour = true;
    result.cell_stats[0].contour_codepoint = '|';
    result.cell_stats[0].contour_strength = 0.5f;

    Config config = Config::defaults();
    TemporalSmoother smoother;
    CharSelector selector;
    ColorMapper color_mapper(ColorMode::Truecolor);
    BilateralGrid bilateral_grid;
    MotionEstimator motion;
    BlockRenderer block_renderer;
    FrameComposer composer;
    FrameComposer::Context ctx{
        smoother, selector, color_mapper, bilateral_grid, motion, block_renderer,
        config, ColorMode::Truecolor, config.edge.high_threshold
    };

    auto output = composer.compose(result, ctx);
    assert(output.cells.size() == 1);
    assert(output.cells[0].codepoint == '|');
}

TEST(composer_contour_override_can_be_disabled) {
    Pipeline::Result result;
    result.grid_cols = 1;
    result.grid_rows = 1;
    result.luminance = FloatImage(8, 8, 0.5f);
    result.color_buffer = FrameBuffer(8, 8, Color(255, 255, 255));
    result.cell_stats.resize(1);
    result.cell_stats[0].mean_luminance = 0.5f;
    result.cell_stats[0].mean_r = 1.0f;
    result.cell_stats[0].mean_g = 1.0f;
    result.cell_stats[0].mean_b = 1.0f;
    result.cell_stats[0].has_contour = true;
    result.cell_stats[0].contour_codepoint = '|';
    result.cell_stats[0].contour_strength = 0.5f;

    Config config = Config::defaults();
    config.edge.contours_enabled = false;
    TemporalSmoother smoother;
    CharSelector selector;
    ColorMapper color_mapper(ColorMode::Truecolor);
    BilateralGrid bilateral_grid;
    MotionEstimator motion;
    BlockRenderer block_renderer;
    FrameComposer composer;
    FrameComposer::Context ctx{
        smoother, selector, color_mapper, bilateral_grid, motion, block_renderer,
        config, ColorMode::Truecolor, config.edge.high_threshold
    };

    auto output = composer.compose(result, ctx);
    assert(output.cells.size() == 1);
    assert(output.cells[0].codepoint == ' ');
}

TEST(cli_contour_flags) {
    const char* argv_no_contours[] = {"ascii-engine", "--no-contours", "input.png"};
    Args no_contours = parse_args(3, const_cast<char**>(argv_no_contours));
    assert(!no_contours.contours_enabled);

    const char* argv_thresh[] = {"ascii-engine", "--contour-thresh", "0.2", "input.png"};
    Args thresh = parse_args(4, const_cast<char**>(argv_thresh));
    assert(std::abs(thresh.contour_threshold - 0.2f) < 0.001f);

    const char* argv_invalid[] = {"ascii-engine", "--contour-thresh", "2.0", "input.png"};
    Args invalid = parse_args(4, const_cast<char**>(argv_invalid));
    assert(invalid.contour_threshold < 0.0f);
}

TEST(config_cli_contour_overrides) {
    Config config = Config::defaults();
    const char* argv[] = {"ascii-engine", "--no-contours", "--contour-thresh", "0.25", "input.png"};
    Args args = parse_args(5, const_cast<char**>(argv));
    config = apply_cli_overrides(config, args);
    assert(!config.edge.contours_enabled);
    assert(std::abs(config.edge.contour_min_occupancy - 0.25f) < 0.001f);
}

int main() {
    RUN_TEST(vertical_split_produces_pipe);
    RUN_TEST(horizontal_split_produces_dash);
    RUN_TEST(rising_diagonal_produces_slash);
    RUN_TEST(falling_diagonal_produces_backslash);
    RUN_TEST(cross_produces_plus);
    RUN_TEST(inverted_polarity_preserves_line_orientation);
    RUN_TEST(low_occupancy_edge_produces_no_contour);
    RUN_TEST(tangent_conversion_keeps_vertical_split_unrotated);
    RUN_TEST(composer_contour_overrides_selection);
    RUN_TEST(composer_contour_override_can_be_disabled);
    RUN_TEST(cli_contour_flags);
    RUN_TEST(config_cli_contour_overrides);

    if (failures > 0) {
        std::cout << failures << " contour tests failed\n";
        return 1;
    }
    std::cout << "All contour tests passed\n";
    return 0;
}
