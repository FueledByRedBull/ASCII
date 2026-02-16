#include <iostream>
#include <cassert>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <stdexcept>

#include "../src/core/types.hpp"
#include "../src/core/temporal.hpp"
#include "../src/core/edge_detector.hpp"
#include "../src/core/cell_stats.hpp"
#include "../src/glyph/font_loader.hpp"
#include "../src/glyph/char_sets.hpp"
#include "../src/glyph/glyph_stats.hpp"
#include "../src/cli/args.hpp"
#include "../src/terminal/terminal.hpp"
#include "../src/mapping/color_mapper.hpp"
#include "../src/mapping/char_selector.hpp"

using namespace ascii;

#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    try { \
        test_##name(); \
        std::cout << "PASSED\n"; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << "\n"; \
        failures++; \
    } catch (...) { \
        std::cout << "FAILED: unknown exception\n"; \
        failures++; \
    } \
} while(0)

int failures = 0;

TEST(temporal_bounds_checking) {
    TemporalSmoother smoother;
    smoother.initialize(2, 2);
    
    float val = smoother.smooth_luminance(0, 0.5f);
    assert(std::abs(val - 0.5f) < 0.001f);
    
    bool threw = false;
    try {
        smoother.smooth_luminance(100, 0.5f);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw && "Should throw on out of bounds index");
    
    threw = false;
    try {
        smoother.smooth_edge_strength(-1, 0.3f);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    assert(threw && "Should throw on negative index");
}

TEST(temporal_smoothing_formula) {
    TemporalSmoother smoother;
    TemporalSmoother::Config cfg;
    cfg.alpha = 0.5f;
    smoother.set_config(cfg);
    smoother.initialize(1, 1);
    
    float v1 = smoother.smooth_luminance(0, 1.0f);
    assert(std::abs(v1 - 1.0f) < 0.001f);
    
    float v2 = smoother.smooth_luminance(0, 0.0f);
    assert(std::abs(v2 - 0.5f) < 0.001f);
    
    float v3 = smoother.smooth_luminance(0, 0.0f);
    assert(std::abs(v3 - 0.25f) < 0.001f);
}

TEST(temporal_hysteresis) {
    TemporalSmoother smoother;
    TemporalSmoother::Config cfg;
    cfg.hysteresis_margin = 0.2f;
    cfg.enable_hysteresis = true;
    smoother.set_config(cfg);
    smoother.initialize(1, 1);
    
    smoother.update_glyph(0, 65, 0.5f);
    
    assert(!smoother.should_change_glyph(0, 65, 0.6f));
    assert(!smoother.should_change_glyph(0, 66, 0.5f));
    assert(!smoother.should_change_glyph(0, 66, 0.6f));
    assert(smoother.should_change_glyph(0, 66, 0.8f));
}

TEST(edge_detector_bounds) {
    FloatImage small(3, 3);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            small.set(x, y, 0.5f);
        }
    }
    small.set(1, 1, 1.0f);
    
    EdgeDetector detector;
    EdgeDetector::Config cfg;
    cfg.blur_sigma = 0.5f;
    cfg.low_threshold = 0.05f;
    cfg.high_threshold = 0.2f;
    detector.set_config(cfg);
    
    auto edges = detector.detect(small);
    
    assert(!edges.magnitude.empty());
    assert(edges.magnitude.width() == 3);
    assert(edges.magnitude.height() == 3);
}

TEST(edge_detector_gradient) {
    FloatImage img(10, 10);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            img.set(x, y, static_cast<float>(x) / 10.0f);
        }
    }
    
    EdgeDetector detector;
    auto grad = detector.compute_gradients(img);
    
    assert(!grad.gx.empty());
    assert(!grad.gy.empty());
    
    float gx_center = grad.gx.get(5, 5);
    assert(gx_center > 0);
}

TEST(cell_stats_computation) {
    FloatImage lum(16, 16);
    EdgeData edges;
    edges.magnitude = FloatImage(16, 16, 0.1f);
    edges.edge_mask.resize(16 * 16, false);
    edges.edge_mask[8 * 16 + 8] = true;
    
    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            lum.set(x, y, static_cast<float>(y) / 16.0f);
        }
    }
    
    CellStatsAggregator agg;
    CellStatsAggregator::Config cfg;
    cfg.cell_width = 8;
    cfg.cell_height = 16;
    agg.set_config(cfg);
    
    auto stats = agg.compute(lum, edges);
    
    assert(stats.size() == 2);
    assert(agg.grid_cols(16) == 2);
    assert(agg.grid_rows(16) == 1);
}

TEST(font_loader_path_traversal) {
    FontLoader loader;
    
    auto result = loader.load("../../../etc/passwd");
    assert(!result.success());
    
    result = loader.load("..\\..\\..\\windows\\system32");
    assert(!result.success());
    
    result = loader.load("/etc/passwd\x00.ttf");
    assert(!result.success());
}

TEST(font_loader_invalid_data) {
    FontLoader loader;
    
    uint8_t tiny_data[] = {0x00, 0x01, 0x00, 0x00};
    auto result = loader.load_from_memory(tiny_data, 4);
    assert(!result.success());
    
    uint8_t invalid_sig[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 
                              0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    result = loader.load_from_memory(invalid_sig, 13);
    assert(!result.success());
}

TEST(utf8_validation_valid) {
    std::string basic = "Hello World!";
    auto cps = CharSet::to_codepoints(basic);
    assert(cps.size() == 12);
    assert(cps[0] == 'H');
    assert(cps[11] == '!');
    
    std::string unicode = "é";
    cps = CharSet::to_codepoints(unicode);
    assert(cps.size() == 1);
    assert(cps[0] == 0xE9);
}

TEST(utf8_validation_invalid) {
    std::string invalid = "abc\xFFxyz";
    auto cps = CharSet::to_codepoints(invalid);
    assert(cps.size() == 3 || cps.size() == 6);
    
    std::string truncated = "abc\xC2";
    cps = CharSet::to_codepoints(truncated);
    assert(cps.size() >= 3);
}

TEST(utf8_surrogate_rejection) {
    uint8_t surrogate_bytes[] = {0xED, 0xA0, 0x80};
    std::string surrogate(reinterpret_cast<char*>(surrogate_bytes), 3);
    auto cps = CharSet::to_codepoints(surrogate);
    assert(cps.empty() || cps[0] < 0xD800 || cps[0] > 0xDFFF);
}

TEST(char_sets_available) {
    auto basic = CharSet::get_set("basic");
    assert(!basic.empty());
    assert(std::find(basic.begin(), basic.end(), static_cast<uint32_t>(' ')) != basic.end());
    
    auto blocks = CharSet::get_set("blocks");
    assert(!blocks.empty());
    
    auto line_art = CharSet::get_set("line-art");
    assert(!line_art.empty());
    
    auto unknown = CharSet::get_set("unknown");
    assert(!unknown.empty());
}

TEST(color_mapper_modes) {
    ColorMapper mapper(ColorMode::None);
    auto c = mapper.map(255, 0, 0);
    assert(c.r == 128 && c.g == 128 && c.b == 128);
    
    mapper.set_mode(ColorMode::Truecolor);
    c = mapper.map(255, 128, 64);
    assert(c.r == 255 && c.g == 128 && c.b == 64);
    
    c = mapper.map_luminance(0.5f);
    assert(std::abs(c.r - 128) <= 1);
    
    c = mapper.map_rgb(1.0f, 0.0f, 0.0f);
    assert(c.r == 255 && c.g == 0 && c.b == 0);
    
    c = mapper.map_rgb(0.5f, 0.5f, 0.5f);
    assert(std::abs(c.r - 128) <= 1);
}

TEST(color_mapper_hsv) {
    HSV black = ColorMapper::rgb_to_hsv(0.0f, 0.0f, 0.0f);
    assert(black.v == 0.0f);
    
    HSV white = ColorMapper::rgb_to_hsv(1.0f, 1.0f, 1.0f);
    assert(std::abs(white.v - 1.0f) < 0.001f);
    assert(std::abs(white.s - 0.0f) < 0.001f);
    
    HSV red = ColorMapper::rgb_to_hsv(1.0f, 0.0f, 0.0f);
    assert(std::abs(red.h - 0.0f) < 0.001f || std::abs(red.h - 360.0f) < 0.001f);
    assert(std::abs(red.s - 1.0f) < 0.001f);
    
    HSV green = ColorMapper::rgb_to_hsv(0.0f, 1.0f, 0.0f);
    assert(std::abs(green.h - 120.0f) < 1.0f);
    
    HSV blue = ColorMapper::rgb_to_hsv(0.0f, 0.0f, 1.0f);
    assert(std::abs(blue.h - 240.0f) < 1.0f);
    
    uint8_t r, g, b;
    ColorMapper::hsv_to_rgb(0.0f, 1.0f, 1.0f, r, g, b);
    assert(r == 255 && g == 0 && b == 0);
    
    ColorMapper::hsv_to_rgb(120.0f, 1.0f, 1.0f, r, g, b);
    assert(r == 0 && g == 255 && b == 0);
}

TEST(terminal_rgb_to_256) {
    uint8_t idx = Terminal::rgb_to_256(0, 0, 0);
    assert(idx < 256);
    
    idx = Terminal::rgb_to_256(255, 255, 255);
    assert(idx < 256);
    
    idx = Terminal::rgb_to_256(128, 128, 128);
    assert(idx < 256);
    
    idx = Terminal::rgb_to_256(255, 0, 0);
    assert(idx < 256);
}

TEST(terminal_rgb_to_16) {
    uint8_t idx = Terminal::rgb_to_16(0, 0, 0);
    assert(idx == 0);
    
    idx = Terminal::rgb_to_16(255, 255, 255);
    assert(idx == 15);
    
    idx = Terminal::rgb_to_16(255, 0, 0);
    assert(idx == 9);
}

TEST(types_framebuffer) {
    FrameBuffer fb(10, 10);
    assert(fb.width() == 10);
    assert(fb.height() == 10);
    assert(!fb.empty());
    
    fb.set_pixel(5, 5, Color(255, 128, 64));
    Color c = fb.get_pixel(5, 5);
    assert(c.r == 255 && c.g == 128 && c.b == 64);
    
    Color oob = fb.get_pixel(100, 100);
    assert(oob.r == 0 && oob.g == 0 && oob.b == 0);
}

TEST(types_float_image) {
    FloatImage img(10, 10);
    assert(img.width() == 10);
    assert(img.height() == 10);
    
    img.set(5, 5, 0.75f);
    assert(std::abs(img.get(5, 5) - 0.75f) < 0.001f);
    
    float clamped = img.get_clamped(100, 100);
    assert(clamped >= 0.0f && clamped <= 1.0f);
    
    assert(img.get(-1, -1) == 0.0f);
}

TEST(types_color_luminance) {
    Color white(255, 255, 255);
    assert(std::abs(white.luminance() - 1.0f) < 0.001f);
    
    Color black(0, 0, 0);
    assert(std::abs(black.luminance() - 0.0f) < 0.001f);
    
    Color green(0, 255, 0);
    assert(green.luminance() > 0.5f);
    
    Color blue(0, 0, 255);
    assert(blue.luminance() < 0.2f);
}

TEST(edge_detector_nms) {
    FloatImage mag(10, 10, 0.5f);
    mag.set(5, 5, 1.0f);
    
    FloatImage orient(10, 10, 0.0f);
    
    auto nms = EdgeDetector::non_maximum_suppression(mag, orient);
    
    assert(nms.width() == 10);
    assert(nms.height() == 10);
}

TEST(edge_detector_hysteresis) {
    FloatImage mag(5, 5);
    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 5; ++x) {
            float dist = std::sqrt((x-2)*(x-2) + (y-2)*(y-2));
            mag.set(x, y, std::max(0.0f, 1.0f - dist * 0.3f));
        }
    }
    
    auto mask = EdgeDetector::hysteresis_threshold(mag, 5, 5, 0.3f, 0.7f);
    
    assert(mask.size() == 25);
}

TEST(glyph_stats_similarity) {
    GlyphStats s1;
    s1.orientation_hist = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f, 0.03f, 0.02f};
    
    GlyphStats s2;
    s2.orientation_hist = {0.1f, 0.2f, 0.3f, 0.2f, 0.1f, 0.05f, 0.03f, 0.02f};
    
    float sim = s1.orientation_similarity(s2.orientation_hist);
    assert(std::abs(sim - 1.0f) < 0.001f);
    
    s2.orientation_hist = {0.3f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
    sim = s1.orientation_similarity(s2.orientation_hist);
    assert(sim < 1.0f);
}

TEST(glyph_stats_edge_detection) {
    GlyphStats s;
    s.brightness = 0.5f;
    s.orientation_hist = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    
    assert(!s.is_good_edge_glyph());
    
    s.orientation_hist = {0.1f, 0.7f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f};
    assert(s.is_good_edge_glyph());
}

TEST(simple_orientation_mode) {
    CharSelector::Config cfg;
    cfg.use_simple_orientation = true;
    cfg.use_orientation_matching = false;
    CharSelector selector(cfg);
    
    auto sel = selector.select_edge_simple(0.0f);
    assert(sel.codepoint == static_cast<uint32_t>('-'));
    
    sel = selector.select_edge_simple(M_PI / 2.0f);
    assert(sel.codepoint == static_cast<uint32_t>('|'));
    
    sel = selector.select_edge_simple(M_PI / 4.0f);
    assert(sel.codepoint == static_cast<uint32_t>('/'));
    
    sel = selector.select_edge_simple(-M_PI / 4.0f);
    assert(sel.codepoint == static_cast<uint32_t>('\\'));
}

TEST(result_type) {
    Result ok = Result::ok();
    assert(ok.success());
    assert(!ok.failure());
    
    Result fail = Result::fail(ErrorCode::FILE_NOT_FOUND, "test error");
    assert(!fail.success());
    assert(fail.failure());
    assert(fail.message == "test error");
}

TEST(size_type) {
    Size s1{10, 20};
    Size s2{10, 20};
    Size s3{5, 30};
    
    assert(s1 == s2);
    assert(s1 != s3);
    assert(s1.area() == 200);
}

TEST(float_image_from_rgba) {
    uint8_t rgba[] = {
        255, 0, 0, 255,
        0, 255, 0, 255,
        0, 0, 255, 255,
        255, 255, 255, 255
    };
    
    auto img = FloatImage::from_rgba(rgba, 2, 2);
    
    assert(img.width() == 2);
    assert(img.height() == 2);
    
    float r_lum = img.get(0, 0);
    float g_lum = img.get(1, 0);
    float b_lum = img.get(0, 1);
    
    assert(g_lum > r_lum);
    assert(g_lum > b_lum);
}

int main() {
    std::cout << "=== ASCII Engine Comprehensive Test Suite ===\n\n";
    
    std::cout << "--- Temporal Smoother Tests ---\n";
    RUN_TEST(temporal_bounds_checking);
    RUN_TEST(temporal_smoothing_formula);
    RUN_TEST(temporal_hysteresis);
    
    std::cout << "\n--- Edge Detector Tests ---\n";
    RUN_TEST(edge_detector_bounds);
    RUN_TEST(edge_detector_gradient);
    RUN_TEST(edge_detector_nms);
    RUN_TEST(edge_detector_hysteresis);
    
    std::cout << "\n--- Cell Stats Tests ---\n";
    RUN_TEST(cell_stats_computation);
    
    std::cout << "\n--- Font Loader Tests ---\n";
    RUN_TEST(font_loader_path_traversal);
    RUN_TEST(font_loader_invalid_data);
    
    std::cout << "\n--- UTF-8 Validation Tests ---\n";
    RUN_TEST(utf8_validation_valid);
    RUN_TEST(utf8_validation_invalid);
    RUN_TEST(utf8_surrogate_rejection);
    
    std::cout << "\n--- Character Set Tests ---\n";
    RUN_TEST(char_sets_available);
    
    std::cout << "\n--- Color Mapper Tests ---\n";
    RUN_TEST(color_mapper_modes);
    RUN_TEST(color_mapper_hsv);
    
    std::cout << "\n--- Terminal Tests ---\n";
    RUN_TEST(terminal_rgb_to_256);
    RUN_TEST(terminal_rgb_to_16);
    
    std::cout << "\n--- Types Tests ---\n";
    RUN_TEST(types_framebuffer);
    RUN_TEST(types_float_image);
    RUN_TEST(types_color_luminance);
    RUN_TEST(float_image_from_rgba);
    
    std::cout << "\n--- Glyph Stats Tests ---\n";
    RUN_TEST(glyph_stats_similarity);
    RUN_TEST(glyph_stats_edge_detection);
    
    std::cout << "\n--- Simple Orientation Tests ---\n";
    RUN_TEST(simple_orientation_mode);
    
    std::cout << "=== Test Summary ===\n";
    std::cout << "Failures: " << failures << "\n";
    
    if (failures == 0) {
        std::cout << "\n✓ All tests passed!\n";
        return 0;
    } else {
        std::cout << "\n✗ Some tests failed!\n";
        return 1;
    }
}
