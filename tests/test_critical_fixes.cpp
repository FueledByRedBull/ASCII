#include <iostream>
#include <cassert>
#include <fstream>
#include <algorithm>
#include <filesystem>
#include "../src/core/temporal.hpp"
#include "../src/core/edge_detector.hpp"
#include "../src/core/types.hpp"
#include "../src/glyph/font_loader.hpp"
#include "../src/glyph/glyph_cache.hpp"
#include "../src/glyph/char_sets.hpp"
#include "../src/mapping/char_selector.hpp"
#include "../src/cli/args.hpp"
#include "../src/core/config.hpp"
#include "../src/render/dither.hpp"
#include "../src/core/frame_source.hpp"
#include "../src/core/motion.hpp"
#include "../src/core/pipeline.hpp"
#include "../src/core/pipeline_runtime_cache.hpp"
#include "../src/mapping/bilateral_grid.hpp"
#include "../src/render/bitmap_renderer.hpp"

using namespace ascii;

Args parse(std::initializer_list<const char*> values) {
    std::vector<std::string> storage;
    std::vector<char*> argv;
    storage.reserve(values.size());
    argv.reserve(values.size());
    for (const char* value : values) storage.emplace_back(value);
    for (auto& value : storage) argv.push_back(value.data());
    return parse_args(static_cast<int>(argv.size()), argv.data());
}

void test_cli_validation_and_precedence() {
    std::cout << "Testing CLI validation and config precedence...\n";

    assert(!parse({"ascii-engine", "--fps"}).valid);
    assert(!parse({"ascii-engine", "--fps", "abc"}).valid);
    assert(!parse({"ascii-engine", "--color", "invalid"}).valid);
    assert(!parse({"ascii-engine", "--unknown"}).valid);

    Config configured = Config::defaults();
    configured.selector.char_set = "line-art";
    configured.grid.scale_mode = "fill";
    configured.edge.use_hysteresis = false;
    configured.selector.use_orientation_matching = false;
    configured.no_audio = true;
    configured.debug.profile_live = true;
    configured.debug.strict_memory = true;

    Args absent = parse({"ascii-engine", "input.jpg"});
    assert(absent.valid);
    Config preserved = apply_cli_overrides(configured, absent);
    assert(preserved.selector.char_set == "line-art");
    assert(preserved.grid.scale_mode == "fill");
    assert(!preserved.edge.use_hysteresis);
    assert(!preserved.selector.use_orientation_matching);
    assert(preserved.no_audio);
    assert(preserved.debug.profile_live);
    assert(preserved.debug.strict_memory);

    Args explicit_args = parse({"ascii-engine", "input.jpg", "--char-set", "blocks", "--scale", "stretch", "--no-hysteresis", "--no-orientation", "--fps", "24"});
    assert(explicit_args.valid);
    Config overridden = apply_cli_overrides(Config::defaults(), explicit_args);
    assert(overridden.selector.char_set == "blocks");
    assert(overridden.grid.scale_mode == "stretch");
    assert(!overridden.edge.use_hysteresis);
    assert(!overridden.selector.use_orientation_matching);
    assert(overridden.fps == 24);

    const auto config_path = std::filesystem::temp_directory_path() / "ascii_engine_precedence.toml";
    {
        std::ofstream file(config_path);
        file << "profile = \"anime\"\n"
                "no_audio = true\n"
                "[grid]\nscale_mode = \"fill\"\n"
                "[edge]\nuse_hysteresis = false\n"
                "[selector]\nchar_set = \"line-art\"\n"
                "weight_brightness = 0.45\nuse_orientation_matching = false\n"
                "[debug]\nprofile_live = true\nstrict_memory = true\n";
    }
    const auto loaded = Config::load(config_path.string());
    assert(loaded);
    Config from_file = *loaded;
    from_file = apply_cli_overrides(from_file, absent);
    assert(from_file.selector.char_set == "line-art");
    assert(from_file.grid.scale_mode == "fill");
    assert(!from_file.edge.use_hysteresis);
    assert(!from_file.selector.use_orientation_matching);
    assert(std::abs(from_file.selector.weight_brightness - 0.45f) < 1e-6f);
    assert(from_file.no_audio);
    assert(from_file.debug.profile_live);
    assert(from_file.debug.strict_memory);
    std::filesystem::remove(config_path);

    std::cout << "[OK] CLI validation and config precedence test passed\n";
}

void test_temporal_initialization() {
    std::cout << "Testing temporal smoothing initialization...\n";
    
    TemporalSmoother smoother;
    smoother.initialize(2, 2);
    
    // Test that both smoothing methods set initialized flag properly
    float lum1 = smoother.smooth_luminance(0, 0.5f);
    assert(smoother.frame_state()[0].initialized);
    assert(lum1 == 0.5f);
    float edge_after_lum = smoother.smooth_edge_strength(0, 0.3f);
    float coherence_after_lum = smoother.smooth_coherence(0, 0.7f);
    assert(edge_after_lum == 0.3f);
    assert(coherence_after_lum == 0.7f);
    
    // Reset for edge test
    smoother.reset();
    float edge1 = smoother.smooth_edge_strength(0, 0.3f);
    assert(smoother.frame_state()[0].initialized);
    assert(edge1 == 0.3f);
    
    std::cout << "[OK] Temporal initialization test passed\n";
}

void test_current_frame_hysteresis_and_orientation_controls() {
    std::cout << "Testing current-frame hysteresis and orientation controls...\n";

    TemporalSmoother::Config temporal_config;
    temporal_config.hysteresis_margin = 0.1f;
    TemporalSmoother smoother(temporal_config);
    smoother.initialize(1, 1);
    smoother.update_glyph_with_loss(0, 'A', 0.9f, 0.1f);
    smoother.begin_frame();
    assert(smoother.should_change_glyph_with_loss(0, 'B', 0.2f, 0.8f, 0.05f));
    assert(!smoother.should_change_glyph_with_loss(0, 'B', 0.7f, 0.5f, 0.05f));

    CharSelector::Config selector_config;
    selector_config.loss_weights.brightness = 0.0f;
    selector_config.loss_weights.orientation = 1.0f;
    selector_config.loss_weights.contrast = 0.0f;
    selector_config.loss_weights.frequency = 0.0f;
    selector_config.loss_weights.texture = 0.0f;
    selector_config.enable_frequency_matching = false;
    selector_config.enable_texture_matching = false;
    CellStats cell_stats;
    cell_stats.orientation_histogram[0] = 1.0f;
    GlyphStats glyph_stats;
    glyph_stats.orientation_hist.assign(8, 0.0f);
    glyph_stats.orientation_hist[2] = 1.0f;
    CharSelector enabled(selector_config);
    assert(enabled.compute_loss(cell_stats, glyph_stats) > 0.9f);
    selector_config.use_orientation_matching = false;
    CharSelector disabled(selector_config);
    assert(disabled.compute_loss(cell_stats, glyph_stats) == 0.0f);

    FontLoader loader;
    assert(loader.load_system_fallback(16.0f).success());
    GlyphCache cache;
    assert(cache.initialize(&loader, {' ', '-', '|', '/', '\\'}, 8, 16));
    const auto* space = cache.get_bitmap(' ');
    assert(space && space->width == 8 && space->height == 16);
    assert(std::all_of(space->pixels.begin(), space->pixels.end(), [](uint8_t v) { return v == 0; }));
    assert(cache.get_bitmap(0x2580));
    assert(cache.get_bitmap(0x2599));

    selector_config.use_simple_orientation = true;
    selector_config.use_orientation_matching = false;
    CharSelector simple(selector_config);
    simple.set_cache(&cache);
    assert(simple.select_edge_simple(0.0f).codepoint == static_cast<uint32_t>('|'));
    constexpr float kPi = 3.14159265358979323846f;
    assert(simple.select_edge_simple(0.5f * kPi).codepoint == static_cast<uint32_t>('-'));
    assert(simple.select_edge_simple(kPi).codepoint == static_cast<uint32_t>('|'));

    std::cout << "[OK] Current-frame hysteresis and orientation controls test passed\n";
}

void test_dither_error_propagation() {
    DitherBuffer buffer(3, 2);
    buffer.distribute_error_serpentine(1, 0, true, 0.16f, 0.0f, 0.0f);
    assert(std::abs(buffer.get_error_r(2, 0) - 0.07f) < 1e-6f);
    assert(std::abs(buffer.get_error_r(0, 1) - 0.03f) < 1e-6f);
    assert(std::abs(buffer.get_error_r(1, 1) - 0.05f) < 1e-6f);
    assert(std::abs(buffer.get_error_r(2, 1) - 0.01f) < 1e-6f);
}

class CountingSource final : public FrameSource {
public:
    bool open(const std::string&) override { return true; }
    FrameReadStatus read_next(FrameBuffer& out) override {
        ++read_count;
        out = FrameBuffer(1, 1);
        return FrameReadStatus::Frame;
    }
    double fps() const override { return 30.0; }
    Size frame_size() const override { return {1, 1}; }
    bool is_open() const override { return true; }
    void reset() override {}
    int read_count = 0;
};

void test_pause_does_not_advance_source() {
    CountingSource source;
    FrameBuffer frame;
    assert(read_frame_if_ready(source, true, frame) == FrameReadStatus::Paused);
    assert(source.read_count == 0);
    assert(read_frame_if_ready(source, false, frame) == FrameReadStatus::Frame);
    assert(source.read_count == 1);
}

void test_motion_confidence_decay_affects_reuse() {
    FloatImage previous(48, 48, 0.0f);
    FloatImage current(48, 48, 0.0f);
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x) {
            previous.set(x, y, ((x / 3 + y / 3) % 2) ? 1.0f : 0.0f);
        }
    }
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x) {
            current.set(x, y, previous.get_clamped(x - 2, y));
        }
    }

    MotionEstimator::Config config;
    config.solve_divisor = 1;
    config.motion_cap = 4.0f;
    config.use_phase_correlation = false;
    config.still_scene_threshold = 0.0f;
    config.reuse_scene_threshold = 1.0f;
    config.max_reuse_frames = 2;
    config.reuse_confidence_decay = 0.5f;
    MotionEstimator estimator(config);
    estimator.compute_flow(previous, current);

    float before = 0.0f;
    int best_x = 0;
    int best_y = 0;
    for (int y = 0; y < 48; ++y) {
        for (int x = 0; x < 48; ++x) {
            if (estimator.get_motion(x, y).confidence > before) {
                before = estimator.get_motion(x, y).confidence;
                best_x = x;
                best_y = y;
            }
        }
    }
    assert(before > 0.01f);
    estimator.compute_flow(previous, current);
    const float after = estimator.get_motion(best_x, best_y).confidence;
    assert(std::abs(after - before * 0.5f) < 1e-5f);
}

int edge_count(const EdgeData& edges) {
    return static_cast<int>(std::count(edges.edge_mask.begin(), edges.edge_mask.end(), true));
}

void test_edge_controls_are_effective() {
    FloatImage image(64, 64, 0.0f);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const float contrast = x < 32 ? 1.0f : 0.18f;
            image.set(x, y, ((x / 4 + y / 4) % 2) ? contrast : 0.0f);
        }
    }

    EdgeDetector::Config config;
    config.multi_scale = true;
    config.adaptive_mode = "local";
    config.high_threshold = 0.02f;
    config.low_threshold = 0.01f;
    config.blur_sigma = 0.1f;
    EdgeDetector detector(config);
    const EdgeData sharp = detector.detect(image);
    config.blur_sigma = 3.0f;
    detector.set_config(config);
    const EdgeData blurred = detector.detect(image);
    double sharp_sum = 0.0;
    double blurred_sum = 0.0;
    for (size_t i = 0; i < sharp.magnitude.size_in_elements(); ++i) {
        sharp_sum += sharp.magnitude.data()[i];
        blurred_sum += blurred.magnitude.data()[i];
    }
    assert(std::abs(sharp_sum - blurred_sum) > 1.0);

    config.blur_sigma = 0.1f;
    config.scale_variance_floor = 0.0f;
    config.scale_variance_ceil = 0.001f;
    detector.set_config(config);
    const EdgeData fine_scale = detector.detect(image);
    config.scale_variance_floor = 1.0f;
    config.scale_variance_ceil = 2.0f;
    detector.set_config(config);
    const EdgeData coarse_scale = detector.detect(image);
    double scale_difference = 0.0;
    for (size_t i = 0; i < fine_scale.magnitude.size_in_elements(); ++i) {
        scale_difference += std::abs(fine_scale.magnitude.data()[i] - coarse_scale.magnitude.data()[i]);
    }
    assert(scale_difference > 0.1);

    config.scale_variance_floor = 0.0005f;
    config.scale_variance_ceil = 0.02f;
    config.adaptive_mode = "local";
    config.high_threshold = 0.01f;
    detector.set_config(config);
    const int low_threshold_count = edge_count(detector.detect(image));
    config.high_threshold = 0.8f;
    detector.set_config(config);
    const int high_threshold_count = edge_count(detector.detect(image));
    assert(low_threshold_count > high_threshold_count);

    config.high_threshold = 0.001f;
    config.global_percentile = 0.9f;
    config.tile_size = 16;
    config.use_hysteresis = false;
    config.adaptive_mode = "local";
    detector.set_config(config);
    const int local_count = edge_count(detector.detect(image));
    config.adaptive_mode = "hybrid";
    detector.set_config(config);
    const int hybrid_count = edge_count(detector.detect(image));
    assert(local_count != hybrid_count);
}

void test_runtime_cache_local_motion_and_area_downsample() {
    FrameBuffer first(64, 64, Color(0, 0, 0));
    PipelineRuntimeCache cache;
    PipelineRuntimeCache::Query query;
    query.reuse_limit = 4;
    query.still_threshold = 1.0f;
    assert(!cache.begin_frame(first, query).reuse_pipeline_result);
    Pipeline::Result cached_result;
    cached_result.cell_stats.resize(1);
    cache.commit_processed_result(cached_result, true, false);
    assert(cache.begin_frame(first, query).reuse_pipeline_result);

    FrameBuffer localized = first;
    localized.set_pixel(63, 63, Color(255, 255, 255));
    const auto changed = cache.begin_frame(localized, query);
    assert(!changed.reuse_pipeline_result);
    assert(!changed.reuse_cell_stats);

    FrameBuffer checker(64, 64);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const uint8_t value = ((x + y) % 2) ? 255 : 0;
            checker.set_pixel(x, y, Color(value, value, value));
        }
    }
    Pipeline::Config pipeline_config;
    pipeline_config.target_cols = 1;
    pipeline_config.target_rows = 1;
    pipeline_config.cell_width = 8;
    pipeline_config.cell_height = 8;
    pipeline_config.scale_mode = "stretch";
    pipeline_config.multi_scale = false;
    Pipeline pipeline(pipeline_config);
    const auto result = pipeline.process(checker);
    for (size_t i = 0; i < result.luminance.size_in_elements(); ++i) {
        assert(std::abs(result.luminance.data()[i] - 0.5f) < 0.01f);
    }
}

void test_bilateral_spatial_resolution_and_checked_allocations() {
    std::vector<CellStats> cells(80 * 20);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 80; ++x) {
            auto& cell = cells[y * 80 + x];
            cell.mean_luminance = static_cast<float>(x) / 79.0f;
            cell.mean_r = cell.mean_luminance;
        }
    }
    BilateralGrid::Config config;
    config.enabled = true;
    config.spatial_bins = 8;
    BilateralGrid coarse(config);
    coarse.build(cells, 80, 20);
    assert(coarse.valid());
    assert(coarse.grid_cols() == 8);
    assert(coarse.grid_rows() == 2);
    config.spatial_bins = 40;
    BilateralGrid fine(config);
    fine.build(cells, 80, 20);
    assert(fine.grid_cols() == 40);
    assert(fine.grid_rows() == 10);

    bool threw = false;
    try {
        FrameBuffer invalid(-1, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
}

float temporal_step_after_one_second(float fps) {
    TemporalSmoother::Config config;
    config.alpha = 0.3f;
    config.frame_rate = fps;
    config.use_wavelet_flicker = false;
    TemporalSmoother smoother(config);
    smoother.initialize(1, 1);
    smoother.smooth_luminance(0, 0.0f);
    float value = 0.0f;
    for (int i = 0; i < static_cast<int>(fps); ++i) {
        value = smoother.smooth_luminance(0, 1.0f);
    }
    return value;
}

void test_frame_rate_invariant_temporal_and_motion_reference() {
    const float at_15 = temporal_step_after_one_second(15.0f);
    const float at_24 = temporal_step_after_one_second(24.0f);
    const float at_30 = temporal_step_after_one_second(30.0f);
    const float at_60 = temporal_step_after_one_second(60.0f);
    assert(std::abs(at_15 - at_24) < 1e-5f);
    assert(std::abs(at_24 - at_30) < 1e-5f);
    assert(std::abs(at_15 - at_30) < 1e-5f);
    assert(std::abs(at_30 - at_60) < 1e-5f);

    TemporalSmoother smoother;
    smoother.initialize(2, 1);
    smoother.update_glyph_with_loss(0, 'A', 1.0f, 0.0f);
    smoother.update_glyph_with_loss(1, 'B', 1.0f, 0.0f);
    smoother.begin_frame();
    smoother.set_motion_offset(1, 1.0f, 0.0f);
    smoother.update_glyph_with_loss(0, 'C', 1.0f, 0.0f);
    assert(smoother.reference_glyph(1) == static_cast<uint32_t>('A'));
}

void test_exhaustive_glyph_selection_scene_cut_and_render_tolerance() {
    FontLoader loader;
    assert(loader.load_system_fallback(16.0f).success());
    GlyphCache cache;
    const auto codepoints = CharSet::get_set("basic");
    assert(cache.initialize(&loader, codepoints, 8, 16));

    CharSelector::Config selector_config;
    selector_config.transition_penalty = 0.0f;
    CharSelector selector(selector_config);
    selector.set_cache(&cache);
    for (int step = 0; step <= 10; ++step) {
        CellStats stats;
        stats.mean_luminance = step / 10.0f;
        stats.adaptive_level = 1;
        const auto selected = selector.select_unified(stats, 0);
        float exhaustive_loss = std::numeric_limits<float>::infinity();
        uint32_t exhaustive_glyph = 0;
        for (uint32_t glyph : cache.get_by_brightness()) {
            const auto* glyph_stats = cache.get_stats(glyph);
            if (!glyph_stats) continue;
            const float loss = selector.compute_loss(stats, *glyph_stats);
            if (loss < exhaustive_loss) {
                exhaustive_loss = loss;
                exhaustive_glyph = glyph;
            }
        }
        assert(selected.codepoint == exhaustive_glyph);
        assert(std::abs(selected.loss - exhaustive_loss) < 1e-6f);
    }

    TemporalSmoother smoother;
    smoother.initialize(1, 1);
    smoother.update_glyph_with_loss(0, ' ', 1.0f, 0.0f);
    smoother.begin_frame();
    assert(smoother.should_change_glyph_with_loss(0, '#', 0.0f, 1.0f, 0.0f));
    smoother.update_glyph_with_loss(0, '#', 1.0f, 0.0f);
    assert(smoother.frame_state()[0].last_glyph == static_cast<uint32_t>('#'));

    BitmapRenderer renderer;
    renderer.set_cache(&cache);
    renderer.set_cell_size(8, 16);
    ASCIICell blank;
    blank.codepoint = ' ';
    blank.fg_r = blank.fg_g = blank.fg_b = 255;
    const FrameBuffer blank_image = renderer.render({blank}, 1, 1);
    for (int y = 0; y < blank_image.height(); ++y) {
        for (int x = 0; x < blank_image.width(); ++x) {
            const Color pixel = blank_image.get_pixel(x, y);
            assert(pixel.r == 0 && pixel.g == 0 && pixel.b == 0);
        }
    }

    ASCIICell block;
    block.codepoint = 0x2588;
    block.fg_r = 255;
    block.fg_g = 0;
    block.fg_b = 0;
    const FrameBuffer block_image = renderer.render({block}, 1, 1);
    int red_pixels = 0;
    for (int y = 0; y < block_image.height(); ++y) {
        for (int x = 0; x < block_image.width(); ++x) {
            const Color pixel = block_image.get_pixel(x, y);
            if (pixel.r > 180 && pixel.g < 40 && pixel.b < 40) ++red_pixels;
        }
    }
    assert(red_pixels >= block_image.width() * block_image.height() / 2);
}

void test_edge_bounds_checking() {
    std::cout << "Testing edge detection bounds checking...\n";
    
    // Create a small test image
    FloatImage test_img(3, 3);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            test_img.set(x, y, 0.5f);
        }
    }
    
    // Set center to higher value
    test_img.set(1, 1, 1.0f);
    
    EdgeDetector detector;
    EdgeDetector::Config config;
    config.blur_sigma = 1.0f;
    config.low_threshold = 0.1f;
    config.high_threshold = 0.3f;
    detector.set_config(config);
    
    auto edges = detector.detect(test_img);
    
    // Should not crash and should produce valid results
    assert(!edges.magnitude.empty());
    assert(edges.magnitude.width() == 3);
    assert(edges.magnitude.height() == 3);
    
    std::cout << "[OK] Edge bounds checking test passed\n";
}

void test_font_validation() {
    std::cout << "Testing font security validation...\n";
    
    // Test font loader with invalid file
    FontLoader loader;
    auto result = loader.load("nonexistent_font.ttf");
    assert(!result.success());
    
    // Test font loader with invalid data in memory (too small)
    uint8_t small_data[] = {0x00, 0x01, 0x00, 0x00}; // TrueType signature but tiny
    auto mem_result = loader.load_from_memory(small_data, 4);
    assert(!mem_result.success());
    
    std::cout << "[OK] Font validation test passed\n";
}

int main() {
    std::cout << "Running critical fixes validation tests...\n\n";
    
    try {
        test_temporal_initialization();
        test_current_frame_hysteresis_and_orientation_controls();
        test_dither_error_propagation();
        test_pause_does_not_advance_source();
        test_motion_confidence_decay_affects_reuse();
        test_edge_controls_are_effective();
        test_runtime_cache_local_motion_and_area_downsample();
        test_bilateral_spatial_resolution_and_checked_allocations();
        test_frame_rate_invariant_temporal_and_motion_reference();
        test_exhaustive_glyph_selection_scene_cut_and_render_tolerance();
        test_cli_validation_and_precedence();
        test_edge_bounds_checking();
        test_font_validation();
        
        std::cout << "\n[OK] All critical fixes tests passed!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "[FAIL] Test failed with unknown exception\n";
        return 1;
    }
}
