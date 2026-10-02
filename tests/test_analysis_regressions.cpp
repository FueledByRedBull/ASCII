#include "core/pipeline.hpp"
#include "core/motion.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <stdexcept>
#include <array>
#include <thread>
#include <barrier>

using namespace ascii;

namespace {
int failures = 0;
void expect_true(const char* name, bool pass) {
    std::cout << (pass ? "PASS " : "FAIL ") << name << '\n';
    failures += !pass;
}
void expect_near(const char* name, float actual, float expected, float tolerance) {
    const bool pass = std::isfinite(actual) && std::abs(actual - expected) <= tolerance;
    std::cout << (pass ? "PASS " : "FAIL ") << name << " actual=" << actual
              << " expected=" << expected << " tolerance=" << tolerance << '\n';
    failures += !pass;
}

Pipeline::Config pipeline_config() {
    Pipeline::Config config;
    config.target_cols = 1;
    config.target_rows = 1;
    config.cell_width = 8;
    config.cell_height = 8;
    config.scale_mode = "stretch";
    config.multi_scale = false;
    config.contours_enabled = false;
    config.enable_frequency_signature = false;
    config.enable_texture_signature = false;
    return config;
}

void color_stats() {
    FrameBuffer checker(64, 64);
    for (int y = 0; y < 64; ++y) {
        for (int x = 0; x < 64; ++x) {
            const uint8_t v = (x + y) % 2 ? 255 : 0;
            checker.set_pixel(x, y, Color(v, v, v));
        }
    }
    Pipeline pipeline(pipeline_config());
    Pipeline::ProcessOptions options;
    options.need_color_buffer = true;
    auto buffered = pipeline.process(checker, options);
    options.need_color_buffer = false;
    auto direct = pipeline.process(checker, options);
    expect_near("checker buffered linear mean", buffered.cell_stats[0].mean_r, 0.5f, 0.001f);
    expect_near("checker direct linear mean", direct.cell_stats[0].mean_r, 0.5f, 0.001f);

    auto config = pipeline_config();
    config.scale_mode = "fit";
    config.char_aspect = 1.0f;
    pipeline.set_config(config);
    const FrameBuffer white(12, 8, Color(255, 255, 255));
    options.need_color_buffer = true;
    buffered = pipeline.process(white, options);
    options.need_color_buffer = false;
    direct = pipeline.process(white, options);
    // Width 12 -> 8; height floor(8/1.5) -> 5. Three rows are black padding.
    expect_near("fit buffered padded mean", buffered.cell_stats[0].mean_r, 0.625f, 0.01f);
    expect_near("fit direct padded mean", direct.cell_stats[0].mean_r, 0.625f, 0.001f);

    // A fit image must preserve both source endpoint bars, even when display
    // aspect and the analysis raster cell ratio differ.
    config.char_aspect = 2.0f;
    pipeline.set_config(config);
    FrameBuffer endpoint_bars(60, 80, Color(0, 0, 0));
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 60; ++x) {
            endpoint_bars.set_pixel(x, y, Color(255, 255, 255));
            endpoint_bars.set_pixel(x, 79-y, Color(255, 255, 255));
        }
    }
    auto fit = pipeline.process(endpoint_bars, options);
    float maximum = 0.0f;
    for (size_t i = 0; i < fit.luminance.size_in_elements(); ++i)
        maximum = std::max(maximum, fit.luminance.data()[i]);
    std::cout << (maximum > 0.05f ? "PASS " : "FAIL ")
              << "fit preserves endpoint bars actual_max=" << maximum
              << " required_max>0.05\n";
    failures += !(maximum > 0.05f);

    config.scale_mode = "fill";
    pipeline.set_config(config);
    auto fill = pipeline.process(FrameBuffer(60, 80, Color(255, 255, 255)), options);
    float minimum = 1.0f;
    for (size_t i = 0; i < fill.luminance.size_in_elements(); ++i)
        minimum = std::min(minimum, fill.luminance.data()[i]);
    expect_near("fill has no padding", minimum, 1.0f, 0.001f);
}

FloatImage texture(int width, int height, int block) {
    FloatImage image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Deterministic, nonperiodic local texture; no runtime RNG.
            uint32_t value = static_cast<uint32_t>(x / block + 1) * 0x45d9f3bu;
            value ^= static_cast<uint32_t>(y / block + 1) * 0x27d4eb2du;
            value ^= value >> 16;
            value *= 0x85ebca6bu;
            value ^= value >> 13;
            image.set(x, y, static_cast<float>(value & 255u) / 255.0f);
        }
    }
    return image;
}

FloatImage translate(const FloatImage& image, int dx, int dy) {
    FloatImage shifted(image.width(), image.height());
    for (int y = 0; y < image.height(); ++y)
        for (int x = 0; x < image.width(); ++x)
            shifted.set(x, y, image.get_clamped(x - dx, y - dy));
    return shifted;
}

MotionEstimator::Config motion_config() {
    MotionEstimator::Config config;
    config.solve_divisor = 1;
    config.pyramid_levels = 1;
    config.motion_cap = 8.0f;
    config.still_scene_threshold = 0.0f;
    config.max_reuse_frames = 0;
    config.use_phase_correlation = false;
    config.phase_blend = 1.0f;
    config.phase_search_radius = 8;
    return config;
}

void motion_sign_scale() {
    const auto image = texture(256, 192, 4);
    for (const int direction : {-1, 1}) {
        const auto shifted = translate(image, direction * 4, direction * -4);
        for (const int divisor : {1, 4}) {
            auto config = motion_config();
            config.solve_divisor = divisor;
            MotionEstimator estimator(config);
            estimator.compute_flow(image, shifted);
            const auto baseline = estimator.get_motion(100, 96);
            std::string label = "translation phase-off divisor=" + std::to_string(divisor);
            expect_near(label.c_str(), baseline.dx, direction * 4.0f, 0.1f);
            expect_near("translation phase-off dy", baseline.dy, direction * -4.0f, 0.1f);
            config.use_phase_correlation = true;
            estimator = MotionEstimator(config);
            estimator.compute_flow(image, shifted);
            const auto refined = estimator.get_motion(100, 96);
            label = "translation phase-on divisor=" + std::to_string(divisor);
            expect_near(label.c_str(), refined.dx, direction * 4.0f, 0.3f);
            expect_near("translation phase-on dy", refined.dy, direction * -4.0f, 0.3f);
        }
    }
}

void motion_unequal_axis_scaling() {
    for (const bool horizontal : {false, true}) {
        // Only the moving axis is divisible by eight. Scaling both components
        // by an averaged ratio changes the known eight-pixel displacement.
        const auto image = texture(horizontal ? 128 : 143, horizontal ? 143 : 128, 8);
        auto config = motion_config();
        config.solve_divisor = 8;
        config.motion_cap = 16.0f;
        MotionEstimator estimator(config);
        estimator.compute_flow(image, translate(image, horizontal ? 8 : 0, horizontal ? 0 : 8));
        const auto flow = estimator.get_motion(64, 64);
        expect_near("unequal resize dx", flow.dx, horizontal ? 8.0f : 0.0f, 0.001f);
        expect_near("unequal resize dy", flow.dy, horizontal ? 0.0f : 8.0f, 0.001f);
    }
}

void sparse_tail() {
    const auto image = texture(100, 100, 1);
    const auto shifted = translate(image, 1, 1);
    auto config = motion_config();
    config.motion_cap = 2.0f;
    MotionEstimator estimator(config);
    estimator.compute_flow(image, shifted);
    expect_near("sparse last solved tile dx", estimator.get_motion(92, 92).dx, 1.0f, 0.001f);
    expect_near("sparse uncovered tail dx", estimator.get_motion(95, 92).dx, 1.0f, 0.001f);
    expect_near("sparse uncovered tail dy", estimator.get_motion(92, 95).dy, 1.0f, 0.001f);
    expect_near("sparse border replication dx", estimator.get_motion(99, 92).dx, 1.0f, 0.001f);
}

void integral_reuse() {
    IntegralImage integral(FloatImage(2, 2, 1.0f));
    integral.compute(FloatImage(4, 1, 1.0f));
    expect_near("integral shape-change sum", integral.sum(0, 0, 4, 1), 4.0f, 0.001f);
    expect_near("integral clipped mean", integral.mean(-1, -1, 5, 2), 1.0f, 0.001f);
    integral.compute(FloatImage(1, 4, 2.0f));
    expect_near("integral tall shape-change sum", integral.sum(0, 0, 1, 4), 8.0f, 0.001f);
}

void adaptive_threshold_controls() {
    FloatImage step(64, 64, 0.0f);
    for (int y = 0; y < 64; ++y)
        for (int x = 32; x < 64; ++x)
            step.set(x, y, y < 24 ? 1.0f : 0.2f);
    for (const std::string mode : {"global", "local", "hybrid"}) {
        EdgeDetector::Config config;
        config.multi_scale = false;
        config.blur_sigma = 0.0f;
        config.adaptive_mode = mode;
        config.tile_size = 64;
        config.high_threshold = 0.5f;
        config.low_threshold = 0.01f;
        const auto loose = EdgeDetector(config).detect(step);
        config.low_threshold = 0.5f;
        const auto tight = EdgeDetector(config).detect(step);
        auto count = [](const EdgeData& edges) {
            return std::count(edges.edge_mask.begin(), edges.edge_mask.end(), true);
        };
        expect_true((mode + " low threshold controls weak-edge retention").c_str(), count(loose) > count(tight));
        config.high_threshold = 10.0f;
        config.low_threshold = 5.0f;
        expect_true((mode + " high threshold suppresses all weaker edges").c_str(),
                    count(EdgeDetector(config).detect(step)) == 0);
    }
}

void cell_histogram_bounds() {
    for (const int bins : {0, 9}) {
        CellStatsAggregator::Config config;
        config.orientation_bins = bins;
        bool rejected = false;
        try { CellStatsAggregator aggregator(config); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect_true("reject histogram bins outside fixed storage", rejected);
    }
}

void motion_reconfiguration() {
    const auto image = texture(96, 96, 1);
    const auto shifted = translate(image, 4, 0);
    auto config = motion_config();
    MotionEstimator estimator(config);
    estimator.compute_flow(image, shifted);
    config.motion_cap = 1.0f;
    config.max_reuse_frames = 4;
    config.reuse_scene_threshold = 1.0f;
    estimator.set_config(config);
    estimator.compute_flow(image, shifted);
    expect_true("configuration change discards cached flow", std::abs(estimator.get_motion(48, 48).dx) <= 1.0f);

    config.motion_cap = 8.0f;
    estimator.set_config(config);
    estimator.compute_flow(image, shifted);
    const auto wider = texture(100, 96, 1);
    estimator.compute_flow(wider, translate(wider, 1, 0));
    expect_near("dimension change solves new flow", estimator.get_motion(48, 48).dx, 1.0f, 0.001f);
}

void contour_reuse_boundaries() {
    ContourExtractor extractor;
    std::vector<CellStats> cells(1);
    cells[0].has_contour = true;
    cells[0].contour_codepoint = '|';
    cells[0].contour_strength = 0.5f;
    cells[0].mean_luminance = 0.3f;
    extractor.apply(FloatImage(8, 8, 0.0f), EdgeData{}, 8, 8, 1, 1, cells);
    expect_true("flat frame clears previous contour", !cells[0].has_contour &&
                cells[0].contour_codepoint == 0 && cells[0].contour_strength == 0.0f);
    expect_near("contour reset preserves cell statistics", cells[0].mean_luminance, 0.3f, 0.0f);

    auto config = extractor.config();
    config.enabled = false;
    extractor.set_config(config);
    cells[0].has_contour = true;
    extractor.apply(FloatImage(), EdgeData{}, 0, 0, 0, 0, cells);
    expect_true("disabled contours clear reused flags", !cells[0].has_contour);
    config.enabled = true;
    extractor.set_config(config);

    bool rejected = false;
    try { extractor.apply(FloatImage(16, 8), EdgeData{}, 8, 8, 2, 1, cells); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect_true("short contour cell vector is rejected", rejected);

    rejected = false;
    try { extractor.apply(FloatImage(8, 8), EdgeData{}, -1, 8, 1, 1, cells); }
    catch (const std::invalid_argument&) { rejected = true; }
    expect_true("negative contour geometry is rejected", rejected);

    std::vector<CellStats> empty_cells;
    extractor.apply(FloatImage(), EdgeData{}, 0, 0, 0, 0, empty_cells);
    expect_true("default empty contour inputs are usable", empty_cells.empty());
}

void concurrent_motion_instances() {
    constexpr size_t workers = 4;
    std::array<MotionVector, workers> results{};
    std::barrier ready(static_cast<std::ptrdiff_t>(workers));
    std::array<std::thread, workers> threads;
    for (size_t i = 0; i < workers; ++i) {
        threads[i] = std::thread([i, &ready, &results] {
            auto config = motion_config();
            config.use_phase_correlation = true;
            config.pyramid_levels = 3;
            config.solve_divisor = 2;
            const auto image = texture(112 + static_cast<int>(i) * 16, 96, 2);
            const int direction = i % 2 ? -1 : 1;
            const auto shifted = translate(image, direction * 2, direction * -2);
            ready.arrive_and_wait();
            MotionEstimator estimator(config);
            estimator.compute_flow(image, shifted);
            results[i] = estimator.get_motion(56, 48);
        });
    }
    for (auto& thread : threads) thread.join();
    for (size_t i = 0; i < workers; ++i) {
        const float direction = i % 2 ? -1.0f : 1.0f;
        expect_near("concurrent phase dx", results[i].dx, direction * 2.0f, 0.3f);
        expect_near("concurrent phase dy", results[i].dy, direction * -2.0f, 0.3f);
    }
}

void motion_cell_clipping() {
    const auto image = texture(96, 96, 1);
    auto config = motion_config();
    MotionEstimator estimator(config);
    estimator.compute_flow(image, translate(image, 1, 0));
    float expected_x = 0.0f, expected_y = 0.0f;
    estimator.get_motion_for_cell(0, 40, 4, 8, expected_x, expected_y);
    float actual_x = 0.0f, actual_y = 0.0f;
    estimator.get_motion_for_cell(-4, 40, 8, 8, actual_x, actual_y);
    expect_near("motion cell clips left border", actual_x, expected_x, 0.0001f);
    expect_near("motion cell clipping preserves dy", actual_y, expected_y, 0.0001f);
}

void image_access_boundaries() {
    FloatImage empty;
    expect_near("default empty clamped sample is zero", empty.get_clamped(0, 0), 0.0f, 0.0f);
    expect_near("zero-width clamped sample is zero", FloatImage(0, 3).get_clamped(2, -1), 0.0f, 0.0f);
    FloatImage image(2, 2, 0.25f);
    image.set(1, 1, 0.75f);
    expect_near("clamped sample respects image border", image.get_clamped(4, 4), 0.75f, 0.0f);
    image.set(-1, 0, 1.0f);
    expect_near("out-of-range write preserves pixels", image.get(0, 0), 0.25f, 0.0f);
    FrameBuffer frame(2, 2, Color(1, 2, 3));
    frame.set_pixel(1, 1, Color(4, 5, 6));
    const auto pixel = frame.get_pixel(1, 1);
    expect_true("RGBA pixel indexing preserves channels", pixel.r == 4 && pixel.g == 5 && pixel.b == 6);
    expect_true("empty frame reads default color", FrameBuffer().get_pixel(0, 0).r == 0);
    for (const bool floating : {false, true}) {
        bool rejected = false;
        try {
            if (floating) { FloatImage invalid(-1, 2); }
            else { FrameBuffer invalid(2, -1); }
        } catch (const std::invalid_argument&) { rejected = true; }
        expect_true("negative image dimensions are rejected", rejected);
    }
    expect_true("empty RGBA conversion accepts no storage", FloatImage::from_rgba(nullptr, 0, 0).empty());
    EdgeData edges;
    edges.magnitude = FloatImage(2, 2);
    expect_true("missing edge mask reads no edge", !edges.is_edge(0, 0));
    edges.edge_mask = {true};
    expect_true("partial edge mask bounds are safe", edges.is_edge(0, 0) && !edges.is_edge(1, 1));
}

void motion_limit_validation() {
    for (const float cap : {-1.0f, std::numeric_limits<float>::quiet_NaN()}) {
        auto config = motion_config();
        config.motion_cap = cap;
        bool rejected = false;
        try { MotionEstimator invalid(config); }
        catch (const std::invalid_argument&) { rejected = true; }
        expect_true("invalid motion limit is rejected", rejected);
    }
}

void changing_motion_geometry() {
    auto config = motion_config();
    config.use_phase_correlation = true;
    config.pyramid_levels = 3;
    config.solve_divisor = 2;
    MotionEstimator estimator(config);
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        const auto image = texture(96 + (i % 5) * 16, 96 + (i % 3) * 8, 2);
        estimator.compute_flow(image, translate(image, 2, -2));
        const auto flow = estimator.get_motion(48, 48);
        correct &= std::abs(flow.dx - 2.0f) < 0.3f && std::abs(flow.dy + 2.0f) < 0.3f;
    }
    expect_true("phase caches preserve changing image geometries", correct);
    const auto invalid = estimator.get_motion_interpolated(std::numeric_limits<float>::quiet_NaN(), 0.0f);
    expect_true("non-finite motion coordinates return no motion", invalid.dx == 0.0f && invalid.dy == 0.0f);
}

void color_buffer_resampling() {
    auto config = pipeline_config();
    config.cell_width = 1;
    config.cell_height = 1;
    Pipeline pipeline(config);
    FrameBuffer pair(2, 1);
    pair.set_pixel(0, 0, Color(255, 0, 0, 64));
    pair.set_pixel(1, 0, Color(0, 0, 255, 192));
    auto downsampled = pipeline.process(pair);
    auto color = downsampled.color_buffer.get_pixel(0, 0);
    // IEC sRGB encoding: E(0.5) rounds to 188; alpha is arithmetic coverage.
    expect_near("linear-light red/blue downsample red", color.r, 188.0f, 1.0f);
    expect_near("linear-light red/blue downsample blue", color.b, 188.0f, 1.0f);
    expect_near("downsample alpha remains linear", color.a, 128.0f, 0.0f);
    config.cell_width = 4;
    pipeline.set_config(config);
    auto enlarged = pipeline.process(pair);
    color = enlarged.color_buffer.get_pixel(1, 0);
    // Pixel-center bilinear weights are 3/4 red and 1/4 blue.
    expect_near("linear-light bilinear red", color.r, 225.0f, 1.0f);
    expect_near("linear-light bilinear blue", color.b, 137.0f, 1.0f);
    expect_near("bilinear alpha remains linear", color.a, 96.0f, 0.0f);
    color = enlarged.color_buffer.get_pixel(2, 0);
    expect_near("linear-light bilinear opposite red", color.r, 137.0f, 1.0f);
    expect_near("linear-light bilinear opposite blue", color.b, 225.0f, 1.0f);
    config.cell_width = 1;
    pipeline.set_config(config);
    FrameBuffer checker(8, 8);
    for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 8; ++x) {
            const uint8_t v = (x + y) % 2 ? 255 : 0;
            checker.set_pixel(x, y, Color(v, v, v));
        }
    color = pipeline.process(checker).color_buffer.get_pixel(0, 0);
    expect_near("linear-light checker buffer mean", color.r, 188.0f, 1.0f);
}

void pipeline_boundaries() {
    Pipeline pipeline(pipeline_config());
    const auto empty = pipeline.process(FrameBuffer{});
    expect_true("empty pipeline input returns empty result", empty.luminance.empty() &&
                empty.cell_stats.empty() && empty.grid_cols == 0 && empty.grid_rows == 0);
    for (int scenario = 0; scenario < 7; ++scenario) {
        auto config = pipeline_config();
        if (scenario == 0) config.cell_width = 0;
        if (scenario == 1) config.target_cols = -1;
        if (scenario == 2) config.char_aspect = 0.0f;
        if (scenario == 3) config.char_aspect = std::numeric_limits<float>::quiet_NaN();
        if (scenario == 4) config.target_cols = std::numeric_limits<int>::max();
        if (scenario == 5) config.scale_mode = "unknown";
        if (scenario == 6) config.tile_size = 0;
        bool rejected = false;
        try { Pipeline invalid(config); }
        catch (const std::invalid_argument&) { rejected = true; }
        catch (const std::length_error&) { rejected = true; }
        expect_true("invalid pipeline configuration rejected before allocation", rejected);
    }
    auto config = pipeline_config();
    config.target_cols = 1000;
    config.cell_width = 1;
    config.cell_height = 64;
    config.char_aspect = 1.0f / 32.0f;
    config.scale_mode = "fill";
    pipeline.set_config(config);
    bool rejected = false;
    try { pipeline.process(FrameBuffer(1, 2048)); }
    catch (const std::length_error&) { rejected = true; }
    expect_true("overflowing resize extent rejected with small input", rejected);
}
}

int main() {
    color_stats();
    motion_sign_scale();
    motion_unequal_axis_scaling();
    sparse_tail();
    integral_reuse();
    adaptive_threshold_controls();
    cell_histogram_bounds();
    motion_reconfiguration();
    contour_reuse_boundaries();
    concurrent_motion_instances();
    motion_cell_clipping();
    image_access_boundaries();
    motion_limit_validation();
    changing_motion_geometry();
    color_buffer_resampling();
    pipeline_boundaries();
    std::cout << "Total failed expectations: " << failures << '\n';
    return failures == 0 ? 0 : 1;
}
