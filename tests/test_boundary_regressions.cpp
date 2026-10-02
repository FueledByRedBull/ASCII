#include "core/edge_detector.hpp"
#include "core/cell_stats.hpp"
#include "mapping/bilateral_grid.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace ascii;

namespace {
int failures = 0;
void check(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; ++failures; }
}
template<class Function>
void rejects(Function operation, const char* message) {
    try { operation(); }
    catch (const std::invalid_argument&) { return; }
    check(false, message);
}

void test_cell_geometry() {
    for (int dimension : {0, -1}) {
        CellStatsAggregator::Config config;
        config.cell_width = dimension;
        rejects([&] { CellStatsAggregator invalid(config); }, "Cell aggregation rejects nonpositive width");
        config.cell_width = 8;
        config.cell_height = dimension;
        rejects([&] { CellStatsAggregator invalid(config); }, "Cell aggregation rejects nonpositive height");
    }
    CellStatsAggregator aggregator;
    const int largest = std::numeric_limits<int>::max();
    check(aggregator.grid_cols(largest) == 1 + (largest - 1) / 8 &&
          aggregator.grid_rows(largest) == 1 + (largest - 1) / 16,
          "Cell grid ceiling division does not overflow");
    check(aggregator.grid_cols(0) == 0 && aggregator.grid_rows(0) == 0,
          "Empty cell grid has zero extent");
    check(aggregator.compute(FloatImage(0, largest), {}).empty(),
          "Zero-area cell image ignores an unused large extent");
    rejects([&] { aggregator.grid_cols(-1); }, "Cell grid rejects negative image width");
    rejects([&] { aggregator.grid_rows(-1); }, "Cell grid rejects negative image height");
}

void test_frame_fill() {
    for (const auto size : {Size{0, std::numeric_limits<int>::max()}, Size{7, 0}, Size{3, 5}}) {
        FrameBuffer image(size.width, size.height, Color(255, 254, 253, 252));
        image.fill(Color(7, 13, 29, 41));
        check(image.size() == size && image.byte_size() == size.area() * 4,
              "Frame fill preserves dimensions and logical storage");
        const FrameBuffer constructed(size.width, size.height, Color(7, 13, 29, 41));
        for (size_t i = 0; i < image.byte_size(); ++i) {
            const uint8_t expected[4] = {7, 13, 29, 41};
            check(image.data()[i] == expected[i % 4] && constructed.data()[i] == expected[i % 4],
                  "Frame fill and construction set every RGBA channel");
        }
    }
}

void test_gaussian_parameters() {
    FloatImage input(2, 2, 0.5f);
    input.set(0, 0, 1.0f);
    rejects([&] { EdgeDetector::gaussian_blur(input, -std::numeric_limits<float>::infinity()); },
            "Gaussian rejects nonfinite sigma before its identity path");
    const auto tiny = EdgeDetector::gaussian_blur(input, 1e-30f);
    const auto zero = EdgeDetector::gaussian_blur(input, 0.0f);
    const auto negative = EdgeDetector::gaussian_blur(input, -1.0f);
    for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        check(tiny.get(x, y) == input.get(x, y), "Tiny Gaussian sigma is a finite identity kernel");
        check(zero.get(x, y) == input.get(x, y) && negative.get(x, y) == input.get(x, y),
              "Finite nonpositive Gaussian sigma preserves the no-blur API");
    }
    const auto constant = EdgeDetector::gaussian_blur(FloatImage(7, 5, 0.25f), 1.6f);
    for (int y = 0; y < 5; ++y) for (int x = 0; x < 7; ++x)
        check(std::abs(constant.get(x, y) - 0.25f) < 1e-6f, "Normal Gaussian blur preserves a constant field");
}

void test_tile_parameters() {
    FloatImage input(3, 2, 0.2f);
    input.set(1, 0, 0.8f);
    rejects([&] { EdgeDetector::compute_tile_threshold(input, 0, 0, -1, 3, 2, 0.5f); },
            "Tile threshold rejects nonpositive tile size");
    rejects([&] { EdgeDetector::compute_tile_threshold(input, 0, 0, 1, -1, 2, 0.5f); },
            "Tile threshold rejects negative image bounds");
    rejects([&] { EdgeDetector::compute_global_percentile_threshold(input, 1.1f); },
            "Global percentile rejects values outside its probability range");
    rejects([&] { EdgeDetector::compute_tile_threshold(input, 0, 0, 2, 3, 2, 1.1f); },
            "Tile percentile rejects values outside its probability range");
    rejects([&] { EdgeDetector::compute_adaptive_threshold_map(input, 2, 1.1f, 0.0f); },
            "Adaptive percentile rejects values outside its probability range");
    rejects([&] { EdgeDetector::compute_adaptive_threshold_map(input, 2, 0.5f,
        std::numeric_limits<float>::infinity()); }, "Adaptive map rejects a nonfinite floor");
    check(EdgeDetector::compute_tile_threshold(input, -1, -1, 2, 3, 2, 0.5f) == 0.2f,
          "Tile ROI intersects negative origins with the image");
    check(EdgeDetector::compute_tile_threshold(input, 5, 5, 2, 3, 2, 0.5f) == 0.1f,
          "Empty tile ROI retains its threshold default");
    const auto map = EdgeDetector::compute_adaptive_threshold_map(input, 2, 0.5f, 0.3f);
    check(map.width() == 2 && map.height() == 1 && map.get(0, 0) == 0.3f && map.get(1, 0) == 0.3f,
          "Adaptive tile map keeps normal edge-tile dimensions and floor");
}

void test_bilateral_parameters() {
    BilateralGrid::Config config;
    config.enabled = true;
    config.spatial_bins = 2;
    config.range_bins = 4;
    config.spatial_sigma = config.range_sigma = 0.0f;
    std::vector<CellStats> cells(4);
    for (auto& cell : cells) { cell.mean_luminance = 0.5f; cell.mean_r = 0.25f; }
    BilateralGrid grid(config);
    grid.build(cells, 2, 2);
    const auto sample = grid.sample(0, 0, 0.5f);
    check(sample.has_support && std::abs(sample.r - 0.25f) < 1e-6f,
          "Zero-sigma bilateral API retains constant-color support");
    for (int bins : {1, 257}) {
        auto bad = config;
        bad.spatial_bins = bins;
        grid.set_config(bad);
        rejects([&] { grid.build(cells, 2, 2); }, "Bilateral rejects unsupported spatial bin counts");
    }
    for (int bins : {3, 65}) {
        auto bad = config;
        bad.range_bins = bins;
        grid.set_config(bad);
        rejects([&] { grid.build(cells, 2, 2); }, "Bilateral rejects unsupported range bin counts");
    }
    auto bad = config;
    bad.spatial_sigma = -1.0f;
    grid.set_config(bad);
    rejects([&] { grid.build(cells, 2, 2); }, "Bilateral rejects a negative spatial sigma");
    bad = config;
    bad.range_sigma = -1.0f;
    grid.set_config(bad);
    rejects([&] { grid.build(cells, 2, 2); }, "Bilateral rejects a negative range sigma");
    grid.set_config(config);
    rejects([&] { grid.build(cells, -1, 2); }, "Bilateral rejects negative source dimensions");
    grid.build({}, 2, 2);
    check(!grid.valid(), "Short bilateral source data leaves no valid samples");
    config.enabled = false;
    grid.set_config(config);
    grid.build(cells, 2, 2);
    check(!grid.valid(), "Disabled bilateral grid leaves no valid samples");
}

void test_guarded_extremes() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const int largest = std::numeric_limits<int>::max();
    FloatImage input(2, 2, 0.4f);
    for (float sigma : {nan, infinity, 1e30f}) {
        rejects([&] { EdgeDetector::gaussian_blur(input, sigma); },
                "Gaussian rejects nonfinite or unrepresentable kernel radii");
    }
    rejects([&] { EdgeDetector::compute_adaptive_threshold_map(input, 0, 0.5f, 0.0f); },
            "Adaptive tile map rejects zero tile size before division");
    rejects([&] { EdgeDetector::compute_tile_threshold(input, 0, 0, 1, 2, 2, nan); },
            "Tile threshold rejects nonfinite percentile before conversion");
    check(EdgeDetector::compute_tile_threshold(input, 1, 1, largest, largest, largest, 0.5f) == 0.4f,
          "Oversized tile is clipped before reserving storage or adding endpoints");
    check(EdgeDetector::compute_tile_threshold(input, largest, largest, largest, largest, largest, 0.5f) == 0.1f,
          "Far outside tile endpoints do not overflow");
    const auto map = EdgeDetector::compute_adaptive_threshold_map(input, largest, 0.5f, 0.0f);
    check(map.width() == 1 && map.height() == 1 && map.get(0, 0) == 0.4f,
          "Adaptive ceil division and clipped reserve support a tile larger than the image");
    check(EdgeDetector::gaussian_blur(FloatImage{}, 1.0f).empty(), "Empty Gaussian input stays empty");
    BilateralGrid::Config config;
    config.spatial_bins = config.range_bins = largest;
    BilateralGrid grid(config);
    grid.build({}, largest, largest);
    check(!grid.valid() && grid.grid_cols() == 0 && grid.grid_rows() == 0,
          "Disabled bilateral build does not construct grid storage");
    config.enabled = true;
    grid.set_config(config);
    grid.build({}, largest, largest);
    check(!grid.valid() && grid.grid_cols() == 0 && grid.grid_rows() == 0,
          "Short bilateral input returns before bin allocation");
    config.spatial_bins = 2;
    config.range_bins = 4;
    std::vector<CellStats> cells(1);
    cells[0].mean_luminance = 0.5f;
    for (float sigma : {nan, infinity, 1e30f}) {
        config.spatial_sigma = sigma;
        grid.set_config(config);
        rejects([&] { grid.build(cells, 1, 1); }, "Bilateral rejects nonfinite or excessive sigmas");
    }
    config.spatial_sigma = 1.0f;
    grid.set_config(config);
    grid.build(cells, 1, 1);
    check(!grid.sample(0, 0, nan).has_support && !grid.sample(0, 0, infinity).has_support,
          "Bilateral nonfinite query has no support");
    cells[0].mean_luminance = nan;
    rejects([&] { grid.build(cells, 1, 1); }, "Bilateral rejects nonfinite cell values before bin conversion");
    check(!grid.valid(), "Failed bilateral rebuild leaves no valid samples");
}
}

int main() {
    test_cell_geometry();
    test_frame_fill();
    test_gaussian_parameters();
    test_tile_parameters();
    test_bilateral_parameters();
    test_guarded_extremes();
    std::cout << "FAILURES=" << failures << '\n';
    return failures ? 1 : 0;
}
