#include "core/edge_detector.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

using namespace ascii;

namespace {
int failures = 0;
size_t checks = 0;
uint32_t random_state = 0x193af845u;

void check(bool condition, const char* message) {
    ++checks;
    if (!condition) {
        if (++failures <= 10) std::cerr << message << '\n';
    }
}

bool same_image(const FloatImage& a, const FloatImage& b) {
    if (a.width() != b.width() || a.height() != b.height()) return false;
    for (size_t i = 0; i < a.size_in_elements(); ++i) {
        if (std::bit_cast<uint32_t>(a.data()[i]) != std::bit_cast<uint32_t>(b.data()[i])) return false;
    }
    return true;
}

FloatImage pattern(int w, int h, int kind) {
    FloatImage image(w, h);
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        random_state = 1664525u * random_state + 1013904223u;
        const float noise = static_cast<float>((random_state >> 24) & 255) / 255.0f;
        const float value = kind == 0 ? 0.0f : kind == 1 ? 1.0f :
            kind == 2 ? (x >= w / 2 ? 1.0f : 0.0f) : kind == 3 ? (x == y ? 1.0f : 0.0f) :
            kind == 4 ? static_cast<float>((x + y) % 2) :
            kind == 6 ? .3f + .35f * noise : noise;
        image.set(x, y, value);
    }
    return image;
}

// Deliberately separate masks and a coordinate queue provide an independent
// reference for the packed output and the production flood's state transitions.
std::vector<bool> reference_mask(const FloatImage& image, const FloatImage& lows,
                                 const FloatImage& highs, bool hysteresis) {
    const int w = image.width(), h = image.height();
    std::vector<bool> strong(image.size_in_elements()), weak(strong.size()), result(strong.size());
    std::queue<std::pair<int, int>> pending;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        const int i = y * w + x;
        strong[i] = image.get(x, y) >= highs.get(x, y);
        weak[i] = !strong[i] && image.get(x, y) >= lows.get(x, y);
    }
    if (!hysteresis) return strong;
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        if (strong[y * w + x]) { result[y * w + x] = true; pending.push({x, y}); }
    }
    while (!pending.empty()) {
        const auto [cx, cy] = pending.front();
        pending.pop();
        for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
            const int x = cx + dx, y = cy + dy;
            if (x < 0 || x >= w || y < 0 || y >= h) continue;
            const int i = y * w + x;
            if (weak[i] && !result[i]) { result[i] = true; pending.push({x, y}); }
        }
    }
    return result;
}

std::vector<bool> reference_detection(const FloatImage& nms, const EdgeDetector::Config& config) {
    const int w = nms.width(), h = nms.height();
    FloatImage lows(w, h, config.low_threshold), highs(w, h, config.high_threshold);
    const float ratio = config.high_threshold > 0
        ? std::clamp(config.low_threshold / config.high_threshold, 0.0f, 1.0f) : 0.0f;
    if (config.adaptive_mode == "global") {
        const float high = std::max({EdgeDetector::compute_global_percentile_threshold(
            nms, config.global_percentile), config.dark_scene_floor, config.high_threshold});
        lows.fill(high * ratio);
        highs.fill(high);
    } else if (config.adaptive_mode == "local" || config.adaptive_mode == "hybrid") {
        const float floor = std::max(config.dark_scene_floor, config.high_threshold);
        const auto map = EdgeDetector::compute_adaptive_threshold_map(
            nms, config.tile_size, config.global_percentile, floor);
        const float global = std::max(EdgeDetector::compute_global_percentile_threshold(
            nms, config.global_percentile), floor);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            float high = map.get(x / config.tile_size, y / config.tile_size);
            if (config.adaptive_mode == "hybrid") high = .5f * high + .5f * global;
            highs.set(x, y, high);
            lows.set(x, y, high * ratio);
        }
    }
    return reference_mask(nms, lows, highs, config.use_hysteresis);
}

MultiScaleGradientData reference_scales(const FloatImage& input, const EdgeDetector::Config& config) {
    const int w = input.width(), h = input.height();
    const float first = std::max(.1f, config.scale_sigma_0);
    const std::array<float, 2> sigmas{first, std::max(first + 1e-4f, config.scale_sigma_1)};
    std::array<GradientData, 2> gradients;
    std::array<FloatImage, 2> laplacians;
    for (size_t s = 0; s < sigmas.size(); ++s) {
        const auto blurred = EdgeDetector::gaussian_blur(input,
            std::sqrt(sigmas[s] * sigmas[s] + config.blur_sigma * config.blur_sigma));
        auto& grad = gradients[s];
        EdgeDetector::sobel(blurred, grad.gx, grad.gy);
        grad.magnitude = FloatImage(w, h);
        grad.orientation = FloatImage(w, h);
        laplacians[s] = FloatImage(w, h);
        for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
            const float sx = grad.gx.get(x, y), sy = grad.gy.get(x, y);
            grad.magnitude.set(x, y, std::sqrt(sx * sx + sy * sy));
            grad.orientation.set(x, y, std::atan2(sy, sx));
            if (x > 0 && y > 0 && x + 1 < w && y + 1 < h) {
                const float lap = blurred.get(x - 1, y) + blurred.get(x + 1, y) +
                    blurred.get(x, y - 1) + blurred.get(x, y + 1) - 4.0f * blurred.get(x, y);
                laplacians[s].set(x, y, sigmas[s] * sigmas[s] * std::abs(lap));
            }
        }
    }
    MultiScaleGradientData result{FloatImage(w, h), FloatImage(w, h), FloatImage(w, h),
                                 FloatImage(w, h), config.adaptive_scale_selection ? -1 : 0};
    for (int y = 0; y < h; ++y) for (int x = 0; x < w; ++x) {
        size_t selected = 0;
        if (config.adaptive_scale_selection) {
            selected = laplacians[1].get(x, y) > laplacians[0].get(x, y) ? 1 : 0;
            float sum = 0, squared = 0;
            for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx) {
                const float value = input.get_clamped(x + dx, y + dy);
                sum += value;
                squared += value * value;
            }
            const float mean = sum / 9.0f;
            const float variance = std::max(0.0f, squared / 9.0f - mean * mean);
            const float span = std::max(config.scale_variance_ceil - config.scale_variance_floor, 1e-8f);
            const float detail = std::clamp((variance - config.scale_variance_floor) / span, 0.0f, 1.0f);
            if (detail <= .33f || detail >= .67f) selected = static_cast<size_t>(std::lround(1.0f - detail));
        }
        const auto& grad = gradients[selected];
        result.magnitude.set(x, y, grad.magnitude.get(x, y));
        result.orientation.set(x, y, grad.orientation.get(x, y));
        result.gx.set(x, y, grad.gx.get(x, y));
        result.gy.set(x, y, grad.gy.get(x, y));
    }
    return result;
}

void test_hysteresis() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const std::array<std::pair<float, float>, 9> thresholds{{
        {0.0f, 0.0f}, {0.0f, .75f}, {.25f, .75f}, {.75f, .25f}, {.5f, .5f},
        {-1.0f, 2.0f}, {nan, .75f}, {.25f, nan}, {inf, inf}}};
    for (int encoding = 0; encoding < 729; ++encoding) {
        FloatImage image(2, 3);
        int digits = encoding;
        for (int i = 0; i < 6; ++i, digits /= 3) image.data()[i] = (digits % 3) * .5f;
        for (const auto [low, high] : thresholds) {
            check(EdgeDetector::hysteresis_threshold(image, 2, 3, low, high) ==
                  reference_mask(image, FloatImage(2, 3, low), FloatImage(2, 3, high), true),
                  "Hysteresis preserves every reference mask bit");
        }
    }
    const auto zero = EdgeDetector::hysteresis_threshold(FloatImage(3, 2), 3, 2, 0, 0);
    check(std::all_of(zero.begin(), zero.end(), [](bool value) { return value; }),
          "Equal zero thresholds accept all pixels including zero borders");
    check(EdgeDetector::hysteresis_threshold(FloatImage(1, 1, 1.0f), 3, 2, .5f, .5f) ==
          std::vector<bool>({true, false, false, false, false, false}),
          "Explicit hysteresis bounds preserve out-of-image zero sampling");
    bool rejected = false;
    try { EdgeDetector::hysteresis_threshold(FloatImage(), -1, 0, 0, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Hysteresis rejects negative dimensions before allocation");
    rejected = false;
    try { EdgeDetector::hysteresis_threshold(FloatImage(), 46341, 46341, 0, 0); }
    catch (const std::length_error&) { rejected = true; }
    check(rejected, "Hysteresis rejects dimensions exceeding integer-index storage");
}

void test_detection_and_scales() {
    for (const auto [w, h] : std::array<std::pair<int, int>, 17>{{
             {0, 0}, {0, 3}, {1, 1}, {1, 7}, {7, 1}, {2, 2}, {3, 3},
             {5, 7}, {6, 7}, {7, 7}, {8, 7}, {9, 7}, {10, 7}, {17, 13},
             {31, 19}, {32, 19}, {65, 33}}}) {
        for (int kind = 0; kind < 7; ++kind) {
            const auto input = pattern(w, h, kind);
            const auto original = input;
            EdgeDetector::Config config;
            config.adaptive_scale_selection = kind % 2 == 0;
            config.scale_sigma_0 = kind % 3 == 0 ? .3f : .8f;
            config.scale_sigma_1 = kind % 3 == 0 ? 2.0f : 1.6f;
            config.blur_sigma = kind % 2 == 0 ? 0.0f : 1.0f;
            const auto expected = reference_scales(input, config);
            const auto actual = EdgeDetector(config).compute_multi_scale_gradients(input);
            check(same_image(expected.magnitude, actual.magnitude) &&
                  same_image(expected.orientation, actual.orientation) &&
                  same_image(expected.gx, actual.gx) && same_image(expected.gy, actual.gy) &&
                  expected.best_scale == actual.best_scale,
                  "Two-scale selection preserves all gradient float bits and scale metadata");
            config.multi_scale = kind % 2 == 0;
            config.tile_size = std::array<int, 3>{1, 4, 31}[kind % 3];
            config.global_percentile = std::array<float, 3>{0, .7f, 1}[kind % 3];
            config.low_threshold = kind % 3 == 0 ? 0.0f : .05f;
            config.high_threshold = kind % 3 == 0 ? 0.0f : .15f;
            config.dark_scene_floor = kind % 2 == 0 ? 0.0f : .02f;
            for (bool hysteresis : {false, true}) for (const char* mode : {"none", "global", "local", "hybrid"}) {
                config.use_hysteresis = hysteresis;
                config.adaptive_mode = mode;
                const auto edges = EdgeDetector(config).detect(input);
                const auto nms = EdgeDetector::non_maximum_suppression(edges.magnitude, edges.orientation);
                check(edges.edge_mask == reference_detection(nms, config),
                      "Detection preserves reference masks across modes, tiles, and borders");
            }
            check(same_image(input, original), "Gradient and detection calls leave the source image unchanged");
        }
    }
}

void test_empty_geometry() {
    for (const auto [w, h] : std::array<std::pair<int, int>, 5>{{
             {0, 0}, {0, 17}, {17, 0}, {0, 65536}, {65536, 0}}}) {
        const FloatImage input(w, h);
        EdgeDetector detector;
        const auto gradient = detector.compute_gradients(input);
        check(gradient.magnitude.size() == input.size() && gradient.orientation.size() == input.size() &&
              gradient.gx.size() == input.size() && gradient.gy.size() == input.size(),
              "Empty single-scale gradients preserve both dimensions");
        const auto multi = detector.compute_multi_scale_gradients(input);
        check(multi.magnitude.size() == input.size() && multi.orientation.size() == input.size() &&
              multi.gx.size() == input.size() && multi.gy.size() == input.size() && multi.best_scale == -1,
              "Empty multiscale gradients preserve geometry and scale metadata");
        check(EdgeDetector::non_maximum_suppression(input, input).size() == input.size(),
              "Empty NMS preserves both dimensions");
        check(EdgeDetector::compute_global_percentile_threshold(input, .7f) == .1f,
              "Empty global percentile retains its no-values threshold");
        check(EdgeDetector::compute_tile_threshold(input, 0, 0, 65536, w, h, .7f) == .1f,
              "Empty tile percentile retains its no-values threshold");
        check(EdgeDetector::hysteresis_threshold(input, w, h, 0, 0).empty(),
              "Zero-area explicit hysteresis bounds return an empty mask");
        GradientData selected;
        const auto edges = detector.detect(input, &selected);
        check(edges.magnitude.size() == input.size() && edges.orientation.size() == input.size() &&
              selected.gx.size() == input.size() && selected.gy.size() == input.size() && edges.edge_mask.empty(),
              "Empty detection preserves selected-gradient geometry");
    }
    bool rejected = false;
    try { EdgeDetector::compute_global_percentile_threshold(FloatImage(), -1); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Empty global percentile still validates its percentile");
    rejected = false;
    try { EdgeDetector::compute_tile_threshold(FloatImage(), 0, 0, 0, 0, 0, .7f); }
    catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Empty tile percentile still validates its tile size");
}

void test_nms_sectors() {
    constexpr float pi = 3.14159265358979323846f;
    const float inf = std::numeric_limits<float>::infinity();
    std::vector<std::pair<float, int>> angles{{0.0f, 0}, {-0.0f, 0}, {pi, 0}, {-pi, 0},
        {inf, 0}, {-inf, 0}, {std::numeric_limits<float>::quiet_NaN(), 3}};
    const std::array<float, 4> boundaries{pi / 8, 3 * pi / 8, 5 * pi / 8, 7 * pi / 8};
    for (int sector = 0; sector < 4; ++sector) {
        angles.push_back({std::nextafter(boundaries[sector], -inf), sector});
        angles.push_back({boundaries[sector], (sector + 1) % 4});
        angles.push_back({std::nextafter(boundaries[sector], inf), (sector + 1) % 4});
    }
    const std::array<std::pair<int, int>, 8> neighbors{{
        {-1, 0}, {1, 0}, {-1, -1}, {1, 1}, {0, -1}, {0, 1}, {1, -1}, {-1, 1}}};
    for (const auto [angle, sector] : angles) for (size_t i = 0; i < neighbors.size(); ++i) {
        FloatImage mag(10, 5, .25f), orientation(10, 5, angle);
        mag.set(4, 2, 1.0f);
        mag.set(4 + neighbors[i].first, 2 + neighbors[i].second, 2.0f);
        const auto result = EdgeDetector::non_maximum_suppression(mag, orientation);
        check(result.get(4, 2) == (static_cast<int>(i / 2) == sector ? 0.0f : 1.0f),
              "NMS sector boundaries select exactly their two reference neighbors");
    }
    FloatImage signed_zero(10, 5, 0.0f);
    signed_zero.set(4, 2, -0.0f);
    check(std::bit_cast<uint32_t>(EdgeDetector::non_maximum_suppression(
              signed_zero, FloatImage(10, 5)).get(4, 2)) == 0x80000000u,
          "NMS keeps the sign bit of accepted negative zero");
    FloatImage mag(10, 5, 0.0f);
    mag.set(4, 2, 1.0f);
    mag.set(4, 1, 2.0f);
    check(EdgeDetector::non_maximum_suppression(mag, FloatImage()).get(4, 2) == 1.0f,
          "Missing orientation pixels retain zero-angle sampling");
}
}

int main() {
    test_hysteresis();
    test_detection_and_scales();
    test_empty_geometry();
    test_nms_sectors();
    std::cout << "CHECKS=" << checks << " FAILURES=" << failures << '\n';
    return failures ? 1 : 0;
}
