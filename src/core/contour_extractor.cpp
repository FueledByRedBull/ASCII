#include "contour_extractor.hpp"

#include "edge_detector.hpp"

#include <algorithm>
#include <array>
#include <cmath>

namespace ascii {

namespace {

constexpr float kPi = 3.14159265358979323846f;

int contour_bin(float gradient_angle) {
    float tangent = gradient_angle + 0.5f * kPi;
    while (tangent < 0.0f) tangent += kPi;
    while (tangent >= kPi) tangent -= kPi;

    const float deg = tangent * 180.0f / kPi;
    if (deg < 22.5f || deg >= 157.5f) return 0;
    if (deg < 67.5f) return 1;
    if (deg < 112.5f) return 2;
    return 3;
}

uint32_t glyph_for_bin(int bin) {
    switch (bin) {
    case 0: return static_cast<uint32_t>('-');
    case 1: return static_cast<uint32_t>('\\');
    case 2: return static_cast<uint32_t>('|');
    default: return static_cast<uint32_t>('/');
    }
}

}  // namespace

void ContourExtractor::apply(const FloatImage& luminance,
                             const EdgeData& edges,
                             int cell_width,
                             int cell_height,
                             int grid_cols,
                             int grid_rows,
                             std::vector<CellStats>& cells) const {
    if (!config_.enabled || luminance.empty() || grid_cols <= 0 || grid_rows <= 0 ||
        cell_width <= 0 || cell_height <= 0 || cells.empty()) {
        return;
    }

    const int w = luminance.width();
    const int h = luminance.height();

    FloatImage inner = EdgeDetector::gaussian_blur(luminance, config_.dog_sigma_inner);
    FloatImage outer = EdgeDetector::gaussian_blur(luminance, config_.dog_sigma_outer);
    FloatImage dog(w, h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            dog.set(x, y, inner.get(x, y) - outer.get(x, y));
        }
    }

    FloatImage gx;
    FloatImage gy;
    EdgeDetector::sobel(dog, gx, gy);

    FloatImage magnitude(w, h, 0.0f);
    FloatImage orientation(w, h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const float sx = gx.get(x, y);
            const float sy = gy.get(x, y);
            magnitude.set(x, y, std::sqrt(sx * sx + sy * sy));
            orientation.set(x, y, std::atan2(sy, sx));
        }
    }

    FloatImage nms = EdgeDetector::non_maximum_suppression(magnitude, orientation);
    const int threshold_tile = std::max(4, std::min(cell_width, cell_height) * 2);
    FloatImage threshold_map = EdgeDetector::compute_adaptive_threshold_map(
        nms, threshold_tile, 0.72f, 0.001f);

    for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
            const int x0 = col * cell_width;
            const int y0 = row * cell_height;
            const int x1 = std::min(x0 + cell_width, w);
            const int y1 = std::min(y0 + cell_height, h);
            const int area = (x1 - x0) * (y1 - y0);
            if (area <= 0) continue;

            std::array<float, 4> hist{0.0f, 0.0f, 0.0f, 0.0f};
            int contour_pixels = 0;
            float strength = 0.0f;

            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const int tx = std::min(threshold_map.width() - 1, x / threshold_tile);
                    const int ty = std::min(threshold_map.height() - 1, y / threshold_tile);
                    const float threshold = std::max(threshold_map.get(tx, ty), 0.001f);
                    const float mag = nms.get(x, y);
                    const bool source_edge = edges.empty() || edges.is_edge(x, y);
                    if (!source_edge || mag < threshold) {
                        continue;
                    }

                    const int bin = contour_bin(orientation.get(x, y));
                    hist[static_cast<size_t>(bin)] += mag;
                    strength += mag;
                    contour_pixels++;
                }
            }

            const float occupancy = static_cast<float>(contour_pixels) / static_cast<float>(area);
            if (contour_pixels < config_.min_pixels || occupancy < config_.min_occupancy) {
                continue;
            }

            int best_bin = 0;
            float best = hist[0];
            float second = 0.0f;
            for (int i = 1; i < 4; ++i) {
                if (hist[static_cast<size_t>(i)] > best) {
                    second = best;
                    best = hist[static_cast<size_t>(i)];
                    best_bin = i;
                } else {
                    second = std::max(second, hist[static_cast<size_t>(i)]);
                }
            }

            const bool mixed = second > 0.0f &&
                               (best / second < config_.intersection_ratio ||
                                (occupancy >= config_.min_occupancy * 2.0f && second >= best * 0.15f));
            const bool dominant = second <= 0.0f || best / second >= config_.dominance_ratio;
            CellStats& stats = cells[static_cast<size_t>(row) * grid_cols + col];
            stats.has_contour = true;
            stats.contour_codepoint = (mixed || !dominant) ? static_cast<uint32_t>('+') : glyph_for_bin(best_bin);
            stats.contour_strength = strength / static_cast<float>(std::max(1, contour_pixels));
        }
    }
}

}  // namespace ascii
