#include "bilateral_grid.hpp"
#include <algorithm>
#include <cmath>

namespace ascii {

void BilateralGrid::build(const std::vector<CellStats>& cells, int cols, int rows) {
    built_ = false;
    source_cols_ = std::max(1, cols);
    source_rows_ = std::max(1, rows);
    const int spatial_limit = std::max(2, config_.spatial_bins);
    if (source_cols_ >= source_rows_) {
        cols_ = std::min(source_cols_, spatial_limit);
        rows_ = std::min(source_rows_, std::max(1, static_cast<int>(std::lround(
            static_cast<double>(source_rows_) * cols_ / source_cols_))));
    } else {
        rows_ = std::min(source_rows_, spatial_limit);
        cols_ = std::min(source_cols_, std::max(1, static_cast<int>(std::lround(
            static_cast<double>(source_cols_) * rows_ / source_rows_))));
    }
    range_bins_ = std::max(4, config_.range_bins);
    int total = cols_ * rows_ * range_bins_;
    sum_r_.assign(total, 0.0f);
    sum_g_.assign(total, 0.0f);
    sum_b_.assign(total, 0.0f);
    weight_.assign(total, 0.0f);

    if (!config_.enabled || static_cast<int>(cells.size()) < source_cols_ * source_rows_) {
        return;
    }

    for (int y = 0; y < source_rows_; ++y) {
        const float yf = source_rows_ > 1
            ? static_cast<float>(y) * (rows_ - 1) / (source_rows_ - 1) : 0.0f;
        const int y0 = std::clamp(static_cast<int>(std::floor(yf)), 0, rows_ - 1);
        const int y1 = std::min(y0 + 1, rows_ - 1);
        const float ty = yf - y0;
        for (int x = 0; x < source_cols_; ++x) {
            const float xf = source_cols_ > 1
                ? static_cast<float>(x) * (cols_ - 1) / (source_cols_ - 1) : 0.0f;
            const int x0 = std::clamp(static_cast<int>(std::floor(xf)), 0, cols_ - 1);
            const int x1 = std::min(x0 + 1, cols_ - 1);
            const float tx = xf - x0;
            int ci = y * source_cols_ + x;
            const auto& c = cells[ci];
            float lum = std::clamp(c.mean_luminance, 0.0f, 1.0f);
            float zf = lum * (range_bins_ - 1);
            int z0 = std::clamp(static_cast<int>(std::floor(zf)), 0, range_bins_ - 1);
            int z1 = std::min(z0 + 1, range_bins_ - 1);
            float tz = zf - z0;

            const int xs[2] = {x0, x1};
            const int ys[2] = {y0, y1};
            const int zs[2] = {z0, z1};
            const float xw[2] = {1.0f - tx, tx};
            const float yw[2] = {1.0f - ty, ty};
            const float zw[2] = {1.0f - tz, tz};
            for (int iz = 0; iz < 2; ++iz) {
                for (int iy = 0; iy < 2; ++iy) {
                    for (int ix = 0; ix < 2; ++ix) {
                        const float weight = xw[ix] * yw[iy] * zw[iz];
                        const int grid_index = idx(xs[ix], ys[iy], zs[iz]);
                        sum_r_[grid_index] += c.mean_r * weight;
                        sum_g_[grid_index] += c.mean_g * weight;
                        sum_b_[grid_index] += c.mean_b * weight;
                        weight_[grid_index] += weight;
                    }
                }
            }
        }
    }

    const float spatial_scale = 0.5f * (
        static_cast<float>(cols_) / source_cols_ + static_cast<float>(rows_) / source_rows_);
    const float spatial_sigma = std::max(0.0f, config_.spatial_sigma) * spatial_scale;
    const float range_sigma_bins = std::max(0.0f, config_.range_sigma) *
                                   static_cast<float>(range_bins_ - 1);

    const std::vector<float> kx = make_gaussian_kernel(spatial_sigma, cols_ - 1);
    const std::vector<float> ky = make_gaussian_kernel(spatial_sigma, rows_ - 1);
    const std::vector<float> kz = make_gaussian_kernel(range_sigma_bins, range_bins_ - 1);

    convolve_x(sum_r_, kx); convolve_x(sum_g_, kx); convolve_x(sum_b_, kx); convolve_x(weight_, kx);
    convolve_y(sum_r_, ky); convolve_y(sum_g_, ky); convolve_y(sum_b_, ky); convolve_y(weight_, ky);
    convolve_z(sum_r_, kz); convolve_z(sum_g_, kz); convolve_z(sum_b_, kz); convolve_z(weight_, kz);

    built_ = true;
}

BilateralGrid::Sample BilateralGrid::sample(int x, int y, float luminance) const {
    Sample s{};
    if (!valid()) {
        return s;
    }

    x = std::clamp(x, 0, source_cols_ - 1);
    y = std::clamp(y, 0, source_rows_ - 1);
    const float xf = source_cols_ > 1
        ? static_cast<float>(x) * (cols_ - 1) / (source_cols_ - 1) : 0.0f;
    const float yf = source_rows_ > 1
        ? static_cast<float>(y) * (rows_ - 1) / (source_rows_ - 1) : 0.0f;
    const int x0 = std::clamp(static_cast<int>(std::floor(xf)), 0, cols_ - 1);
    const int x1 = std::min(x0 + 1, cols_ - 1);
    const int y0 = std::clamp(static_cast<int>(std::floor(yf)), 0, rows_ - 1);
    const int y1 = std::min(y0 + 1, rows_ - 1);
    const float tx = xf - x0;
    const float ty = yf - y0;
    float zf = std::clamp(luminance, 0.0f, 1.0f) * (range_bins_ - 1);
    int z0 = std::clamp(static_cast<int>(std::floor(zf)), 0, range_bins_ - 1);
    int z1 = std::min(z0 + 1, range_bins_ - 1);
    float tz = zf - z0;

    const int xs[2] = {x0, x1};
    const int ys[2] = {y0, y1};
    const int zs[2] = {z0, z1};
    const float xw[2] = {1.0f - tx, tx};
    const float yw[2] = {1.0f - ty, ty};
    const float zw[2] = {1.0f - tz, tz};
    float accumulated_weight = 0.0f;
    for (int iz = 0; iz < 2; ++iz) {
        for (int iy = 0; iy < 2; ++iy) {
            for (int ix = 0; ix < 2; ++ix) {
                const int grid_index = idx(xs[ix], ys[iy], zs[iz]);
                if (weight_[grid_index] <= 1e-6f) continue;
                const float interpolation_weight = xw[ix] * yw[iy] * zw[iz];
                s.r += (sum_r_[grid_index] / weight_[grid_index]) * interpolation_weight;
                s.g += (sum_g_[grid_index] / weight_[grid_index]) * interpolation_weight;
                s.b += (sum_b_[grid_index] / weight_[grid_index]) * interpolation_weight;
                accumulated_weight += interpolation_weight;
            }
        }
    }
    if (accumulated_weight > 1e-6f) {
        s.r /= accumulated_weight;
        s.g /= accumulated_weight;
        s.b /= accumulated_weight;
    }
    s.r = std::clamp(s.r, 0.0f, 1.0f);
    s.g = std::clamp(s.g, 0.0f, 1.0f);
    s.b = std::clamp(s.b, 0.0f, 1.0f);
    return s;
}

std::vector<float> BilateralGrid::make_gaussian_kernel(float sigma, int max_radius) {
    if (sigma <= 1e-4f || max_radius <= 0) {
        return {1.0f};
    }

    int radius = static_cast<int>(std::ceil(3.0f * sigma));
    radius = std::clamp(radius, 1, max_radius);

    std::vector<float> kernel(2 * radius + 1, 0.0f);
    const float denom = 2.0f * sigma * sigma;
    float sum = 0.0f;

    for (int k = -radius; k <= radius; ++k) {
        const float wk = std::exp(-(static_cast<float>(k * k)) / denom);
        kernel[static_cast<size_t>(k + radius)] = wk;
        sum += wk;
    }

    if (sum > 0.0f) {
        for (float& v : kernel) {
            v /= sum;
        }
    }

    return kernel;
}

void BilateralGrid::convolve_x(std::vector<float>& v, const std::vector<float>& kernel) const {
    if (kernel.size() <= 1) {
        return;
    }

    const int radius = static_cast<int>(kernel.size() / 2);
    std::vector<float> out(v.size(), 0.0f);
    for (int z = 0; z < range_bins_; ++z) {
        for (int y = 0; y < rows_; ++y) {
            for (int x = 0; x < cols_; ++x) {
                int i = idx(x, y, z);
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int xs = std::clamp(x + k, 0, cols_ - 1);
                    acc += kernel[static_cast<size_t>(k + radius)] * v[idx(xs, y, z)];
                }
                out[i] = acc;
            }
        }
    }
    v.swap(out);
}

void BilateralGrid::convolve_y(std::vector<float>& v, const std::vector<float>& kernel) const {
    if (kernel.size() <= 1) {
        return;
    }

    const int radius = static_cast<int>(kernel.size() / 2);
    std::vector<float> out(v.size(), 0.0f);
    for (int z = 0; z < range_bins_; ++z) {
        for (int y = 0; y < rows_; ++y) {
            for (int x = 0; x < cols_; ++x) {
                int i = idx(x, y, z);
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int ys = std::clamp(y + k, 0, rows_ - 1);
                    acc += kernel[static_cast<size_t>(k + radius)] * v[idx(x, ys, z)];
                }
                out[i] = acc;
            }
        }
    }
    v.swap(out);
}

void BilateralGrid::convolve_z(std::vector<float>& v, const std::vector<float>& kernel) const {
    if (kernel.size() <= 1) {
        return;
    }

    const int radius = static_cast<int>(kernel.size() / 2);
    std::vector<float> out(v.size(), 0.0f);
    for (int z = 0; z < range_bins_; ++z) {
        for (int y = 0; y < rows_; ++y) {
            for (int x = 0; x < cols_; ++x) {
                int i = idx(x, y, z);
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int zs = std::clamp(z + k, 0, range_bins_ - 1);
                    acc += kernel[static_cast<size_t>(k + radius)] * v[idx(x, y, zs)];
                }
                out[i] = acc;
            }
        }
    }
    v.swap(out);
}

}  // namespace ascii
