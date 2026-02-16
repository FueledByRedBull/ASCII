#include "bilateral_grid.hpp"
#include <algorithm>
#include <cmath>

namespace ascii {

void BilateralGrid::build(const std::vector<CellStats>& cells, int cols, int rows) {
    built_ = false;
    cols_ = std::max(1, cols);
    rows_ = std::max(1, rows);
    range_bins_ = std::max(4, config_.range_bins);
    int total = cols_ * rows_ * range_bins_;
    sum_r_.assign(total, 0.0f);
    sum_g_.assign(total, 0.0f);
    sum_b_.assign(total, 0.0f);
    weight_.assign(total, 0.0f);

    if (!config_.enabled || static_cast<int>(cells.size()) < cols_ * rows_) {
        return;
    }

    for (int y = 0; y < rows_; ++y) {
        for (int x = 0; x < cols_; ++x) {
            int ci = y * cols_ + x;
            const auto& c = cells[ci];
            float lum = std::clamp(c.mean_luminance, 0.0f, 1.0f);
            float zf = lum * (range_bins_ - 1);
            int z0 = std::clamp(static_cast<int>(std::floor(zf)), 0, range_bins_ - 1);
            int z1 = std::min(z0 + 1, range_bins_ - 1);
            float tz = zf - z0;

            float w0 = 1.0f - tz;
            float w1 = tz;
            int i0 = idx(x, y, z0);
            int i1 = idx(x, y, z1);
            sum_r_[i0] += c.mean_r * w0;
            sum_g_[i0] += c.mean_g * w0;
            sum_b_[i0] += c.mean_b * w0;
            weight_[i0] += w0;
            sum_r_[i1] += c.mean_r * w1;
            sum_g_[i1] += c.mean_g * w1;
            sum_b_[i1] += c.mean_b * w1;
            weight_[i1] += w1;
        }
    }

    const float spatial_sigma = std::max(0.0f, config_.spatial_sigma);
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

    x = std::clamp(x, 0, cols_ - 1);
    y = std::clamp(y, 0, rows_ - 1);
    float zf = std::clamp(luminance, 0.0f, 1.0f) * (range_bins_ - 1);
    int z0 = std::clamp(static_cast<int>(std::floor(zf)), 0, range_bins_ - 1);
    int z1 = std::min(z0 + 1, range_bins_ - 1);
    float tz = zf - z0;

    int i0 = idx(x, y, z0);
    int i1 = idx(x, y, z1);
    float w0 = std::max(weight_[i0], 1e-6f);
    float w1 = std::max(weight_[i1], 1e-6f);
    float r0 = sum_r_[i0] / w0;
    float g0 = sum_g_[i0] / w0;
    float b0 = sum_b_[i0] / w0;
    float r1 = sum_r_[i1] / w1;
    float g1 = sum_g_[i1] / w1;
    float b1 = sum_b_[i1] / w1;

    s.r = std::clamp((1.0f - tz) * r0 + tz * r1, 0.0f, 1.0f);
    s.g = std::clamp((1.0f - tz) * g0 + tz * g1, 0.0f, 1.0f);
    s.b = std::clamp((1.0f - tz) * b0 + tz * b1, 0.0f, 1.0f);
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
