#include "cell_stats.hpp"
#include "core/color_space.hpp"
#include <cmath>
#include <algorithm>
#include <array>
#include <utility>
#include <stdexcept>

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <immintrin.h>
#endif

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace ascii {

namespace {

constexpr int kFreqBins = 8;
constexpr int kTextureBins = 8;
constexpr float kPi = 3.14159265358979323846f;

struct DCTBasis {
    static constexpr int W = 8;
    static constexpr int H = 16;
    float cos_w[W][W] = {};
    float cos_h[H][H] = {};
    float alpha_w[W] = {};
    float alpha_h[H] = {};
};

const DCTBasis& dct_basis() {
    static const DCTBasis basis = [] {
        DCTBasis b{};
        for (int u = 0; u < DCTBasis::W; ++u) {
            b.alpha_w[u] = (u == 0) ? std::sqrt(1.0f / DCTBasis::W) : std::sqrt(2.0f / DCTBasis::W);
            for (int x = 0; x < DCTBasis::W; ++x) {
                b.cos_w[u][x] = std::cos((kPi * (2.0f * x + 1.0f) * u) / (2.0f * DCTBasis::W));
            }
        }
        for (int v = 0; v < DCTBasis::H; ++v) {
            b.alpha_h[v] = (v == 0) ? std::sqrt(1.0f / DCTBasis::H) : std::sqrt(2.0f / DCTBasis::H);
            for (int y = 0; y < DCTBasis::H; ++y) {
                b.cos_h[v][y] = std::cos((kPi * (2.0f * y + 1.0f) * v) / (2.0f * DCTBasis::H));
            }
        }
        return b;
    }();
    return basis;
}

struct GaborBank {
    static constexpr int Radius = 2;
    static constexpr int Size = 2 * Radius + 1;
    static constexpr int Orientations = 4;
    static constexpr int Frequencies = 2;
    float kernel[Frequencies][Orientations][Size][Size] = {};
};

const GaborBank& gabor_bank() {
    static const GaborBank bank = [] {
        GaborBank g{};
        constexpr float kSigma = 1.3f;
        constexpr float kGamma = 0.6f;
        constexpr std::array<float, GaborBank::Orientations> kAngles = {
            0.0f, 0.25f * kPi, 0.5f * kPi, 0.75f * kPi
        };
        constexpr std::array<float, GaborBank::Frequencies> kLambdas = {3.2f, 6.4f};

        for (int fi = 0; fi < GaborBank::Frequencies; ++fi) {
            const float lambda = kLambdas[fi];
            for (int oi = 0; oi < GaborBank::Orientations; ++oi) {
                const float theta = kAngles[oi];
                const float ct = std::cos(theta);
                const float st = std::sin(theta);
                for (int ky = -GaborBank::Radius; ky <= GaborBank::Radius; ++ky) {
                    for (int kx = -GaborBank::Radius; kx <= GaborBank::Radius; ++kx) {
                        const float xr = kx * ct + ky * st;
                        const float yr = -kx * st + ky * ct;
                        const float gauss = std::exp(-(xr * xr + (kGamma * kGamma) * yr * yr) / (2.0f * kSigma * kSigma));
                        const float carrier = std::cos((2.0f * kPi * xr) / lambda);
                        g.kernel[fi][oi][ky + GaborBank::Radius][kx + GaborBank::Radius] = gauss * carrier;
                    }
                }
            }
        }
        return g;
    }();
    return bank;
}

void compute_cell_frequency_signature(const FloatImage& img,
                                      int x0, int y0, int x1, int y1,
                                      float out[kFreqBins]) {
    for (int i = 0; i < kFreqBins; ++i) out[i] = 0.0f;
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);

    constexpr int W = DCTBasis::W;
    constexpr int H = DCTBasis::H;
    const DCTBasis& basis = dct_basis();
    float sample[H][W] = {};
    for (int sy = 0; sy < H; ++sy) {
        for (int sx = 0; sx < W; ++sx) {
            int px = x0 + (sx * w) / W;
            int py = y0 + (sy * h) / H;
            px = std::clamp(px, x0, x1 - 1);
            py = std::clamp(py, y0, y1 - 1);
            sample[sy][sx] = img.get(px, py);
        }
    }

    // Only horizontal frequencies 0..3 contribute to the retained coefficients.
    float row_dct[H][4] = {};
    for (int yy = 0; yy < H; ++yy) {
        for (int u = 0; u < 4; ++u) {
            float sum = 0.0f;
            for (int xx = 0; xx < W; ++xx) {
                sum += sample[yy][xx] * basis.cos_w[u][xx];
            }
            row_dct[yy][u] = sum;
        }
    }

    static constexpr std::array<std::pair<int, int>, kFreqBins> kZigZag = {
        std::pair<int, int>{1, 0}, {0, 1}, {2, 0}, {1, 1},
        {0, 2}, {3, 0}, {2, 1}, {0, 3}
    };
    for (int i = 0; i < kFreqBins; ++i) {
        const auto [u, v] = kZigZag[i];
        float sum = 0.0f;
        for (int yy = 0; yy < H; ++yy) {
            sum += row_dct[yy][u] * basis.cos_h[v][yy];
        }
        out[i] = basis.alpha_w[u] * basis.alpha_h[v] * sum;
    }

    float norm = 0.0f;
    for (int i = 0; i < kFreqBins; ++i) norm += out[i] * out[i];
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < kFreqBins; ++i) out[i] /= norm;
    }
}

void compute_cell_texture_signature(const FloatImage& img,
                                    int x0, int y0, int x1, int y1,
                                    float out[kTextureBins]) {
    for (int i = 0; i < kTextureBins; ++i) out[i] = 0.0f;
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);
    if (w < 5 || h < 5) return;

    constexpr int kRadius = GaborBank::Radius;
    const GaborBank& bank = gabor_bank();
    const float* src = img.data();
    const int stride = img.width();

    for (int fi = 0; fi < GaborBank::Frequencies; ++fi) {
        for (int oi = 0; oi < GaborBank::Orientations; ++oi) {
            float energy = 0.0f;
            int samples = 0;

            for (int y = y0 + kRadius; y < y1 - kRadius; ++y) {
                int x = x0 + kRadius;
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
                for (; x <= x1 - kRadius - 4; x += 4) {
                    __m128 responses = _mm_setzero_ps();
                    const float* row_base = src + static_cast<size_t>(y) * stride + x;
                    for (int ky = -kRadius; ky <= kRadius; ++ky) {
                        const float* src_row = row_base + ky * stride;
                        const float* kernel_row = bank.kernel[fi][oi][ky + kRadius];
                        for (int kx = -kRadius; kx <= kRadius; ++kx) {
                            responses = _mm_add_ps(responses, _mm_mul_ps(
                                _mm_set1_ps(kernel_row[kx + kRadius]), _mm_loadu_ps(src_row + kx)));
                        }
                    }
                    float values[4];
                    _mm_storeu_ps(values, responses);
                    // Keep the scalar left-to-right energy reduction unchanged.
                    for (int lane = 0; lane < 4; ++lane) energy += std::abs(values[lane]);
                    samples += 4;
                }
#endif
                for (; x < x1 - kRadius; ++x) {
                    float resp = 0.0f;
                    const float* row_base = src + static_cast<size_t>(y) * stride + x;
                    for (int ky = -kRadius; ky <= kRadius; ++ky) {
                        const float* src_row = row_base + ky * stride;
                        const float* kernel_row = bank.kernel[fi][oi][ky + kRadius];
                        for (int kx = -kRadius; kx <= kRadius; ++kx) {
                            resp += kernel_row[kx + kRadius] * src_row[kx];
                        }
                    }
                    energy += std::abs(resp);
                    samples++;
                }
            }
            out[fi * GaborBank::Orientations + oi] = samples > 0 ? (energy / samples) : 0.0f;
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < kTextureBins; ++i) norm += out[i] * out[i];
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < kTextureBins; ++i) out[i] /= norm;
    }
}

int estimate_adaptive_level(float variance, float edge_occ, float coherence) {
    float score = 0.55f * std::sqrt(std::max(variance, 0.0f)) +
                  0.30f * edge_occ +
                  0.15f * coherence;
    if (score > 0.28f) return 2;
    if (score > 0.14f) return 1;
    return 0;
}

void assign_quadtree_levels(std::vector<CellStats>& stats, int cols, int rows,
                            int x0, int y0, int x1, int y1,
                            int depth, int max_depth, float var_threshold) {
    if (x0 >= x1 || y0 >= y1) return;

    float mean_var = 0.0f;
    int count = 0;
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const auto& s = stats[static_cast<size_t>(y) * cols + x];
            mean_var += s.luminance_variance;
            count++;
        }
    }
    mean_var = count > 0 ? (mean_var / count) : 0.0f;

    bool leaf = (depth >= max_depth) || (mean_var < var_threshold) || ((x1 - x0) <= 1 && (y1 - y0) <= 1);
    if (leaf) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                auto& s = stats[static_cast<size_t>(y) * cols + x];
                s.adaptive_level = std::max(s.adaptive_level, depth);
            }
        }
        return;
    }

    int xm = (x0 + x1) / 2;
    int ym = (y0 + y1) / 2;
    if (xm == x0 && x1 - x0 > 1) xm++;
    if (ym == y0 && y1 - y0 > 1) ym++;
    assign_quadtree_levels(stats, cols, rows, x0, y0, xm, ym, depth + 1, max_depth, var_threshold);
    assign_quadtree_levels(stats, cols, rows, xm, y0, x1, ym, depth + 1, max_depth, var_threshold);
    assign_quadtree_levels(stats, cols, rows, x0, ym, xm, y1, depth + 1, max_depth, var_threshold);
    assign_quadtree_levels(stats, cols, rows, xm, ym, x1, y1, depth + 1, max_depth, var_threshold);
}

}  // namespace

IntegralImage::IntegralImage(const FloatImage& input) {
    compute(input);
}

void IntegralImage::compute(const FloatImage& input) {
    width_ = input.width();
    height_ = input.height();
    
    data_.assign((static_cast<size_t>(width_) + 1) * (static_cast<size_t>(height_) + 1), 0.0);
    
    for (int y = 0; y < height_; ++y) {
        double row_sum = 0.0;
        for (int x = 0; x < width_; ++x) {
            row_sum += input.get(x, y);
            data_[(y + 1) * (width_ + 1) + (x + 1)] = 
                data_[y * (width_ + 1) + (x + 1)] + row_sum;
        }
    }
}

float IntegralImage::sum(int x0, int y0, int x1, int y1) const {
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(width_, x1);
    y1 = std::min(height_, y1);
    
    if (x0 >= x1 || y0 >= y1) return 0.0f;
    
    double a = data_[y0 * (width_ + 1) + x0];
    double b = data_[y0 * (width_ + 1) + x1];
    double c = data_[y1 * (width_ + 1) + x0];
    double d = data_[y1 * (width_ + 1) + x1];
    
    return static_cast<float>(d - b - c + a);
}

float IntegralImage::mean(int x0, int y0, int x1, int y1) const {
    x0 = std::max(0, x0);
    y0 = std::max(0, y0);
    x1 = std::min(width_, x1);
    y1 = std::min(height_, y1);
    if (x0 >= x1 || y0 >= y1) return 0.0f;
    const double area = static_cast<double>(x1 - x0) * (y1 - y0);
    return static_cast<float>(sum(x0, y0, x1, y1) / area);
}

CellStatsAggregator::CellStatsAggregator(const Config& config) {
    set_config(config);
}

void CellStatsAggregator::set_config(const Config& config) {
    if (config.cell_width <= 0 || config.cell_height <= 0) {
        throw std::invalid_argument("Cell dimensions must be positive");
    }
    if (config.orientation_bins < 1 || config.orientation_bins > 8) {
        throw std::invalid_argument("Cell orientation bins must be between 1 and 8");
    }
    config_ = config;
}

int CellStatsAggregator::grid_cols(int image_width) const {
    if (image_width < 0) throw std::invalid_argument("Image width must be non-negative");
    return image_width == 0 ? 0 : 1 + (image_width - 1) / config_.cell_width;
}

int CellStatsAggregator::grid_rows(int image_height) const {
    if (image_height < 0) throw std::invalid_argument("Image height must be non-negative");
    return image_height == 0 ? 0 : 1 + (image_height - 1) / config_.cell_height;
}

void CellStatsAggregator::compute_orientation_histogram(const FloatImage& gx, const FloatImage& gy,
                                                         int x0, int y0, int x1, int y1,
                                                         float* histogram, int bins) const {
    std::fill(histogram, histogram + bins, 0.0f);
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            float grad_x = gx.get(x, y);
            float grad_y = gy.get(x, y);
            float mag = std::sqrt(grad_x * grad_x + grad_y * grad_y);
            
            if (mag < 0.001f) continue;
            
            float angle = std::atan2(grad_y, grad_x);
            angle += kPi;
            
            int bin = static_cast<int>(angle / (2.0f * kPi) * bins);
            bin = bin % bins;
            
            histogram[bin] += mag;
        }
    }
    
    float total = 0.0f;
    for (int i = 0; i < bins; ++i) {
        total += histogram[i];
    }
    if (total > 0.0f) {
        for (int i = 0; i < bins; ++i) {
            histogram[i] /= total;
        }
    }
}

void CellStatsAggregator::compute_structure_tensor(const FloatImage& gx, const FloatImage& gy,
                                                    int x0, int y0, int x1, int y1,
                                                    float& coherence, float& orientation) const {
    double Jxx = 0.0, Jxy = 0.0, Jyy = 0.0;
    int count = 0;
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            float grad_x = gx.get(x, y);
            float grad_y = gy.get(x, y);
            
            Jxx += grad_x * grad_x;
            Jxy += grad_x * grad_y;
            Jyy += grad_y * grad_y;
            count++;
        }
    }
    
    if (count > 0) {
        Jxx /= count;
        Jxy /= count;
        Jyy /= count;
    }
    
    double trace = Jxx + Jyy;
    double det = Jxx * Jyy - Jxy * Jxy;
    double discriminant = std::max(0.0, trace * trace - 4.0 * det);
    
    double lambda1 = (trace + std::sqrt(discriminant)) / 2.0;
    double lambda2 = (trace - std::sqrt(discriminant)) / 2.0;
    
    constexpr double eps = 1e-8;
    coherence = static_cast<float>((lambda1 - lambda2) / (lambda1 + lambda2 + eps));
    coherence = std::clamp(coherence, 0.0f, 1.0f);
    
    orientation = static_cast<float>(0.5 * std::atan2(2.0 * Jxy, Jxx - Jyy));
}

std::vector<CellStats> CellStatsAggregator::compute(const FloatImage& luminance, const EdgeData& edges,
                                                     const FrameBuffer* color, const GradientData* grad) const {
    if (luminance.empty()) return {};
    int cols = grid_cols(luminance.width());
    int rows = grid_rows(luminance.height());
    
    std::vector<CellStats> result(static_cast<size_t>(cols) * rows);
    
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(dynamic)
#endif
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int x0 = col * config_.cell_width;
            int y0 = row * config_.cell_height;
            int x1 = x0 + std::min(config_.cell_width, luminance.width() - x0);
            int y1 = y0 + std::min(config_.cell_height, luminance.height() - y0);
            
            CellStats& stats = result[static_cast<size_t>(row) * cols + col];
            
            int count = (x1 - x0) * (y1 - y0);
            if (count <= 0) continue;
            
            double luminance_sum = 0.0, luminance_square_sum = 0.0, magnitude_sum = 0.0;
            float max_mag = 0.0f;
            int edge_count = 0;
            for (int y = y0; y < y1; ++y) {
                for (int x = x0; x < x1; ++x) {
                    const float value = luminance.get(x, y);
                    luminance_sum += value;
                    luminance_square_sum += value * value;
                    if (!edges.magnitude.empty()) {
                        const float magnitude = edges.magnitude.get(x, y);
                        magnitude_sum += magnitude;
                        max_mag = std::max(max_mag, magnitude);
                        if (edges.is_edge(x, y)) ++edge_count;
                    }
                }
            }
            // Preserve the integral mean's float rounding before division.
            const auto mean = [count](double sum) {
                return static_cast<float>(static_cast<float>(sum) / static_cast<double>(count));
            };
            stats.mean_luminance = mean(luminance_sum);
            const float mean_sq = mean(luminance_square_sum);
            stats.luminance_variance = std::max(0.0f, mean_sq - stats.mean_luminance * stats.mean_luminance);
            stats.local_contrast = std::sqrt(stats.luminance_variance);
            if (!edges.magnitude.empty()) {
                stats.edge_strength = mean(magnitude_sum);
                stats.edge_strength_max = max_mag;
                stats.edge_occupancy = static_cast<float>(edge_count) / count;
                stats.is_edge_cell = stats.edge_occupancy >= config_.edge_threshold || max_mag >= config_.edge_threshold;
            }
            if (grad) {
                if (config_.enable_orientation_histogram) {
                    compute_orientation_histogram(grad->gx, grad->gy, x0, y0, x1, y1,
                                                  stats.orientation_histogram, config_.orientation_bins);
                }
                
                compute_structure_tensor(grad->gx, grad->gy, x0, y0, x1, y1,
                                         stats.structure_coherence, stats.dominant_orientation);
                
                float sum_gx = 0.0f, sum_gy = 0.0f;
                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        sum_gx += grad->gx.get(x, y);
                        sum_gy += grad->gy.get(x, y);
                    }
                }
                stats.mean_gx = sum_gx / count;
                stats.mean_gy = sum_gy / count;
                stats.cell_orientation = std::atan2(stats.mean_gy, stats.mean_gx);
            }

            if (config_.enable_frequency_signature) {
                compute_cell_frequency_signature(luminance, x0, y0, x1, y1, stats.frequency_signature);
            }
            if (config_.enable_texture_signature) {
                compute_cell_texture_signature(luminance, x0, y0, x1, y1, stats.texture_signature);
            }
            stats.adaptive_level = estimate_adaptive_level(
                stats.luminance_variance,
                stats.edge_occupancy,
                stats.structure_coherence
            );
            
            if (color) {
                float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;
                for (int y = y0; y < y1; ++y) {
                    for (int x = x0; x < x1; ++x) {
                        Color c = color->get_pixel(x, y);
                        LinearColor linear = ColorSpace::srgb_to_linear(c.r, c.g, c.b);
                        sum_r += linear.r;
                        sum_g += linear.g;
                        sum_b += linear.b;
                    }
                }
                stats.mean_r = sum_r / count;
                stats.mean_g = sum_g / count;
                stats.mean_b = sum_b / count;
            }
        }
    }

    if (config_.quad_tree_adaptive && !result.empty()) {
        assign_quadtree_levels(
            result, cols, rows,
            0, 0, cols, rows,
            0, std::max(0, config_.quad_tree_max_depth),
            config_.quad_tree_variance_threshold
        );
    }
    
    return result;
}

}
