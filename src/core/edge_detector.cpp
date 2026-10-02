#include "edge_detector.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <numeric>
#include <cmath>
#include <utility>
#include <memory>

#if defined(__AVX2__) || defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <immintrin.h>
#endif

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace ascii {

namespace {

constexpr int kCacheTile = 64;

std::vector<bool> connect_weak_edges(std::vector<uint8_t>& state,
                                    std::vector<int>& pending, int w, int h) {
    // 0 = rejected, 1 = weak, 2 = accepted. Mark before enqueue so each pixel
    // enters the queue at most once; an all-rejected image allocates no queue.
    std::vector<bool> result(state.size(), false);
    for (size_t head = 0; head < pending.size(); ++head) {
        const int index = pending[head];
        result[index] = true;
        const int cy = index / w;
        const int cx = index - cy * w;
        for (int y = std::max(0, cy - 1); y <= std::min(h - 1, cy + 1); ++y) {
            for (int x = std::max(0, cx - 1); x <= std::min(w - 1, cx + 1); ++x) {
                const int next = y * w + x;
                if (state[next] == 1) {
                    state[next] = 2;
                    pending.push_back(next);
                }
            }
        }
    }
    return result;
}

}  // namespace

EdgeDetector::EdgeDetector(const Config& config) : config_(config) {}

GradientData EdgeDetector::compute_gradients(const FloatImage& input) {
    GradientData result;

    FloatImage diffused;
    const bool use_diffusion = config_.use_anisotropic_diffusion && config_.diffusion_iterations > 0;
    if (use_diffusion) {
        diffused = anisotropic_diffusion(
            input,
            config_.diffusion_iterations,
            config_.diffusion_kappa,
            config_.diffusion_lambda
        );
    }
    const FloatImage& working = use_diffusion ? diffused : input;

    FloatImage blurred = gaussian_blur(working, config_.blur_sigma);
    
    sobel(blurred, result.gx, result.gy);
    
    int w = blurred.width();
    int h = blurred.height();
    result.magnitude = FloatImage(w, h);
    result.orientation = FloatImage(w, h);
    if (blurred.empty()) return result;
    const float* gx_data = result.gx.data();
    const float* gy_data = result.gy.data();
    float* mag_data = result.magnitude.data();
    float* ori_data = result.orientation.data();
    
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int ty = 0; ty < h; ty += kCacheTile) {
        int y_end = std::min(ty + kCacheTile, h);
        for (int y = ty; y < y_end; ++y) {
            int base = y * w;
            int x = 0;

#if defined(__AVX2__)
            for (; x + 7 < w; x += 8) {
                __m256 vx = _mm256_loadu_ps(gx_data + base + x);
                __m256 vy = _mm256_loadu_ps(gy_data + base + x);
                __m256 mag = _mm256_sqrt_ps(_mm256_add_ps(_mm256_mul_ps(vx, vx), _mm256_mul_ps(vy, vy)));
                _mm256_storeu_ps(mag_data + base + x, mag);
            }
#endif

#if !defined(__AVX2__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
            for (; x + 3 < w; x += 4) {
                __m128 vx = _mm_loadu_ps(gx_data + base + x);
                __m128 vy = _mm_loadu_ps(gy_data + base + x);
                __m128 mag = _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(vx, vx), _mm_mul_ps(vy, vy)));
                _mm_storeu_ps(mag_data + base + x, mag);
            }
#endif

            for (; x < w; ++x) {
                float gxv = gx_data[base + x];
                float gyv = gy_data[base + x];
                mag_data[base + x] = std::sqrt(gxv * gxv + gyv * gyv);
            }

            for (int ox = 0; ox < w; ++ox) {
                ori_data[base + ox] = std::atan2(gy_data[base + ox], gx_data[base + ox]);
            }
        }
    }
    
    return result;
}

MultiScaleGradientData EdgeDetector::compute_multi_scale_gradients(const FloatImage& input) {
    MultiScaleGradientData result;
    int w = input.width();
    int h = input.height();

    FloatImage diffused;
    const bool use_diffusion = config_.use_anisotropic_diffusion && config_.diffusion_iterations > 0;
    if (use_diffusion) {
        diffused = anisotropic_diffusion(
            input,
            config_.diffusion_iterations,
            config_.diffusion_kappa,
            config_.diffusion_lambda
        );
    }
    const FloatImage& working = use_diffusion ? diffused : input;

    // Compare the two configured normalized-Laplacian responses, retaining the
    // first scale on ties. Variance can override this choice in flat/detail regions.
    const float sigma0 = std::max(0.1f, config_.scale_sigma_0);
    const float sigmaN = std::max(sigma0 + 1e-4f, config_.scale_sigma_1);
    const std::array<float, 2> sigmas = {sigma0, sigmaN};
    std::array<FloatImage, 2> gx, gy, norm_lap;
    const FloatImage variance = local_variance_3x3(working);

    for (size_t s = 0; s < sigmas.size(); ++s) {
        const float effective_sigma = std::sqrt(
            sigmas[s] * sigmas[s] + config_.blur_sigma * config_.blur_sigma);
        const FloatImage blurred = gaussian_blur(working, effective_sigma);
        sobel(blurred, gx[s], gy[s]);
        norm_lap[s] = FloatImage(w, h);
        if (working.empty()) continue;
        const float* pixels = blurred.data();
        float* responses = norm_lap[s].data();
        const float sigma_squared = sigmas[s] * sigmas[s];

#ifdef HAS_OPENMP
        #pragma omp parallel for
#endif
        for (int y = 1; y < h - 1; ++y) {
            const int row = y * w;
            int x = 1;
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
            const __m128 four = _mm_set1_ps(4.0f);
            const __m128 scale = _mm_set1_ps(sigma_squared);
            const __m128 sign = _mm_set1_ps(-0.0f);
            for (; x < w - 4; x += 4) {
                const int i = row + x;
                __m128 sum = _mm_add_ps(_mm_loadu_ps(pixels + i - 1), _mm_loadu_ps(pixels + i + 1));
                sum = _mm_add_ps(sum, _mm_loadu_ps(pixels + i - w));
                sum = _mm_add_ps(sum, _mm_loadu_ps(pixels + i + w));
                const __m128 lap = _mm_sub_ps(sum, _mm_mul_ps(four, _mm_loadu_ps(pixels + i)));
                _mm_storeu_ps(responses + i, _mm_mul_ps(scale, _mm_andnot_ps(sign, lap)));
            }
#endif
            for (; x < w - 1; ++x) {
                const int i = row + x;
                const float lap = pixels[i - 1] + pixels[i + 1] + pixels[i - w] +
                    pixels[i + w] - 4.0f * pixels[i];
                responses[i] = sigma_squared * std::abs(lap);
            }
        }
    }

    const float variance_span = std::max(
        config_.scale_variance_ceil - config_.scale_variance_floor, 1e-8f);
    const float* variance_data = variance.data();
    const float* response0 = norm_lap[0].data();
    const float* response1 = norm_lap[1].data();
    const std::array<const float*, 2> gx_data = {gx[0].data(), gx[1].data()};
    const std::array<const float*, 2> gy_data = {gy[0].data(), gy[1].data()};
    // Each source pixel is consumed once before these buffers are overwritten.
    result.magnitude = std::move(norm_lap[0]);
    result.orientation = std::move(norm_lap[1]);
    result.gx = std::move(gx[0]);
    result.gy = std::move(gy[0]);
    result.best_scale = config_.adaptive_scale_selection ? -1 : 0;
    if (working.empty()) return result;
    float* out_gx = result.gx.data();
    float* out_gy = result.gy.data();
    float* out_magnitude = result.magnitude.data();
    float* out_orientation = result.orientation.data();

#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            int best_scale = 0;
            if (config_.adaptive_scale_selection) {
                const float detail = std::clamp(
                    (variance_data[i] - config_.scale_variance_floor) / variance_span,
                    0.0f, 1.0f);
                if (detail <= 0.33f) best_scale = 1;
                else if (detail >= 0.67f) best_scale = 0;
                else best_scale = response1[i] > response0[i] ? 1 : 0;
            }

            const float sx = gx_data[best_scale][i];
            const float sy = gy_data[best_scale][i];
            out_gx[i] = sx;
            out_gy[i] = sy;
            out_magnitude[i] = std::sqrt(sx * sx + sy * sy);
            out_orientation[i] = std::atan2(sy, sx);
        }
    }

    return result;
}

float EdgeDetector::compute_global_percentile_threshold(const FloatImage& magnitude, float percentile) {
    if (!std::isfinite(percentile) || percentile < 0.0f || percentile > 1.0f) {
        throw std::invalid_argument("Edge percentile must be finite and between zero and one");
    }
    if (magnitude.empty()) return 0.1f;
    int w = magnitude.width();
    int h = magnitude.height();
    
    std::vector<float> values;
    values.reserve(checked_image_size(w, h));
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = magnitude.get(x, y);
            if (v > 0.001f) {
                values.push_back(v);
            }
        }
    }
    
    if (values.empty()) return 0.1f;
    
    size_t idx = static_cast<size_t>(values.size() * percentile);
    if (idx >= values.size()) idx = values.size() - 1;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx), values.end());
    return values[idx];
}

float EdgeDetector::compute_tile_threshold(const FloatImage& magnitude, int x0, int y0, int tile_size,
                                            int img_w, int img_h, float percentile) {
    if (tile_size <= 0 || img_w < 0 || img_h < 0 ||
        !std::isfinite(percentile) || percentile < 0.0f || percentile > 1.0f) {
        throw std::invalid_argument("Invalid edge tile size, bounds, or percentile");
    }
    img_w = std::min(img_w, magnitude.width());
    img_h = std::min(img_h, magnitude.height());
    const int x1 = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(x0) + tile_size, 0, img_w));
    const int y1 = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(y0) + tile_size, 0, img_h));
    x0 = std::clamp(x0, 0, img_w);
    y0 = std::clamp(y0, 0, img_h);
    if (x0 >= x1 || y0 >= y1) return 0.1f;
    std::vector<float> values;
    values.reserve(checked_image_size(x1 - x0, y1 - y0));
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            float v = magnitude.get(x, y);
            if (v > 0.001f) {
                values.push_back(v);
            }
        }
    }
    
    if (values.empty()) return 0.1f;
    
    size_t idx = static_cast<size_t>(values.size() * percentile);
    if (idx >= values.size()) idx = values.size() - 1;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(idx), values.end());
    return values[idx];
}

FloatImage EdgeDetector::compute_adaptive_threshold_map(const FloatImage& magnitude, int tile_size,
                                                         float percentile, float floor) {
    if (tile_size <= 0 || !std::isfinite(percentile) || percentile < 0.0f || percentile > 1.0f ||
        !std::isfinite(floor)) {
        throw std::invalid_argument("Invalid adaptive edge tile size, percentile, or floor");
    }
    int w = magnitude.width();
    int h = magnitude.height();
    const int tw = w / tile_size + (w % tile_size != 0);
    const int th = h / tile_size + (h % tile_size != 0);
    const size_t tiles = checked_image_size(tw, th);
    if (tiles > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("Adaptive edge grid exceeds supported tile count");
    }
    FloatImage result(tw, th);

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int tile = 0; tile < static_cast<int>(tiles); ++tile) {
        const int tx = tile % tw;
        const int ty = tile / tw;
        const int x0 = tx * tile_size;
        const int y0 = ty * tile_size;
        const float local_thresh = compute_tile_threshold(
            magnitude, x0, y0, tile_size, w, h, percentile);
        result.set(tx, ty, std::max(local_thresh, floor));
    }
    
    return result;
}

EdgeData EdgeDetector::detect(const FloatImage& input, GradientData* selected_gradients) {
    if (input.size_in_elements() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("Edge image exceeds supported pixel count");
    }
    EdgeData result;
    
    GradientData grad;
    MultiScaleGradientData ms_grad;
    
    if (config_.multi_scale) {
        ms_grad = compute_multi_scale_gradients(input);
        result.magnitude = std::move(ms_grad.magnitude);
        result.orientation = std::move(ms_grad.orientation);
        if (selected_gradients) {
            selected_gradients->gx = std::move(ms_grad.gx);
            selected_gradients->gy = std::move(ms_grad.gy);
        }
    } else {
        grad = compute_gradients(input);
        result.magnitude = std::move(grad.magnitude);
        result.orientation = std::move(grad.orientation);
        if (selected_gradients) {
            selected_gradients->gx = std::move(grad.gx);
            selected_gradients->gy = std::move(grad.gy);
        }
    }
    
    FloatImage nms = non_maximum_suppression(result.magnitude, result.orientation);
    
    int w = result.magnitude.width();
    int h = result.magnitude.height();
    
    float low_thresh = config_.low_threshold;
    float high_thresh = config_.high_threshold;
    const float low_ratio = high_thresh > 0.0f
        ? std::clamp(low_thresh / high_thresh, 0.0f, 1.0f) : 0.0f;
    
    if (config_.adaptive_mode == "global") {
        high_thresh = compute_global_percentile_threshold(nms, config_.global_percentile);
        high_thresh = std::max({high_thresh, config_.dark_scene_floor, config_.high_threshold});
        low_thresh = high_thresh * low_ratio;
    } else if (config_.adaptive_mode == "local" || config_.adaptive_mode == "hybrid") {
        const float configured_floor = std::max(config_.dark_scene_floor, config_.high_threshold);
        FloatImage thresh_map = compute_adaptive_threshold_map(nms, config_.tile_size,
                                                                config_.global_percentile,
                                                                configured_floor);
        const float global_high = std::max(
            compute_global_percentile_threshold(nms, config_.global_percentile), configured_floor);
        
        const size_t count = nms.size_in_elements();
        if (count == 0) return result;
        std::vector<uint8_t> state(config_.use_hysteresis ? count : 0, 0);
        std::vector<int> pending;
        if (!config_.use_hysteresis) result.edge_mask.resize(count, false);
        for (int y = 0; y < h; ++y) {
            const int ty = y / config_.tile_size;
            for (int tx = 0, x0 = 0; x0 < w; ++tx) {
                const int x1 = x0 + std::min(config_.tile_size, w - x0);
                float local_high = thresh_map.get(tx, ty);
                if (config_.adaptive_mode == "hybrid") {
                    local_high = 0.5f * local_high + 0.5f * global_high;
                }
                const float local_low = local_high * low_ratio;
                for (int x = x0; x < x1; ++x) {
                    const int i = y * w + x;
                    const float mag = nms.data()[i];
                    if (!config_.use_hysteresis) result.edge_mask[i] = mag >= local_high;
                    else if (mag >= local_high) {
                        state[i] = 2;
                        pending.push_back(i);
                    } else if (mag >= local_low) state[i] = 1;
                }
                x0 = x1;
            }
        }
        if (config_.use_hysteresis) {
            result.edge_mask = connect_weak_edges(state, pending, w, h);
        }
        
        return result;
    }
    
    if (config_.use_hysteresis) {
        result.edge_mask = hysteresis_threshold(nms, w, h, low_thresh, high_thresh);
    } else {
        result.edge_mask.resize(w * h, false);
        for (int i = 0; i < w * h; ++i) {
            result.edge_mask[i] = nms.data()[i] >= high_thresh;
        }
    }
    
    return result;
}

FloatImage EdgeDetector::anisotropic_diffusion(const FloatImage& input, int iterations, float kappa, float lambda) {
    int w = input.width();
    int h = input.height();
    if (w <= 2 || h <= 2 || iterations <= 0) {
        return input;
    }

    FloatImage curr = input;
    FloatImage next(w, h);

    const float inv_kappa2 = 1.0f / (kappa * kappa + 1e-8f);
    for (int it = 0; it < iterations; ++it) {
#ifdef HAS_OPENMP
        #pragma omp parallel for
#endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float c = curr.get(x, y);
                float n = curr.get_clamped(x, y - 1);
                float s = curr.get_clamped(x, y + 1);
                float e = curr.get_clamped(x + 1, y);
                float wv = curr.get_clamped(x - 1, y);

                float dn = n - c;
                float ds = s - c;
                float de = e - c;
                float dw = wv - c;

                float cn = std::exp(-(dn * dn) * inv_kappa2);
                float cs = std::exp(-(ds * ds) * inv_kappa2);
                float ce = std::exp(-(de * de) * inv_kappa2);
                float cw = std::exp(-(dw * dw) * inv_kappa2);

                float update = cn * dn + cs * ds + ce * de + cw * dw;
                next.set(x, y, c + lambda * update);
            }
        }
        curr = next;
    }

    return curr;
}

FloatImage EdgeDetector::local_variance_3x3(const FloatImage& input) {
    int w = input.width();
    int h = input.height();
    FloatImage var(w, h, 0.0f);
    if (w <= 0 || h <= 0) {
        return var;
    }

#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        const std::array<const float*, 3> rows = {
            input.data() + static_cast<size_t>(std::max(0, y - 1)) * w,
            input.data() + static_cast<size_t>(y) * w,
            input.data() + static_cast<size_t>(std::min(h - 1, y + 1)) * w
        };
        const auto scalar_variance = [&](int x) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (const float* row : rows) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const float v = row[std::clamp(x + dx, 0, w - 1)];
                    sum += v;
                    sum_sq += v * v;
                }
            }
            const float mean = sum / 9.0f;
            return std::max(0.0f, sum_sq / 9.0f - mean * mean);
        };
        float* out = var.data() + static_cast<size_t>(y) * w;
        out[0] = scalar_variance(0);
        int x = 1;
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
        const __m128 nine = _mm_set1_ps(9.0f);
        for (; x < w - 4; x += 4) {
            __m128 sum = _mm_setzero_ps();
            __m128 sum_sq = _mm_setzero_ps();
            for (const float* row : rows) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const __m128 v = _mm_loadu_ps(row + x + dx);
                    sum = _mm_add_ps(sum, v);
                    sum_sq = _mm_add_ps(sum_sq, _mm_mul_ps(v, v));
                }
            }
            const __m128 mean = _mm_div_ps(sum, nine);
            const __m128 variance = _mm_sub_ps(_mm_div_ps(sum_sq, nine), _mm_mul_ps(mean, mean));
            _mm_storeu_ps(out + x, _mm_max_ps(variance, _mm_setzero_ps()));
        }
#endif
        for (; x < w; ++x) out[x] = scalar_variance(x);
    }
    return var;
}

FloatImage EdgeDetector::gaussian_blur(const FloatImage& input, float sigma) {
    int w = input.width();
    int h = input.height();
    if (!std::isfinite(sigma)) {
        throw std::invalid_argument("Gaussian sigma must be finite");
    }
    // Below this scale all noncentral coefficients round to zero in float.
    if (sigma <= 1e-4f) return input;
    const double radius_value = std::ceil(static_cast<double>(sigma * 3.0f));
    if (radius_value > (std::numeric_limits<int>::max() - 1) / 2) {
        throw std::invalid_argument("Gaussian kernel radius exceeds supported size");
    }
    if (input.empty()) return input;
    int radius = static_cast<int>(radius_value);
    int ksize = 2 * radius + 1;
    
    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    for (int i = 0; i < ksize; ++i) {
        float x = static_cast<float>(i - radius);
        kernel[i] = std::exp(-x * x / (2 * sigma * sigma));
        sum += kernel[i];
    }
    if (sum > 1e-12f) {
        for (float& k : kernel) {
            k /= sum;
        }
    }

#if defined(__AVX2__)
    std::vector<__m256> kernel_avx(static_cast<size_t>(ksize));
    for (int i = 0; i < ksize; ++i) {
        kernel_avx[static_cast<size_t>(i)] = _mm256_set1_ps(kernel[static_cast<size_t>(i)]);
    }
#endif

#if !defined(__AVX2__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
    std::vector<__m128> kernel_sse(static_cast<size_t>(ksize));
    for (int i = 0; i < ksize; ++i) {
        kernel_sse[static_cast<size_t>(i)] = _mm_set1_ps(kernel[static_cast<size_t>(i)]);
    }
#endif

    // Separable Gaussian: horizontal pass then vertical pass.
    // Border handling uses clamped replication to preserve energy.
    auto temp = std::make_unique_for_overwrite<float[]>(input.size_in_elements());
    const float* in_data = input.data();
    float* temp_data = temp.get();
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        const float* in_row = in_data + static_cast<size_t>(y) * w;
        float* out_row = temp_data + static_cast<size_t>(y) * w;

        int x = 0;
        const int interior_begin = std::min(std::max(radius, 0), w);
        const int interior_end = std::max(interior_begin, w - radius);

        // Left border (clamped)
        for (; x < interior_begin; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int nx = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(x) + k, 0, w - 1));
                acc += in_row[nx] * kernel[k + radius];
            }
            out_row[x] = acc;
        }

#if defined(__AVX2__)
        for (; x + 7 < interior_end; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int k = -radius; k <= radius; ++k) {
                __m256 s = _mm256_loadu_ps(in_row + x + k);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(s, kernel_avx[static_cast<size_t>(k + radius)]));
            }
            _mm256_storeu_ps(out_row + x, acc);
        }
#endif

#if !defined(__AVX2__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
        for (; x + 3 < interior_end; x += 4) {
            __m128 acc = _mm_setzero_ps();
            for (int k = -radius; k <= radius; ++k) {
                __m128 s = _mm_loadu_ps(in_row + x + k);
                acc = _mm_add_ps(acc, _mm_mul_ps(s, kernel_sse[static_cast<size_t>(k + radius)]));
            }
            _mm_storeu_ps(out_row + x, acc);
        }
#endif

        // Scalar interior remainder (no clamping needed)
        for (; x < interior_end; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                acc += in_row[x + k] * kernel[k + radius];
            }
            out_row[x] = acc;
        }

        // Right border (clamped)
        for (; x < w; ++x) {
            float acc = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int nx = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(x) + k, 0, w - 1));
                acc += in_row[nx] * kernel[k + radius];
            }
            out_row[x] = acc;
        }
    }

    FloatImage result(w, h, 0.0f);
    const float* temp_ro = temp.get();
    float* out_data = result.data();
    std::vector<const float*> row_ptrs(static_cast<size_t>(ksize), nullptr);
#ifdef HAS_OPENMP
    #pragma omp parallel for firstprivate(row_ptrs)
#endif
    for (int y = 0; y < h; ++y) {
        for (int k = -radius; k <= radius; ++k) {
            int ny = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(y) + k, 0, h - 1));
            row_ptrs[k + radius] = temp_ro + static_cast<size_t>(ny) * w;
        }

        float* out_row = out_data + static_cast<size_t>(y) * w;
        int x = 0;

#if defined(__AVX2__)
        for (; x + 7 < w; x += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ki = 0; ki < ksize; ++ki) {
                __m256 s = _mm256_loadu_ps(row_ptrs[ki] + x);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(s, kernel_avx[static_cast<size_t>(ki)]));
            }
            _mm256_storeu_ps(out_row + x, acc);
        }
#endif

#if !defined(__AVX2__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
        for (; x + 3 < w; x += 4) {
            __m128 acc = _mm_setzero_ps();
            for (int ki = 0; ki < ksize; ++ki) {
                __m128 s = _mm_loadu_ps(row_ptrs[ki] + x);
                acc = _mm_add_ps(acc, _mm_mul_ps(s, kernel_sse[static_cast<size_t>(ki)]));
            }
            _mm_storeu_ps(out_row + x, acc);
        }
#endif

        for (; x < w; ++x) {
            float acc = 0.0f;
            for (int ki = 0; ki < ksize; ++ki) {
                acc += row_ptrs[ki][x] * kernel[ki];
            }
            out_row[x] = acc;
        }
    }

    return result;
}

void EdgeDetector::sobel(const FloatImage& input, FloatImage& gx, FloatImage& gy) {
    int w = input.width();
    int h = input.height();
    
    gx = FloatImage(w, h);
    gy = FloatImage(w, h);

    if (w < 3 || h < 3) {
        return;
    }

    const float* src = input.data();
    float* gx_data = gx.data();
    float* gy_data = gy.data();

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int ty = 1; ty < h - 1; ty += kCacheTile) {
        int y_end = std::min(ty + kCacheTile, h - 1);
        for (int y = ty; y < y_end; ++y) {
            const float* row0 = src + static_cast<size_t>(y - 1) * w;
            const float* row1 = src + static_cast<size_t>(y) * w;
            const float* row2 = src + static_cast<size_t>(y + 1) * w;
            float* gx_row = gx_data + static_cast<size_t>(y) * w;
            float* gy_row = gy_data + static_cast<size_t>(y) * w;

            int x = 1;

#if defined(__AVX2__)
                const __m256 v_two = _mm256_set1_ps(2.0f);
                const __m256 v_quarter = _mm256_set1_ps(0.25f);
                for (; x + 7 < w - 1; x += 8) {
                    __m256 tl = _mm256_loadu_ps(row0 + x - 1);
                    __m256 tc = _mm256_loadu_ps(row0 + x);
                    __m256 tr = _mm256_loadu_ps(row0 + x + 1);
                    __m256 ml = _mm256_loadu_ps(row1 + x - 1);
                    __m256 mr = _mm256_loadu_ps(row1 + x + 1);
                    __m256 bl = _mm256_loadu_ps(row2 + x - 1);
                    __m256 bc = _mm256_loadu_ps(row2 + x);
                    __m256 br = _mm256_loadu_ps(row2 + x + 1);

                    __m256 sx = _mm256_add_ps(
                        _mm256_add_ps(_mm256_sub_ps(tr, tl), _mm256_mul_ps(v_two, _mm256_sub_ps(mr, ml))),
                        _mm256_sub_ps(br, bl));
                    __m256 sy = _mm256_add_ps(
                        _mm256_add_ps(_mm256_sub_ps(bl, tl), _mm256_mul_ps(v_two, _mm256_sub_ps(bc, tc))),
                        _mm256_sub_ps(br, tr));

                    _mm256_storeu_ps(gx_row + x, _mm256_mul_ps(sx, v_quarter));
                    _mm256_storeu_ps(gy_row + x, _mm256_mul_ps(sy, v_quarter));
                }
#endif

#if !defined(__AVX2__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
                const __m128 v_two = _mm_set1_ps(2.0f);
                const __m128 v_quarter = _mm_set1_ps(0.25f);
                for (; x + 3 < w - 1; x += 4) {
                    __m128 tl = _mm_loadu_ps(row0 + x - 1);
                    __m128 tc = _mm_loadu_ps(row0 + x);
                    __m128 tr = _mm_loadu_ps(row0 + x + 1);
                    __m128 ml = _mm_loadu_ps(row1 + x - 1);
                    __m128 mr = _mm_loadu_ps(row1 + x + 1);
                    __m128 bl = _mm_loadu_ps(row2 + x - 1);
                    __m128 bc = _mm_loadu_ps(row2 + x);
                    __m128 br = _mm_loadu_ps(row2 + x + 1);

                    __m128 sx = _mm_add_ps(
                        _mm_add_ps(_mm_sub_ps(tr, tl), _mm_mul_ps(v_two, _mm_sub_ps(mr, ml))),
                        _mm_sub_ps(br, bl));
                    __m128 sy = _mm_add_ps(
                        _mm_add_ps(_mm_sub_ps(bl, tl), _mm_mul_ps(v_two, _mm_sub_ps(bc, tc))),
                        _mm_sub_ps(br, tr));

                    _mm_storeu_ps(gx_row + x, _mm_mul_ps(sx, v_quarter));
                    _mm_storeu_ps(gy_row + x, _mm_mul_ps(sy, v_quarter));
                }
#endif

            for (; x < w - 1; ++x) {
                float tl = row0[x - 1];
                float tc = row0[x];
                float tr = row0[x + 1];
                float ml = row1[x - 1];
                float mr = row1[x + 1];
                float bl = row2[x - 1];
                float bc = row2[x];
                float br = row2[x + 1];

                float sx = (tr - tl) + 2.0f * (mr - ml) + (br - bl);
                float sy = (bl - tl) + 2.0f * (bc - tc) + (br - tr);
                gx_row[x] = sx * 0.25f;
                gy_row[x] = sy * 0.25f;
            }
        }
    }
}

FloatImage EdgeDetector::non_maximum_suppression(const FloatImage& magnitude, const FloatImage& orientation) {
    int w = magnitude.width();
    int h = magnitude.height();
    FloatImage result(w, h, 0.0f);
    if (w < 3 || h < 3) return result;
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            const float mag = magnitude.get(x, y);
            float angle = orientation.get(x, y);
            constexpr float kPi = 3.14159265358979323846f;
            if (angle < 0.0f) angle += kPi;
            if (angle >= kPi) angle -= kPi;

            float mag1 = 0.0f;
            float mag2 = 0.0f;
            if (angle < kPi / 8.0f || angle >= 7.0f * kPi / 8.0f) {
                mag1 = magnitude.get(x - 1, y);
                mag2 = magnitude.get(x + 1, y);
            } else if (angle < 3.0f * kPi / 8.0f) {
                mag1 = magnitude.get(x - 1, y - 1);
                mag2 = magnitude.get(x + 1, y + 1);
            } else if (angle < 5.0f * kPi / 8.0f) {
                mag1 = magnitude.get(x, y - 1);
                mag2 = magnitude.get(x, y + 1);
            } else {
                mag1 = magnitude.get(x + 1, y - 1);
                mag2 = magnitude.get(x - 1, y + 1);
            }
            
            if (mag >= mag1 && mag >= mag2) {
                result.set(x, y, mag);
            }
        }
    }
    
    return result;
}

std::vector<bool> EdgeDetector::hysteresis_threshold(const FloatImage& magnitude, int w, int h, float low, float high) {
    const size_t count = checked_image_size(w, h);
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::length_error("Edge image exceeds supported pixel count");
    }
    if (count == 0) return {};
    std::vector<uint8_t> state(count, 0);
    std::vector<int> pending;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            float mag = magnitude.get(x, y);
            if (mag >= high) {
                state[idx] = 2;
                pending.push_back(idx);
            } else if (mag >= low) {
                state[idx] = 1;
            }
        }
    }
    return connect_weak_edges(state, pending, w, h);
}

}
