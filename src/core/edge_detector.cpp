#include "edge_detector.hpp"
#include <algorithm>
#include <queue>
#include <cstring>
#include <numeric>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace ascii {

EdgeDetector::EdgeDetector(const Config& config) : config_(config) {}

GradientData EdgeDetector::compute_gradients(const FloatImage& input) {
    GradientData result;

    FloatImage working = input;
    if (config_.use_anisotropic_diffusion && config_.diffusion_iterations > 0) {
        working = anisotropic_diffusion(
            input,
            config_.diffusion_iterations,
            config_.diffusion_kappa,
            config_.diffusion_lambda
        );
    }

    FloatImage blurred = gaussian_blur(working, config_.blur_sigma);
    
    sobel(blurred, result.gx, result.gy);
    
    int w = blurred.width();
    int h = blurred.height();
    result.magnitude = FloatImage(w, h);
    result.orientation = FloatImage(w, h);
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float gx = result.gx.get(x, y);
            float gy = result.gy.get(x, y);
            result.magnitude.set(x, y, std::sqrt(gx*gx + gy*gy));
            result.orientation.set(x, y, std::atan2(gy, gx));
        }
    }
    
    return result;
}

MultiScaleGradientData EdgeDetector::compute_multi_scale_gradients(const FloatImage& input) {
    MultiScaleGradientData result;
    int w = input.width();
    int h = input.height();

    FloatImage working = input;
    if (config_.use_anisotropic_diffusion && config_.diffusion_iterations > 0) {
        working = anisotropic_diffusion(
            input,
            config_.diffusion_iterations,
            config_.diffusion_kappa,
            config_.diffusion_lambda
        );
    }

    // Lindeberg-style normalized Laplacian scale selection over a small geometric scale stack.
    const float sigma0 = std::max(0.1f, config_.scale_sigma_0);
    const float sigmaN = std::max(sigma0 + 1e-4f, config_.scale_sigma_1);
    constexpr int kScaleLevels = 5;

    std::vector<float> sigmas(kScaleLevels, sigma0);
    if constexpr (kScaleLevels > 1) {
        const float ratio = sigmaN / sigma0;
        for (int s = 0; s < kScaleLevels; ++s) {
            float t = static_cast<float>(s) / static_cast<float>(kScaleLevels - 1);
            sigmas[s] = sigma0 * std::pow(ratio, t);
        }
    }

    std::vector<FloatImage> blurred(kScaleLevels);
    std::vector<FloatImage> gx(kScaleLevels);
    std::vector<FloatImage> gy(kScaleLevels);
    std::vector<FloatImage> mag(kScaleLevels);
    std::vector<FloatImage> orient(kScaleLevels);
    std::vector<FloatImage> lap(kScaleLevels);
    std::vector<FloatImage> norm_lap(kScaleLevels);

    auto compute_laplacian = [](const FloatImage& img) {
        FloatImage out(img.width(), img.height(), 0.0f);
        for (int y = 1; y < img.height() - 1; ++y) {
            for (int x = 1; x < img.width() - 1; ++x) {
                float c = img.get(x, y);
                float l = img.get(x - 1, y);
                float r = img.get(x + 1, y);
                float u = img.get(x, y - 1);
                float d = img.get(x, y + 1);
                out.set(x, y, l + r + u + d - 4.0f * c);
            }
        }
        return out;
    };

    for (int s = 0; s < kScaleLevels; ++s) {
        blurred[s] = gaussian_blur(working, sigmas[s]);
        sobel(blurred[s], gx[s], gy[s]);
        mag[s] = FloatImage(w, h);
        orient[s] = FloatImage(w, h);
        lap[s] = compute_laplacian(blurred[s]);
        norm_lap[s] = FloatImage(w, h);

#ifdef HAS_OPENMP
        #pragma omp parallel for
#endif
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float sx = gx[s].get(x, y);
                float sy = gy[s].get(x, y);
                float m = std::sqrt(sx * sx + sy * sy);
                mag[s].set(x, y, m);
                orient[s].set(x, y, std::atan2(sy, sx));
                float nlap = sigmas[s] * sigmas[s] * std::abs(lap[s].get(x, y));
                norm_lap[s].set(x, y, nlap);
            }
        }
    }

    result.magnitude = FloatImage(w, h);
    result.orientation = FloatImage(w, h);
    result.gx = FloatImage(w, h);
    result.gy = FloatImage(w, h);

#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int best_scale = 0;
            if (config_.adaptive_scale_selection) {
                float peak_response = -1.0f;
                int peak_index = -1;

                for (int s = 1; s < kScaleLevels - 1; ++s) {
                    float prev = norm_lap[s - 1].get(x, y);
                    float curr = norm_lap[s].get(x, y);
                    float next = norm_lap[s + 1].get(x, y);
                    if (curr >= prev && curr >= next && curr > peak_response) {
                        peak_response = curr;
                        peak_index = s;
                    }
                }

                if (peak_index >= 0) {
                    best_scale = peak_index;
                } else {
                    float best_response = norm_lap[0].get(x, y);
                    best_scale = 0;
                    for (int s = 1; s < kScaleLevels; ++s) {
                        float rs = norm_lap[s].get(x, y);
                        if (rs > best_response) {
                            best_response = rs;
                            best_scale = s;
                        }
                    }
                }
            } else {
                // Backward-compatible non-adaptive mode.
                best_scale = 0;
            }

            result.magnitude.set(x, y, mag[best_scale].get(x, y));
            result.orientation.set(x, y, orient[best_scale].get(x, y));
            result.gx.set(x, y, gx[best_scale].get(x, y));
            result.gy.set(x, y, gy[best_scale].get(x, y));
        }
    }

    result.best_scale = config_.adaptive_scale_selection ? -1 : 0;
    return result;
}

FloatImage EdgeDetector::fuse_multi_scale_magnitude(const FloatImage& mag0, const FloatImage& mag1,
                                                     float w0, float w1) {
    int w = mag0.width();
    int h = mag0.height();
    FloatImage result(w, h);
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            result.set(x, y, mag0.get(x, y) * w0 + mag1.get(x, y) * w1);
        }
    }
    
    return result;
}

void EdgeDetector::fuse_multi_scale_orientation(const FloatImage& orient0, const FloatImage& mag0,
                                                 const FloatImage& orient1, const FloatImage& mag1,
                                                 FloatImage& out_orientation) {
    int w = mag0.width();
    int h = mag0.height();
    out_orientation = FloatImage(w, h);
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float m0 = mag0.get(x, y);
            float m1 = mag1.get(x, y);
            
            if (m0 >= m1) {
                out_orientation.set(x, y, orient0.get(x, y));
            } else {
                out_orientation.set(x, y, orient1.get(x, y));
            }
        }
    }
}

float EdgeDetector::compute_global_percentile_threshold(const FloatImage& magnitude, float percentile) {
    int w = magnitude.width();
    int h = magnitude.height();
    
    std::vector<float> values;
    values.reserve(w * h);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float v = magnitude.get(x, y);
            if (v > 0.001f) {
                values.push_back(v);
            }
        }
    }
    
    if (values.empty()) return 0.1f;
    
    std::sort(values.begin(), values.end());
    
    size_t idx = static_cast<size_t>(values.size() * percentile);
    if (idx >= values.size()) idx = values.size() - 1;
    
    return values[idx];
}

float EdgeDetector::compute_tile_threshold(const FloatImage& magnitude, int x0, int y0, int tile_size,
                                            int img_w, int img_h, float percentile) {
    std::vector<float> values;
    values.reserve(tile_size * tile_size);
    
    int x1 = std::min(x0 + tile_size, img_w);
    int y1 = std::min(y0 + tile_size, img_h);
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            float v = magnitude.get(x, y);
            if (v > 0.001f) {
                values.push_back(v);
            }
        }
    }
    
    if (values.empty()) return 0.1f;
    
    std::sort(values.begin(), values.end());
    
    size_t idx = static_cast<size_t>(values.size() * percentile);
    if (idx >= values.size()) idx = values.size() - 1;
    
    return values[idx];
}

FloatImage EdgeDetector::compute_adaptive_threshold_map(const FloatImage& magnitude, int tile_size,
                                                         float percentile, float floor) {
    int w = magnitude.width();
    int h = magnitude.height();
    FloatImage result((w + tile_size - 1) / tile_size, (h + tile_size - 1) / tile_size);
    
    int tw = result.width();
    int th = result.height();
    
    float global_thresh = compute_global_percentile_threshold(magnitude, percentile);
    
    for (int ty = 0; ty < th; ++ty) {
        for (int tx = 0; tx < tw; ++tx) {
            int x0 = tx * tile_size;
            int y0 = ty * tile_size;
            
            float local_thresh = compute_tile_threshold(magnitude, x0, y0, tile_size, w, h, percentile);
            
            float combined = std::max(local_thresh, global_thresh * 0.5f);
            combined = std::max(combined, floor);
            
            result.set(tx, ty, combined);
        }
    }
    
    return result;
}

EdgeData EdgeDetector::detect(const FloatImage& input) {
    EdgeData result;
    
    GradientData grad;
    MultiScaleGradientData ms_grad;
    
    if (config_.multi_scale) {
        ms_grad = compute_multi_scale_gradients(input);
        result.magnitude = std::move(ms_grad.magnitude);
        result.orientation = std::move(ms_grad.orientation);
    } else {
        grad = compute_gradients(input);
        result.magnitude = std::move(grad.magnitude);
        result.orientation = std::move(grad.orientation);
    }
    
    FloatImage nms = non_maximum_suppression(result.magnitude, result.orientation);
    
    int w = result.magnitude.width();
    int h = result.magnitude.height();
    
    float low_thresh = config_.low_threshold;
    float high_thresh = config_.high_threshold;
    
    if (config_.adaptive_mode == "global") {
        high_thresh = compute_global_percentile_threshold(nms, config_.global_percentile);
        high_thresh = std::max(high_thresh, config_.dark_scene_floor);
        low_thresh = high_thresh * 0.5f;
    } else if (config_.adaptive_mode == "local" || config_.adaptive_mode == "hybrid") {
        FloatImage thresh_map = compute_adaptive_threshold_map(nms, config_.tile_size,
                                                                config_.global_percentile,
                                                                config_.dark_scene_floor);
        
        result.edge_mask.resize(w * h, false);
        
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int tx = x / config_.tile_size;
                int ty = y / config_.tile_size;
                if (tx >= thresh_map.width()) tx = thresh_map.width() - 1;
                if (ty >= thresh_map.height()) ty = thresh_map.height() - 1;
                
                float local_high = thresh_map.get(tx, ty);
                
                float mag = nms.get(x, y);
                result.edge_mask[y * w + x] = mag >= local_high;
            }
        }
        
        if (config_.use_hysteresis) {
            std::vector<bool> strong = result.edge_mask;
            std::vector<bool> weak(w * h, false);
            
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int idx = y * w + x;
                    int tx = x / config_.tile_size;
                    int ty = y / config_.tile_size;
                    if (tx >= thresh_map.width()) tx = thresh_map.width() - 1;
                    if (ty >= thresh_map.height()) ty = thresh_map.height() - 1;
                    
                    float local_high = thresh_map.get(tx, ty);
                    float local_low = local_high * 0.5f;
                    
                    float mag = nms.get(x, y);
                    
                    if (strong[idx]) {
                        continue;
                    } else if (mag >= local_low) {
                        weak[idx] = true;
                    }
                }
            }
            
            std::queue<std::pair<int, int>> queue;
            std::fill(result.edge_mask.begin(), result.edge_mask.end(), false);
            
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    int idx = y * w + x;
                    if (strong[idx]) {
                        result.edge_mask[idx] = true;
                        queue.push({x, y});
                    }
                }
            }
            
            while (!queue.empty()) {
                auto [cx, cy] = queue.front();
                queue.pop();
                
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) continue;
                        int nx = cx + dx;
                        int ny = cy + dy;
                        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                        
                        int idx = ny * w + nx;
                        if (weak[idx] && !result.edge_mask[idx]) {
                            result.edge_mask[idx] = true;
                            queue.push({nx, ny});
                        }
                    }
                }
            }
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
        for (int x = 0; x < w; ++x) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            int count = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    float v = input.get_clamped(x + dx, y + dy);
                    sum += v;
                    sum_sq += v * v;
                    count++;
                }
            }
            float mean = sum / static_cast<float>(count);
            float variance = std::max(0.0f, sum_sq / static_cast<float>(count) - mean * mean);
            var.set(x, y, variance);
        }
    }
    return var;
}

FloatImage EdgeDetector::gaussian_blur(const FloatImage& input, float sigma) {
    int w = input.width();
    int h = input.height();
    
    if (sigma <= 0.0f) {
        FloatImage result(w, h);
        std::memcpy(result.data(), input.data(), w * h * sizeof(float));
        return result;
    }
    
    int radius = static_cast<int>(std::ceil(sigma * 3));
    int ksize = 2 * radius + 1;
    
    std::vector<float> kernel(ksize);
    float sum = 0.0f;
    for (int i = 0; i < ksize; ++i) {
        float x = static_cast<float>(i - radius);
        kernel[i] = std::exp(-x * x / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (float& k : kernel) k /= sum;
    
    FloatImage temp(w, h);
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float val = 0.0f;
            float wsum = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                int nx = x + k - radius;
                if (nx >= 0 && nx < w) {
                    val += input.get(nx, y) * kernel[k];
                    wsum += kernel[k];
                }
            }
            temp.set(x, y, val / wsum);
        }
    }
    
    FloatImage result(w, h);
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float val = 0.0f;
            float wsum = 0.0f;
            for (int k = 0; k < ksize; ++k) {
                int ny = y + k - radius;
                if (ny >= 0 && ny < h) {
                    val += temp.get(x, ny) * kernel[k];
                    wsum += kernel[k];
                }
            }
            result.set(x, y, val / wsum);
        }
    }
    
    return result;
}

void EdgeDetector::sobel(const FloatImage& input, FloatImage& gx, FloatImage& gy) {
    int w = input.width();
    int h = input.height();
    
    gx = FloatImage(w, h);
    gy = FloatImage(w, h);
    
    constexpr float kx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    constexpr float ky[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float sx = 0.0f, sy = 0.0f;
            for (int ky_idx = 0; ky_idx < 3; ++ky_idx) {
                for (int kx_idx = 0; kx_idx < 3; ++kx_idx) {
                    float val = input.get(x + kx_idx - 1, y + ky_idx - 1);
                    sx += val * kx[ky_idx][kx_idx];
                    sy += val * ky[ky_idx][kx_idx];
                }
            }
            gx.set(x, y, sx / 4.0f);
            gy.set(x, y, sy / 4.0f);
        }
    }
}

FloatImage EdgeDetector::non_maximum_suppression(const FloatImage& magnitude, const FloatImage& orientation) {
    int w = magnitude.width();
    int h = magnitude.height();
    FloatImage result(w, h, 0.0f);
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float mag = magnitude.get(x, y);
            float angle = orientation.get(x, y);
            
            float nx = std::cos(angle);
            float ny = std::sin(angle);
            
            int x1 = std::max(0, std::min(w - 1, static_cast<int>(x + nx + 0.5f)));
            int y1 = std::max(0, std::min(h - 1, static_cast<int>(y + ny + 0.5f)));
            int x2 = std::max(0, std::min(w - 1, static_cast<int>(x - nx + 0.5f)));
            int y2 = std::max(0, std::min(h - 1, static_cast<int>(y - ny + 0.5f)));
            
            float mag1 = magnitude.get(x1, y1);
            float mag2 = magnitude.get(x2, y2);
            
            if (mag >= mag1 && mag >= mag2) {
                result.set(x, y, mag);
            }
        }
    }
    
    return result;
}

std::vector<bool> EdgeDetector::hysteresis_threshold(const FloatImage& magnitude, int w, int h, float low, float high) {
    std::vector<bool> strong(w * h, false);
    std::vector<bool> weak(w * h, false);
    std::vector<bool> result(w * h, false);
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            float mag = magnitude.get(x, y);
            if (mag >= high) {
                strong[idx] = true;
            } else if (mag >= low) {
                weak[idx] = true;
            }
        }
    }
    
    std::queue<std::pair<int, int>> queue;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int idx = y * w + x;
            if (strong[idx]) {
                result[idx] = true;
                queue.push({x, y});
            }
        }
    }
    
    while (!queue.empty()) {
        auto [cx, cy] = queue.front();
        queue.pop();
        
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int nx = cx + dx;
                int ny = cy + dy;
                if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
                
                int idx = ny * w + nx;
                if (weak[idx] && !result[idx]) {
                    result[idx] = true;
                    queue.push({nx, ny});
                }
            }
        }
    }
    
    return result;
}

}
