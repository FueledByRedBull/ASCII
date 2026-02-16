#include "motion.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace ascii {

namespace {

int next_power_of_two(int v) {
    int n = 1;
    while (n < v) {
        n <<= 1;
    }
    return n;
}

float quadratic_peak_offset(float left, float center, float right) {
    float denom = left - 2.0f * center + right;
    if (std::abs(denom) < 1e-6f) {
        return 0.0f;
    }
    return std::clamp(0.5f * (left - right) / denom, -1.0f, 1.0f);
}

void fft_1d(std::complex<float>* data, int n, bool inverse) {
    if (n <= 1) {
        return;
    }

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    const float pi = 3.14159265358979323846f;
    for (int len = 2; len <= n; len <<= 1) {
        float angle = (inverse ? 2.0f : -2.0f) * pi / static_cast<float>(len);
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            int half = len >> 1;
            for (int j = 0; j < half; ++j) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + half] * w;
                data[i + j] = u + v;
                data[i + j + half] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        float inv_n = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) {
            data[i] *= inv_n;
        }
    }
}

void fft_2d(std::complex<float>* data, int w, int h, bool inverse) {
    for (int y = 0; y < h; ++y) {
        fft_1d(data + static_cast<size_t>(y) * w, w, inverse);
    }

    std::vector<std::complex<float>> col(h);
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            col[y] = data[static_cast<size_t>(y) * w + x];
        }
        fft_1d(col.data(), h, inverse);
        for (int y = 0; y < h; ++y) {
            data[static_cast<size_t>(y) * w + x] = col[y];
        }
    }
}

void load_zero_mean_hann(const FloatImage& img, std::vector<std::complex<float>>& out, int n) {
    const int w = img.width();
    const int h = img.height();
    if (w <= 0 || h <= 0) {
        return;
    }

    double sum = 0.0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            sum += img.get(x, y);
        }
    }
    const float mean = static_cast<float>(sum / static_cast<double>(w * h));
    const float pi = 3.14159265358979323846f;

    const float wx_denom = static_cast<float>(std::max(1, w - 1));
    const float wy_denom = static_cast<float>(std::max(1, h - 1));
    for (int y = 0; y < h; ++y) {
        float wy = 0.5f * (1.0f - std::cos(2.0f * pi * static_cast<float>(y) / wy_denom));
        for (int x = 0; x < w; ++x) {
            float wx = 0.5f * (1.0f - std::cos(2.0f * pi * static_cast<float>(x) / wx_denom));
            float v = (img.get(x, y) - mean) * wx * wy;
            out[static_cast<size_t>(y) * n + x] = std::complex<float>(v, 0.0f);
        }
    }
}

float corr_magnitude(const std::vector<std::complex<float>>& corr, int n, int ix, int iy) {
    ix = (ix % n + n) % n;
    iy = (iy % n + n) % n;
    return std::abs(corr[static_cast<size_t>(iy) * n + ix]);
}

int wrap_signed_shift(int s, int n) {
    const int half = n / 2;
    while (s > half) s -= n;
    while (s < -half) s += n;
    return s;
}

FloatImage downsample_half(const FloatImage& src) {
    const int src_w = src.width();
    const int src_h = src.height();
    const int dst_w = std::max(1, src_w / 2);
    const int dst_h = std::max(1, src_h / 2);
    FloatImage dst(dst_w, dst_h, 0.0f);

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const int sx = x * 2;
            const int sy = y * 2;
            float sum = src.get(sx, sy) +
                        src.get(std::min(sx + 1, src_w - 1), sy) +
                        src.get(sx, std::min(sy + 1, src_h - 1)) +
                        src.get(std::min(sx + 1, src_w - 1), std::min(sy + 1, src_h - 1));
            dst.set(x, y, 0.25f * sum);
        }
    }
    return dst;
}

FloatImage upsample_flow_field(const FloatImage& src, int dst_w, int dst_h, float scale) {
    FloatImage dst(dst_w, dst_h, 0.0f);
    if (src.empty() || dst_w <= 0 || dst_h <= 0) {
        return dst;
    }

    const float sx_scale = static_cast<float>(src.width()) / static_cast<float>(std::max(1, dst_w));
    const float sy_scale = static_cast<float>(src.height()) / static_cast<float>(std::max(1, dst_h));

    for (int y = 0; y < dst_h; ++y) {
        float sy = (static_cast<float>(y) + 0.5f) * sy_scale - 0.5f;
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = y0 + 1;
        float ty = sy - static_cast<float>(y0);
        y0 = std::clamp(y0, 0, src.height() - 1);
        y1 = std::clamp(y1, 0, src.height() - 1);

        for (int x = 0; x < dst_w; ++x) {
            float sx = (static_cast<float>(x) + 0.5f) * sx_scale - 0.5f;
            int x0 = static_cast<int>(std::floor(sx));
            int x1 = x0 + 1;
            float tx = sx - static_cast<float>(x0);
            x0 = std::clamp(x0, 0, src.width() - 1);
            x1 = std::clamp(x1, 0, src.width() - 1);

            float v00 = src.get(x0, y0);
            float v10 = src.get(x1, y0);
            float v01 = src.get(x0, y1);
            float v11 = src.get(x1, y1);
            float v0 = v00 + (v10 - v00) * tx;
            float v1 = v01 + (v11 - v01) * tx;
            dst.set(x, y, (v0 + (v1 - v0) * ty) * scale);
        }
    }

    return dst;
}

MotionVector estimate_phase_correlation_shift(
    const FloatImage& prev,
    const FloatImage& curr,
    int radius,
    float center_dx = 0.0f,
    float center_dy = 0.0f) {
    MotionVector mv{};
    if (radius < 1) {
        return mv;
    }

    const int w = prev.width();
    const int h = prev.height();
    if (w <= 0 || h <= 0 || w != curr.width() || h != curr.height()) {
        return mv;
    }

    const int n = next_power_of_two(std::max(w, h));
    if (n < 2) {
        return mv;
    }

    std::vector<std::complex<float>> f(static_cast<size_t>(n) * n, std::complex<float>(0.0f, 0.0f));
    std::vector<std::complex<float>> g(static_cast<size_t>(n) * n, std::complex<float>(0.0f, 0.0f));
    load_zero_mean_hann(prev, f, n);
    load_zero_mean_hann(curr, g, n);

    fft_2d(f.data(), n, n, false);
    fft_2d(g.data(), n, n, false);

    std::vector<std::complex<float>> r(static_cast<size_t>(n) * n, std::complex<float>(0.0f, 0.0f));
    for (size_t i = 0; i < r.size(); ++i) {
        std::complex<float> cross = f[i] * std::conj(g[i]);
        float mag = std::abs(cross);
        if (mag > 1e-12f) {
            r[i] = cross / mag;
        }
    }
    fft_2d(r.data(), n, n, true);

    const int search = std::clamp(radius, 1, std::max(1, n / 2 - 1));
    const int center_ix = static_cast<int>(std::lround(center_dx));
    const int center_iy = static_cast<int>(std::lround(center_dy));
    float best = -1.0f;
    float second = -1.0f;
    int best_dx = 0;
    int best_dy = 0;
    double sum = 0.0;
    int samples = 0;
    for (int dy = -search; dy <= search; ++dy) {
        for (int dx = -search; dx <= search; ++dx) {
            int cand_dx = wrap_signed_shift(center_ix + dx, n);
            int cand_dy = wrap_signed_shift(center_iy + dy, n);
            const int ix = (cand_dx >= 0) ? cand_dx : (n + cand_dx);
            const int iy = (cand_dy >= 0) ? cand_dy : (n + cand_dy);
            float v = std::abs(r[static_cast<size_t>(iy) * n + ix]);
            sum += v;
            samples++;
            if (v > best) {
                second = best;
                best = v;
                best_dx = cand_dx;
                best_dy = cand_dy;
            } else if (v > second) {
                second = v;
            }
        }
    }

    if (!(best > 1e-8f) || !std::isfinite(best)) {
        return mv;
    }

    const int best_ix = (best_dx >= 0) ? best_dx : (n + best_dx);
    const int best_iy = (best_dy >= 0) ? best_dy : (n + best_dy);

    float c = corr_magnitude(r, n, best_ix, best_iy);
    float l = corr_magnitude(r, n, best_ix - 1, best_iy);
    float rr = corr_magnitude(r, n, best_ix + 1, best_iy);
    float u = corr_magnitude(r, n, best_ix, best_iy - 1);
    float d = corr_magnitude(r, n, best_ix, best_iy + 1);

    float sub_dx = quadratic_peak_offset(l, c, rr);
    float sub_dy = quadratic_peak_offset(u, c, d);

    mv.dx = static_cast<float>(best_dx) + sub_dx;
    mv.dy = static_cast<float>(best_dy) + sub_dy;

    float ratio = best / std::max(second, 1e-8f);
    float mean = static_cast<float>(sum / std::max(samples, 1));
    float sharpness = std::clamp((ratio - 1.35f) / 1.65f, 0.0f, 1.0f);
    float prominence = std::clamp((best - mean) / std::max(best, 1e-8f), 0.0f, 1.0f);
    float isolation = std::clamp((c - 0.25f * (l + rr + u + d)) / std::max(c, 1e-8f), 0.0f, 1.0f);
    mv.confidence = std::clamp(0.55f * sharpness + 0.25f * prominence + 0.20f * isolation, 0.0f, 1.0f);
    return mv;
}

MotionVector estimate_phase_correlation_hierarchical(const FloatImage& prev,
                                                     const FloatImage& curr,
                                                     int radius,
                                                     int max_levels) {
    MotionVector mv{};
    if (prev.empty() || curr.empty() || prev.width() != curr.width() || prev.height() != curr.height()) {
        return mv;
    }

    std::vector<FloatImage> prev_levels;
    std::vector<FloatImage> curr_levels;
    prev_levels.push_back(prev);
    curr_levels.push_back(curr);

    const int levels_target = std::clamp(max_levels, 1, 4);
    for (int l = 1; l < levels_target; ++l) {
        const FloatImage& p = prev_levels.back();
        const FloatImage& c = curr_levels.back();
        if (p.width() < 32 || p.height() < 32 || c.width() < 32 || c.height() < 32) {
            break;
        }
        prev_levels.push_back(downsample_half(p));
        curr_levels.push_back(downsample_half(c));
    }

    float pred_dx = 0.0f;
    float pred_dy = 0.0f;
    MotionVector last{};
    const int top = static_cast<int>(prev_levels.size()) - 1;
    for (int l = top; l >= 0; --l) {
        if (l != top) {
            pred_dx *= 2.0f;
            pred_dy *= 2.0f;
        }
        int local_radius = std::max(1, radius >> l);
        local_radius = std::min(local_radius, std::max(2, radius));

        MotionVector level_mv = estimate_phase_correlation_shift(
            prev_levels[static_cast<size_t>(l)],
            curr_levels[static_cast<size_t>(l)],
            local_radius,
            pred_dx,
            pred_dy);

        if (l != top && level_mv.confidence < 0.02f) {
            level_mv.dx = pred_dx;
            level_mv.dy = pred_dy;
            level_mv.confidence = last.confidence * 0.85f;
        }

        pred_dx = level_mv.dx;
        pred_dy = level_mv.dy;
        last = level_mv;
    }

    return last;
}

}  // namespace

MotionEstimator::MotionEstimator(const Config& config) : config_(config) {}

void MotionEstimator::reset() {
    flow_.clear();
    width_ = 0;
    height_ = 0;
    for (int i = 0; i < 4; ++i) {
        prev_pyramid_[i] = FloatImage();
        curr_pyramid_[i] = FloatImage();
    }
}

void MotionEstimator::build_pyramid(const FloatImage& img, FloatImage* pyramid, int levels) {
    for (int i = 1; i < 4; ++i) {
        pyramid[i] = FloatImage();
    }
    pyramid[0] = img;
    
    int capped_levels = std::clamp(levels, 1, 4);
    for (int l = 1; l < capped_levels; ++l) {
        int src_w = pyramid[l-1].width();
        int src_h = pyramid[l-1].height();
        int dst_w = src_w / 2;
        int dst_h = src_h / 2;
        
        if (dst_w < 8 || dst_h < 8) {
            break;
        }
        
        pyramid[l] = FloatImage(dst_w, dst_h);
        
        for (int y = 0; y < dst_h; ++y) {
            for (int x = 0; x < dst_w; ++x) {
                int sx = x * 2;
                int sy = y * 2;
                float sum = 0.0f;
                int cnt = 0;
                
                for (int dy = 0; dy < 2 && sy + dy < src_h; ++dy) {
                    for (int dx = 0; dx < 2 && sx + dx < src_w; ++dx) {
                        sum += pyramid[l-1].get(sx + dx, sy + dy);
                        cnt++;
                    }
                }
                pyramid[l].set(x, y, sum / cnt);
            }
        }
    }
}

void MotionEstimator::compute_farneback_level(const FloatImage& prev, const FloatImage& curr,
                                               FloatImage& flow_x, FloatImage& flow_y) {
    int w = prev.width();
    int h = prev.height();
    
    flow_x = FloatImage(w, h, 0.0f);
    flow_y = FloatImage(w, h, 0.0f);
    
    int half_win = config_.window_size / 2;
    int poly_n = config_.poly_n;
    float poly_sigma = config_.poly_sigma;
    
    std::vector<float> poly_prev(w * h * 6, 0.0f);
    std::vector<float> poly_curr(w * h * 6, 0.0f);
    
    auto compute_poly_expansion = [&](const FloatImage& img, std::vector<float>& poly) {
        int radius = poly_n / 2;
        
        for (int y = radius; y < h - radius; ++y) {
            for (int x = radius; x < w - radius; ++x) {
                float sum_xx = 0, sum_xy = 0, sum_yy = 0;
                float sum_x = 0, sum_y = 0, sum_v = 0;
                float sum_w = 0;
                
                for (int dy = -radius; dy <= radius; ++dy) {
                    for (int dx = -radius; dx <= radius; ++dx) {
                        float fdx = static_cast<float>(dx);
                        float fdy = static_cast<float>(dy);
                        float dist = fdx * fdx + fdy * fdy;
                        float weight = std::exp(-dist / (2.0f * poly_sigma * poly_sigma));
                        float v = img.get(x + dx, y + dy);
                        
                        sum_xx += weight * fdx * fdx;
                        sum_xy += weight * fdx * fdy;
                        sum_yy += weight * fdy * fdy;
                        sum_x += weight * fdx * v;
                        sum_y += weight * fdy * v;
                        sum_v += weight * v;
                        sum_w += weight;
                    }
                }
                
                float det = sum_xx * sum_yy - sum_xy * sum_xy;
                if (std::abs(det) > 1e-6f) {
                    int idx = (y * w + x) * 6;
                    poly[idx + 0] = sum_v / sum_w;
                    poly[idx + 1] = (sum_yy * sum_x - sum_xy * sum_y) / det;
                    poly[idx + 2] = (sum_xx * sum_y - sum_xy * sum_x) / det;
                    poly[idx + 3] = sum_xx / sum_w;
                    poly[idx + 4] = sum_xy / sum_w;
                    poly[idx + 5] = sum_yy / sum_w;
                }
            }
        }
    };
    
    compute_poly_expansion(prev, poly_prev);
    compute_poly_expansion(curr, poly_curr);
    
    for (int y = half_win; y < h - half_win; ++y) {
        for (int x = half_win; x < w - half_win; ++x) {
            float sum_a00 = 0, sum_a01 = 0, sum_a11 = 0;
            float sum_b0 = 0, sum_b1 = 0;
            float sum_w = 0;
            
            for (int dy = -half_win; dy <= half_win; ++dy) {
                for (int dx = -half_win; dx <= half_win; ++dx) {
                    int nx = x + dx;
                    int ny = y + dy;
                    
                    if (nx < poly_n/2 || nx >= w - poly_n/2 || 
                        ny < poly_n/2 || ny >= h - poly_n/2) continue;
                    
                    int idx = (ny * w + nx) * 6;
                    
                    float r0 = poly_prev[idx + 0];
                    float r1 = poly_prev[idx + 1];
                    float r2 = poly_prev[idx + 2];
                    float r3 = poly_prev[idx + 3];
                    float r4 = poly_prev[idx + 4];
                    float r5 = poly_prev[idx + 5];
                    
                    float s0 = poly_curr[idx + 0];
                    float s1 = poly_curr[idx + 1];
                    float s2 = poly_curr[idx + 2];
                    
                    float a00 = r1 * r1 + r2 * r2 + r3 * r3 + r4 * r4 + r5 * r5;
                    float a01 = r1 * s1 + r2 * s2;
                    float a11 = s1 * s1 + s2 * s2;
                    float b0 = r1 * (s0 - r0);
                    float b1 = s1 * (s0 - r0);
                    
                    float gw = 1.0f;
                    sum_a00 += gw * a00;
                    sum_a01 += gw * a01;
                    sum_a11 += gw * a11;
                    sum_b0 += gw * b0;
                    sum_b1 += gw * b1;
                    sum_w += gw;
                }
            }
            
            if (sum_w > 0) {
                sum_a00 /= sum_w;
                sum_a01 /= sum_w;
                sum_a11 /= sum_w;
                sum_b0 /= sum_w;
                sum_b1 /= sum_w;
            }
            
            float det = sum_a00 * sum_a11 - sum_a01 * sum_a01;
            if (std::abs(det) > 1e-6f) {
                float fx = (sum_a11 * sum_b0 - sum_a01 * sum_b1) / det;
                float fy = (sum_a00 * sum_b1 - sum_a01 * sum_b0) / det;
                
                fx = std::clamp(fx, -config_.motion_cap, config_.motion_cap);
                fy = std::clamp(fy, -config_.motion_cap, config_.motion_cap);
                
                flow_x.set(x, y, fx);
                flow_y.set(x, y, fy);
            }
        }
    }
}

void MotionEstimator::compute_flow(const FloatImage& prev, const FloatImage& curr) {
    if (prev.width() != curr.width() || prev.height() != curr.height()) {
        reset();
        return;
    }
    
    width_ = prev.width();
    height_ = prev.height();
    flow_.assign(width_ * height_, MotionVector{});
    
    int requested_levels = std::clamp(config_.pyramid_levels, 1, 4);
    build_pyramid(prev, prev_pyramid_, requested_levels);
    build_pyramid(curr, curr_pyramid_, requested_levels);

    int levels_available = 1;
    for (int l = 1; l < requested_levels; ++l) {
        if (prev_pyramid_[l].empty() || curr_pyramid_[l].empty()) {
            break;
        }
        levels_available = l + 1;
    }
    
    FloatImage flow_x, flow_y;

    // Coarse-to-fine estimation: significantly cheaper than repeated full-resolution solves.
    for (int l = levels_available - 1; l >= 0; --l) {
        const FloatImage& p = prev_pyramid_[l];
        const FloatImage& c = curr_pyramid_[l];
        if (p.empty() || c.empty()) {
            continue;
        }

        FloatImage level_fx, level_fy;
        compute_farneback_level(p, c, level_fx, level_fy);

        if (flow_x.empty()) {
            flow_x = std::move(level_fx);
            flow_y = std::move(level_fy);
            continue;
        }

        float scale_x = static_cast<float>(p.width()) / static_cast<float>(std::max(1, flow_x.width()));
        float scale_y = static_cast<float>(p.height()) / static_cast<float>(std::max(1, flow_y.height()));
        float scale = 0.5f * (scale_x + scale_y);
        FloatImage up_x = upsample_flow_field(flow_x, p.width(), p.height(), scale);
        FloatImage up_y = upsample_flow_field(flow_y, p.width(), p.height(), scale);
        const float refine_weight = (l == 0) ? 0.65f : 0.50f;
        const float keep_weight = 1.0f - refine_weight;
        const int n = p.width() * p.height();
        float* up_x_data = up_x.data();
        float* up_y_data = up_y.data();
        const float* level_x_data = level_fx.data();
        const float* level_y_data = level_fy.data();
        for (int i = 0; i < n; ++i) {
            up_x_data[i] = keep_weight * up_x_data[i] + refine_weight * level_x_data[i];
            up_y_data[i] = keep_weight * up_y_data[i] + refine_weight * level_y_data[i];
        }

        flow_x = std::move(up_x);
        flow_y = std::move(up_y);
    }

    if (flow_x.empty() || flow_y.empty()) {
        flow_x = FloatImage(width_, height_, 0.0f);
        flow_y = FloatImage(width_, height_, 0.0f);
    }

    int extra_iters = std::max(0, config_.iterations - levels_available);
    for (int iter = 0; iter < extra_iters; ++iter) {
        FloatImage fx, fy;
        compute_farneback_level(prev, curr, fx, fy);
        const float blend = 0.35f;
        const float keep = 1.0f - blend;
        const int n = width_ * height_;
        float* flow_x_data = flow_x.data();
        float* flow_y_data = flow_y.data();
        const float* fx_data = fx.data();
        const float* fy_data = fy.data();
        for (int i = 0; i < n; ++i) {
            flow_x_data[i] = keep * flow_x_data[i] + blend * fx_data[i];
            flow_y_data[i] = keep * flow_y_data[i] + blend * fy_data[i];
        }
    }

    MotionVector phase_mv{};
    if (config_.use_phase_correlation) {
        phase_mv = estimate_phase_correlation_hierarchical(
            prev, curr, config_.phase_search_radius, levels_available);
    }
    
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int idx = y * width_ + x;
            float dx = flow_x.get(x, y);
            float dy = flow_y.get(x, y);
            if (config_.use_phase_correlation) {
                float gated = std::clamp((phase_mv.confidence - 0.2f) / 0.8f, 0.0f, 1.0f);
                float blend = std::clamp(config_.phase_blend * gated * gated, 0.0f, 1.0f);
                dx = (1.0f - blend) * dx + blend * phase_mv.dx;
                dy = (1.0f - blend) * dy + blend * phase_mv.dy;
            }
            flow_[idx].dx = std::clamp(dx, -config_.motion_cap, config_.motion_cap);
            flow_[idx].dy = std::clamp(dy, -config_.motion_cap, config_.motion_cap);
            
            float mag = std::sqrt(flow_[idx].dx * flow_[idx].dx + flow_[idx].dy * flow_[idx].dy);
            flow_[idx].confidence = std::min(mag / std::max(config_.motion_cap, 1e-6f), 1.0f);
            if (config_.use_phase_correlation) {
                flow_[idx].confidence = std::max(flow_[idx].confidence, phase_mv.confidence * 0.7f);
            }
        }
    }
}

const MotionVector& MotionEstimator::get_motion(int x, int y) const {
    static const MotionVector empty;
    if (x < 0 || x >= width_ || y < 0 || y >= height_ || flow_.empty()) {
        return empty;
    }
    return flow_[y * width_ + x];
}

MotionVector MotionEstimator::get_motion_interpolated(float x, float y) const {
    if (flow_.empty() || width_ == 0 || height_ == 0) {
        return MotionVector{};
    }
    
    x = std::clamp(x, 0.0f, static_cast<float>(width_ - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(height_ - 1));
    
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = std::min(x0 + 1, width_ - 1);
    int y1 = std::min(y0 + 1, height_ - 1);
    
    float fx = x - x0;
    float fy = y - y0;
    
    const MotionVector& v00 = get_motion(x0, y0);
    const MotionVector& v10 = get_motion(x1, y0);
    const MotionVector& v01 = get_motion(x0, y1);
    const MotionVector& v11 = get_motion(x1, y1);
    
    MotionVector result;
    result.dx = (1-fx)*(1-fy)*v00.dx + fx*(1-fy)*v10.dx + (1-fx)*fy*v01.dx + fx*fy*v11.dx;
    result.dy = (1-fx)*(1-fy)*v00.dy + fx*(1-fy)*v10.dy + (1-fx)*fy*v01.dy + fx*fy*v11.dy;
    result.confidence = (1-fx)*(1-fy)*v00.confidence + fx*(1-fy)*v10.confidence + 
                        (1-fx)*fy*v01.confidence + fx*fy*v11.confidence;
    
    return result;
}

void MotionEstimator::average_flow_for_cell(int x0, int y0, int w, int h,
                                             float& dx, float& dy) const {
    if (flow_.empty()) {
        dx = 0.0f;
        dy = 0.0f;
        return;
    }
    
    float sum_dx = 0.0f, sum_dy = 0.0f;
    int count = 0;
    
    int x1 = std::min(x0 + w, width_);
    int y1 = std::min(y0 + h, height_);
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const auto& mv = get_motion(x, y);
            sum_dx += mv.dx;
            sum_dy += mv.dy;
            count++;
        }
    }
    
    if (count > 0) {
        dx = sum_dx / count;
        dy = sum_dy / count;
    } else {
        dx = 0.0f;
        dy = 0.0f;
    }
}

void MotionEstimator::get_motion_for_cell(int cell_x, int cell_y, int cell_w, int cell_h,
                                           float& out_dx, float& out_dy) const {
    average_flow_for_cell(cell_x, cell_y, cell_w, cell_h, out_dx, out_dy);
    
    out_dx = std::clamp(out_dx, -config_.motion_cap, config_.motion_cap);
    out_dy = std::clamp(out_dy, -config_.motion_cap, config_.motion_cap);
}

}
