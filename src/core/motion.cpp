#include "motion.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <unordered_map>
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

struct FFTPlan {
    int n = 0;
    std::vector<int> bitrev;
    std::vector<std::vector<std::complex<float>>> twiddles_fwd;
    std::vector<std::vector<std::complex<float>>> twiddles_inv;
};

const FFTPlan& get_fft_plan(int n) {
    static std::unordered_map<int, FFTPlan> cache;
    auto it = cache.find(n);
    if (it != cache.end()) {
        return it->second;
    }

    FFTPlan plan;
    plan.n = n;
    plan.bitrev.resize(n);

    int bits = 0;
    while ((1 << bits) < n) {
        ++bits;
    }
    for (int i = 0; i < n; ++i) {
        int r = 0;
        for (int b = 0; b < bits; ++b) {
            r = (r << 1) | ((i >> b) & 1);
        }
        plan.bitrev[i] = r;
    }

    constexpr float kTwoPi = 6.28318530717958647692f;
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1;
        std::vector<std::complex<float>> fwd(half);
        std::vector<std::complex<float>> inv(half);
        for (int j = 0; j < half; ++j) {
            float angle = -kTwoPi * static_cast<float>(j) / static_cast<float>(len);
            fwd[j] = std::complex<float>(std::cos(angle), std::sin(angle));
            inv[j] = std::conj(fwd[j]);
        }
        plan.twiddles_fwd.push_back(std::move(fwd));
        plan.twiddles_inv.push_back(std::move(inv));
    }

    auto [inserted_it, _] = cache.emplace(n, std::move(plan));
    return inserted_it->second;
}

void fft_1d(std::complex<float>* data, int n, bool inverse) {
    if (n <= 1) {
        return;
    }

    const FFTPlan& plan = get_fft_plan(n);
    for (int i = 0; i < n; ++i) {
        int j = plan.bitrev[i];
        if (i < j) {
            std::swap(data[i], data[j]);
        }
    }

    int stage = 0;
    for (int len = 2; len <= n; len <<= 1, ++stage) {
        const int half = len >> 1;
        const auto& tw = inverse ? plan.twiddles_inv[stage] : plan.twiddles_fwd[stage];
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half; ++j) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + half] * tw[j];
                data[i + j] = u + v;
                data[i + j + half] = u - v;
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

void fft_2d(std::complex<float>* data, int w, int h, bool inverse,
            std::vector<std::complex<float>>& column_scratch) {
    for (int y = 0; y < h; ++y) {
        fft_1d(data + static_cast<size_t>(y) * w, w, inverse);
    }

    if (static_cast<int>(column_scratch.size()) < h) {
        column_scratch.resize(h);
    }
    std::complex<float>* col = column_scratch.data();
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            col[y] = data[static_cast<size_t>(y) * w + x];
        }
        fft_1d(col, h, inverse);
        for (int y = 0; y < h; ++y) {
            data[static_cast<size_t>(y) * w + x] = col[y];
        }
    }
}

const std::vector<float>& hann_window(int n) {
    static std::unordered_map<int, std::vector<float>> cache;
    auto it = cache.find(n);
    if (it != cache.end()) {
        return it->second;
    }

    std::vector<float> weights(std::max(1, n), 1.0f);
    if (n > 1) {
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float denom = static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) {
            weights[i] = 0.5f * (1.0f - std::cos(kTwoPi * static_cast<float>(i) / denom));
        }
    }

    auto [inserted_it, _] = cache.emplace(n, std::move(weights));
    return inserted_it->second;
}

struct PhaseCorrelationWorkspace {
    int fft_w = 0;
    int fft_h = 0;
    std::vector<std::complex<float>> prev_fft;
    std::vector<std::complex<float>> curr_fft;
    std::vector<std::complex<float>> cross_power;
    std::vector<std::complex<float>> column;
};

PhaseCorrelationWorkspace& get_phase_workspace(int fft_w, int fft_h) {
    static thread_local PhaseCorrelationWorkspace ws;
    if (ws.fft_w != fft_w || ws.fft_h != fft_h) {
        ws.fft_w = fft_w;
        ws.fft_h = fft_h;
        const size_t fft_size = static_cast<size_t>(fft_w) * fft_h;
        ws.prev_fft.assign(fft_size, std::complex<float>(0.0f, 0.0f));
        ws.curr_fft.assign(fft_size, std::complex<float>(0.0f, 0.0f));
        ws.cross_power.assign(fft_size, std::complex<float>(0.0f, 0.0f));
        ws.column.assign(std::max(fft_w, fft_h), std::complex<float>(0.0f, 0.0f));
    }
    return ws;
}

void load_zero_mean_hann(const FloatImage& img, std::complex<float>* out, int fft_w) {
    const int w = img.width();
    const int h = img.height();
    if (w <= 0 || h <= 0) {
        return;
    }

    const float* src = img.data();
    double sum = 0.0;
    for (int i = 0; i < w * h; ++i) {
        sum += src[i];
    }
    const float mean = static_cast<float>(sum / static_cast<double>(w * h));

    const auto& wx = hann_window(w);
    const auto& wy = hann_window(h);
    for (int y = 0; y < h; ++y) {
        const float* src_row = src + static_cast<size_t>(y) * w;
        std::complex<float>* dst_row = out + static_cast<size_t>(y) * fft_w;
        const float wyv = wy[y];
        for (int x = 0; x < w; ++x) {
            float v = (src_row[x] - mean) * wx[x] * wyv;
            dst_row[x] = std::complex<float>(v, 0.0f);
        }
    }
}

float corr_magnitude(const std::vector<std::complex<float>>& corr, int fft_w, int fft_h, int ix, int iy) {
    ix = (ix % fft_w + fft_w) % fft_w;
    iy = (iy % fft_h + fft_h) % fft_h;
    return std::abs(corr[static_cast<size_t>(iy) * fft_w + ix]);
}

int wrap_signed_shift(int s, int n) {
    const int half = n / 2;
    while (s > half) s -= n;
    while (s < -half) s += n;
    return s;
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

    const int fft_w = next_power_of_two(w);
    const int fft_h = next_power_of_two(h);
    if (fft_w < 2 || fft_h < 2) {
        return mv;
    }

    PhaseCorrelationWorkspace& ws = get_phase_workspace(fft_w, fft_h);
    std::fill(ws.prev_fft.begin(), ws.prev_fft.end(), std::complex<float>(0.0f, 0.0f));
    std::fill(ws.curr_fft.begin(), ws.curr_fft.end(), std::complex<float>(0.0f, 0.0f));
    std::fill(ws.cross_power.begin(), ws.cross_power.end(), std::complex<float>(0.0f, 0.0f));
    load_zero_mean_hann(prev, ws.prev_fft.data(), fft_w);
    load_zero_mean_hann(curr, ws.curr_fft.data(), fft_w);

    fft_2d(ws.prev_fft.data(), fft_w, fft_h, false, ws.column);
    fft_2d(ws.curr_fft.data(), fft_w, fft_h, false, ws.column);

    for (size_t i = 0; i < ws.cross_power.size(); ++i) {
        std::complex<float> cross = ws.prev_fft[i] * std::conj(ws.curr_fft[i]);
        float mag = std::abs(cross);
        if (mag > 1e-12f) {
            ws.cross_power[i] = cross / mag;
        }
    }
    fft_2d(ws.cross_power.data(), fft_w, fft_h, true, ws.column);

    const int search_x = std::clamp(radius, 1, std::max(1, fft_w / 2 - 1));
    const int search_y = std::clamp(radius, 1, std::max(1, fft_h / 2 - 1));
    const int center_ix = static_cast<int>(std::lround(center_dx));
    const int center_iy = static_cast<int>(std::lround(center_dy));
    float best = -1.0f;
    float second = -1.0f;
    int best_dx = 0;
    int best_dy = 0;
    double sum = 0.0;
    int samples = 0;
    for (int dy = -search_y; dy <= search_y; ++dy) {
        for (int dx = -search_x; dx <= search_x; ++dx) {
            int cand_dx = wrap_signed_shift(center_ix + dx, fft_w);
            int cand_dy = wrap_signed_shift(center_iy + dy, fft_h);
            const int ix = (cand_dx >= 0) ? cand_dx : (fft_w + cand_dx);
            const int iy = (cand_dy >= 0) ? cand_dy : (fft_h + cand_dy);
            float v = std::abs(ws.cross_power[static_cast<size_t>(iy) * fft_w + ix]);
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

    const int best_ix = (best_dx >= 0) ? best_dx : (fft_w + best_dx);
    const int best_iy = (best_dy >= 0) ? best_dy : (fft_h + best_dy);

    float c = corr_magnitude(ws.cross_power, fft_w, fft_h, best_ix, best_iy);
    float l = corr_magnitude(ws.cross_power, fft_w, fft_h, best_ix - 1, best_iy);
    float rr = corr_magnitude(ws.cross_power, fft_w, fft_h, best_ix + 1, best_iy);
    float u = corr_magnitude(ws.cross_power, fft_w, fft_h, best_ix, best_iy - 1);
    float d = corr_magnitude(ws.cross_power, fft_w, fft_h, best_ix, best_iy + 1);

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

MotionVector estimate_phase_correlation_hierarchical(const FloatImage* prev_levels,
                                                     const FloatImage* curr_levels,
                                                     int levels_available,
                                                     int radius) {
    MotionVector mv{};
    if (!prev_levels || !curr_levels || levels_available <= 0) {
        return mv;
    }

    float pred_dx = 0.0f;
    float pred_dy = 0.0f;
    MotionVector last{};
    const int top = levels_available - 1;
    for (int l = top; l >= 0; --l) {
        if (prev_levels[l].empty() || curr_levels[l].empty()) {
            continue;
        }
        if (l != top) {
            pred_dx *= 2.0f;
            pred_dy *= 2.0f;
        }
        int local_radius = std::max(1, radius >> l);
        local_radius = std::min(local_radius, std::max(2, radius));

        MotionVector level_mv = estimate_phase_correlation_shift(
            prev_levels[l],
            curr_levels[l],
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
            prev_pyramid_, curr_pyramid_, levels_available, config_.phase_search_radius);
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
