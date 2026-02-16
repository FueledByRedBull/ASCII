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

MotionVector estimate_phase_correlation_shift(const FloatImage& prev, const FloatImage& curr, int radius) {
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
    float best = -1.0f;
    float second = -1.0f;
    int best_dx = 0;
    int best_dy = 0;
    double sum = 0.0;
    int samples = 0;
    for (int dy = -search; dy <= search; ++dy) {
        for (int dx = -search; dx <= search; ++dx) {
            const int ix = (dx >= 0) ? dx : (n + dx);
            const int iy = (dy >= 0) ? dy : (n + dy);
            float v = std::abs(r[static_cast<size_t>(iy) * n + ix]);
            sum += v;
            samples++;
            if (v > best) {
                second = best;
                best = v;
                best_dx = dx;
                best_dy = dy;
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
    pyramid[0] = img;
    
    for (int l = 1; l < levels; ++l) {
        int src_w = pyramid[l-1].width();
        int src_h = pyramid[l-1].height();
        int dst_w = src_w / 2;
        int dst_h = src_h / 2;
        
        if (dst_w < 8 || dst_h < 8) {
            levels = l;
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
    
    build_pyramid(prev, prev_pyramid_, config_.pyramid_levels);
    build_pyramid(curr, curr_pyramid_, config_.pyramid_levels);
    
    FloatImage flow_x, flow_y;
    
    for (int iter = 0; iter < config_.iterations; ++iter) {
        FloatImage fx, fy;
        compute_farneback_level(prev, curr, fx, fy);
        
        if (flow_x.empty()) {
            flow_x = fx;
            flow_y = fy;
        } else {
            for (int y = 0; y < height_; ++y) {
                for (int x = 0; x < width_; ++x) {
                    float old_x = flow_x.get(x, y);
                    float old_y = flow_y.get(x, y);
                    float new_x = fx.get(x, y);
                    float new_y = fy.get(x, y);
                    flow_x.set(x, y, old_x * 0.5f + new_x * 0.5f);
                    flow_y.set(x, y, old_y * 0.5f + new_y * 0.5f);
                }
            }
        }
    }

    MotionVector phase_mv{};
    if (config_.use_phase_correlation) {
        phase_mv = estimate_phase_correlation_shift(prev, curr, config_.phase_search_radius);
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
