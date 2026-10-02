#include "motion.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <array>
#include <vector>

#if defined(__AVX2__) || defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <immintrin.h>
#endif

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
    // Four pyramid levels need at most two axis lengths each.
    static thread_local std::array<FFTPlan, 8> cache;
    static thread_local size_t next = 0;
    for (const auto& plan : cache) {
        if (plan.n == n) return plan;
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

    auto& slot = cache[next++ % cache.size()];
    slot = std::move(plan);
    return slot;
}

#if defined(__AVX2__)
inline __m256 complex_mul4_interleaved_avx(__m256 a, __m256 b) {
    const __m256 ac_bd = _mm256_mul_ps(a, b);
    const __m256 b_swap = _mm256_permute_ps(b, 0xB1);
    const __m256 ad_bc = _mm256_mul_ps(a, b_swap);

    const __m256 real_dup = _mm256_hsub_ps(ac_bd, ac_bd);
    const __m256 imag_dup = _mm256_hadd_ps(ad_bc, ad_bc);

    const __m128 real_lo = _mm256_castps256_ps128(real_dup);
    const __m128 real_hi = _mm256_extractf128_ps(real_dup, 1);
    const __m128 imag_lo = _mm256_castps256_ps128(imag_dup);
    const __m128 imag_hi = _mm256_extractf128_ps(imag_dup, 1);

    const __m128 real = _mm_movelh_ps(real_lo, real_hi);  // [r0 r1 r2 r3]
    const __m128 imag = _mm_movelh_ps(imag_lo, imag_hi);  // [i0 i1 i2 i3]

    const __m128 out_lo = _mm_unpacklo_ps(real, imag);    // [r0 i0 r1 i1]
    const __m128 out_hi = _mm_unpackhi_ps(real, imag);    // [r2 i2 r3 i3]
    __m256 out = _mm256_castps128_ps256(out_lo);
    out = _mm256_insertf128_ps(out, out_hi, 1);
    return out;
}
#endif

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
            int j = 0;

#if defined(__AVX2__)
            // 4 butterflies at once (8 floats / 4 complex numbers).
            for (; j + 3 < half; j += 4) {
                std::complex<float>* u_ptr = data + i + j;
                std::complex<float>* v_src_ptr = data + i + j + half;
                const std::complex<float>* w_ptr = tw.data() + j;

                const __m256 u = _mm256_loadu_ps(reinterpret_cast<const float*>(u_ptr));
                const __m256 v_src = _mm256_loadu_ps(reinterpret_cast<const float*>(v_src_ptr));
                const __m256 w = _mm256_loadu_ps(reinterpret_cast<const float*>(w_ptr));
                const __m256 v = complex_mul4_interleaved_avx(v_src, w);

                _mm256_storeu_ps(reinterpret_cast<float*>(u_ptr), _mm256_add_ps(u, v));
                _mm256_storeu_ps(reinterpret_cast<float*>(v_src_ptr), _mm256_sub_ps(u, v));
            }
#endif

#if !defined(__AVX2__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
            // 2 butterflies at once (SSE2): scalar complex multiply + SIMD combine.
            for (; j + 1 < half; j += 2) {
                std::complex<float>* u_ptr = data + i + j;
                std::complex<float>* v_src_ptr = data + i + j + half;
                const std::complex<float>* w_ptr = tw.data() + j;

                const std::complex<float> v0 = v_src_ptr[0] * w_ptr[0];
                const std::complex<float> v1 = v_src_ptr[1] * w_ptr[1];

                const __m128 u = _mm_loadu_ps(reinterpret_cast<const float*>(u_ptr));
                const __m128 v = _mm_set_ps(v1.imag(), v1.real(), v0.imag(), v0.real());
                _mm_storeu_ps(reinterpret_cast<float*>(u_ptr), _mm_add_ps(u, v));
                _mm_storeu_ps(reinterpret_cast<float*>(v_src_ptr), _mm_sub_ps(u, v));
            }
#endif

            for (; j < half; ++j) {
                const std::complex<float> u = data[i + j];
                const std::complex<float> v = data[i + j + half] * tw[j];
                data[i + j] = u + v;
                data[i + j + half] = u - v;
            }
        }
    }

    if (inverse) {
        const float inv_n = 1.0f / static_cast<float>(n);
        int i = 0;
#if defined(__AVX2__)
        const __m256 scale = _mm256_set1_ps(inv_n);
        for (; i + 3 < n; i += 4) {
            __m256 v = _mm256_loadu_ps(reinterpret_cast<const float*>(data + i));
            v = _mm256_mul_ps(v, scale);
            _mm256_storeu_ps(reinterpret_cast<float*>(data + i), v);
        }
#endif
#if !defined(__AVX2__) && (defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
        const __m128 scale = _mm_set1_ps(inv_n);
        for (; i + 1 < n; i += 2) {
            __m128 v = _mm_loadu_ps(reinterpret_cast<const float*>(data + i));
            v = _mm_mul_ps(v, scale);
            _mm_storeu_ps(reinterpret_cast<float*>(data + i), v);
        }
#endif
        for (; i < n; ++i) {
            data[i] *= inv_n;
        }
    }
}

void fft_2d(std::complex<float>* data, int w, int h, bool inverse,
            std::vector<std::complex<float>>& column_scratch) {
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static) if(static_cast<size_t>(w) * h >= 4096)
#endif
    for (int y = 0; y < h; ++y) {
        fft_1d(data + static_cast<size_t>(y) * w, w, inverse);
    }

    const size_t scratch_size = static_cast<size_t>(w) * h;
    if (column_scratch.size() < scratch_size) {
        column_scratch.resize(scratch_size);
    }

    std::complex<float>* transposed = column_scratch.data();

    // Cache-friendly blocked transpose: columns become contiguous rows.
    constexpr int kTile = 32;
    for (int ty = 0; ty < h; ty += kTile) {
        int y_end = std::min(ty + kTile, h);
        for (int tx = 0; tx < w; tx += kTile) {
            int x_end = std::min(tx + kTile, w);
            for (int y = ty; y < y_end; ++y) {
                const std::complex<float>* src_row = data + static_cast<size_t>(y) * w;
                for (int x = tx; x < x_end; ++x) {
                    transposed[static_cast<size_t>(x) * h + y] = src_row[x];
                }
            }
        }
    }

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static) if(static_cast<size_t>(w) * h >= 4096)
#endif
    for (int x = 0; x < w; ++x) {
        fft_1d(transposed + static_cast<size_t>(x) * h, h, inverse);
    }

    for (int tx = 0; tx < w; tx += kTile) {
        int x_end = std::min(tx + kTile, w);
        for (int ty = 0; ty < h; ty += kTile) {
            int y_end = std::min(ty + kTile, h);
            for (int x = tx; x < x_end; ++x) {
                const std::complex<float>* src_col = transposed + static_cast<size_t>(x) * h;
                for (int y = ty; y < y_end; ++y) {
                    data[static_cast<size_t>(y) * w + x] = src_col[y];
                }
            }
        }
    }
}

const std::vector<float>& hann_window(int n, bool vertical) {
    static thread_local std::array<std::vector<float>, 8> cache;
    static thread_local std::array<size_t, 2> next{};
    const size_t axis = vertical ? 1 : 0;
    const size_t first = axis * 4;
    // Separate axis slots keep the horizontal reference valid while fetching vertical weights.
    for (size_t i = first; i < first + 4; ++i) {
        const auto& weights = cache[i];
        if (weights.size() == static_cast<size_t>(n)) return weights;
    }

    std::vector<float> weights(std::max(1, n), 1.0f);
    if (n > 1) {
        constexpr float kTwoPi = 6.28318530717958647692f;
        const float denom = static_cast<float>(n - 1);
        for (int i = 0; i < n; ++i) {
            weights[i] = 0.5f * (1.0f - std::cos(kTwoPi * static_cast<float>(i) / denom));
        }
    }

    auto& slot = cache[first + next[axis]++ % 4];
    slot = std::move(weights);
    return slot;
}

struct PhaseCorrelationWorkspace {
    int fft_w = 0;
    int fft_h = 0;
    std::vector<std::complex<float>> prev_fft;
    std::vector<std::complex<float>> curr_fft;
    std::vector<std::complex<float>> cross_power;
    std::vector<std::complex<float>> scratch;
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
        ws.scratch.assign(fft_size, std::complex<float>(0.0f, 0.0f));
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

    const auto& wx = hann_window(w, false);
    const auto& wy = hann_window(h, true);
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

    struct HorizontalSample { int x0, x1; float weight; };
    std::vector<HorizontalSample> horizontal(dst_w);
    for (int x = 0; x < dst_w; ++x) {
        float sx = (static_cast<float>(x) + 0.5f) * sx_scale - 0.5f;
        int x0 = static_cast<int>(std::floor(sx));
        horizontal[x] = {std::clamp(x0, 0, src.width() - 1),
                         std::clamp(x0 + 1, 0, src.width() - 1),
                         sx - static_cast<float>(x0)};
    }

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst_h; ++y) {
        float sy = (static_cast<float>(y) + 0.5f) * sy_scale - 0.5f;
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = y0 + 1;
        float ty = sy - static_cast<float>(y0);
        y0 = std::clamp(y0, 0, src.height() - 1);
        y1 = std::clamp(y1, 0, src.height() - 1);

        for (int x = 0; x < dst_w; ++x) {
            const auto& sample = horizontal[x];
            const int x0 = sample.x0, x1 = sample.x1;
            const float tx = sample.weight;

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

FloatImage downsample_area_average(const FloatImage& src, int dst_w, int dst_h) {
    dst_w = std::max(1, dst_w);
    dst_h = std::max(1, dst_h);
    FloatImage dst(dst_w, dst_h, 0.0f);
    if (src.empty()) {
        return dst;
    }

    const int src_w = src.width();
    const int src_h = src.height();
    const float* s = src.data();
    float* d = dst.data();

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < dst_h; ++y) {
        int y0 = (y * src_h) / dst_h;
        int y1 = ((y + 1) * src_h) / dst_h;
        y1 = std::max(y0 + 1, y1);
        y1 = std::min(y1, src_h);

        for (int x = 0; x < dst_w; ++x) {
            int x0 = (x * src_w) / dst_w;
            int x1 = ((x + 1) * src_w) / dst_w;
            x1 = std::max(x0 + 1, x1);
            x1 = std::min(x1, src_w);

            double sum = 0.0;
            int count = 0;
            for (int yy = y0; yy < y1; ++yy) {
                const float* row = s + static_cast<size_t>(yy) * src_w;
                for (int xx = x0; xx < x1; ++xx) {
                    sum += row[xx];
                    ++count;
                }
            }
            d[static_cast<size_t>(y) * dst_w + x] =
                (count > 0) ? static_cast<float>(sum / count) : 0.0f;
        }
    }
    return dst;
}

float estimate_scene_change_score(const FloatImage& prev, const FloatImage& curr) {
    if (prev.empty() || curr.empty() ||
        prev.width() != curr.width() || prev.height() != curr.height()) {
        return 1.0f;
    }

    const int w = prev.width();
    const int h = prev.height();
    const float* a = prev.data();
    const float* b = curr.data();

    int stride = std::max(1, std::min(w, h) / 64);
    double sum_abs = 0.0;
    int count = 0;
    for (int y = 0; y < h; y += stride) {
        size_t row = static_cast<size_t>(y) * w;
        for (int x = 0; x < w; x += stride) {
            float d = std::abs(a[row + x] - b[row + x]);
            sum_abs += d;
            ++count;
        }
    }
    return (count > 0) ? static_cast<float>(sum_abs / count) : 1.0f;
}

void decay_flow_confidence(std::vector<MotionVector>& flow, float decay) {
    decay = std::clamp(decay, 0.0f, 1.0f);
    if (decay >= 0.9999f) {
        return;
    }
    for (auto& mv : flow) {
        mv.confidence *= decay;
    }
}

void compute_sparse_block_matching_flow(const FloatImage& prev,
                                        const FloatImage& curr,
                                        int search_radius,
                                        int block_radius,
                                        int step,
                                        float motion_cap,
                                        FloatImage& flow_x,
                                        FloatImage& flow_y) {
    const int w = prev.width();
    const int h = prev.height();
    flow_x = FloatImage(w, h, 0.0f);
    flow_y = FloatImage(w, h, 0.0f);
    if (w <= 0 || h <= 0 || w != curr.width() || h != curr.height()) {
        return;
    }

    search_radius = std::max(1, search_radius);
    block_radius = std::max(1, block_radius);
    step = std::max(1, step);
    motion_cap = std::max(0.0f, motion_cap);

    const int margin = search_radius + block_radius;
    if (w <= 2 * margin || h <= 2 * margin) {
        return;
    }

    const float* a = prev.data();
    const float* b = curr.data();
    float* fx = flow_x.data();
    float* fy = flow_y.data();

    // Each sparse sample owns a disjoint half-open tile, so rows can be solved in parallel.
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int y = margin; y < h - margin; y += step) {
        const int y0 = y == margin ? 0 : y - step / 2;
        const int y1 = y + step >= h - margin ? h : y + (step + 1) / 2;
        for (int x = margin; x < w - margin; x += step) {
            float best_cost = std::numeric_limits<float>::infinity();
            int best_dx = 0;
            int best_dy = 0;

            for (int dy = -search_radius; dy <= search_radius; ++dy) {
                for (int dx = -search_radius; dx <= search_radius; ++dx) {
                    float sad = 0.0f;
                    for (int ky = -block_radius; ky <= block_radius; ++ky) {
                        const float* row_a = a + static_cast<size_t>(y + ky) * w;
                        const float* row_b = b + static_cast<size_t>(y + dy + ky) * w;
                        for (int kx = -block_radius; kx <= block_radius; ++kx) {
                            sad += std::abs(row_a[x + kx] - row_b[x + dx + kx]);
                        }
                        if (sad >= best_cost) {
                            break;
                        }
                    }
                    if (sad < best_cost) {
                        best_cost = sad;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            const float best_fx = std::clamp(static_cast<float>(best_dx), -motion_cap, motion_cap);
            const float best_fy = std::clamp(static_cast<float>(best_dy), -motion_cap, motion_cap);
            const int x0 = x == margin ? 0 : x - step / 2;
            const int x1 = x + step >= w - margin ? w : x + (step + 1) / 2;
            for (int yy = y0; yy < y1; ++yy) {
                float* row_fx = fx + static_cast<size_t>(yy) * w;
                float* row_fy = fy + static_cast<size_t>(yy) * w;
                for (int xx = x0; xx < x1; ++xx) {
                    row_fx[xx] = best_fx;
                    row_fy[xx] = best_fy;
                }
            }
        }
    }

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

    fft_2d(ws.prev_fft.data(), fft_w, fft_h, false, ws.scratch);
    fft_2d(ws.curr_fft.data(), fft_w, fft_h, false, ws.scratch);

    for (size_t i = 0; i < ws.cross_power.size(); ++i) {
        std::complex<float> cross = ws.curr_fft[i] * std::conj(ws.prev_fft[i]);
        float mag = std::abs(cross);
        if (mag > 1e-12f) {
            ws.cross_power[i] = cross / mag;
        }
    }
    fft_2d(ws.cross_power.data(), fft_w, fft_h, true, ws.scratch);

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

MotionEstimator::MotionEstimator(const Config& config) {
    set_config(config);
}

void MotionEstimator::set_config(const Config& config) {
    if (!std::isfinite(config.motion_cap) || config.motion_cap < 0.0f) {
        throw std::invalid_argument("Motion limit must be finite and non-negative");
    }
    config_ = config;
    reset();
}

void MotionEstimator::reset() {
    flow_.clear();
    width_ = 0;
    height_ = 0;
    frame_counter_ = 0;
    reused_frames_ = 0;
    last_phase_mv_ = MotionVector{};
    has_last_phase_ = false;
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
        
#ifdef HAS_OPENMP
        #pragma omp parallel for schedule(static)
#endif
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

void MotionEstimator::compute_flow(const FloatImage& prev, const FloatImage& curr) {
    if (prev.empty() || curr.empty() ||
        prev.width() != curr.width() || prev.height() != curr.height()) {
        reset();
        return;
    }
    if (width_ != prev.width() || height_ != prev.height()) {
        reset();
    }
    
    width_ = prev.width();
    height_ = prev.height();
    const int full_count = width_ * height_;
    if (static_cast<int>(flow_.size()) != full_count) {
        flow_.assign(full_count, MotionVector{});
    }

    const float scene_change = estimate_scene_change_score(prev, curr);
    if (config_.still_scene_threshold > 0.0f &&
        scene_change < config_.still_scene_threshold) {
        std::fill(flow_.begin(), flow_.end(), MotionVector{});
        reused_frames_ = 0;
        has_last_phase_ = false;
        ++frame_counter_;
        return;
    }

    const bool can_reuse =
        frame_counter_ > 0 &&
        config_.max_reuse_frames > 0 &&
        reused_frames_ < config_.max_reuse_frames &&
        scene_change < config_.reuse_scene_threshold;
    if (can_reuse) {
        decay_flow_confidence(flow_, config_.reuse_confidence_decay);
        ++reused_frames_;
        ++frame_counter_;
        return;
    }
    reused_frames_ = 0;

    int solve_div = std::clamp(config_.solve_divisor, 1, 8);
    const int dst_w = std::max(1, width_ / solve_div);
    const int dst_h = std::max(1, height_ / solve_div);
    const bool downsample = solve_div > 1 && dst_w >= 16 && dst_h >= 16;
    FloatImage prev_work = downsample ? downsample_area_average(prev, dst_w, dst_h) : prev;
    FloatImage curr_work = downsample ? downsample_area_average(curr, dst_w, dst_h) : curr;
    float flow_scale_x = downsample ? static_cast<float>(width_) / prev_work.width() : 1.0f;
    float flow_scale_y = downsample ? static_cast<float>(height_) / prev_work.height() : 1.0f;

    int requested_levels = std::clamp(config_.pyramid_levels, 1, 4);
    build_pyramid(prev_work, prev_pyramid_, requested_levels);
    build_pyramid(curr_work, curr_pyramid_, requested_levels);

    int levels_available = 1;
    for (int l = 1; l < requested_levels; ++l) {
        if (prev_pyramid_[l].empty() || curr_pyramid_[l].empty()) {
            break;
        }
        levels_available = l + 1;
    }
    
    FloatImage flow_x, flow_y;
    const float work_scale = std::min(flow_scale_x, flow_scale_y);
    const float work_motion_cap = std::max(1.0f, config_.motion_cap / work_scale);
    const int max_search = std::max(1, std::min(prev_work.width(), prev_work.height()) / 6);
    const int search_radius = static_cast<int>(std::min(std::ceil(work_motion_cap), static_cast<float>(max_search)));
    const int block_radius = std::clamp(config_.block_match_radius, 1, 8);
    const int step = std::clamp(config_.block_match_step, 1, 16);
    compute_sparse_block_matching_flow(
        prev_work, curr_work, search_radius, block_radius, step,
        work_motion_cap, flow_x, flow_y);

    MotionVector phase_mv{};
    if (config_.use_phase_correlation) {
        const int phase_interval = std::max(1, config_.phase_interval);
        const bool refresh_phase =
            !has_last_phase_ ||
            (frame_counter_ % phase_interval == 0) ||
            (scene_change >= config_.phase_scene_trigger);
        if (refresh_phase) {
            phase_mv = estimate_phase_correlation_hierarchical(
                prev_pyramid_, curr_pyramid_, levels_available, config_.phase_search_radius);
            last_phase_mv_ = phase_mv;
            has_last_phase_ = true;
        } else {
            phase_mv = last_phase_mv_;
        }
    } else {
        has_last_phase_ = false;
    }

    if (flow_x.width() != width_ || flow_x.height() != height_) {
        flow_x = upsample_flow_field(flow_x, width_, height_, flow_scale_x);
        flow_y = upsample_flow_field(flow_y, width_, height_, flow_scale_y);
    }
    phase_mv.dx *= flow_scale_x;
    phase_mv.dy *= flow_scale_y;
    
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            int idx = y * width_ + x;
            float dx = flow_x.get_clamped(x, y);
            float dy = flow_y.get_clamped(x, y);
            if (config_.use_phase_correlation) {
                float gated = std::clamp((phase_mv.confidence - 0.2f) / 0.8f, 0.0f, 1.0f);
                float blend = std::clamp(config_.phase_blend * gated * gated, 0.0f, 1.0f);
                dx = (1.0f - blend) * dx + blend * phase_mv.dx;
                dy = (1.0f - blend) * dy + blend * phase_mv.dy;
            }
            flow_[idx].dx = std::clamp(dx, -config_.motion_cap, config_.motion_cap);
            flow_[idx].dy = std::clamp(dy, -config_.motion_cap, config_.motion_cap);
        }
    }

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            const int idx = y * width_ + x;
            const auto& mv = flow_[idx];
            const int px = std::clamp(static_cast<int>(std::lround(x - mv.dx)), 0, width_ - 1);
            const int py = std::clamp(static_cast<int>(std::lround(y - mv.dy)), 0, height_ - 1);
            const float residual = std::abs(curr.get(x, y) - prev.get(px, py));
            const float photometric = 1.0f - std::clamp(residual / 0.25f, 0.0f, 1.0f);
            const float gx = std::abs(prev.get_clamped(px + 1, py) - prev.get_clamped(px - 1, py));
            const float gy = std::abs(prev.get_clamped(px, py + 1) - prev.get_clamped(px, py - 1));
            const float texture = std::clamp((gx + gy) / 0.2f, 0.0f, 1.0f);

            float consistency_error = 0.0f;
            int neighbors = 0;
            constexpr int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
            for (const auto& offset : offsets) {
                const int nx = x + offset[0];
                const int ny = y + offset[1];
                if (nx < 0 || nx >= width_ || ny < 0 || ny >= height_) continue;
                const auto& neighbor = flow_[ny * width_ + nx];
                const float delta_x = mv.dx - neighbor.dx;
                const float delta_y = mv.dy - neighbor.dy;
                // Axis-aligned differences have exact length without a hypot call.
                consistency_error += delta_x == 0.0f ? std::abs(delta_y)
                    : delta_y == 0.0f ? std::abs(delta_x)
                    : std::hypot(delta_x, delta_y);
                ++neighbors;
            }
            const float consistency = neighbors > 0
                ? 1.0f - std::clamp(
                    consistency_error / (neighbors * std::max(config_.motion_cap, 1.0f)),
                    0.0f, 1.0f)
                : 0.0f;
            float confidence = photometric * (0.2f + 0.8f * texture) * consistency;
            if (config_.use_phase_correlation) {
                confidence = std::max(confidence, phase_mv.confidence * photometric * 0.7f);
            }
            flow_[idx].confidence = std::clamp(confidence, 0.0f, 1.0f);
        }
    }

    ++frame_counter_;
}

const MotionVector& MotionEstimator::get_motion(int x, int y) const {
    static const MotionVector empty;
    if (x < 0 || x >= width_ || y < 0 || y >= height_ || flow_.empty()) {
        return empty;
    }
    return flow_[y * width_ + x];
}

MotionVector MotionEstimator::get_motion_interpolated(float x, float y) const {
    if (flow_.empty() || width_ == 0 || height_ == 0 || !std::isfinite(x) || !std::isfinite(y)) {
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
    if (flow_.empty() || w <= 0 || h <= 0) {
        dx = 0.0f;
        dy = 0.0f;
        return;
    }
    
    float sum_dx = 0.0f, sum_dy = 0.0f;
    float sum_confidence = 0.0f;
    int count = 0;
    
    const int x1 = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(x0) + w, 0, width_));
    const int y1 = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(y0) + h, 0, height_));
    x0 = std::clamp(x0, 0, width_);
    y0 = std::clamp(y0, 0, height_);
    
    for (int y = y0; y < y1; ++y) {
        for (int x = x0; x < x1; ++x) {
            const auto& mv = get_motion(x, y);
            sum_dx += mv.dx * mv.confidence;
            sum_dy += mv.dy * mv.confidence;
            sum_confidence += mv.confidence;
            count++;
        }
    }
    
    if (count > 0 && sum_confidence > 1e-6f) {
        const float mean_confidence = sum_confidence / count;
        dx = (sum_dx / sum_confidence) * mean_confidence;
        dy = (sum_dy / sum_confidence) * mean_confidence;
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
