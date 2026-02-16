#include "glyph_cache.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace {

constexpr int kFreqBins = 8;
constexpr int kTextureBins = 8;

std::vector<float> compute_frequency_signature(const ascii::GlyphBitmap& bmp) {
    std::vector<float> signature(kFreqBins, 0.0f);
    if (bmp.width <= 0 || bmp.height <= 0 || bmp.pixels.empty()) {
        return signature;
    }

    constexpr int W = 8;
    constexpr int H = 16;
    constexpr float kPi = 3.14159265358979323846f;
    float sample[H][W] = {};
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int sx = std::clamp((x * bmp.width) / W, 0, bmp.width - 1);
            int sy = std::clamp((y * bmp.height) / H, 0, bmp.height - 1);
            sample[y][x] = bmp.pixels[sy * bmp.width + sx] / 255.0f;
        }
    }

    float dct[H][W] = {};
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            float sum = 0.0f;
            for (int y = 0; y < H; ++y) {
                for (int x = 0; x < W; ++x) {
                    float cx = std::cos((kPi * (2.0f * x + 1.0f) * u) / (2.0f * W));
                    float cy = std::cos((kPi * (2.0f * y + 1.0f) * v) / (2.0f * H));
                    sum += sample[y][x] * cx * cy;
                }
            }
            float au = (u == 0) ? std::sqrt(1.0f / W) : std::sqrt(2.0f / W);
            float av = (v == 0) ? std::sqrt(1.0f / H) : std::sqrt(2.0f / H);
            dct[v][u] = au * av * sum;
        }
    }

    // Low-frequency terms for an 8x16 basis, excluding DC.
    static constexpr std::array<std::pair<int, int>, kFreqBins> kZigZag = {
        std::pair<int, int>{1, 0}, {0, 1}, {2, 0}, {1, 1},
        {0, 2}, {3, 0}, {2, 1}, {0, 3}
    };
    for (int i = 0; i < kFreqBins; ++i) {
        int u = kZigZag[i].first;
        int v = kZigZag[i].second;
        signature[i] = dct[v][u];
    }

    float norm = 0.0f;
    for (float v : signature) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& v : signature) v /= norm;
    }
    return signature;
}

std::vector<float> compute_texture_signature(const ascii::GlyphBitmap& bmp) {
    std::vector<float> out(kTextureBins, 0.0f);
    if (bmp.width < 5 || bmp.height < 5 || bmp.pixels.empty()) {
        return out;
    }

    constexpr int kRadius = 2;
    constexpr float kSigma = 1.3f;
    constexpr float kGamma = 0.6f;
    constexpr float kPi = 3.14159265358979323846f;
    constexpr std::array<float, 4> kAngles = {
        0.0f,
        0.25f * kPi,
        0.5f * kPi,
        0.75f * kPi
    };
    constexpr std::array<float, 2> kLambdas = {3.2f, 6.4f};

    for (int fi = 0; fi < static_cast<int>(kLambdas.size()); ++fi) {
        float lambda = kLambdas[fi];
        for (int oi = 0; oi < static_cast<int>(kAngles.size()); ++oi) {
            float theta = kAngles[oi];
            float ct = std::cos(theta);
            float st = std::sin(theta);
            float energy = 0.0f;

            for (int y = kRadius; y < bmp.height - kRadius; ++y) {
                for (int x = kRadius; x < bmp.width - kRadius; ++x) {
                    float resp = 0.0f;
                    for (int ky = -kRadius; ky <= kRadius; ++ky) {
                        for (int kx = -kRadius; kx <= kRadius; ++kx) {
                            float xr = kx * ct + ky * st;
                            float yr = -kx * st + ky * ct;
                            float gauss = std::exp(-(xr * xr + (kGamma * kGamma) * yr * yr) / (2.0f * kSigma * kSigma));
                            float carrier = std::cos((2.0f * kPi * xr) / lambda);
                            float kernel = gauss * carrier;
                            float v = bmp.pixels[(y + ky) * bmp.width + (x + kx)] / 255.0f;
                            resp += kernel * v;
                        }
                    }
                    energy += std::abs(resp);
                }
            }
            out[fi * 4 + oi] = energy;
        }
    }

    float norm = 0.0f;
    for (float v : out) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (float& v : out) v /= norm;
    }
    return out;
}

}

namespace ascii {

GlyphCache::GlyphCache() = default;

bool GlyphCache::initialize(FontLoader* loader, const std::vector<uint32_t>& codepoints, int target_width, int target_height) {
    if (!loader) return false;
    
    loader_ = loader;
    cell_width_ = target_width;
    cell_height_ = target_height;
    
    bitmaps_.clear();
    stats_.clear();
    brightness_sorted_.clear();
    edge_glyphs_.clear();
    
    for (uint32_t cp : codepoints) {
        render_and_analyze(cp);
    }
    
    brightness_sorted_.reserve(stats_.size());
    for (const auto& [cp, _] : stats_) {
        brightness_sorted_.push_back(cp);
    }
    
    std::sort(brightness_sorted_.begin(), brightness_sorted_.end(), [this](uint32_t a, uint32_t b) {
        auto* sa = get_stats(a);
        auto* sb = get_stats(b);
        if (!sa || !sb) return false;
        return sa->brightness < sb->brightness;
    });
    
    for (uint32_t cp : brightness_sorted_) {
        auto* s = get_stats(cp);
        if (s && s->is_good_edge_glyph()) {
            edge_glyphs_.push_back(cp);
        }
    }
    
    return true;
}

const GlyphStats* GlyphCache::get_stats(uint32_t codepoint) const {
    auto it = stats_.find(codepoint);
    return it != stats_.end() ? &it->second : nullptr;
}

const GlyphBitmap* GlyphCache::get_bitmap(uint32_t codepoint) const {
    auto it = bitmaps_.find(codepoint);
    return it != bitmaps_.end() ? &it->second : nullptr;
}

std::vector<uint32_t> GlyphCache::get_by_brightness() const {
    return brightness_sorted_;
}

std::vector<uint32_t> GlyphCache::get_edge_glyphs() const {
    return edge_glyphs_;
}

void GlyphCache::render_and_analyze(uint32_t codepoint) {
    if (!loader_) return;
    
    GlyphBitmap src = loader_->render_glyph(codepoint);
    if (src.empty()) return;
    
    GlyphBitmap scaled;
    scaled.width = cell_width_;
    scaled.height = cell_height_;
    scaled.advance = cell_width_;
    scaled.bearing_x = 0;
    scaled.bearing_y = 0;
    scaled.pixels.resize(cell_width_ * cell_height_, 0);
    
    float scale_x = static_cast<float>(src.width) / cell_width_;
    float scale_y = static_cast<float>(src.height) / cell_height_;
    
    for (int y = 0; y < cell_height_; ++y) {
        for (int x = 0; x < cell_width_; ++x) {
            int sx = static_cast<int>(x * scale_x);
            int sy = static_cast<int>(y * scale_y);
            sx = std::min(sx, src.width - 1);
            sy = std::min(sy, src.height - 1);
            scaled.pixels[y * cell_width_ + x] = src.pixels[sy * src.width + sx];
        }
    }
    
    bitmaps_[codepoint] = std::move(scaled);
    
    GlyphStats stats;
    stats.codepoint = codepoint;
    stats.brightness = bitmaps_[codepoint].brightness();
    stats.orientation_hist = bitmaps_[codepoint].orientation_histogram(8);
    stats.contrast = 0.0f;
    stats.frequency_signature = compute_frequency_signature(bitmaps_[codepoint]);
    stats.texture_signature = compute_texture_signature(bitmaps_[codepoint]);
    
    const auto& bmp = bitmaps_[codepoint];
    float mean = stats.brightness;
    float var_sum = 0.0f;
    for (uint8_t p : bmp.pixels) {
        float diff = p / 255.0f - mean;
        var_sum += diff * diff;
    }
    stats.contrast = std::sqrt(var_sum / bmp.pixels.size());
    
    stats_[codepoint] = stats;
}

}
