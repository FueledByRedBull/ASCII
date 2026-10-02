#include "glyph_cache.hpp"
#include "core/cell_stats.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace {

constexpr int kFreqBins = 8;
constexpr int kTextureBins = 8;

std::vector<float> compute_orientation_signature(const ascii::GlyphBitmap& bitmap,
                                                 const ascii::EdgeDetector::Config& config) {
    ascii::FloatImage coverage(bitmap.width, bitmap.height);
    for (size_t i = 0; i < bitmap.pixels.size(); ++i) coverage.data()[i] = bitmap.pixels[i] / 255.0f;
    ascii::EdgeDetector detector(config);
    ascii::GradientData gradients;
    if (config.multi_scale) {
        auto selected = detector.compute_multi_scale_gradients(coverage);
        gradients.gx = std::move(selected.gx);
        gradients.gy = std::move(selected.gy);
    } else {
        gradients = detector.compute_gradients(coverage);
    }
    ascii::CellStatsAggregator::Config cell_config;
    cell_config.cell_width = bitmap.width;
    cell_config.cell_height = bitmap.height;
    cell_config.enable_frequency_signature = false;
    cell_config.enable_texture_signature = false;
    const auto stats = ascii::CellStatsAggregator(cell_config).compute(coverage, {}, nullptr, &gradients);
    return {stats[0].orientation_histogram, stats[0].orientation_histogram + 8};
}

constexpr std::array<uint32_t, 32> kRendererGlyphs = {
    0x0020, 0x2580, 0x2584, 0x2588, 0x258C, 0x2590, 0x2591,
    0x2592, 0x2593, 0x2596, 0x2597, 0x2598, 0x2599, 0x259A,
    0x259B, 0x259C, 0x259D, 0x259E, 0x259F, '-', '|', '/', '\\', '+',
    '.', ':', '=', '*', '#', '%', '@', 'O'
};

uint8_t bilinear_sample(const ascii::GlyphBitmap& src, float x, float y) {
    if (src.pixels.empty() || src.width <= 0 || src.height <= 0) return 0;
    x = std::clamp(x, 0.0f, static_cast<float>(src.width - 1));
    y = std::clamp(y, 0.0f, static_cast<float>(src.height - 1));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, src.width - 1);
    const int y1 = std::min(y0 + 1, src.height - 1);
    const float fx = x - x0;
    const float fy = y - y0;
    const float top = src.pixels[y0 * src.width + x0] * (1.0f - fx) +
                      src.pixels[y0 * src.width + x1] * fx;
    const float bottom = src.pixels[y1 * src.width + x0] * (1.0f - fx) +
                         src.pixels[y1 * src.width + x1] * fx;
    return static_cast<uint8_t>(std::lround(top * (1.0f - fy) + bottom * fy));
}

ascii::GlyphBitmap procedural_block(uint32_t codepoint, int width, int height) {
    ascii::GlyphBitmap bitmap;
    bitmap.width = width;
    bitmap.height = height;
    bitmap.advance = width;
    bitmap.pixels.assign(static_cast<size_t>(width) * height, 0);
    const int half_x = width / 2;
    const int half_y = height / 2;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            bool on = false;
            switch (codepoint) {
                case 0x2580: on = y < half_y; break;
                case 0x2584: on = y >= half_y; break;
                case 0x2588: on = true; break;
                case 0x258C: on = x < half_x; break;
                case 0x2590: on = x >= half_x; break;
                case 0x2591: on = ((x + 2 * y) % 4) == 0; break;
                case 0x2592: on = ((x + y) % 2) == 0; break;
                case 0x2593: on = ((x + 2 * y) % 4) != 0; break;
                case 0x2596: on = x < half_x && y >= half_y; break;
                case 0x2597: on = x >= half_x && y >= half_y; break;
                case 0x2598: on = x < half_x && y < half_y; break;
                case 0x259D: on = x >= half_x && y < half_y; break;
                case 0x2599: on = x < half_x || y >= half_y; break;
                case 0x259A: on = (x < half_x) == (y < half_y); break;
                case 0x259B: on = x < half_x || y < half_y; break;
                case 0x259C: on = x >= half_x || y < half_y; break;
                case 0x259E: on = (x < half_x) != (y < half_y); break;
                case 0x259F: on = x >= half_x || y >= half_y; break;
                default: break;
            }
            bitmap.pixels[static_cast<size_t>(y) * width + x] = on ? 255 : 0;
        }
    }
    return bitmap;
}

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

bool GlyphCache::initialize(FontLoader* loader, const std::vector<uint32_t>& codepoints, int target_width, int target_height,
                            const EdgeDetector::Config& orientation_config) {
    if (!loader || !loader->is_loaded() || target_width <= 0 || target_height <= 0 ||
        target_width > 1024 || target_height > 1024) return false;
    
    loader_ = loader;
    cell_width_ = target_width;
    cell_height_ = target_height;
    orientation_config_ = orientation_config;
    
    bitmaps_.clear();
    stats_.clear();
    brightness_sorted_.clear();
    edge_glyphs_.clear();
    
    std::vector<uint32_t> all_codepoints = codepoints;
    for (uint32_t cp : kRendererGlyphs) {
        if (std::find(all_codepoints.begin(), all_codepoints.end(), cp) == all_codepoints.end()) {
            all_codepoints.push_back(cp);
        }
    }
    for (uint32_t cp : all_codepoints) {
        render_and_analyze(cp);
    }
    
    brightness_sorted_.reserve(codepoints.size());
    for (uint32_t cp : codepoints) {
        if (stats_.find(cp) != stats_.end() &&
            std::find(brightness_sorted_.begin(), brightness_sorted_.end(), cp) == brightness_sorted_.end()) {
            brightness_sorted_.push_back(cp);
        }
    }
    
    std::sort(brightness_sorted_.begin(), brightness_sorted_.end(), [this](uint32_t a, uint32_t b) {
        auto* sa = get_stats(a);
        auto* sb = get_stats(b);
        if (!sa || !sb) return a < b;
        if (sa->brightness == sb->brightness) return a < b;
        return sa->brightness < sb->brightness;
    });
    
    for (uint32_t cp : brightness_sorted_) {
        auto* s = get_stats(cp);
        if (s && s->is_good_edge_glyph()) {
            edge_glyphs_.push_back(cp);
        }
    }
    
    return !brightness_sorted_.empty();
}

const GlyphStats* GlyphCache::get_stats(uint32_t codepoint) const {
    auto it = stats_.find(codepoint);
    return it != stats_.end() ? &it->second : nullptr;
}

const GlyphBitmap* GlyphCache::get_bitmap(uint32_t codepoint) const {
    auto it = bitmaps_.find(codepoint);
    return it != bitmaps_.end() ? &it->second : nullptr;
}

const std::vector<uint32_t>& GlyphCache::get_by_brightness() const {
    return brightness_sorted_;
}

const std::vector<uint32_t>& GlyphCache::get_edge_glyphs() const {
    return edge_glyphs_;
}

void GlyphCache::render_and_analyze(uint32_t codepoint) {
    if (!loader_) return;

    const bool is_renderer_block = codepoint >= 0x2580 && codepoint <= 0x259F &&
        std::find(kRendererGlyphs.begin(), kRendererGlyphs.end(), codepoint) != kRendererGlyphs.end();
    if (!is_renderer_block && codepoint != 0x0020 && !loader_->has_glyph(codepoint)) return;
    GlyphBitmap src = is_renderer_block ? GlyphBitmap{} : loader_->render_glyph(codepoint);
    if (src.empty() && codepoint != 0x0020 && !is_renderer_block) return;

    GlyphBitmap scaled;
    scaled.width = cell_width_;
    scaled.height = cell_height_;
    scaled.advance = src.advance > 0 ? src.advance : cell_width_;
    scaled.bearing_x = src.bearing_x;
    scaled.bearing_y = src.bearing_y;
    scaled.pixels.assign(static_cast<size_t>(cell_width_) * cell_height_, 0);

    if (is_renderer_block) {
        // Block elements are renderer primitives, so their cell coverage must
        // not depend on the platform font's bearings or internal padding.
        scaled = procedural_block(codepoint, cell_width_, cell_height_);
    } else if (!src.empty()) {
        const float shrink = std::min({1.0f,
            static_cast<float>(cell_width_) / std::max(1, src.width),
            static_cast<float>(cell_height_) / std::max(1, src.height)});
        const int draw_width = std::max(1, static_cast<int>(std::lround(src.width * shrink)));
        const int draw_height = std::max(1, static_cast<int>(std::lround(src.height * shrink)));
        const int advance = std::max(1, static_cast<int>(std::lround(src.advance * shrink)));
        const int pen_x = (cell_width_ - advance) / 2;
        const int origin_x = pen_x + static_cast<int>(std::lround(src.bearing_x * shrink));
        const int baseline = std::clamp(loader_->ascent_pixels(), 0, cell_height_ - 1);
        const int origin_y = baseline + static_cast<int>(std::lround(src.bearing_y * shrink));

        for (int y = 0; y < draw_height; ++y) {
            const int dy = origin_y + y;
            if (dy < 0 || dy >= cell_height_) continue;
            for (int x = 0; x < draw_width; ++x) {
                const int dx = origin_x + x;
                if (dx < 0 || dx >= cell_width_) continue;
                const float sx = ((x + 0.5f) * src.width / draw_width) - 0.5f;
                const float sy = ((y + 0.5f) * src.height / draw_height) - 0.5f;
                scaled.pixels[static_cast<size_t>(dy) * cell_width_ + dx] = bilinear_sample(src, sx, sy);
            }
        }
    }
    
    bitmaps_[codepoint] = std::move(scaled);
    
    GlyphStats stats;
    stats.codepoint = codepoint;
    stats.brightness = bitmaps_[codepoint].brightness();
    stats.orientation_hist = compute_orientation_signature(bitmaps_[codepoint], orientation_config_);
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
