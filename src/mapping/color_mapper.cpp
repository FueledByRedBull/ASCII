#include "color_mapper.hpp"
#include <algorithm>
#include <cmath>

namespace ascii {

OKLab ColorMapper::palette16_oklab_[16];
OKLab ColorMapper::palette256_oklab_[256];
bool ColorMapper::palettes_initialized_ = false;

ColorMapper::ColorMapper(ColorMode mode) : mode_(mode) {
    init_palettes();
}

void ColorMapper::init_palettes() {
    if (palettes_initialized_) return;
    
    ColorSpace::init();
    
    static const uint8_t palette16[16][3] = {
        {0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
        {0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {192, 192, 192},
        {128, 128, 128}, {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
        {0, 0, 255}, {255, 0, 255}, {0, 255, 255}, {255, 255, 255}
    };
    
    for (int i = 0; i < 16; ++i) {
        palette16_oklab_[i] = ColorSpace::srgb_to_oklab(
            palette16[i][0], palette16[i][1], palette16[i][2]);
    }
    
    for (int i = 0; i < 256; ++i) {
        uint8_t r, g, b;
        if (i < 16) {
            r = palette16[i][0];
            g = palette16[i][1];
            b = palette16[i][2];
        } else if (i < 232) {
            int idx = i - 16;
            int rv = idx / 36;
            int gv = (idx % 36) / 6;
            int bv = idx % 6;
            r = rv ? static_cast<uint8_t>(55 + rv * 40) : 0;
            g = gv ? static_cast<uint8_t>(55 + gv * 40) : 0;
            b = bv ? static_cast<uint8_t>(55 + bv * 40) : 0;
        } else {
            uint8_t gray = static_cast<uint8_t>(8 + (i - 232) * 10);
            r = g = b = gray;
        }
        palette256_oklab_[i] = ColorSpace::srgb_to_oklab(r, g, b);
    }
    
    palettes_initialized_ = true;
}

uint8_t ColorMapper::find_nearest_palette_oklab(float r, float g, float b,
                                                  const OKLab* palette, int palette_size) {
    OKLab target = ColorSpace::srgb_to_oklab(
        static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f)),
        static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f)));
    
    float best_dist = 1e10f;
    uint8_t best_idx = 0;
    
    for (int i = 0; i < palette_size; ++i) {
        float dist = OKLab::distance(target, palette[i]);
        if (dist < best_dist) {
            best_dist = dist;
            best_idx = static_cast<uint8_t>(i);
        }
    }
    
    return best_idx;
}

uint8_t ColorMapper::find_nearest_16_oklab(uint8_t r, uint8_t g, uint8_t b) {
    return find_nearest_palette_oklab(r / 255.0f, g / 255.0f, b / 255.0f,
                                       palette16_oklab_, 16);
}

uint8_t ColorMapper::find_nearest_256_oklab(uint8_t r, uint8_t g, uint8_t b) {
    return find_nearest_palette_oklab(r / 255.0f, g / 255.0f, b / 255.0f,
                                       palette256_oklab_, 256);
}

HSV ColorMapper::rgb_to_hsv(float r, float g, float b) {
    HSV hsv;
    float max_val = std::max({r, g, b});
    float min_val = std::min({r, g, b});
    float delta = max_val - min_val;
    
    hsv.v = max_val;
    
    if (delta < 0.00001f) {
        hsv.h = 0.0f;
        hsv.s = 0.0f;
        return hsv;
    }
    
    if (max_val > 0.0f) {
        hsv.s = delta / max_val;
    } else {
        hsv.s = 0.0f;
        hsv.h = 0.0f;
        return hsv;
    }
    
    if (r >= max_val) {
        hsv.h = (g - b) / delta;
    } else if (g >= max_val) {
        hsv.h = 2.0f + (b - r) / delta;
    } else {
        hsv.h = 4.0f + (r - g) / delta;
    }
    
    hsv.h *= 60.0f;
    if (hsv.h < 0.0f) {
        hsv.h += 360.0f;
    }
    
    return hsv;
}

void ColorMapper::hsv_to_rgb(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b) {
    if (s <= 0.0f) {
        r = g = b = static_cast<uint8_t>(std::clamp(v * 255.0f, 0.0f, 255.0f));
        return;
    }
    
    while (h >= 360.0f) h -= 360.0f;
    while (h < 0.0f) h += 360.0f;
    
    h /= 60.0f;
    int i = static_cast<int>(h);
    float ff = h - i;
    float p = v * (1.0f - s);
    float q = v * (1.0f - (s * ff));
    float t = v * (1.0f - (s * (1.0f - ff)));
    
    float rf, gf, bf;
    
    switch (i) {
        case 0:  rf = v; gf = t; bf = p; break;
        case 1:  rf = q; gf = v; bf = p; break;
        case 2:  rf = p; gf = v; bf = t; break;
        case 3:  rf = p; gf = q; bf = v; break;
        case 4:  rf = t; gf = p; bf = v; break;
        default: rf = v; gf = p; bf = q; break;
    }
    
    r = static_cast<uint8_t>(std::clamp(rf * 255.0f, 0.0f, 255.0f));
    g = static_cast<uint8_t>(std::clamp(gf * 255.0f, 0.0f, 255.0f));
    b = static_cast<uint8_t>(std::clamp(bf * 255.0f, 0.0f, 255.0f));
}

ColorMapper::MappedColor ColorMapper::map(uint8_t r, uint8_t g, uint8_t b) const {
    switch (mode_) {
        case ColorMode::None:
            return {128, 128, 128};
        case ColorMode::Ansi16: {
            uint8_t idx = find_nearest_16_oklab(r, g, b);
            static const uint8_t palette[16][3] = {
                {0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
                {0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {192, 192, 192},
                {128, 128, 128}, {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
                {0, 0, 255}, {255, 0, 255}, {0, 255, 255}, {255, 255, 255}
            };
            return {palette[idx][0], palette[idx][1], palette[idx][2]};
        }
        case ColorMode::Ansi256: {
            uint8_t idx = find_nearest_256_oklab(r, g, b);
            if (idx < 16) {
                static const uint8_t palette[16][3] = {
                    {0, 0, 0}, {128, 0, 0}, {0, 128, 0}, {128, 128, 0},
                    {0, 0, 128}, {128, 0, 128}, {0, 128, 128}, {192, 192, 192},
                    {128, 128, 128}, {255, 0, 0}, {0, 255, 0}, {255, 255, 0},
                    {0, 0, 255}, {255, 0, 255}, {0, 255, 255}, {255, 255, 255}
                };
                return {palette[idx][0], palette[idx][1], palette[idx][2]};
            }
            if (idx >= 232) {
                uint8_t gray = 8 + (idx - 232) * 10;
                return {gray, gray, gray};
            }
            idx -= 16;
            uint8_t red = (idx / 36) ? 55 + (idx / 36) * 40 : 0;
            uint8_t green = ((idx % 36) / 6) ? 55 + ((idx % 36) / 6) * 40 : 0;
            uint8_t blue = (idx % 6) ? 55 + (idx % 6) * 40 : 0;
            return {red, green, blue};
        }
        case ColorMode::BlockArt:
        case ColorMode::Truecolor:
        default:
            return {r, g, b};
    }
}

ColorMapper::MappedColor ColorMapper::map_with_dither(int x, int y, int row_dir,
                                                       float r, float g, float b, bool is_edge) const {
    if (ditherer_ && ditherer_->should_dither_cell(is_edge)) {
        ditherer_->apply_dithering(x, y, r, g, b);
    }
    
    uint8_t ri = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
    uint8_t gi = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
    uint8_t bi = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
    
    MappedColor result = map(ri, gi, bi);
    
    if (ditherer_ && ditherer_->should_dither_cell(is_edge)) {
        float er = r - (result.r / 255.0f);
        float eg = g - (result.g / 255.0f);
        float eb = b - (result.b / 255.0f);
        ditherer_->distribute_error(x, y, row_dir, er, eg, eb);
    }
    
    return result;
}

ColorMapper::MappedColor ColorMapper::map_luminance(float lum) const {
    uint8_t v = static_cast<uint8_t>(std::clamp(lum * 255.0f, 0.0f, 255.0f));
    return {v, v, v};
}

ColorMapper::MappedColor ColorMapper::map_rgb(float r, float g, float b) const {
    uint8_t ri = static_cast<uint8_t>(std::clamp(r * 255.0f, 0.0f, 255.0f));
    uint8_t gi = static_cast<uint8_t>(std::clamp(g * 255.0f, 0.0f, 255.0f));
    uint8_t bi = static_cast<uint8_t>(std::clamp(b * 255.0f, 0.0f, 255.0f));
    
    if (mode_ == ColorMode::None) {
        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        return map_luminance(lum);
    }
    
    return map(ri, gi, bi);
}

ColorMapper::BlockArtResult ColorMapper::map_block_art(uint8_t r, uint8_t g, uint8_t b) const {
    ColorMapper::BlockArtResult result;
    
    LinearColor linear = ColorSpace::srgb_to_linear(r, g, b);
    float lum = linear.luminance();
    
    static const uint32_t block_chars[] = {
        0x0020,
        0x2591,
        0x2592,
        0x2593,
        0x2588
    };
    
    if (lum < 0.125f) {
        result.codepoint = block_chars[0];
        result.fg_r = result.fg_g = result.fg_b = 0;
        result.bg_r = result.bg_g = result.bg_b = 0;
    } else if (lum < 0.375f) {
        result.codepoint = block_chars[1];
        result.fg_r = r; result.fg_g = g; result.fg_b = b;
        result.bg_r = 0; result.bg_g = 0; result.bg_b = 0;
    } else if (lum < 0.625f) {
        result.codepoint = block_chars[2];
        result.fg_r = r; result.fg_g = g; result.fg_b = b;
        result.bg_r = 0; result.bg_g = 0; result.bg_b = 0;
    } else if (lum < 0.875f) {
        result.codepoint = block_chars[3];
        result.fg_r = r; result.fg_g = g; result.fg_b = b;
        result.bg_r = 0; result.bg_g = 0; result.bg_b = 0;
    } else {
        result.codepoint = block_chars[4];
        result.fg_r = r; result.fg_g = g; result.fg_b = b;
        result.bg_r = r; result.bg_g = g; result.bg_b = b;
    }
    
    return result;
}

}
