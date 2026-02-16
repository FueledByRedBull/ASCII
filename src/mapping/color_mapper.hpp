#pragma once

#include "core/types.hpp"
#include "core/color_space.hpp"
#include "terminal/terminal.hpp"
#include "render/dither.hpp"
#include <vector>
#include <cstdint>

namespace ascii {

struct HSV {
    float h = 0.0f;
    float s = 0.0f;
    float v = 0.0f;
};

class ColorMapper {
public:
    struct MappedColor {
        uint8_t r, g, b;
    };
    
    struct BlockArtResult {
        uint32_t codepoint;
        uint8_t fg_r, fg_g, fg_b;
        uint8_t bg_r, bg_g, bg_b;
    };
    
    explicit ColorMapper(ColorMode mode = ColorMode::Truecolor);
    
    void set_mode(ColorMode mode) { mode_ = mode; }
    ColorMode mode() const { return mode_; }
    
    void set_ditherer(Ditherer* ditherer) { ditherer_ = ditherer; }
    
    MappedColor map(uint8_t r, uint8_t g, uint8_t b) const;
    MappedColor map_luminance(float lum) const;
    MappedColor map_rgb(float r, float g, float b) const;
    
    MappedColor map_with_dither(int x, int y, int row_dir, float r, float g, float b, bool is_edge) const;
    
    BlockArtResult map_block_art(uint8_t r, uint8_t g, uint8_t b) const;
    
    static HSV rgb_to_hsv(float r, float g, float b);
    static void hsv_to_rgb(float h, float s, float v, uint8_t& r, uint8_t& g, uint8_t& b);
    
    static uint8_t find_nearest_16_oklab(uint8_t r, uint8_t g, uint8_t b);
    static uint8_t find_nearest_256_oklab(uint8_t r, uint8_t g, uint8_t b);
    
private:
    ColorMode mode_;
    Ditherer* ditherer_ = nullptr;
    
    static OKLab palette16_oklab_[16];
    static OKLab palette256_oklab_[256];
    static bool palettes_initialized_;
    
    static void init_palettes();
    
    static uint8_t find_nearest_palette_oklab(float r, float g, float b, 
                                               const OKLab* palette, int palette_size);
};

}
