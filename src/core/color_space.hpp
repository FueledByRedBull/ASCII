#pragma once

#include <cstdint>
#include <cmath>

namespace ascii {

struct LinearColor {
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    
    LinearColor() = default;
    LinearColor(float r, float g, float b) : r(r), g(g), b(b) {}
    
    float luminance() const {
        return 0.2126f * r + 0.7152f * g + 0.0722f * b;
    }
    
    LinearColor operator+(const LinearColor& o) const { return {r + o.r, g + o.g, b + o.b}; }
    LinearColor operator-(const LinearColor& o) const { return {r - o.r, g - o.g, b - o.b}; }
    LinearColor operator*(float s) const { return {r * s, g * s, b * s}; }
};

struct OKLab {
    float L = 0.0f;
    float a = 0.0f;
    float b = 0.0f;
    
    OKLab() = default;
    OKLab(float L, float a, float b) : L(L), a(a), b(b) {}
    
    static float distance(const OKLab& c1, const OKLab& c2) {
        float dL = c1.L - c2.L;
        float da = c1.a - c2.a;
        float db = c1.b - c2.b;
        return std::sqrt(dL * dL + da * da + db * db);
    }
};

class ColorSpace {
public:
    static void init();
    
    static float srgb_to_linear(uint8_t srgb);
    static uint8_t linear_to_srgb(float linear);
    
    static LinearColor srgb_to_linear(uint8_t r, uint8_t g, uint8_t b);
    static void linear_to_srgb(const LinearColor& linear, uint8_t& r, uint8_t& g, uint8_t& b);
    
    static LinearColor srgb_to_linear_fast(uint8_t r, uint8_t g, uint8_t b);
    static void linear_to_srgb_fast(const LinearColor& linear, uint8_t& r, uint8_t& g, uint8_t& b);
    
    static OKLab to_oklab(const LinearColor& linear);
    static OKLab srgb_to_oklab(uint8_t r, uint8_t g, uint8_t b);
    
    static LinearColor from_oklab(const OKLab& lab);
    static void oklab_to_srgb(const OKLab& lab, uint8_t& r, uint8_t& g, uint8_t& b);
    
    static float oklab_distance_srgb(uint8_t r1, uint8_t g1, uint8_t b1,
                                      uint8_t r2, uint8_t g2, uint8_t b2);
    
    static float perceptual_luminance(uint8_t r, uint8_t g, uint8_t b);
    
private:
    static float srgb_decode_lut_[256];
    static uint8_t srgb_encode_lut_[4096];
    static bool initialized_;
    
    static float srgb_decode(uint8_t c);
    static uint8_t srgb_encode(float c);
    
    static float lab_f(float t);
    static float lab_f_inv(float t);
};

}
