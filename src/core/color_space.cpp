#include "core/color_space.hpp"
#include <cmath>
#include <algorithm>

namespace ascii {

float ColorSpace::srgb_decode_lut_[256];
uint8_t ColorSpace::srgb_encode_lut_[4096];
bool ColorSpace::initialized_ = false;

void ColorSpace::init() {
    if (initialized_) return;
    
    for (int i = 0; i < 256; ++i) {
        float c = i / 255.0f;
        if (c <= 0.04045f) {
            srgb_decode_lut_[i] = c / 12.92f;
        } else {
            srgb_decode_lut_[i] = std::pow((c + 0.055f) / 1.055f, 2.4f);
        }
    }
    
    for (int i = 0; i < 4096; ++i) {
        float c = i / 4095.0f;
        float result;
        if (c <= 0.0031308f) {
            result = 12.92f * c;
        } else {
            result = 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
        }
        srgb_encode_lut_[i] = static_cast<uint8_t>(std::clamp(std::round(result * 255.0f), 0.0f, 255.0f));
    }
    
    initialized_ = true;
}

float ColorSpace::srgb_decode(uint8_t c) {
    float cv = c / 255.0f;
    if (cv <= 0.04045f) {
        return cv / 12.92f;
    }
    return std::pow((cv + 0.055f) / 1.055f, 2.4f);
}

uint8_t ColorSpace::srgb_encode(float c) {
    c = std::clamp(c, 0.0f, 1.0f);
    float result;
    if (c <= 0.0031308f) {
        result = 12.92f * c;
    } else {
        result = 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
    }
    return static_cast<uint8_t>(std::clamp(std::round(result * 255.0f), 0.0f, 255.0f));
}

float ColorSpace::srgb_to_linear(uint8_t srgb) {
    if (initialized_) {
        return srgb_decode_lut_[srgb];
    }
    return srgb_decode(srgb);
}

uint8_t ColorSpace::linear_to_srgb(float linear) {
    if (initialized_) {
        int idx = static_cast<int>(std::clamp(linear, 0.0f, 1.0f) * 4095.0f + 0.5f);
        return srgb_encode_lut_[idx];
    }
    return srgb_encode(linear);
}

LinearColor ColorSpace::srgb_to_linear(uint8_t r, uint8_t g, uint8_t b) {
    return {srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)};
}

void ColorSpace::linear_to_srgb(const LinearColor& linear, uint8_t& r, uint8_t& g, uint8_t& b) {
    r = linear_to_srgb(linear.r);
    g = linear_to_srgb(linear.g);
    b = linear_to_srgb(linear.b);
}

LinearColor ColorSpace::srgb_to_linear_fast(uint8_t r, uint8_t g, uint8_t b) {
    return srgb_to_linear(r, g, b);
}

void ColorSpace::linear_to_srgb_fast(const LinearColor& linear, uint8_t& r, uint8_t& g, uint8_t& b) {
    linear_to_srgb(linear, r, g, b);
}

float ColorSpace::lab_f(float t) {
    constexpr float delta = 6.0f / 29.0f;
    if (t > delta * delta * delta) {
        return std::cbrtf(t);
    }
    return t / (3.0f * delta * delta) + 4.0f / 29.0f;
}

float ColorSpace::lab_f_inv(float t) {
    constexpr float delta = 6.0f / 29.0f;
    if (t > delta) {
        return t * t * t;
    }
    return 3.0f * delta * delta * (t - 4.0f / 29.0f);
}

OKLab ColorSpace::to_oklab(const LinearColor& linear) {
    float r = linear.r;
    float g = linear.g;
    float b = linear.b;
    
    float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
    float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
    float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;
    
    float l_ = std::cbrtf(l);
    float m_ = std::cbrtf(m);
    float s_ = std::cbrtf(s);
    
    float L = 0.2104542553f * l_ + 0.7936177850f * m_ - 0.0040720468f * s_;
    float A = 1.9779984951f * l_ - 2.4285922050f * m_ + 0.4505937099f * s_;
    float B = 0.0259040371f * l_ + 0.7827717662f * m_ - 0.8086757660f * s_;
    
    return {L, A, B};
}

OKLab ColorSpace::srgb_to_oklab(uint8_t r, uint8_t g, uint8_t b) {
    LinearColor linear = srgb_to_linear(r, g, b);
    return to_oklab(linear);
}

LinearColor ColorSpace::from_oklab(const OKLab& lab) {
    float L = lab.L;
    float A = lab.a;
    float B = lab.b;
    
    float l_ = L + 0.3963377774f * A + 0.2158037573f * B;
    float m_ = L - 0.1055613458f * A - 0.0638541728f * B;
    float s_ = L - 0.0894841775f * A - 1.2914855480f * B;
    
    float l = l_ * l_ * l_;
    float m = m_ * m_ * m_;
    float s = s_ * s_ * s_;
    
    float r = 4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s;
    float g = -1.2684380046f * l + 2.3809276044f * m - 0.0913817355f * s;
    float b = -0.0041960863f * l - 0.0748454739f * m + 1.0915434158f * s;
    
    return {std::clamp(r, 0.0f, 1.0f), 
            std::clamp(g, 0.0f, 1.0f), 
            std::clamp(b, 0.0f, 1.0f)};
}

void ColorSpace::oklab_to_srgb(const OKLab& lab, uint8_t& r, uint8_t& g, uint8_t& b) {
    LinearColor linear = from_oklab(lab);
    linear_to_srgb(linear, r, g, b);
}

float ColorSpace::oklab_distance_srgb(uint8_t r1, uint8_t g1, uint8_t b1,
                                       uint8_t r2, uint8_t g2, uint8_t b2) {
    OKLab c1 = srgb_to_oklab(r1, g1, b1);
    OKLab c2 = srgb_to_oklab(r2, g2, b2);
    return OKLab::distance(c1, c2);
}

float ColorSpace::perceptual_luminance(uint8_t r, uint8_t g, uint8_t b) {
    LinearColor linear = srgb_to_linear(r, g, b);
    return linear.luminance();
}

}
