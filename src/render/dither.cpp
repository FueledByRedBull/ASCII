#include "dither.hpp"
#include "blue_noise_64.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ascii {

DitherBuffer::DitherBuffer(int width, int height) {
    resize(width, height);
}

void DitherBuffer::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    width_ = width;
    height_ = height;
    stride_ = width + 2;
    const size_t padded = static_cast<size_t>(height + 2) * stride_;
    error_r_.assign(padded, 0.0f);
    error_g_.assign(padded, 0.0f);
    error_b_.assign(padded, 0.0f);
}

void DitherBuffer::reset() {
    std::fill(error_r_.begin(), error_r_.end(), 0.0f);
    std::fill(error_g_.begin(), error_g_.end(), 0.0f);
    std::fill(error_b_.begin(), error_b_.end(), 0.0f);
}

float DitherBuffer::get_error_r(int x, int y) const {
    return error_r_[index(x, y)];
}

float DitherBuffer::get_error_g(int x, int y) const {
    return error_g_[index(x, y)];
}

float DitherBuffer::get_error_b(int x, int y) const {
    return error_b_[index(x, y)];
}

void DitherBuffer::add_error(int x, int y, float er, float eg, float eb) {
    size_t idx = index(x, y);
    error_r_[idx] = clamp_error(error_r_[idx] + er);
    error_g_[idx] = clamp_error(error_g_[idx] + eg);
    error_b_[idx] = clamp_error(error_b_[idx] + eb);
}

void DitherBuffer::distribute_error_serpentine(int x, int y, bool left_to_right,
                                                float er, float eg, float eb) {
    int dx = left_to_right ? 1 : -1;
    
    add_error(x + dx, y,     er * 7.0f / 16.0f, eg * 7.0f / 16.0f, eb * 7.0f / 16.0f);
    add_error(x - dx, y + 1, er * 3.0f / 16.0f, eg * 3.0f / 16.0f, eb * 3.0f / 16.0f);
    add_error(x,      y + 1, er * 5.0f / 16.0f, eg * 5.0f / 16.0f, eb * 5.0f / 16.0f);
    add_error(x + dx, y + 1, er * 1.0f / 16.0f, eg * 1.0f / 16.0f, eb * 1.0f / 16.0f);
}

Ditherer::Ditherer(const Config& config) : config_(config) {}

float Ditherer::blue_noise(int x, int y) {
    int ix = ((x % 64) + 64) % 64;
    int iy = ((y % 64) + 64) % 64;
    constexpr float kInvMaxRank = 1.0f / 4095.0f;
    return static_cast<float>(kBlueNoiseRank64[static_cast<size_t>(iy) * 64 + ix]) * kInvMaxRank;
}

void Ditherer::begin_frame(int width, int height) {
    buffer_.resize(width, height);
    buffer_.reset();
}

void Ditherer::apply_dithering(int x, int y, float& r, float& g, float& b) {
    if (!config_.enabled) return;
    
    r = std::clamp(r + buffer_.get_error_r(x, y), 0.0f, 1.0f);
    g = std::clamp(g + buffer_.get_error_g(x, y), 0.0f, 1.0f);
    b = std::clamp(b + buffer_.get_error_b(x, y), 0.0f, 1.0f);

    if (config_.use_blue_noise_halftone) {
        float luma = (r + g + b) / 3.0f;
        float bn1 = blue_noise(x, y);
        float bn2 = blue_noise(x + 37, y + 53);
        float tpdf = (bn1 + bn2) - 1.0f;
        int cell = std::max(2, config_.halftone_cell_size);
        float ux = (static_cast<float>(x % cell) + 0.5f) / cell - 0.5f;
        float uy = (static_cast<float>(y % cell) + 0.5f) / cell - 0.5f;
        float radial = std::sqrt(ux * ux + uy * uy);
        float dot = std::exp(-8.0f * radial * radial);
        float strength = config_.halftone_strength * (0.35f + 0.65f * (1.0f - luma));
        float modulation = strength * (0.65f * (tpdf * 0.5f) + 0.35f * (dot - 0.5f));
        r = std::clamp(r + modulation, 0.0f, 1.0f);
        g = std::clamp(g + modulation, 0.0f, 1.0f);
        b = std::clamp(b + modulation, 0.0f, 1.0f);
    }
}

void Ditherer::distribute_error(int x, int y, int row_direction,
                                 float er, float eg, float eb) {
    if (!config_.enabled) return;
    
    er = std::clamp(er, -config_.error_clamp, config_.error_clamp);
    eg = std::clamp(eg, -config_.error_clamp, config_.error_clamp);
    eb = std::clamp(eb, -config_.error_clamp, config_.error_clamp);
    
    bool left_to_right = (row_direction > 0);
    int dx = left_to_right ? 1 : -1;
    
    float e7 = er * config_.distribution_7;
    float e3 = er * config_.distribution_3;
    float e5 = er * config_.distribution_5;
    float e1 = er * config_.distribution_1;
    
    buffer_.add_error(x + dx, y,     e7, eg * config_.distribution_7, eb * config_.distribution_7);
    buffer_.add_error(x - dx, y + 1, e3, eg * config_.distribution_3, eb * config_.distribution_3);
    buffer_.add_error(x,      y + 1, e5, eg * config_.distribution_5, eb * config_.distribution_5);
    buffer_.add_error(x + dx, y + 1, e1, eg * config_.distribution_1, eb * config_.distribution_1);
}

bool Ditherer::should_dither_cell(bool is_edge_cell) const {
    return config_.enabled && !is_edge_cell;
}

}
