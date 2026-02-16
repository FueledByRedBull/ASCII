#include "dither.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

namespace ascii {

namespace {

std::array<float, 64 * 64> generate_blue_noise_64() {
    constexpr int N = 64;
    constexpr int TOTAL = N * N;
    constexpr int R = 6;
    constexpr float sigma = 1.5f;

    struct KernelTap {
        int dx;
        int dy;
        float w;
    };

    std::vector<KernelTap> kernel;
    kernel.reserve((2 * R + 1) * (2 * R + 1));
    for (int dy = -R; dy <= R; ++dy) {
        for (int dx = -R; dx <= R; ++dx) {
            float d2 = static_cast<float>(dx * dx + dy * dy);
            float w = std::exp(-d2 / (2.0f * sigma * sigma));
            kernel.push_back({dx, dy, w});
        }
    }

    std::array<uint8_t, TOTAL> binary{};
    std::array<float, TOTAL> energy{};
    std::array<float, TOTAL> ranks{};
    energy.fill(0.0f);
    ranks.fill(0.0f);

    auto wrap = [](int v) {
        int r = v % N;
        return (r < 0) ? (r + N) : r;
    };

    auto add_point_energy = [&](std::array<float, TOTAL>& emap, int px, int py, float sign) {
        for (const auto& tap : kernel) {
            int x = wrap(px + tap.dx);
            int y = wrap(py + tap.dy);
            emap[static_cast<size_t>(y) * N + x] += sign * tap.w;
        }
    };

    uint32_t rng = 0x12345678u;
    auto next_rand = [&]() -> uint32_t {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return rng;
    };

    const int initial_ones = TOTAL / 10;
    int seeded = 0;
    while (seeded < initial_ones) {
        int pos = static_cast<int>(next_rand() % TOTAL);
        if (binary[static_cast<size_t>(pos)] == 0) {
            binary[static_cast<size_t>(pos)] = 1;
            add_point_energy(energy, pos % N, pos / N, 1.0f);
            seeded++;
        }
    }

    auto choose_extreme = [&](const std::array<uint8_t, TOTAL>& bits,
                              const std::array<float, TOTAL>& emap,
                              bool choose_on,
                              bool choose_max) -> int {
        int best_i = -1;
        float best_v = choose_max ? -1e30f : 1e30f;
        for (int i = 0; i < TOTAL; ++i) {
            bool is_on = bits[static_cast<size_t>(i)] != 0;
            if (is_on != choose_on) {
                continue;
            }
            float e = emap[static_cast<size_t>(i)];
            if ((choose_max && e > best_v) || (!choose_max && e < best_v)) {
                best_v = e;
                best_i = i;
            }
        }
        return best_i;
    };

    auto temp_binary = binary;
    auto temp_energy = energy;
    for (int rank = initial_ones - 1; rank >= 0; --rank) {
        int idx = choose_extreme(temp_binary, temp_energy, true, true);
        if (idx < 0) {
            break;
        }
        temp_binary[static_cast<size_t>(idx)] = 0;
        ranks[static_cast<size_t>(idx)] = static_cast<float>(rank);
        add_point_energy(temp_energy, idx % N, idx / N, -1.0f);
    }

    temp_binary = binary;
    temp_energy = energy;
    for (int rank = initial_ones; rank < TOTAL; ++rank) {
        int idx = choose_extreme(temp_binary, temp_energy, false, false);
        if (idx < 0) {
            break;
        }
        temp_binary[static_cast<size_t>(idx)] = 1;
        ranks[static_cast<size_t>(idx)] = static_cast<float>(rank);
        add_point_energy(temp_energy, idx % N, idx / N, 1.0f);
    }

    const float inv = 1.0f / static_cast<float>(TOTAL - 1);
    for (auto& v : ranks) {
        v *= inv;
    }
    return ranks;
}

}  // namespace

DitherBuffer::DitherBuffer(int width, int height) {
    resize(width, height);
}

void DitherBuffer::resize(int width, int height) {
    width_ = width;
    height_ = height;
    error_r_.assign(static_cast<size_t>(width) * height, 0.0f);
    error_g_.assign(static_cast<size_t>(width) * height, 0.0f);
    error_b_.assign(static_cast<size_t>(width) * height, 0.0f);
}

void DitherBuffer::reset() {
    std::fill(error_r_.begin(), error_r_.end(), 0.0f);
    std::fill(error_g_.begin(), error_g_.end(), 0.0f);
    std::fill(error_b_.begin(), error_b_.end(), 0.0f);
}

float DitherBuffer::get_error_r(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return 0.0f;
    return error_r_[y * width_ + x];
}

float DitherBuffer::get_error_g(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return 0.0f;
    return error_g_[y * width_ + x];
}

float DitherBuffer::get_error_b(int x, int y) const {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return 0.0f;
    return error_b_[y * width_ + x];
}

void DitherBuffer::add_error(int x, int y, float er, float eg, float eb) {
    if (x < 0 || x >= width_ || y < 0 || y >= height_) return;
    size_t idx = y * width_ + x;
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
    static const std::array<float, 64 * 64> kBlueNoise64 = generate_blue_noise_64();
    int ix = ((x % 64) + 64) % 64;
    int iy = ((y % 64) + 64) % 64;
    return kBlueNoise64[static_cast<size_t>(iy) * 64 + ix];
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
