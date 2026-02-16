#pragma once

#include "core/types.hpp"
#include <vector>
#include <cstdint>

namespace ascii {

class DitherBuffer {
public:
    DitherBuffer() = default;
    DitherBuffer(int width, int height);
    
    void resize(int width, int height);
    void reset();
    
    int width() const { return width_; }
    int height() const { return height_; }
    
    float get_error_r(int x, int y) const;
    float get_error_g(int x, int y) const;
    float get_error_b(int x, int y) const;
    
    void add_error(int x, int y, float er, float eg, float eb);
    
    void distribute_error_serpentine(int x, int y, bool left_to_right,
                                      float er, float eg, float eb);
    
private:
    std::vector<float> error_r_;
    std::vector<float> error_g_;
    std::vector<float> error_b_;
    int width_ = 0;
    int height_ = 0;
    int stride_ = 0;
    
    static constexpr float ERROR_CLAMP = 0.12f;
    
    float clamp_error(float e) const {
        return std::clamp(e, -ERROR_CLAMP, ERROR_CLAMP);
    }

    size_t index(int x, int y) const {
        return static_cast<size_t>(y + 1) * stride_ + static_cast<size_t>(x + 1);
    }
};

class Ditherer {
public:
    struct Config {
        bool enabled = true;
        float error_clamp = 0.12f;
        float distribution_7 = 7.0f / 16.0f;
        float distribution_3 = 3.0f / 16.0f;
        float distribution_5 = 5.0f / 16.0f;
        float distribution_1 = 1.0f / 16.0f;
        bool use_blue_noise_halftone = false;
        float halftone_strength = 0.24f;
        int halftone_cell_size = 6;
    };
    
    Ditherer() = default;
    explicit Ditherer(const Config& config);
    
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    
    void begin_frame(int width, int height);
    
    void apply_dithering(int x, int y, float& r, float& g, float& b);
    
    void distribute_error(int x, int y, int row_direction,
                           float er, float eg, float eb);
    
    bool should_dither_cell(bool is_edge_cell) const;
    
private:
    Config config_;
    DitherBuffer buffer_;

    static float blue_noise(int x, int y);
};

}
