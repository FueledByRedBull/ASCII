#pragma once

#include "core/types.hpp"
#include <vector>
#include <cmath>

namespace ascii {

struct MotionVector {
    float dx = 0.0f;
    float dy = 0.0f;
    float confidence = 0.0f;
};

class MotionEstimator {
public:
    struct Config {
        int pyramid_levels = 3;
        int window_size = 15;
        int iterations = 3;
        int poly_n = 5;
        float poly_sigma = 1.1f;
        float motion_cap = 6.0f;
        float flow_scale = 1.0f;
        // Backward-compatible name: implementation uses FFT phase-correlation refinement.
        bool use_phase_correlation = true;
        int phase_search_radius = 6;
        float phase_blend = 0.28f;
    };
    
    MotionEstimator() = default;
    explicit MotionEstimator(const Config& config);
    
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    
    void compute_flow(const FloatImage& prev, const FloatImage& curr);
    
    const MotionVector& get_motion(int x, int y) const;
    MotionVector get_motion_interpolated(float x, float y) const;
    
    int width() const { return width_; }
    int height() const { return height_; }
    bool has_motion() const { return !flow_.empty(); }
    
    void get_motion_for_cell(int cell_x, int cell_y, int cell_w, int cell_h,
                              float& out_dx, float& out_dy) const;
    
    void reset();
    
private:
    Config config_;
    std::vector<MotionVector> flow_;
    int width_ = 0;
    int height_ = 0;
    
    FloatImage prev_pyramid_[4];
    FloatImage curr_pyramid_[4];
    
    void build_pyramid(const FloatImage& img, FloatImage* pyramid, int levels);
    void compute_farneback_level(const FloatImage& prev, const FloatImage& curr,
                                  FloatImage& flow_x, FloatImage& flow_y);
    void average_flow_for_cell(int x0, int y0, int w, int h,
                                float& dx, float& dy) const;
};

}
