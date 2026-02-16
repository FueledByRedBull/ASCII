#pragma once

#include "core/types.hpp"
#include <vector>

namespace ascii {

class BilateralGrid {
public:
    struct Config {
        bool enabled = false;
        int spatial_bins = 32;
        int range_bins = 16;
        float spatial_sigma = 2.0f;
        float range_sigma = 0.15f;
    };

    struct Sample {
        float r = 0.0f;
        float g = 0.0f;
        float b = 0.0f;
    };

    BilateralGrid() = default;
    explicit BilateralGrid(const Config& cfg) : config_(cfg) {}

    void set_config(const Config& cfg) { config_ = cfg; }
    const Config& config() const { return config_; }

    void build(const std::vector<CellStats>& cells, int cols, int rows);
    Sample sample(int x, int y, float luminance) const;
    bool valid() const { return built_ && !weight_.empty(); }

private:
    Config config_;
    int cols_ = 0;
    int rows_ = 0;
    int range_bins_ = 16;
    bool built_ = false;

    std::vector<float> sum_r_;
    std::vector<float> sum_g_;
    std::vector<float> sum_b_;
    std::vector<float> weight_;

    int idx(int x, int y, int z) const {
        return (z * rows_ + y) * cols_ + x;
    }

    static std::vector<float> make_gaussian_kernel(float sigma, int max_radius);
    void convolve_x(std::vector<float>& v, const std::vector<float>& kernel) const;
    void convolve_y(std::vector<float>& v, const std::vector<float>& kernel) const;
    void convolve_z(std::vector<float>& v, const std::vector<float>& kernel) const;
};

}  // namespace ascii
