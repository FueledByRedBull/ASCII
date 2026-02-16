#pragma once

#include "core/types.hpp"
#include "edge_detector.hpp"
#include <vector>

namespace ascii {

class IntegralImage {
public:
    IntegralImage() = default;
    explicit IntegralImage(const FloatImage& input);
    
    void compute(const FloatImage& input);
    
    float sum(int x0, int y0, int x1, int y1) const;
    float mean(int x0, int y0, int x1, int y1) const;
    
    int width() const { return width_; }
    int height() const { return height_; }
    
private:
    std::vector<double> data_;
    int width_ = 0;
    int height_ = 0;
};

class CellStatsAggregator {
public:
    struct Config {
        int cell_width = 8;
        int cell_height = 16;
        float edge_threshold = 0.1f;
        int orientation_bins = 8;
        bool enable_orientation_histogram = true;
        bool enable_frequency_signature = true;
        bool enable_texture_signature = true;
        bool quad_tree_adaptive = false;
        int quad_tree_max_depth = 2;
        float quad_tree_variance_threshold = 0.01f;
    };
    
    CellStatsAggregator() = default;
    explicit CellStatsAggregator(const Config& config);
    
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    
    std::vector<CellStats> compute(const FloatImage& luminance, const EdgeData& edges, 
                                    const FrameBuffer* color = nullptr, const GradientData* grad = nullptr) const;
    
    int grid_cols(int image_width) const;
    int grid_rows(int image_height) const;
    
private:
    Config config_;
    
    void compute_orientation_histogram(const FloatImage& gx, const FloatImage& gy,
                                        int x0, int y0, int x1, int y1,
                                        float* histogram, int bins) const;
    
    void compute_structure_tensor(const FloatImage& gx, const FloatImage& gy,
                                   int x0, int y0, int x1, int y1,
                                   float& coherence, float& orientation) const;
};

}
