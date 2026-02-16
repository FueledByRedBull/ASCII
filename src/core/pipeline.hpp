#pragma once

#include "core/types.hpp"
#include "cell_stats.hpp"
#include "edge_detector.hpp"
#include <functional>
#include <string>

namespace ascii {

class Pipeline {
public:
    struct Config {
        int target_cols = 80;
        int target_rows = 24;
        int cell_width = 8;
        int cell_height = 16;
        float char_aspect = 2.0f;
        float blur_sigma = 1.0f;
        float edge_low = 0.05f;
        float edge_high = 0.15f;
        bool use_hysteresis = true;
        std::string scale_mode = "fit";
        bool quad_tree_adaptive = false;
        int quad_tree_max_depth = 2;
        float quad_tree_variance_threshold = 0.01f;
        
        bool multi_scale = true;
        float scale_sigma_0 = 0.8f;
        float scale_sigma_1 = 1.6f;
        bool adaptive_scale_selection = true;
        float scale_variance_floor = 0.0005f;
        float scale_variance_ceil = 0.02f;
        bool use_anisotropic_diffusion = false;
        int diffusion_iterations = 4;
        float diffusion_kappa = 0.08f;
        float diffusion_lambda = 0.2f;
        
        std::string adaptive_mode = "hybrid";
        int tile_size = 16;
        float dark_scene_floor = 0.02f;
        float global_percentile = 0.7f;
    };
    
    Pipeline() : Pipeline(Config{}) {}
    explicit Pipeline(const Config& config);
    
    void set_config(const Config& config);
    const Config& config() const { return config_; }
    
    struct Result {
        FloatImage luminance;
        EdgeData edges;
        std::vector<CellStats> cell_stats;
        int grid_cols;
        int grid_rows;
        FrameBuffer color_buffer;
        GradientData gradients;
    };
    
    Result process(const FrameBuffer& input);
    
private:
    struct ResizePlan {
        int target_w = 0;
        int target_h = 0;
        int scale_w = 0;
        int scale_h = 0;
        int offset_x = 0;
        int offset_y = 0;
    };
    
    Config config_;
    EdgeDetector edge_detector_;
    CellStatsAggregator cell_aggregator_;
    
    FloatImage to_grayscale(const FrameBuffer& input) const;
    ResizePlan compute_resize_plan(int src_w, int src_h) const;
    FloatImage resize_for_cells(const FloatImage& input) const;
    FrameBuffer resize_color_for_cells(const FrameBuffer& input, int target_w, int target_h) const;
};

}
