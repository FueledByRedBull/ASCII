#pragma once

#include "core/types.hpp"
#include <vector>
#include <cmath>
#include <string>

namespace ascii {

struct MultiScaleGradientData {
    FloatImage magnitude;
    FloatImage orientation;
    FloatImage gx;
    FloatImage gy;
    int best_scale;
};

class EdgeDetector {
public:
    struct Config {
        float low_threshold = 0.05f;
        float high_threshold = 0.15f;
        bool use_hysteresis = true;
        float blur_sigma = 1.0f;
        
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
    
    EdgeDetector() = default;
    explicit EdgeDetector(const Config& config);
    
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    
    GradientData compute_gradients(const FloatImage& input);
    MultiScaleGradientData compute_multi_scale_gradients(const FloatImage& input);
    EdgeData detect(const FloatImage& input);
    
    static FloatImage gaussian_blur(const FloatImage& input, float sigma);
    static void sobel(const FloatImage& input, FloatImage& gx, FloatImage& gy);
    static FloatImage non_maximum_suppression(const FloatImage& magnitude, const FloatImage& orientation);
    static std::vector<bool> hysteresis_threshold(const FloatImage& magnitude, int w, int h, float low, float high);
    
    static float compute_global_percentile_threshold(const FloatImage& magnitude, float percentile);
    static float compute_tile_threshold(const FloatImage& magnitude, int x0, int y0, int tile_size, 
                                         int img_w, int img_h, float percentile);
    static FloatImage compute_adaptive_threshold_map(const FloatImage& magnitude, int tile_size, 
                                                      float percentile, float floor);
    
private:
    Config config_;
    
    static FloatImage fuse_multi_scale_magnitude(const FloatImage& mag0, const FloatImage& mag1,
                                                  float w0, float w1);
    static void fuse_multi_scale_orientation(const FloatImage& orient0, const FloatImage& mag0,
                                             const FloatImage& orient1, const FloatImage& mag1,
                                             FloatImage& out_orientation);
    static FloatImage anisotropic_diffusion(const FloatImage& input, int iterations, float kappa, float lambda);
    static FloatImage local_variance_3x3(const FloatImage& input);
};

}
