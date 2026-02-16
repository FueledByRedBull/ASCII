#include "pipeline.hpp"
#include "core/color_space.hpp"
#include <algorithm>
#include <cstring>

#ifdef ASCII_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace ascii {

Pipeline::Pipeline(const Config& config) : config_(config) {
    ColorSpace::init();
    init_luminance_lut();
    
    EdgeDetector::Config edge_cfg;
    edge_cfg.blur_sigma = config.blur_sigma;
    edge_cfg.low_threshold = config.edge_low;
    edge_cfg.high_threshold = config.edge_high;
    edge_cfg.use_hysteresis = config.use_hysteresis;
    edge_cfg.multi_scale = config.multi_scale;
    edge_cfg.scale_sigma_0 = config.scale_sigma_0;
    edge_cfg.scale_sigma_1 = config.scale_sigma_1;
    edge_cfg.adaptive_scale_selection = config.adaptive_scale_selection;
    edge_cfg.scale_variance_floor = config.scale_variance_floor;
    edge_cfg.scale_variance_ceil = config.scale_variance_ceil;
    edge_cfg.use_anisotropic_diffusion = config.use_anisotropic_diffusion;
    edge_cfg.diffusion_iterations = config.diffusion_iterations;
    edge_cfg.diffusion_kappa = config.diffusion_kappa;
    edge_cfg.diffusion_lambda = config.diffusion_lambda;
    edge_cfg.adaptive_mode = config.adaptive_mode;
    edge_cfg.tile_size = config.tile_size;
    edge_cfg.dark_scene_floor = config.dark_scene_floor;
    edge_cfg.global_percentile = config.global_percentile;
    edge_detector_.set_config(edge_cfg);
    
    CellStatsAggregator::Config cell_cfg;
    cell_cfg.cell_width = config.cell_width;
    cell_cfg.cell_height = config.cell_height;
    cell_cfg.quad_tree_adaptive = config.quad_tree_adaptive;
    cell_cfg.quad_tree_max_depth = config.quad_tree_max_depth;
    cell_cfg.quad_tree_variance_threshold = config.quad_tree_variance_threshold;
    cell_aggregator_.set_config(cell_cfg);
}

void Pipeline::set_config(const Config& config) {
    config_ = config;
    
    EdgeDetector::Config edge_cfg;
    edge_cfg.blur_sigma = config.blur_sigma;
    edge_cfg.low_threshold = config.edge_low;
    edge_cfg.high_threshold = config.edge_high;
    edge_cfg.use_hysteresis = config.use_hysteresis;
    edge_cfg.multi_scale = config.multi_scale;
    edge_cfg.scale_sigma_0 = config.scale_sigma_0;
    edge_cfg.scale_sigma_1 = config.scale_sigma_1;
    edge_cfg.adaptive_scale_selection = config.adaptive_scale_selection;
    edge_cfg.scale_variance_floor = config.scale_variance_floor;
    edge_cfg.scale_variance_ceil = config.scale_variance_ceil;
    edge_cfg.use_anisotropic_diffusion = config.use_anisotropic_diffusion;
    edge_cfg.diffusion_iterations = config.diffusion_iterations;
    edge_cfg.diffusion_kappa = config.diffusion_kappa;
    edge_cfg.diffusion_lambda = config.diffusion_lambda;
    edge_cfg.adaptive_mode = config.adaptive_mode;
    edge_cfg.tile_size = config.tile_size;
    edge_cfg.dark_scene_floor = config.dark_scene_floor;
    edge_cfg.global_percentile = config.global_percentile;
    edge_detector_.set_config(edge_cfg);
    
    CellStatsAggregator::Config cell_cfg;
    cell_cfg.cell_width = config.cell_width;
    cell_cfg.cell_height = config.cell_height;
    cell_cfg.quad_tree_adaptive = config.quad_tree_adaptive;
    cell_cfg.quad_tree_max_depth = config.quad_tree_max_depth;
    cell_cfg.quad_tree_variance_threshold = config.quad_tree_variance_threshold;
    cell_aggregator_.set_config(cell_cfg);
}

void Pipeline::init_luminance_lut() {
    for (int i = 0; i < 256; ++i) {
        const float lin = ColorSpace::srgb_to_linear(static_cast<uint8_t>(i));
        lum_r_lut_[i] = 0.2126f * lin;
        lum_g_lut_[i] = 0.7152f * lin;
        lum_b_lut_[i] = 0.0722f * lin;
    }
}

void Pipeline::to_grayscale(const FrameBuffer& input, FloatImage& output) const {
    if (output.width() != input.width() || output.height() != input.height()) {
        output = FloatImage(input.width(), input.height());
    }

    const int total_pixels = input.width() * input.height();
    const uint8_t* src = input.data();
    float* dst = output.data();
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < total_pixels; ++i) {
        int idx = i * 4;
        dst[i] = lum_r_lut_[src[idx]] + lum_g_lut_[src[idx + 1]] + lum_b_lut_[src[idx + 2]];
    }
}

Pipeline::ResizePlan Pipeline::compute_resize_plan(int src_w, int src_h) const {
    ResizePlan plan;
    plan.target_w = config_.target_cols * config_.cell_width;
    plan.target_h = config_.target_rows * config_.cell_height;
    
    float src_aspect = static_cast<float>(src_w) / src_h;
    float dst_aspect = static_cast<float>(config_.target_cols) / config_.target_rows / config_.char_aspect;
    
    if (config_.scale_mode == "stretch") {
        plan.scale_w = plan.target_w;
        plan.scale_h = plan.target_h;
    } else if (config_.scale_mode == "fill") {
        if (src_aspect > dst_aspect) {
            plan.scale_h = plan.target_h;
            plan.scale_w = static_cast<int>(plan.scale_h * src_aspect);
        } else {
            plan.scale_w = plan.target_w;
            plan.scale_h = static_cast<int>(plan.scale_w / src_aspect);
        }
    } else {
        if (src_aspect > dst_aspect) {
            plan.scale_w = plan.target_w;
            plan.scale_h = static_cast<int>(plan.scale_w / src_aspect);
        } else {
            plan.scale_h = plan.target_h;
            plan.scale_w = static_cast<int>(plan.scale_h * src_aspect);
        }
    }
    
    plan.scale_w = std::max(1, plan.scale_w);
    plan.scale_h = std::max(1, plan.scale_h);
    
    if (config_.scale_mode == "fit" || config_.scale_mode == "fill") {
        plan.offset_x = (plan.target_w - plan.scale_w) / 2;
        plan.offset_y = (plan.target_h - plan.scale_h) / 2;
    }
    
    return plan;
}

void Pipeline::resize_for_cells(const FloatImage& input, FloatImage& output) const {
    ResizePlan plan = compute_resize_plan(input.width(), input.height());
    
#ifdef ASCII_USE_OPENCV
    cv::Mat input_mat(input.height(), input.width(), CV_32F, 
                      const_cast<float*>(input.data()));
    
    cv::Mat scaled_mat;
    cv::Size scale_size(plan.scale_w, plan.scale_h);
    cv::resize(input_mat, scaled_mat, scale_size, 0, 0, cv::INTER_LINEAR);
    
    if (output.width() != plan.target_w || output.height() != plan.target_h) {
        output = FloatImage(plan.target_w, plan.target_h, 0.0f);
    } else {
        output.fill(0.0f);
    }
    
    for (int y = 0; y < plan.scale_h; ++y) {
        int dst_y = y + plan.offset_y;
        if (dst_y < 0 || dst_y >= plan.target_h) continue;
        for (int x = 0; x < plan.scale_w; ++x) {
            int dst_x = x + plan.offset_x;
            if (dst_x < 0 || dst_x >= plan.target_w) continue;
            output.set(dst_x, dst_y, scaled_mat.at<float>(y, x));
        }
    }
#else
    if (output.width() != plan.target_w || output.height() != plan.target_h) {
        output = FloatImage(plan.target_w, plan.target_h, 0.0f);
    } else {
        output.fill(0.0f);
    }
    
    float x_ratio = static_cast<float>(input.width()) / plan.scale_w;
    float y_ratio = static_cast<float>(input.height()) / plan.scale_h;
    
    for (int y = 0; y < plan.scale_h; ++y) {
        int dst_y = y + plan.offset_y;
        if (dst_y < 0 || dst_y >= plan.target_h) continue;
        float src_y = y * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, input.height() - 1);
        float fy = src_y - y0;
        
        for (int x = 0; x < plan.scale_w; ++x) {
            int dst_x = x + plan.offset_x;
            if (dst_x < 0 || dst_x >= plan.target_w) continue;
            float src_x = x * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, input.width() - 1);
            float fx = src_x - x0;
            
            float v00 = input.get_clamped(x0, y0);
            float v10 = input.get_clamped(x1, y0);
            float v01 = input.get_clamped(x0, y1);
            float v11 = input.get_clamped(x1, y1);
            
            float v0 = v00 * (1 - fx) + v10 * fx;
            float v1 = v01 * (1 - fx) + v11 * fx;
            float v = v0 * (1 - fy) + v1 * fy;
            
            output.set(dst_x, dst_y, v);
        }
    }
#endif
}

void Pipeline::resize_color_for_cells(const FrameBuffer& input, int target_w, int target_h, FrameBuffer& output) const {
    ResizePlan plan = compute_resize_plan(input.width(), input.height());
    if (input.width() == plan.target_w && input.height() == plan.target_h && config_.scale_mode == "stretch") {
        output = input;
        return;
    }
    
#ifdef ASCII_USE_OPENCV
    cv::Mat input_mat(input.height(), input.width(), CV_8UC4, 
                      const_cast<uint8_t*>(input.data()));
    
    cv::Mat scaled_mat;
    cv::Size scale_size(plan.scale_w, plan.scale_h);
    cv::resize(input_mat, scaled_mat, scale_size, 0, 0, cv::INTER_LINEAR);
    
    if (output.width() != target_w || output.height() != target_h) {
        output = FrameBuffer(target_w, target_h, Color(0, 0, 0, 255));
    } else {
        output.fill(Color(0, 0, 0, 255));
    }
    
    for (int y = 0; y < plan.scale_h; ++y) {
        int dst_y = y + plan.offset_y;
        if (dst_y < 0 || dst_y >= plan.target_h) continue;
        for (int x = 0; x < plan.scale_w; ++x) {
            int dst_x = x + plan.offset_x;
            if (dst_x < 0 || dst_x >= plan.target_w) continue;
            cv::Vec4b pixel = scaled_mat.at<cv::Vec4b>(y, x);
            output.set_pixel(dst_x, dst_y, Color(pixel[0], pixel[1], pixel[2], pixel[3]));
        }
    }
#else
    if (output.width() != target_w || output.height() != target_h) {
        output = FrameBuffer(target_w, target_h, Color(0, 0, 0, 255));
    } else {
        output.fill(Color(0, 0, 0, 255));
    }
    
    float x_ratio = static_cast<float>(input.width()) / plan.scale_w;
    float y_ratio = static_cast<float>(input.height()) / plan.scale_h;
    
    for (int y = 0; y < plan.scale_h; ++y) {
        int dst_y = y + plan.offset_y;
        if (dst_y < 0 || dst_y >= plan.target_h) continue;
        float src_y = y * y_ratio;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, input.height() - 1);
        float fy = src_y - y0;
        
        for (int x = 0; x < plan.scale_w; ++x) {
            int dst_x = x + plan.offset_x;
            if (dst_x < 0 || dst_x >= plan.target_w) continue;
            float src_x = x * x_ratio;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, input.width() - 1);
            float fx = src_x - x0;
            
            auto bilinear = [&](int c) -> uint8_t {
                Color p00 = input.get_pixel(x0, y0);
                Color p10 = input.get_pixel(x1, y0);
                Color p01 = input.get_pixel(x0, y1);
                Color p11 = input.get_pixel(x1, y1);
                
                const float channels[4][4] = {
                    {static_cast<float>(p00.r), static_cast<float>(p10.r), static_cast<float>(p01.r), static_cast<float>(p11.r)},
                    {static_cast<float>(p00.g), static_cast<float>(p10.g), static_cast<float>(p01.g), static_cast<float>(p11.g)},
                    {static_cast<float>(p00.b), static_cast<float>(p10.b), static_cast<float>(p01.b), static_cast<float>(p11.b)},
                    {static_cast<float>(p00.a), static_cast<float>(p10.a), static_cast<float>(p01.a), static_cast<float>(p11.a)}
                };
                
                float v0 = channels[c][0] * (1 - fx) + channels[c][1] * fx;
                float v1 = channels[c][2] * (1 - fx) + channels[c][3] * fx;
                return static_cast<uint8_t>(std::clamp(v0 * (1 - fy) + v1 * fy, 0.0f, 255.0f));
            };
            
            uint8_t r = bilinear(0);
            uint8_t g = bilinear(1);
            uint8_t b = bilinear(2);
            uint8_t a = bilinear(3);
            
            output.set_pixel(dst_x, dst_y, Color(r, g, b, a));
        }
    }
#endif
}

Pipeline::Result Pipeline::process(const FrameBuffer& input) {
    Result result;

    to_grayscale(input, gray_buffer_);
    resize_for_cells(gray_buffer_, result.luminance);

    resize_color_for_cells(input, result.luminance.width(), result.luminance.height(), result.color_buffer);

    // Drive edge-mask generation through the detector's configured path
    // (multi-scale + adaptive thresholds), then compute gx/gy for cell stats.
    result.edges = edge_detector_.detect(result.luminance);

    if (config_.multi_scale) {
        MultiScaleGradientData ms_grad = edge_detector_.compute_multi_scale_gradients(result.luminance);
        result.gradients.gx = std::move(ms_grad.gx);
        result.gradients.gy = std::move(ms_grad.gy);
        result.gradients.magnitude = std::move(ms_grad.magnitude);
        result.gradients.orientation = std::move(ms_grad.orientation);
    } else {
        result.gradients = edge_detector_.compute_gradients(result.luminance);
    }
    
    result.grid_cols = cell_aggregator_.grid_cols(result.luminance.width());
    result.grid_rows = cell_aggregator_.grid_rows(result.luminance.height());
    
    result.cell_stats = cell_aggregator_.compute(result.luminance, result.edges, 
                                                  &result.color_buffer, &result.gradients);
    
    return result;
}

}
