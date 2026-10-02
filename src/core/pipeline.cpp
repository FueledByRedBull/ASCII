#include "pipeline.hpp"
#include "core/color_space.hpp"
#include <algorithm>
#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

namespace ascii {

Pipeline::Pipeline(const Config& config) : config_(config) {
    ColorSpace::init();
    init_luminance_lut();
    set_config(config);
}

void Pipeline::set_config(const Config& config) {
    if (config.target_cols <= 0 || config.target_rows <= 0 ||
        config.cell_width <= 0 || config.cell_height <= 0 ||
        !std::isfinite(config.char_aspect) || config.char_aspect <= 0.0f ||
        config.tile_size <= 0 ||
        (config.scale_mode != "fit" && config.scale_mode != "fill" && config.scale_mode != "stretch")) {
        throw std::invalid_argument("Invalid pipeline geometry or resize configuration");
    }
    const int64_t width = static_cast<int64_t>(config.target_cols) * config.cell_width;
    const int64_t height = static_cast<int64_t>(config.target_rows) * config.cell_height;
    // Keep the same 100-million-pixel analysis limit as the public configuration.
    if (width > std::numeric_limits<int>::max() || height > std::numeric_limits<int>::max() ||
        width > 100000000 / height) {
        throw std::length_error("Pipeline analysis dimensions are too large");
    }
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
    cell_cfg.enable_orientation_histogram = config.enable_orientation_histogram;
    cell_cfg.enable_frequency_signature = config.enable_frequency_signature;
    cell_cfg.enable_texture_signature = config.enable_texture_signature;
    cell_cfg.quad_tree_adaptive = config.quad_tree_adaptive;
    cell_cfg.quad_tree_max_depth = config.quad_tree_max_depth;
    cell_cfg.quad_tree_variance_threshold = config.quad_tree_variance_threshold;
    cell_aggregator_.set_config(cell_cfg);

    ContourExtractor::Config contour_cfg;
    contour_cfg.enabled = config.contours_enabled;
    contour_cfg.min_occupancy = config.contour_min_occupancy;
    contour_cfg.min_pixels = config.contour_min_pixels;
    contour_cfg.dominance_ratio = config.contour_dominance_ratio;
    contour_cfg.intersection_ratio = config.contour_intersection_ratio;
    contour_cfg.dog_sigma_inner = config.contour_dog_sigma_inner;
    contour_cfg.dog_sigma_outer = config.contour_dog_sigma_outer;
    contour_extractor_.set_config(contour_cfg);
}

void Pipeline::init_luminance_lut() {
    for (int i = 0; i < 256; ++i) {
        const float lin = ColorSpace::srgb_to_linear(static_cast<uint8_t>(i));
        linear_lut_[i] = lin;
        lum_r_lut_[i] = 0.2126f * lin;
        lum_g_lut_[i] = 0.7152f * lin;
        lum_b_lut_[i] = 0.0722f * lin;
    }
}

void Pipeline::to_grayscale(const FrameBuffer& input, FloatImage& output) const {
    if (output.width() != input.width() || output.height() != input.height()) {
        output = FloatImage(input.width(), input.height());
    }

    const int64_t total_pixels = static_cast<int64_t>(input.width()) * input.height();
    const uint8_t* src = input.data();
    float* dst = output.data();
    
#ifdef HAS_OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i = 0; i < total_pixels; ++i) {
        const size_t idx = static_cast<size_t>(i) * 4;
        dst[i] = lum_r_lut_[src[idx]] + lum_g_lut_[src[idx + 1]] + lum_b_lut_[src[idx + 2]];
    }
}

Pipeline::ResizePlan Pipeline::compute_resize_plan(int src_w, int src_h) const {
    ResizePlan plan;
    plan.target_w = config_.target_cols * config_.cell_width;
    plan.target_h = config_.target_rows * config_.cell_height;
    
    // Convert the source's physical aspect to analysis-raster coordinates.
    const double src_aspect = static_cast<double>(src_w) / src_h *
                              (static_cast<double>(config_.char_aspect) * config_.cell_width / config_.cell_height);
    const double dst_aspect = static_cast<double>(plan.target_w) / plan.target_h;
    const auto scaled_extent = [](double extent, bool cover) {
        extent = cover ? std::ceil(extent) : std::floor(extent);
        if (!std::isfinite(extent) || extent > std::numeric_limits<int>::max()) {
            throw std::length_error("Pipeline resize extent is too large");
        }
        return static_cast<int>(std::max(1.0, extent));
    };
    
    if (config_.scale_mode == "stretch") {
        plan.scale_w = plan.target_w;
        plan.scale_h = plan.target_h;
    } else if (config_.scale_mode == "fill") {
        if (src_aspect > dst_aspect) {
            plan.scale_h = plan.target_h;
            plan.scale_w = scaled_extent(plan.scale_h * src_aspect, true);
        } else {
            plan.scale_w = plan.target_w;
            plan.scale_h = scaled_extent(plan.scale_w / src_aspect, true);
        }
    } else {
        if (src_aspect > dst_aspect) {
            plan.scale_w = plan.target_w;
            plan.scale_h = scaled_extent(plan.scale_w / src_aspect, false);
        } else {
            plan.scale_h = plan.target_h;
            plan.scale_w = scaled_extent(plan.scale_h * src_aspect, false);
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
    

    if (output.width() != plan.target_w || output.height() != plan.target_h) {
        output = FloatImage(plan.target_w, plan.target_h, 0.0f);
    } else {
        output.fill(0.0f);
    }
    
    float x_ratio = static_cast<float>(input.width()) / plan.scale_w;
    float y_ratio = static_cast<float>(input.height()) / plan.scale_h;
    
#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int y = std::max(0, -plan.offset_y); y < std::min(plan.scale_h, plan.target_h - plan.offset_y); ++y) {
        int dst_y = y + plan.offset_y;
        if (dst_y < 0 || dst_y >= plan.target_h) continue;
        float src_y = (y + 0.5f) * y_ratio - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        int y1 = std::min(y0 + 1, input.height() - 1);
        float fy = src_y - y0;
        y0 = std::clamp(y0, 0, input.height() - 1);
        y1 = std::clamp(y1, 0, input.height() - 1);
        
        for (int x = std::max(0, -plan.offset_x); x < std::min(plan.scale_w, plan.target_w - plan.offset_x); ++x) {
            int dst_x = x + plan.offset_x;
            if (dst_x < 0 || dst_x >= plan.target_w) continue;
            if (x_ratio > 1.0f || y_ratio > 1.0f) {
                const float sx0 = x * x_ratio;
                const float sx1 = (x + 1) * x_ratio;
                const float sy0 = y * y_ratio;
                const float sy1 = (y + 1) * y_ratio;
                double sum = 0.0;
                double weight_sum = 0.0;
                for (int sy = static_cast<int>(std::floor(sy0)); sy < static_cast<int>(std::ceil(sy1)); ++sy) {
                    if (sy < 0 || sy >= input.height()) continue;
                    const float wy = std::max(0.0f, std::min(sy1, sy + 1.0f) - std::max(sy0, static_cast<float>(sy)));
                    for (int sx = static_cast<int>(std::floor(sx0)); sx < static_cast<int>(std::ceil(sx1)); ++sx) {
                        if (sx < 0 || sx >= input.width()) continue;
                        const float wx = std::max(0.0f, std::min(sx1, sx + 1.0f) - std::max(sx0, static_cast<float>(sx)));
                        const float weight = wx * wy;
                        sum += input.get(sx, sy) * weight;
                        weight_sum += weight;
                    }
                }
                output.set(dst_x, dst_y, weight_sum > 0.0 ? static_cast<float>(sum / weight_sum) : 0.0f);
                continue;
            }
            float src_x = (x + 0.5f) * x_ratio - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            int x1 = std::min(x0 + 1, input.width() - 1);
            float fx = src_x - x0;
            x0 = std::clamp(x0, 0, input.width() - 1);
            x1 = std::clamp(x1, 0, input.width() - 1);
            
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

}

void Pipeline::resize_color_for_cells(const FrameBuffer& input, int target_w, int target_h, FrameBuffer& output) const {
    ResizePlan plan = compute_resize_plan(input.width(), input.height());
    if (input.width() == plan.target_w && input.height() == plan.target_h && config_.scale_mode == "stretch") {
        output = input;
        return;
    }
    

    if (output.width() != target_w || output.height() != target_h) {
        output = FrameBuffer(target_w, target_h, Color(0, 0, 0, 255));
    } else {
        output.fill(Color(0, 0, 0, 255));
    }

    const int src_w = input.width();
    const int src_h = input.height();
    const uint8_t* src = input.data();
    uint8_t* dst = output.data();
    const float x_ratio = static_cast<float>(src_w) / plan.scale_w;
    const float y_ratio = static_cast<float>(src_h) / plan.scale_h;

    for (int y = std::max(0, -plan.offset_y); y < std::min(plan.scale_h, plan.target_h - plan.offset_y); ++y) {
        int dst_y = y + plan.offset_y;
        if (dst_y < 0 || dst_y >= plan.target_h) continue;
        const float src_y = (y + 0.5f) * y_ratio - 0.5f;
        int y0 = static_cast<int>(std::floor(src_y));
        int y1 = std::min(y0 + 1, src_h - 1);
        const float fy = src_y - y0;
        y0 = std::clamp(y0, 0, src_h - 1);
        y1 = std::clamp(y1, 0, src_h - 1);
        const float wy0 = 1.0f - fy;

        const size_t row0 = static_cast<size_t>(y0) * src_w * 4;
        const size_t row1 = static_cast<size_t>(y1) * src_w * 4;
        const size_t dst_row = static_cast<size_t>(dst_y) * target_w * 4;

        for (int x = std::max(0, -plan.offset_x); x < std::min(plan.scale_w, plan.target_w - plan.offset_x); ++x) {
            int dst_x = x + plan.offset_x;
            if (dst_x < 0 || dst_x >= plan.target_w) continue;
            if (x_ratio > 1.0f || y_ratio > 1.0f) {
                const float sx0 = x * x_ratio;
                const float sx1 = (x + 1) * x_ratio;
                const float sy0 = y * y_ratio;
                const float sy1 = (y + 1) * y_ratio;
                double sums[4] = {};
                double weight_sum = 0.0;
                for (int sy = static_cast<int>(std::floor(sy0)); sy < static_cast<int>(std::ceil(sy1)); ++sy) {
                    if (sy < 0 || sy >= src_h) continue;
                    const float wy = std::max(0.0f, std::min(sy1, sy + 1.0f) - std::max(sy0, static_cast<float>(sy)));
                    for (int sx = static_cast<int>(std::floor(sx0)); sx < static_cast<int>(std::ceil(sx1)); ++sx) {
                        if (sx < 0 || sx >= src_w) continue;
                        const float wx = std::max(0.0f, std::min(sx1, sx + 1.0f) - std::max(sx0, static_cast<float>(sx)));
                        const float weight = wx * wy;
                        const size_t source_index = (static_cast<size_t>(sy) * src_w + sx) * 4;
                        for (int c = 0; c < 3; ++c) sums[c] += linear_lut_[src[source_index + c]] * weight;
                        sums[3] += src[source_index + 3] * weight;
                        weight_sum += weight;
                    }
                }
                const size_t output_index = dst_row + static_cast<size_t>(dst_x) * 4;
                for (int c = 0; c < 4; ++c) {
                    const float value = weight_sum > 0.0 ? static_cast<float>(sums[c] / weight_sum) : 0.0f;
                    dst[output_index + c] = c < 3 ? ColorSpace::linear_to_srgb(value)
                        : static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 255.0f)));
                }
                continue;
            }
            const float src_x = (x + 0.5f) * x_ratio - 0.5f;
            int x0 = static_cast<int>(std::floor(src_x));
            int x1 = std::min(x0 + 1, src_w - 1);
            const float fx = src_x - x0;
            x0 = std::clamp(x0, 0, src_w - 1);
            x1 = std::clamp(x1, 0, src_w - 1);
            const float wx0 = 1.0f - fx;

            const size_t i00 = row0 + static_cast<size_t>(x0) * 4;
            const size_t i10 = row0 + static_cast<size_t>(x1) * 4;
            const size_t i01 = row1 + static_cast<size_t>(x0) * 4;
            const size_t i11 = row1 + static_cast<size_t>(x1) * 4;
            const size_t odx = dst_row + static_cast<size_t>(dst_x) * 4;

            for (int c = 0; c < 4; ++c) {
                const float p00 = c < 3 ? linear_lut_[src[i00 + c]] : src[i00 + c];
                const float p10 = c < 3 ? linear_lut_[src[i10 + c]] : src[i10 + c];
                const float p01 = c < 3 ? linear_lut_[src[i01 + c]] : src[i01 + c];
                const float p11 = c < 3 ? linear_lut_[src[i11 + c]] : src[i11 + c];
                const float v0 = p00 * wx0 + p10 * fx;
                const float v1 = p01 * wx0 + p11 * fx;
                const float value = v0 * wy0 + v1 * fy;
                dst[odx + static_cast<size_t>(c)] = c < 3 ? ColorSpace::linear_to_srgb(value)
                    : static_cast<uint8_t>(std::round(std::clamp(value, 0.0f, 255.0f)));
            }
        }
    }

}

void Pipeline::compute_cell_mean_colors(const FrameBuffer& input,
                                        int target_w, int target_h,
                                        int grid_cols, int grid_rows,
                                        std::vector<std::array<float, 3>>& means) const {
    const int cell_count = grid_cols * grid_rows;
    means.assign(static_cast<size_t>(std::max(0, cell_count)), {0.0f, 0.0f, 0.0f});
    if (cell_count <= 0) {
        return;
    }

    const ResizePlan plan = compute_resize_plan(input.width(), input.height());
    const int src_w = input.width();
    const int src_h = input.height();
    const uint8_t* src = input.data();
    const float x_ratio = static_cast<float>(src_w) / static_cast<float>(plan.scale_w);
    const float y_ratio = static_cast<float>(src_h) / static_cast<float>(plan.scale_h);

#ifdef HAS_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int cell = 0; cell < cell_count; ++cell) {
        const int cell_col = cell % grid_cols;
        const int cell_row = cell / grid_cols;
        const int dst_x0 = cell_col * config_.cell_width;
        const int dst_y0 = cell_row * config_.cell_height;
        const int dst_x1 = std::min(dst_x0 + config_.cell_width, target_w);
        const int dst_y1 = std::min(dst_y0 + config_.cell_height, target_h);
        const int image_x0 = std::max(dst_x0, plan.offset_x);
        const int image_y0 = std::max(dst_y0, plan.offset_y);
        const int image_x1 = std::min(dst_x1, plan.offset_x + plan.scale_w);
        const int image_y1 = std::min(dst_y1, plan.offset_y + plan.scale_h);
        if (image_x0 >= image_x1 || image_y0 >= image_y1) continue;

        const float sx0 = (image_x0 - plan.offset_x) * x_ratio;
        const float sx1 = (image_x1 - plan.offset_x) * x_ratio;
        const float sy0 = (image_y0 - plan.offset_y) * y_ratio;
        const float sy1 = (image_y1 - plan.offset_y) * y_ratio;
        double sums[3] = {};
        double weight_sum = 0.0;
        for (int sy = static_cast<int>(std::floor(sy0));
             sy < static_cast<int>(std::ceil(sy1)); ++sy) {
            if (sy < 0 || sy >= src_h) continue;
            const float wy = std::max(
                0.0f, std::min(sy1, sy + 1.0f) - std::max(sy0, static_cast<float>(sy)));
            const size_t row = static_cast<size_t>(sy) * src_w * 4;
            for (int sx = static_cast<int>(std::floor(sx0));
                 sx < static_cast<int>(std::ceil(sx1)); ++sx) {
                if (sx < 0 || sx >= src_w) continue;
                const float wx = std::max(
                    0.0f, std::min(sx1, sx + 1.0f) - std::max(sx0, static_cast<float>(sx)));
                const float weight = wx * wy;
                const size_t pixel = row + static_cast<size_t>(sx) * 4;
                sums[0] += linear_lut_[src[pixel]] * weight;
                sums[1] += linear_lut_[src[pixel + 1]] * weight;
                sums[2] += linear_lut_[src[pixel + 2]] * weight;
                weight_sum += weight;
            }
        }
        if (weight_sum > 0.0) {
            // Uncovered fit padding contributes black to the entire cell.
            const double cell_area = static_cast<double>(dst_x1 - dst_x0) * (dst_y1 - dst_y0);
            const double image_area = static_cast<double>(image_x1 - image_x0) * (image_y1 - image_y0);
            const float inv = static_cast<float>(image_area / (cell_area * weight_sum));
            means[static_cast<size_t>(cell)][0] = static_cast<float>(sums[0]) * inv;
            means[static_cast<size_t>(cell)][1] = static_cast<float>(sums[1]) * inv;
            means[static_cast<size_t>(cell)][2] = static_cast<float>(sums[2]) * inv;
        }
    }
}

Pipeline::Result Pipeline::process(const FrameBuffer& input, const ProcessOptions& options) {
    Result result;
    if (input.empty()) return result;

    to_grayscale(input, gray_buffer_);
    resize_for_cells(gray_buffer_, result.luminance);

    // Drive edge-mask generation through the detector's configured path
    // (multi-scale + adaptive thresholds), then compute gx/gy for cell stats.
    result.edges = edge_detector_.detect(result.luminance, &result.gradients);
    
    result.grid_cols = cell_aggregator_.grid_cols(result.luminance.width());
    result.grid_rows = cell_aggregator_.grid_rows(result.luminance.height());

    if (options.need_color_buffer) {
        resize_color_for_cells(
            input, result.luminance.width(), result.luminance.height(), result.color_buffer);
    }

    const int expected_cells = result.grid_cols * result.grid_rows;
    const bool reuse_stats = options.reuse_cell_stats != nullptr &&
                             static_cast<int>(options.reuse_cell_stats->size()) == expected_cells;
    if (reuse_stats) {
        result.cell_stats = *options.reuse_cell_stats;
        return result;
    }

    result.cell_stats = cell_aggregator_.compute(
        result.luminance, result.edges, nullptr, &result.gradients);

    if (options.need_color_stats) {
        std::vector<std::array<float, 3>> means;
        compute_cell_mean_colors(input,
                                 result.luminance.width(),
                                 result.luminance.height(),
                                 result.grid_cols,
                                 result.grid_rows,
                                 means);
        const size_t n = std::min(result.cell_stats.size(), means.size());
        for (size_t i = 0; i < n; ++i) {
            result.cell_stats[i].mean_r = means[i][0];
            result.cell_stats[i].mean_g = means[i][1];
            result.cell_stats[i].mean_b = means[i][2];
        }
    }

    contour_extractor_.apply(
        result.luminance,
        result.edges,
        config_.cell_width,
        config_.cell_height,
        result.grid_cols,
        result.grid_rows,
        result.cell_stats);
    
    return result;
}

Pipeline::ProcessOptions Pipeline::default_process_options() {
    return ProcessOptions{};
}

}
