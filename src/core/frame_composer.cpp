#include "frame_composer.hpp"

#include "core/color_space.hpp"

#include <algorithm>
#include <cmath>

namespace ascii {

FrameComposer::Output FrameComposer::compose(const Pipeline::Result& result, Context& context) {
    if (!smoother_initialized_ ||
        smoother_cols_ != result.grid_cols ||
        smoother_rows_ != result.grid_rows) {
        context.smoother.initialize(result.grid_cols, result.grid_rows);
        smoother_initialized_ = true;
        smoother_cols_ = result.grid_cols;
        smoother_rows_ = result.grid_rows;
    }

    Output output;
    output.cells.resize(static_cast<size_t>(result.grid_cols) * result.grid_rows);
    context.smoother.begin_frame();

    if (context.color_mode == ColorMode::BlockArt) {
        output.block_cells.resize(output.cells.size());
    }

#ifdef HAS_OPENMP
    const bool cells_are_independent = context.color_mode == ColorMode::Truecolor ||
                                       context.color_mode == ColorMode::None;
    #pragma omp parallel for schedule(static) if(cells_are_independent)
#endif
    for (int i = 0; i < static_cast<int>(result.cell_stats.size()); ++i) {
        const auto& stats = result.cell_stats[i];
        const int cell_x = i % result.grid_cols;
        const int cell_y = i / result.grid_cols;

        const float smoothed_lum = context.smoother.smooth_luminance(i, stats.mean_luminance);
        const float smoothed_edge = context.smoother.smooth_edge_strength(i, stats.edge_strength);
        const float smoothed_coh = context.smoother.smooth_coherence(i, stats.structure_coherence);

        CellStats effective_stats = stats;
        effective_stats.mean_luminance = smoothed_lum;
        effective_stats.edge_strength = smoothed_edge;
        effective_stats.structure_coherence = smoothed_coh;

        if (context.bilateral_grid.valid()) {
            auto smooth_rgb = context.bilateral_grid.sample(cell_x, cell_y, smoothed_lum);
            if (smooth_rgb.has_support) {
                effective_stats.mean_r = smooth_rgb.r;
                effective_stats.mean_g = smooth_rgb.g;
                effective_stats.mean_b = smooth_rgb.b;
            }
        }

        const float adaptive_edge_margin = 0.02f * static_cast<float>(effective_stats.adaptive_level);
        const float adaptive_edge_threshold = std::clamp(
            context.edge_threshold - adaptive_edge_margin, 0.0f, 1.0f);

        const bool edge_candidate = (stats.edge_occupancy >= adaptive_edge_threshold) ||
                                    (smoothed_edge >= adaptive_edge_threshold) ||
                                    stats.has_contour ||
                                    (smoothed_coh >= std::max(
                                        0.05f, 0.2f - 0.04f * effective_stats.adaptive_level));
        context.smoother.update_edge_state(i, edge_candidate);
        effective_stats.is_edge_cell = context.smoother.get_edge_state(i);

        if (context.motion.has_motion()) {
            float dx = 0.0f;
            float dy = 0.0f;
            context.motion.get_motion_for_cell(
                cell_x * context.config.grid.cell_width,
                cell_y * context.config.grid.cell_height,
                context.config.grid.cell_width,
                context.config.grid.cell_height,
                dx,
                dy);
            context.smoother.set_motion_offset(
                i,
                dx / static_cast<float>(std::max(1, context.config.grid.cell_width)),
                dy / static_cast<float>(std::max(1, context.config.grid.cell_height)));
        } else {
            context.smoother.set_motion_offset(i, 0.0f, 0.0f);
        }

        if (context.color_mode == ColorMode::BlockArt) {
            BlockRenderer::CellData block_data = context.block_renderer.analyze_cell(
                result.luminance,
                result.color_buffer,
                cell_x,
                cell_y,
                context.config.grid.cell_width,
                context.config.grid.cell_height,
                effective_stats);

            const auto block_result = context.block_renderer.render_cell(block_data);
            output.block_cells[i] = block_result;
            output.cells[i].codepoint = block_result.codepoint;
            output.cells[i].fg_r = block_result.fg_r;
            output.cells[i].fg_g = block_result.fg_g;
            output.cells[i].fg_b = block_result.fg_b;
            output.cells[i].bg_r = block_result.bg_r;
            output.cells[i].bg_g = block_result.bg_g;
            output.cells[i].bg_b = block_result.bg_b;
            continue;
        }

        auto selection = context.selector.select(effective_stats, context.smoother, i);
        const bool contour_override = context.config.edge.contours_enabled && effective_stats.has_contour &&
                                      effective_stats.contour_codepoint != 0;
        if (context.selector.config().use_unified_loss) {
            const uint32_t previous = context.smoother.reference_glyph(i);
            const float transition_cost = context.selector.compute_transition_cost(
                previous, selection.codepoint);
            const float keep_loss = context.selector.compute_loss_for_glyph(effective_stats, previous);

            if (context.smoother.should_change_glyph_with_loss(
                    i, selection.codepoint, selection.loss, keep_loss, transition_cost)) {
                output.cells[i].codepoint = selection.codepoint;
                context.smoother.update_glyph_with_loss(
                    i,
                    selection.codepoint,
                    selection.score,
                    selection.loss + transition_cost);
            } else {
                output.cells[i].codepoint = previous;
                context.smoother.update_glyph_with_loss(
                    i, previous, 1.0f - std::min(keep_loss, 1.0f), keep_loss);
            }
        } else {
            const uint32_t previous = context.smoother.reference_glyph(i);
            const float keep_loss = context.selector.compute_loss_for_glyph(effective_stats, previous);
            if (context.smoother.should_change_glyph_with_loss(
                    i, selection.codepoint, selection.loss, keep_loss, 0.0f)) {
                output.cells[i].codepoint = selection.codepoint;
                context.smoother.update_glyph_with_loss(
                    i, selection.codepoint, selection.score, selection.loss);
            } else {
                output.cells[i].codepoint = previous;
                context.smoother.update_glyph_with_loss(
                    i, previous, 1.0f - std::min(keep_loss, 1.0f), keep_loss);
            }
        }

        // Overlay contours without retaining them as normal glyph history.
        if (contour_override) output.cells[i].codepoint = effective_stats.contour_codepoint;

        uint8_t sr = 0;
        uint8_t sg = 0;
        uint8_t sb = 0;
        ColorSpace::linear_to_srgb(
            {effective_stats.mean_r, effective_stats.mean_g, effective_stats.mean_b}, sr, sg, sb);
        const int row_dir = 1;
        auto color = context.color_mapper.map_with_dither(
            cell_x, cell_y, row_dir, sr / 255.0f, sg / 255.0f, sb / 255.0f, effective_stats.is_edge_cell);

        output.cells[i].fg_r = color.r;
        output.cells[i].fg_g = color.g;
        output.cells[i].fg_b = color.b;
    }

    if (context.color_mode == ColorMode::BlockArt &&
        !output.block_cells.empty() &&
        context.config.color.block_spectral_palette > 1) {
        context.block_renderer.spectral_quantize_frame(
            output.block_cells,
            context.config.color.block_spectral_palette,
            context.config.color.block_spectral_samples,
            context.config.color.block_spectral_iterations);
        for (size_t i = 0; i < output.cells.size() && i < output.block_cells.size(); ++i) {
            output.cells[i].fg_r = output.block_cells[i].fg_r;
            output.cells[i].fg_g = output.block_cells[i].fg_g;
            output.cells[i].fg_b = output.block_cells[i].fg_b;
            output.cells[i].bg_r = output.block_cells[i].bg_r;
            output.cells[i].bg_g = output.block_cells[i].bg_g;
            output.cells[i].bg_b = output.block_cells[i].bg_b;
        }
    }

    return output;
}

}  // namespace ascii
