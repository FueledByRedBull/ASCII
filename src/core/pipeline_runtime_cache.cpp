#include "pipeline_runtime_cache.hpp"

#include <algorithm>
#include <cmath>

namespace ascii {

void PipelineRuntimeCache::build_scene_signature(const FrameBuffer& frame, std::vector<float>& signature) {
    signature.clear();
    const int w = frame.width();
    const int h = frame.height();
    if (w <= 0 || h <= 0 || frame.empty()) {
        return;
    }

    const int stride = std::max(1, std::min(w, h) / 48);
    const uint8_t* src = frame.data();
    const int samples_x = (w + stride - 1) / stride;
    const int samples_y = (h + stride - 1) / stride;
    signature.reserve(static_cast<size_t>(samples_x) * samples_y);

    for (int y = 0; y < h; y += stride) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w) * 4;
        for (int x = 0; x < w; x += stride) {
            const size_t idx = row + static_cast<size_t>(x) * 4;
            const float lum = (0.2126f * src[idx + 0] +
                               0.7152f * src[idx + 1] +
                               0.0722f * src[idx + 2]) / 255.0f;
            signature.push_back(lum);
        }
    }
}

float PipelineRuntimeCache::signature_scene_change(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 1.0f;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return static_cast<float>(sum / static_cast<double>(a.size()));
}

PipelineRuntimeCache::Decision PipelineRuntimeCache::begin_frame(const FrameBuffer& frame, const Query& query) {
    Decision decision;

    std::vector<float> curr_signature;
    build_scene_signature(frame, curr_signature);
    float scene_change = 1.0f;
    if (!prev_scene_signature_.empty() && prev_scene_signature_.size() == curr_signature.size()) {
        scene_change = signature_scene_change(prev_scene_signature_, curr_signature);
    }
    prev_scene_signature_ = std::move(curr_signature);

    const int reuse_limit = std::max(0, query.reuse_limit);
    const float still_thresh = std::max(0.0f, query.still_threshold);

    if (have_cached_pipeline_result_ && reuse_limit > 0) {
        const bool cached_color_ok = !query.need_color_buffer || cached_pipeline_has_color_buffer_;
        if (scene_change < still_thresh &&
            pipeline_reuse_frames_ < reuse_limit &&
            cached_color_ok) {
            decision.reuse_pipeline_result = true;
            decision.reused_result = &cached_pipeline_result_;
            ++pipeline_reuse_frames_;
            if (have_cached_cell_stats_) {
                cell_stats_reuse_frames_ = std::min(cell_stats_reuse_frames_ + 1, reuse_limit);
            }
            return decision;
        }
    }

    decision.process_options.need_color_buffer = query.need_color_buffer;
    decision.process_options.need_color_stats = query.need_color_stats;
    const float cell_stats_reuse_thresh = still_thresh * query.cell_stats_threshold_scale;
    decision.reuse_cell_stats =
        have_cached_cell_stats_ &&
        reuse_limit > 0 &&
        scene_change < cell_stats_reuse_thresh &&
        cell_stats_reuse_frames_ < reuse_limit;
    if (decision.reuse_cell_stats) {
        decision.process_options.reuse_cell_stats = &cached_cell_stats_;
    }

    return decision;
}

void PipelineRuntimeCache::commit_processed_result(const Pipeline::Result& result,
                                                   bool has_color_buffer,
                                                   bool reused_cell_stats) {
    cached_pipeline_result_ = result;
    have_cached_pipeline_result_ = true;
    cached_pipeline_has_color_buffer_ = has_color_buffer;
    pipeline_reuse_frames_ = 0;

    if (reused_cell_stats) {
        ++cell_stats_reuse_frames_;
    } else {
        cached_cell_stats_ = result.cell_stats;
        have_cached_cell_stats_ = true;
        cell_stats_reuse_frames_ = 0;
    }
}

void PipelineRuntimeCache::invalidate() {
    prev_scene_signature_.clear();
    have_cached_pipeline_result_ = false;
    cached_pipeline_has_color_buffer_ = false;
    pipeline_reuse_frames_ = 0;
    have_cached_cell_stats_ = false;
    cell_stats_reuse_frames_ = 0;
    cached_cell_stats_.clear();
}

}  // namespace ascii
