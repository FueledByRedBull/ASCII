#include "pipeline_runtime_cache.hpp"

#include <algorithm>
#include <cmath>

namespace ascii {

bool PipelineRuntimeCache::is_identical_to_previous(const FrameBuffer& frame) const {
    return frame.width() == previous_frame_width_ &&
           frame.height() == previous_frame_height_ &&
           frame.byte_size() == previous_frame_bytes_.size() &&
           (frame.byte_size() == 0 ||
            std::equal(previous_frame_bytes_.begin(), previous_frame_bytes_.end(), frame.data()));
}

void PipelineRuntimeCache::remember_frame(const FrameBuffer& frame) {
    previous_frame_width_ = frame.width();
    previous_frame_height_ = frame.height();
    previous_frame_bytes_.assign(frame.data(), frame.data() + frame.byte_size());
}

PipelineRuntimeCache::Decision PipelineRuntimeCache::begin_frame(const FrameBuffer& frame, const Query& query) {
    Decision decision;

    const bool identical = is_identical_to_previous(frame);
    remember_frame(frame);

    const int reuse_limit = std::max(0, query.reuse_limit);

    if (have_cached_pipeline_result_ && reuse_limit > 0) {
        const bool cached_color_ok = !query.need_color_buffer || cached_pipeline_has_color_buffer_;
        if (identical &&
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
    decision.reuse_cell_stats =
        have_cached_cell_stats_ &&
        reuse_limit > 0 &&
        identical &&
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
    previous_frame_bytes_.clear();
    previous_frame_width_ = 0;
    previous_frame_height_ = 0;
    have_cached_pipeline_result_ = false;
    cached_pipeline_has_color_buffer_ = false;
    pipeline_reuse_frames_ = 0;
    have_cached_cell_stats_ = false;
    cell_stats_reuse_frames_ = 0;
    cached_cell_stats_.clear();
}

}  // namespace ascii
