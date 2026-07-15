#pragma once

#include "core/types.hpp"
#include "core/pipeline.hpp"
#include <vector>

namespace ascii {

class PipelineRuntimeCache {
public:
    struct Query {
        int reuse_limit = 0;
        float still_threshold = 0.0f;
        float cell_stats_threshold_scale = 0.8f;
        bool need_color_buffer = true;
        bool need_color_stats = true;
    };

    struct Decision {
        bool reuse_pipeline_result = false;
        const Pipeline::Result* reused_result = nullptr;
        Pipeline::ProcessOptions process_options{};
        bool reuse_cell_stats = false;
    };

    Decision begin_frame(const FrameBuffer& frame, const Query& query);

    void commit_processed_result(const Pipeline::Result& result,
                                 bool has_color_buffer,
                                 bool reused_cell_stats);

    const Pipeline::Result& cached_result() const { return cached_pipeline_result_; }

    void invalidate();

private:
    std::vector<uint8_t> previous_frame_bytes_;
    int previous_frame_width_ = 0;
    int previous_frame_height_ = 0;
    Pipeline::Result cached_pipeline_result_;
    bool have_cached_pipeline_result_ = false;
    bool cached_pipeline_has_color_buffer_ = false;
    int pipeline_reuse_frames_ = 0;

    std::vector<CellStats> cached_cell_stats_;
    bool have_cached_cell_stats_ = false;
    int cell_stats_reuse_frames_ = 0;

    bool is_identical_to_previous(const FrameBuffer& frame) const;
    void remember_frame(const FrameBuffer& frame);
};

}  // namespace ascii
