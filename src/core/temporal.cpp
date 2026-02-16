#include "temporal.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace ascii {

TemporalSmoother::TemporalSmoother(const Config& config) : config_(config) {}

void TemporalSmoother::initialize(int grid_cols, int grid_rows) {
    cols_ = grid_cols;
    rows_ = grid_rows;
    frame_state_.assign(static_cast<size_t>(grid_cols) * grid_rows, CellState{});
}

void TemporalSmoother::reset() {
    frame_state_.assign(static_cast<size_t>(cols_) * rows_, CellState{});
}

static void check_bounds(int idx, const std::vector<TemporalSmoother::CellState>& state) {
    if (idx < 0 || static_cast<size_t>(idx) >= state.size()) {
        throw std::out_of_range("TemporalSmoother: index out of bounds");
    }
}

static void push_history(std::array<float, 8>& hist, int& head, int& count, float v) {
    hist[head] = v;
    head = (head + 1) % static_cast<int>(hist.size());
    count = std::min(count + 1, static_cast<int>(hist.size()));
}

static float history_recent(const std::array<float, 8>& hist, int head, int count, int back) {
    if (count <= 0 || back < 0 || back >= count) return 0.0f;
    int idx = (head - 1 - back + static_cast<int>(hist.size())) % static_cast<int>(hist.size());
    return hist[idx];
}

static float soft_threshold(float coeff, float threshold) {
    if (std::abs(coeff) <= threshold) {
        return 0.0f;
    }
    return coeff > 0.0f ? coeff - threshold : coeff + threshold;
}

static float haar_wavelet_recent4(const std::array<float, 8>& hist, int head, int count, float strength) {
    if (count < 4) {
        return history_recent(hist, head, count, 0);
    }

    float s0 = history_recent(hist, head, count, 3);
    float s1 = history_recent(hist, head, count, 2);
    float s2 = history_recent(hist, head, count, 1);
    float s3 = history_recent(hist, head, count, 0);
    constexpr float inv_sqrt2 = 0.7071067811865475f;

    float a0 = (s0 + s1) * inv_sqrt2;
    float d0 = (s0 - s1) * inv_sqrt2;
    float a1 = (s2 + s3) * inv_sqrt2;
    float d1 = (s2 - s3) * inv_sqrt2;
    float a2 = (a0 + a1) * inv_sqrt2;
    float d2 = (a0 - a1) * inv_sqrt2;

    float t_fine = std::max(0.0f, strength) * 0.15f;
    float t_coarse = std::max(0.0f, strength) * 0.04f;
    d0 = soft_threshold(d0, t_fine);
    d1 = soft_threshold(d1, t_fine);
    d2 = soft_threshold(d2, t_coarse);

    float ra0 = (a2 + d2) * inv_sqrt2;
    float ra1 = (a2 - d2) * inv_sqrt2;
    float rs[4];
    rs[0] = (ra0 + d0) * inv_sqrt2;
    rs[1] = (ra0 - d0) * inv_sqrt2;
    rs[2] = (ra1 + d1) * inv_sqrt2;
    rs[3] = (ra1 - d1) * inv_sqrt2;
    return rs[3];
}

static float haar_wavelet_recent8(const std::array<float, 8>& hist, int head, int count, float strength) {
    if (count < 8) {
        return haar_wavelet_recent4(hist, head, count, strength);
    }

    float s[8];
    for (int i = 0; i < 8; ++i) {
        s[i] = history_recent(hist, head, count, 7 - i);
    }
    constexpr float inv_sqrt2 = 0.7071067811865475f;

    float a0 = (s[0] + s[1]) * inv_sqrt2; float d_l1_0 = (s[0] - s[1]) * inv_sqrt2;
    float a1 = (s[2] + s[3]) * inv_sqrt2; float d_l1_1 = (s[2] - s[3]) * inv_sqrt2;
    float a2 = (s[4] + s[5]) * inv_sqrt2; float d_l1_2 = (s[4] - s[5]) * inv_sqrt2;
    float a3 = (s[6] + s[7]) * inv_sqrt2; float d_l1_3 = (s[6] - s[7]) * inv_sqrt2;

    float b0 = (a0 + a1) * inv_sqrt2; float d_l2_0 = (a0 - a1) * inv_sqrt2;
    float b1 = (a2 + a3) * inv_sqrt2; float d_l2_1 = (a2 - a3) * inv_sqrt2;

    float c0 = (b0 + b1) * inv_sqrt2;
    float d_l3 = (b0 - b1) * inv_sqrt2;

    float s_strength = std::max(0.0f, strength);
    float t1 = s_strength * 0.18f;
    float t2 = s_strength * 0.08f;
    float t3 = s_strength * 0.02f;

    d_l1_0 = soft_threshold(d_l1_0, t1);
    d_l1_1 = soft_threshold(d_l1_1, t1);
    d_l1_2 = soft_threshold(d_l1_2, t1);
    d_l1_3 = soft_threshold(d_l1_3, t1);
    d_l2_0 = soft_threshold(d_l2_0, t2);
    d_l2_1 = soft_threshold(d_l2_1, t2);
    d_l3 = soft_threshold(d_l3, t3);

    float rb0 = (c0 + d_l3) * inv_sqrt2;
    float rb1 = (c0 - d_l3) * inv_sqrt2;
    float ra0 = (rb0 + d_l2_0) * inv_sqrt2;
    float ra1 = (rb0 - d_l2_0) * inv_sqrt2;
    float ra2 = (rb1 + d_l2_1) * inv_sqrt2;
    float ra3 = (rb1 - d_l2_1) * inv_sqrt2;

    float rs[8];
    rs[0] = (ra0 + d_l1_0) * inv_sqrt2;
    rs[1] = (ra0 - d_l1_0) * inv_sqrt2;
    rs[2] = (ra1 + d_l1_1) * inv_sqrt2;
    rs[3] = (ra1 - d_l1_1) * inv_sqrt2;
    rs[4] = (ra2 + d_l1_2) * inv_sqrt2;
    rs[5] = (ra2 - d_l1_2) * inv_sqrt2;
    rs[6] = (ra3 + d_l1_3) * inv_sqrt2;
    rs[7] = (ra3 - d_l1_3) * inv_sqrt2;
    return rs[7];
}

float TemporalSmoother::smooth_luminance(int idx, float new_value) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    
    state.prev_luminance = state.smoothed_luminance;
    push_history(state.lum_history, state.lum_history_head, state.lum_history_count, new_value);
    
    if (!state.initialized) {
        state.smoothed_luminance = new_value;
        state.initialized = true;
        return new_value;
    }

    float observed = new_value;
    if (config_.use_wavelet_flicker && config_.wavelet_window >= 8 && state.lum_history_count >= 8) {
        observed = haar_wavelet_recent8(
            state.lum_history, state.lum_history_head, state.lum_history_count, config_.wavelet_strength
        );
    } else if (config_.use_wavelet_flicker && config_.wavelet_window >= 4) {
        observed = haar_wavelet_recent4(
            state.lum_history, state.lum_history_head, state.lum_history_count, config_.wavelet_strength
        );
    }

    state.smoothed_luminance = config_.alpha * observed +
                               (1.0f - config_.alpha) * state.smoothed_luminance;
    return state.smoothed_luminance;
}

float TemporalSmoother::smooth_edge_strength(int idx, float new_value) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    
    state.prev_edge_strength = state.smoothed_edge_strength;
    
    if (!state.initialized) {
        state.smoothed_edge_strength = new_value;
        return new_value;
    }
    
    state.smoothed_edge_strength = config_.alpha * new_value + 
                                   (1.0f - config_.alpha) * state.smoothed_edge_strength;
    return state.smoothed_edge_strength;
}

float TemporalSmoother::smooth_coherence(int idx, float new_value) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    
    if (!state.initialized) {
        state.smoothed_coherence = new_value;
        return new_value;
    }
    
    state.smoothed_coherence = config_.alpha * new_value + (1.0f - config_.alpha) * state.smoothed_coherence;
    return state.smoothed_coherence;
}

int TemporalSmoother::reference_index_for_cell(int idx) const {
    if (cols_ <= 0 || rows_ <= 0) {
        return idx;
    }
    const CellState& state = frame_state_[idx];
    int x = idx % cols_;
    int y = idx / cols_;

    int rx = x - static_cast<int>(std::lround(state.motion_offset_x));
    int ry = y - static_cast<int>(std::lround(state.motion_offset_y));
    if (rx < 0 || rx >= cols_ || ry < 0 || ry >= rows_) {
        return idx;
    }
    int ridx = ry * cols_ + rx;
    if (ridx < 0 || static_cast<size_t>(ridx) >= frame_state_.size()) {
        return idx;
    }
    return ridx;
}

bool TemporalSmoother::should_change_glyph(int idx, uint32_t new_glyph, float new_score) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    int ref_idx = reference_index_for_cell(idx);
    const CellState& ref_state = frame_state_[ref_idx];
    const CellState& baseline = ref_state.initialized ? ref_state : state;
    
    if (!baseline.initialized || baseline.last_glyph == 0) {
        return true;
    }
    
    if (baseline.last_glyph == new_glyph) {
        return false;
    }
    
    if (config_.enable_hysteresis) {
        return new_score > baseline.last_score + config_.hysteresis_margin;
    }
    
    return true;
}

bool TemporalSmoother::should_change_glyph_with_loss(int idx, uint32_t new_glyph, float new_loss, float transition_cost) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    int ref_idx = reference_index_for_cell(idx);
    const CellState& ref_state = frame_state_[ref_idx];
    const CellState& baseline = ref_state.initialized ? ref_state : state;
    
    if (!baseline.initialized || baseline.last_glyph == 0) {
        return true;
    }
    
    if (baseline.last_glyph == new_glyph) {
        return false;
    }
    
    float total_new_loss = new_loss + transition_cost;
    float margin = config_.enable_hysteresis ? config_.hysteresis_margin : 0.0f;
    return total_new_loss + margin < baseline.last_loss;
}

void TemporalSmoother::update_glyph(int idx, uint32_t glyph, float score) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    state.last_glyph = glyph;
    state.last_score = score;
    state.initialized = true;
}

void TemporalSmoother::update_glyph_with_loss(int idx, uint32_t glyph, float score, float loss) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    state.last_glyph = glyph;
    state.last_score = score;
    state.last_loss = loss;
    state.initialized = true;
}

void TemporalSmoother::update_edge_state(int idx, bool is_edge) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    
    if (!state.initialized) {
        state.is_edge_state = is_edge;
        return;
    }
    
    if (state.is_edge_state && !is_edge) {
        if (state.smoothed_edge_strength < config_.edge_exit_threshold) {
            state.is_edge_state = false;
        }
    } else if (!state.is_edge_state && is_edge) {
        if (state.smoothed_edge_strength >= config_.edge_enter_threshold) {
            state.is_edge_state = true;
        }
    }
}

bool TemporalSmoother::get_edge_state(int idx) const {
    if (idx < 0 || static_cast<size_t>(idx) >= frame_state_.size()) {
        return false;
    }
    return frame_state_[idx].is_edge_state;
}

void TemporalSmoother::set_motion_offset(int idx, float ox, float oy) {
    check_bounds(idx, frame_state_);
    CellState& state = frame_state_[idx];
    state.motion_offset_x = ox;
    state.motion_offset_y = oy;
}

}
