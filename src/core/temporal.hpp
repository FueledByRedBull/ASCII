#pragma once

#include "core/types.hpp"
#include <vector>
#include <cstdint>
#include <array>

namespace ascii {

class TemporalSmoother {
public:
    struct Config {
        float alpha = 0.3f;
        float hysteresis_margin = 0.1f;
        bool enable_hysteresis = true;
        float transition_penalty = 0.15f;
        float edge_enter_threshold = 0.08f;
        float edge_exit_threshold = 0.04f;
        bool use_wavelet_flicker = true;
        float wavelet_strength = 0.45f;
        int wavelet_window = 8;
    };
    
    TemporalSmoother() : TemporalSmoother(Config{}) {}
    explicit TemporalSmoother(const Config& config);
    
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    
    void initialize(int grid_cols, int grid_rows);
    void reset();
    
    struct CellState {
        float smoothed_luminance = 0.0f;
        float smoothed_edge_strength = 0.0f;
        float smoothed_coherence = 0.0f;
        uint32_t last_glyph = 0;
        float last_score = 0.0f;
        float last_loss = 0.0f;
        bool is_edge_state = false;
        bool initialized = false;
        
        float prev_luminance = 0.0f;
        float prev_edge_strength = 0.0f;
        float motion_offset_x = 0.0f;
        float motion_offset_y = 0.0f;
        std::array<float, 8> lum_history{};
        int lum_history_count = 0;
        int lum_history_head = 0;
    };
    
    std::vector<CellState>& frame_state() { return frame_state_; }
    const std::vector<CellState>& frame_state() const { return frame_state_; }
    
    float smooth_luminance(int idx, float new_value);
    float smooth_edge_strength(int idx, float new_value);
    float smooth_coherence(int idx, float new_value);
    bool should_change_glyph(int idx, uint32_t new_glyph, float new_score);
    bool should_change_glyph_with_loss(int idx, uint32_t new_glyph, float new_loss, float transition_cost);
    void update_glyph(int idx, uint32_t glyph, float score);
    void update_glyph_with_loss(int idx, uint32_t glyph, float score, float loss);
    
    void update_edge_state(int idx, bool is_edge);
    bool get_edge_state(int idx) const;
    
    void set_motion_offset(int idx, float ox, float oy);
    
private:
    Config config_;
    std::vector<CellState> frame_state_;
    int cols_ = 0;
    int rows_ = 0;

    int reference_index_for_cell(int idx) const;
};

}
