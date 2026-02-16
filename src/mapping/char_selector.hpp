#pragma once

#include "core/types.hpp"
#include "glyph/glyph_cache.hpp"
#include "core/temporal.hpp"
#include <vector>
#include <cstdint>

namespace ascii {

struct UnifiedLossWeights {
    float brightness = 0.45f;
    float orientation = 0.40f;
    float contrast = 0.15f;
    float frequency = 0.20f;
    float texture = 0.15f;
    
    float normalize() const {
        return brightness + orientation + contrast + frequency + texture;
    }
};

class CharSelector {
public:
    struct Config {
        float edge_threshold = 0.1f;
        bool use_orientation_matching = true;
        bool use_simple_orientation = false;
        int orientation_bins = 8;
        
        UnifiedLossWeights loss_weights;
        float transition_penalty = 0.15f;
        bool use_unified_loss = true;
        bool enable_frequency_matching = true;
        bool enable_texture_matching = true;
    };
    
    CharSelector() = default;
    explicit CharSelector(const Config& config);
    
    void set_cache(GlyphCache* cache);
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    
    struct Selection {
        uint32_t codepoint;
        float score;
        float loss;
    };
    
    Selection select(const CellStats& stats, const TemporalSmoother& smoother, int idx);
    Selection select_fill(float luminance);
    Selection select_edge(float orientation);
    Selection select_edge_simple(float orientation);
    
    Selection select_unified(const CellStats& stats, uint32_t prev_glyph);
    
    float compute_loss(const CellStats& cell, const GlyphStats& glyph) const;
    float compute_transition_cost(uint32_t from_glyph, uint32_t to_glyph) const;
    
private:
    Config config_;
    GlyphCache* cache_ = nullptr;
    
    std::vector<float> compute_orientation_hist(float orientation, int bins) const;
    float orientation_hist_distance(const std::vector<float>& a, const std::vector<float>& b) const;
};

}
