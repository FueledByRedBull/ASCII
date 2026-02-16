#pragma once

#include <vector>
#include <cstdint>
#include <cmath>

namespace ascii {

struct GlyphStats {
    uint32_t codepoint = 0;
    float brightness = 0.0f;
    std::vector<float> orientation_hist;
    float contrast = 0.0f;
    std::vector<float> frequency_signature;
    std::vector<float> texture_signature;
    
    float orientation_similarity(const std::vector<float>& other) const {
        if (orientation_hist.size() != other.size()) return 0.0f;
        
        float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
        for (size_t i = 0; i < orientation_hist.size(); ++i) {
            dot += orientation_hist[i] * other[i];
            norm_a += orientation_hist[i] * orientation_hist[i];
            norm_b += other[i] * other[i];
        }
        
        float denom = std::sqrt(norm_a * norm_b);
        return denom > 0.0f ? dot / denom : 0.0f;
    }
    
    float orientation_peak() const {
        if (orientation_hist.empty()) return -1.0f;
        float max_val = 0.0f;
        int max_idx = 0;
        for (size_t i = 0; i < orientation_hist.size(); ++i) {
            if (orientation_hist[i] > max_val) {
                max_val = orientation_hist[i];
                max_idx = static_cast<int>(i);
            }
        }
        return max_val > 0.1f ? static_cast<float>(max_idx) / orientation_hist.size() : -1.0f;
    }
    
    bool is_good_edge_glyph() const {
        if (brightness < 0.1f || brightness > 0.9f) return false;
        float peak = orientation_peak();
        return peak >= 0.0f;
    }
};

}
