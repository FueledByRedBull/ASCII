#include "char_selector.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace ascii {

CharSelector::CharSelector(const Config& config) : config_(config) {}

void CharSelector::set_cache(GlyphCache* cache) {
    cache_ = cache;
}

float CharSelector::compute_loss(const CellStats& cell, const GlyphStats& glyph) const {
    float norm = config_.loss_weights.normalize();
    if (norm < 0.001f) norm = 1.0f;
    
    float brightness_err = std::abs(cell.mean_luminance - glyph.brightness);
    
    float orientation_err = 0.0f;
    if (!glyph.orientation_hist.empty() && cell.orientation_histogram[0] >= 0) {
        std::vector<float> cell_hist(glyph.orientation_hist.size(), 0.0f);
        int bins = static_cast<int>(glyph.orientation_hist.size());
        for (int i = 0; i < bins && i < 8; ++i) {
            cell_hist[i] = cell.orientation_histogram[i];
        }
        orientation_err = orientation_hist_distance(cell_hist, glyph.orientation_hist);
    }
    
    float contrast_err = std::abs(std::sqrt(cell.luminance_variance) - glyph.contrast);
    float frequency_err = 0.0f;
    if (config_.enable_frequency_matching && !glyph.frequency_signature.empty()) {
        int bins = std::min(static_cast<int>(glyph.frequency_signature.size()), 8);
        for (int i = 0; i < bins; ++i) {
            float d = cell.frequency_signature[i] - glyph.frequency_signature[i];
            frequency_err += d * d;
        }
        frequency_err = std::sqrt(frequency_err / std::max(1, bins));
    }

    float texture_err = 0.0f;
    if (config_.enable_texture_matching && !glyph.texture_signature.empty()) {
        int bins = std::min(static_cast<int>(glyph.texture_signature.size()), 8);
        for (int i = 0; i < bins; ++i) {
            float d = cell.texture_signature[i] - glyph.texture_signature[i];
            texture_err += d * d;
        }
        texture_err = std::sqrt(texture_err / std::max(1, bins));
    }
    
    float loss = (config_.loss_weights.brightness * brightness_err +
                  config_.loss_weights.orientation * orientation_err +
                  config_.loss_weights.contrast * contrast_err +
                  config_.loss_weights.frequency * frequency_err +
                  config_.loss_weights.texture * texture_err) / norm;
    
    return loss;
}

float CharSelector::compute_transition_cost(uint32_t from_glyph, uint32_t to_glyph) const {
    if (from_glyph == 0 || to_glyph == 0) return 0.0f;
    if (from_glyph == to_glyph) return 0.0f;
    
    if (!cache_) return config_.transition_penalty;
    
    auto* from_stats = cache_->get_stats(from_glyph);
    auto* to_stats = cache_->get_stats(to_glyph);
    
    if (!from_stats || !to_stats) return config_.transition_penalty;
    
    float brightness_diff = std::abs(from_stats->brightness - to_stats->brightness);
    
    float orientation_diff = 0.0f;
    if (!from_stats->orientation_hist.empty() && !to_stats->orientation_hist.empty()) {
        orientation_diff = orientation_hist_distance(from_stats->orientation_hist, to_stats->orientation_hist);
    }
    
    float cost = config_.transition_penalty * (0.3f + 0.5f * brightness_diff + 0.2f * orientation_diff);
    
    return cost;
}

float CharSelector::orientation_hist_distance(const std::vector<float>& a, const std::vector<float>& b) const {
    if (a.size() != b.size()) return 1.0f;
    
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    float denom = std::sqrt(norm_a * norm_b);
    if (denom < 0.001f) return 1.0f;
    
    float similarity = dot / denom;
    return 1.0f - similarity;
}

CharSelector::Selection CharSelector::select_unified(const CellStats& stats, uint32_t prev_glyph) {
    if (!cache_) return {static_cast<uint32_t>(' '), 0.0f, 1.0f};
    
    auto sorted = cache_->get_by_brightness();
    if (sorted.empty()) return {static_cast<uint32_t>(' '), 0.0f, 1.0f};
    
    uint32_t best_glyph = static_cast<uint32_t>(' ');
    float best_total_loss = 1e10f;
    float best_data_loss = 1.0f;
    int adaptive = std::clamp(stats.adaptive_level, 0, 3);
    
    if (stats.is_edge_cell && !cache_->get_edge_glyphs().empty()) {
        auto edge_glyphs = cache_->get_edge_glyphs();
        int edge_stride = (adaptive <= 0) ? 2 : 1;
        for (size_t ei = 0; ei < edge_glyphs.size(); ei += edge_stride) {
            uint32_t cp = edge_glyphs[ei];
            auto* glyph_stats = cache_->get_stats(cp);
            if (!glyph_stats) continue;
            
            float data_loss = compute_loss(stats, *glyph_stats);
            float transition_cost = compute_transition_cost(prev_glyph, cp);
            float total_loss = data_loss + transition_cost;
            
            if (total_loss < best_total_loss) {
                best_total_loss = total_loss;
                best_glyph = cp;
                best_data_loss = data_loss;
            }
        }
    }
    
    int search_radius = 10 + adaptive * 5;
    int stride = 1;
    int start_idx = static_cast<int>(sorted.size() * stats.mean_luminance);
    start_idx = std::clamp(start_idx - search_radius, 0, static_cast<int>(sorted.size()) - 1);
    int end_idx = std::min(start_idx + (2 * search_radius + 1), static_cast<int>(sorted.size()));
    
    for (int i = start_idx; i < end_idx; i += stride) {
        uint32_t cp = sorted[i];
        auto* glyph_stats = cache_->get_stats(cp);
        if (!glyph_stats) continue;

        // In flat regions, mildly prefer simpler glyphs but don't reject outright.
        // This avoids the "holes" where only spaces survive the filter.
        float complexity_penalty = 0.0f;
        if (adaptive == 0 && !stats.is_edge_cell && glyph_stats->contrast > 0.35f) {
            complexity_penalty = 0.05f * (glyph_stats->contrast - 0.35f);
        }
        
        float data_loss = compute_loss(stats, *glyph_stats) + complexity_penalty;
        float transition_cost = compute_transition_cost(prev_glyph, cp);
        float total_loss = data_loss + transition_cost;
        
        if (total_loss < best_total_loss) {
            best_total_loss = total_loss;
            best_glyph = cp;
            best_data_loss = data_loss;
        }
    }
    
    float score = 1.0f - std::min(best_data_loss, 1.0f);
    return {best_glyph, score, best_data_loss};
}

CharSelector::Selection CharSelector::select(const CellStats& stats, const TemporalSmoother& smoother, int idx) {
    if (config_.use_unified_loss) {
        uint32_t prev_glyph = 0;
        if (idx >= 0 && static_cast<size_t>(idx) < smoother.frame_state().size()) {
            prev_glyph = smoother.frame_state()[idx].last_glyph;
        }
        return select_unified(stats, prev_glyph);
    }
    
    if (stats.is_edge_cell) {
        if (config_.use_simple_orientation) {
            return select_edge_simple(stats.cell_orientation);
        } else if (config_.use_orientation_matching) {
            return select_edge(stats.cell_orientation);
        }
    }
    return select_fill(stats.mean_luminance);
}

CharSelector::Selection CharSelector::select_fill(float luminance) {
    if (!cache_) return {static_cast<uint32_t>(' '), 0.0f, 1.0f};
    
    auto sorted = cache_->get_by_brightness();
    if (sorted.empty()) return {static_cast<uint32_t>(' '), 0.0f, 1.0f};
    
    auto best = std::lower_bound(sorted.begin(), sorted.end(), luminance,
        [this](uint32_t cp, float lum) {
            auto* s = cache_->get_stats(cp);
            return s && s->brightness < lum;
        });
    
    if (best == sorted.end()) best = sorted.end() - 1;
    if (best != sorted.begin()) {
        auto prev = best - 1;
        auto* s_prev = cache_->get_stats(*prev);
        auto* s_best = cache_->get_stats(*best);
        if (s_prev && s_best) {
            if (std::abs(s_prev->brightness - luminance) < std::abs(s_best->brightness - luminance)) {
                best = prev;
            }
        }
    }
    
    auto* glyph_stats = cache_->get_stats(*best);
    float score = glyph_stats ? 1.0f - std::abs(glyph_stats->brightness - luminance) : 0.0f;
    float loss = glyph_stats ? std::abs(glyph_stats->brightness - luminance) : 1.0f;
    
    return {*best, score, loss};
}

CharSelector::Selection CharSelector::select_edge(float orientation) {
    if (!cache_) return {static_cast<uint32_t>(' '), 0.0f, 1.0f};
    
    auto edge_glyphs = cache_->get_edge_glyphs();
    if (edge_glyphs.empty()) {
        return select_fill(0.5f);
    }
    
    auto cell_hist = compute_orientation_hist(orientation, config_.orientation_bins);
    
    uint32_t best_glyph = edge_glyphs[0];
    float best_sim = -1.0f;
    
    for (uint32_t cp : edge_glyphs) {
        auto* stats = cache_->get_stats(cp);
        if (!stats) continue;
        
        float sim = stats->orientation_similarity(cell_hist);
        if (sim > best_sim) {
            best_sim = sim;
            best_glyph = cp;
        }
    }
    
    return {best_glyph, best_sim, 1.0f - best_sim};
}

CharSelector::Selection CharSelector::select_edge_simple(float orientation) {
    static const uint32_t orientation_chars[] = {
        static_cast<uint32_t>('-'),
        static_cast<uint32_t>('/'),
        static_cast<uint32_t>('|'),
        static_cast<uint32_t>('\\'),
        static_cast<uint32_t>('-'),
        static_cast<uint32_t>('/'),
        static_cast<uint32_t>('|'),
        static_cast<uint32_t>('\\')
    };
    constexpr float kPi = 3.14159265358979323846f;
    
    float normalized = (orientation + kPi) / (2.0f * kPi);
    int bin = static_cast<int>(normalized * 8) % 8;
    if (bin < 0) bin += 8;
    
    return {orientation_chars[bin], 1.0f, 0.0f};
}

std::vector<float> CharSelector::compute_orientation_hist(float orientation, int bins) const {
    std::vector<float> hist(bins, 0.0f);
    constexpr float kPi = 3.14159265358979323846f;
    
    float normalized = (orientation + kPi) / (2.0f * kPi);
    int bin = static_cast<int>(normalized * bins) % bins;
    hist[bin] = 1.0f;
    
    int prev_bin = (bin - 1 + bins) % bins;
    int next_bin = (bin + 1) % bins;
    hist[prev_bin] = 0.5f;
    hist[next_bin] = 0.5f;
    
    float sum = 0.0f;
    for (float v : hist) sum += v;
    for (float& v : hist) v /= sum;
    
    return hist;
}

}
