#pragma once

#include "font_loader.hpp"
#include "glyph_stats.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace ascii {

class GlyphCache {
public:
    GlyphCache();
    
    bool initialize(FontLoader* loader, const std::vector<uint32_t>& codepoints, int target_width, int target_height);
    
    const GlyphStats* get_stats(uint32_t codepoint) const;
    const GlyphBitmap* get_bitmap(uint32_t codepoint) const;
    
    std::vector<uint32_t> get_by_brightness() const;
    std::vector<uint32_t> get_edge_glyphs() const;
    
    int cell_width() const { return cell_width_; }
    int cell_height() const { return cell_height_; }
    
private:
    void render_and_analyze(uint32_t codepoint);
    
    FontLoader* loader_ = nullptr;
    int cell_width_ = 8;
    int cell_height_ = 16;
    
    std::unordered_map<uint32_t, GlyphBitmap> bitmaps_;
    std::unordered_map<uint32_t, GlyphStats> stats_;
    
    std::vector<uint32_t> brightness_sorted_;
    std::vector<uint32_t> edge_glyphs_;
};

}
