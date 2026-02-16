#pragma once

#include "core/types.hpp"
#include "glyph/glyph_cache.hpp"
#include <vector>

namespace ascii {

class BitmapRenderer {
public:
    BitmapRenderer();
    
    void set_cache(GlyphCache* cache);
    void set_cell_size(int width, int height);
    
    FrameBuffer render(const std::vector<uint32_t>& codepoints, int cols, int rows);
    
private:
    GlyphCache* cache_ = nullptr;
    int cell_width_ = 8;
    int cell_height_ = 16;
};

}
