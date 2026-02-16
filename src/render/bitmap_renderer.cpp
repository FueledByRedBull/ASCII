#include "bitmap_renderer.hpp"

namespace ascii {

BitmapRenderer::BitmapRenderer() = default;

void BitmapRenderer::set_cache(GlyphCache* cache) {
    cache_ = cache;
}

void BitmapRenderer::set_cell_size(int width, int height) {
    cell_width_ = width;
    cell_height_ = height;
}

FrameBuffer BitmapRenderer::render(const std::vector<uint32_t>& codepoints, int cols, int rows) {
    FrameBuffer result(cols * cell_width_, rows * cell_height_, Color(0, 0, 0, 255));
    
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int idx = row * cols + col;
            if (idx >= static_cast<int>(codepoints.size())) continue;
            
            uint32_t cp = codepoints[idx];
            const GlyphBitmap* glyph = cache_ ? cache_->get_bitmap(cp) : nullptr;
            
            if (!glyph) {
                continue;
            }
            
            int x0 = col * cell_width_;
            int y0 = row * cell_height_;
            
            for (int gy = 0; gy < glyph->height; ++gy) {
                for (int gx = 0; gx < glyph->width; ++gx) {
                    int px = x0 + gx;
                    int py = y0 + gy;
                    
                    if (px >= result.width() || py >= result.height()) continue;
                    
                    uint8_t alpha = glyph->pixels[gy * glyph->width + gx];
                    if (alpha > 0) {
                        float a = alpha / 255.0f;
                        Color existing = result.get_pixel(px, py);
                        Color blended(
                            static_cast<uint8_t>(existing.r * (1 - a) + 255 * a),
                            static_cast<uint8_t>(existing.g * (1 - a) + 255 * a),
                            static_cast<uint8_t>(existing.b * (1 - a) + 255 * a),
                            255
                        );
                        result.set_pixel(px, py, blended);
                    }
                }
            }
        }
    }
    
    return result;
}

}
