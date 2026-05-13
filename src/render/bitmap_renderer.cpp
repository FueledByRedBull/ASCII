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

FrameBuffer BitmapRenderer::render(const std::vector<ASCIICell>& cells, int cols, int rows) {
    FrameBuffer result(cols * cell_width_, rows * cell_height_, Color(0, 0, 0, 255));
    uint8_t* dst = result.data();
    const int dst_w = result.width();

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const int idx = row * cols + col;
            if (idx >= static_cast<int>(cells.size())) continue;

            const ASCIICell& cell = cells[idx];
            const GlyphBitmap* glyph = cache_ ? cache_->get_bitmap(cell.codepoint) : nullptr;

            const int x0 = col * cell_width_;
            const int y0 = row * cell_height_;

            for (int gy = 0; gy < cell_height_; ++gy) {
                const int py = y0 + gy;
                if (py < 0 || py >= result.height()) continue;

                for (int gx = 0; gx < cell_width_; ++gx) {
                    const int px = x0 + gx;
                    if (px < 0 || px >= dst_w) continue;
                    const size_t out = (static_cast<size_t>(py) * dst_w + px) * 4;

                    uint8_t alpha = 0;
                    if (glyph && gx < glyph->width && gy < glyph->height) {
                        alpha = glyph->pixels[gy * glyph->width + gx];
                    }

                    const float a = alpha / 255.0f;
                    dst[out + 0] = static_cast<uint8_t>(cell.bg_r * (1.0f - a) + cell.fg_r * a);
                    dst[out + 1] = static_cast<uint8_t>(cell.bg_g * (1.0f - a) + cell.fg_g * a);
                    dst[out + 2] = static_cast<uint8_t>(cell.bg_b * (1.0f - a) + cell.fg_b * a);
                    dst[out + 3] = 255;
                }
            }
        }
    }

    return result;
}

}
