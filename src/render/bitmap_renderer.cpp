#include "bitmap_renderer.hpp"
#include "core/color_space.hpp"
#include <limits>
#include <stdexcept>

namespace ascii {

namespace {
Size bitmap_size(int cols, int rows, int cell_width, int cell_height) {
    if (cols < 0 || rows < 0 || cell_width <= 0 || cell_height <= 0) {
        throw std::invalid_argument("Invalid bitmap grid or cell dimensions");
    }
    if (cols == 0 || rows == 0) return {};
    const int64_t width = static_cast<int64_t>(cols) * cell_width;
    const int64_t height = static_cast<int64_t>(rows) * cell_height;
    // Match the public configuration's analysis/output raster limit.
    if (width > std::numeric_limits<int>::max() || height > std::numeric_limits<int>::max() ||
        width > 100000000 / height) {
        throw std::length_error("Bitmap dimensions are too large");
    }
    return {static_cast<int>(width), static_cast<int>(height)};
}
}

BitmapRenderer::BitmapRenderer() = default;

void BitmapRenderer::set_cache(GlyphCache* cache) {
    cache_ = cache;
}

void BitmapRenderer::set_cell_size(int width, int height) {
    (void)bitmap_size(1, 1, width, height);
    cell_width_ = width;
    cell_height_ = height;
}

FrameBuffer BitmapRenderer::render(const std::vector<uint32_t>& codepoints, int cols, int rows) {
    const auto size = bitmap_size(cols, rows, cell_width_, cell_height_);
    FrameBuffer result(size.width, size.height, Color(0, 0, 0, 255));
    if (result.empty()) return result;
    
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const size_t idx = static_cast<size_t>(row) * cols + col;
            if (idx >= codepoints.size()) continue;
            
            uint32_t cp = codepoints[idx];
            const GlyphBitmap* glyph = cache_ ? cache_->get_bitmap(cp) : nullptr;
            
            if (!glyph) {
                continue;
            }
            
            int x0 = col * cell_width_;
            int y0 = row * cell_height_;
            
            for (int gy = 0; gy < std::min(glyph->height, cell_height_); ++gy) {
                for (int gx = 0; gx < std::min(glyph->width, cell_width_); ++gx) {
                    int px = x0 + gx;
                    int py = y0 + gy;
                    
                    if (px >= result.width() || py >= result.height()) continue;
                    
                    uint8_t alpha = glyph->pixels[static_cast<size_t>(gy) * glyph->width + gx];
                    if (alpha > 0) {
                        const uint8_t value = ColorSpace::linear_to_srgb(alpha / 255.0f);
                        result.set_pixel(px, py, Color(value, value, value));
                    }
                }
            }
        }
    }
    
    return result;
}

FrameBuffer BitmapRenderer::render(const std::vector<ASCIICell>& cells, int cols, int rows) {
    const auto size = bitmap_size(cols, rows, cell_width_, cell_height_);
    FrameBuffer result(size.width, size.height, Color(0, 0, 0, 255));
    if (result.empty()) return result;
    uint8_t* dst = result.data();
    const int dst_w = result.width();

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            const size_t idx = static_cast<size_t>(row) * cols + col;
            if (idx >= cells.size()) continue;

            const ASCIICell& cell = cells[idx];
            const GlyphBitmap* glyph = cache_ ? cache_->get_bitmap(cell.codepoint) : nullptr;
            const auto foreground = ColorSpace::srgb_to_linear(cell.fg_r, cell.fg_g, cell.fg_b);
            const auto background = ColorSpace::srgb_to_linear(cell.bg_r, cell.bg_g, cell.bg_b);

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
                        alpha = glyph->pixels[static_cast<size_t>(gy) * glyph->width + gx];
                    }

                    const float a = alpha / 255.0f;
                    if (alpha == 0) {
                        dst[out + 0] = cell.bg_r;
                        dst[out + 1] = cell.bg_g;
                        dst[out + 2] = cell.bg_b;
                    } else if (alpha == 255) {
                        dst[out + 0] = cell.fg_r;
                        dst[out + 1] = cell.fg_g;
                        dst[out + 2] = cell.fg_b;
                    } else {
                        ColorSpace::linear_to_srgb(background * (1.0f - a) + foreground * a,
                                                  dst[out + 0], dst[out + 1], dst[out + 2]);
                    }
                    dst[out + 3] = 255;
                }
            }
        }
    }

    return result;
}

}
