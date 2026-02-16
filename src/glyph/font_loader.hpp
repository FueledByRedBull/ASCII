#pragma once

#include "core/types.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace ascii {

struct FontInfoImpl;

struct GlyphBitmap {
    std::vector<uint8_t> pixels;
    int width = 0;
    int height = 0;
    int advance = 0;
    int bearing_x = 0;
    int bearing_y = 0;
    
    bool empty() const { return pixels.empty(); }
    float brightness() const;
    std::vector<float> orientation_histogram(int bins = 8) const;
};

class FontLoader {
public:
    FontLoader();
    ~FontLoader();
    
    Result load(const std::string& path, float pixel_height = 16.0f);
    Result load_from_memory(const uint8_t* data, size_t size, float pixel_height = 16.0f);
    Result load_system_fallback(float pixel_height = 16.0f);
    
    GlyphBitmap render_glyph(uint32_t codepoint) const;
    bool has_glyph(uint32_t codepoint) const;
    bool is_loaded() const { return loaded_; }
    
    int line_height() const { return line_height_; }
    int max_advance() const { return max_advance_; }
    float pixel_height() const { return pixel_height_; }
    
    static std::string find_system_monospace_font();
    
private:
    std::unique_ptr<FontInfoImpl> font_info_;
    std::vector<uint8_t> font_data_;
    float scale_ = 1.0f;
    float pixel_height_ = 16.0f;
    int line_height_ = 0;
    int max_advance_ = 0;
    int ascent_ = 0;
    int descent_ = 0;
    int line_gap_ = 0;
    bool loaded_ = false;
    
    mutable std::unordered_map<uint32_t, GlyphBitmap> cache_;
    
    bool validate_font_file(const std::string& path);
    bool validate_font_data(const uint8_t* data, size_t size);
};

}
