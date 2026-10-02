#include "font_loader.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <limits>

namespace ascii {

struct FontInfoImpl {
    stbtt_fontinfo info;
};

static bool is_safe_font_path(const std::string& path) {
    if (path.empty()) return false;
    if (path.find('\0') != std::string::npos) return false;
    
    size_t max_len = 4096;
    if (path.size() > max_len) return false;
    
    return true;
}

static bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

std::string FontLoader::find_system_monospace_font() {
    static const char* linux_fonts[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/liberation-mono.ttf",
        "/usr/local/share/fonts/DejaVuSansMono.ttf",
        nullptr
    };
    
    static const char* macos_fonts[] = {
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Courier.dfont",
        nullptr
    };
    
    static const char* windows_fonts[] = {
        "C:\\Windows\\Fonts\\consola.ttf",
        "C:\\Windows\\Fonts\\cour.ttf",
        "C:\\Windows\\Fonts\\lucon.ttf",
        nullptr
    };
    
    const char* home = std::getenv("HOME");
    std::string home_font;
    if (home) {
        home_font = std::string(home) + "/.local/share/fonts/DejaVuSansMono.ttf";
        if (file_exists(home_font)) return home_font;
        home_font = std::string(home) + "/.fonts/DejaVuSansMono.ttf";
        if (file_exists(home_font)) return home_font;
    }
    
    for (const char** paths = linux_fonts; *paths; ++paths) {
        if (file_exists(*paths)) return *paths;
    }
    
    for (const char** paths = macos_fonts; *paths; ++paths) {
        if (file_exists(*paths)) return *paths;
    }
    
    for (const char** paths = windows_fonts; *paths; ++paths) {
        if (file_exists(*paths)) return *paths;
    }
    
    const char* xdg_data = std::getenv("XDG_DATA_HOME");
    if (xdg_data) {
        std::string path = std::string(xdg_data) + "/fonts/DejaVuSansMono.ttf";
        if (file_exists(path)) return path;
    }
    
    return "";
}

float GlyphBitmap::brightness() const {
    if (empty()) return 0.0f;
    float sum = 0.0f;
    for (uint8_t p : pixels) {
        sum += p / 255.0f;
    }
    return sum / pixels.size();
}

FontLoader::FontLoader() : font_info_(std::make_unique<FontInfoImpl>()) {}
FontLoader::~FontLoader() = default;

Result FontLoader::load(const std::string& path, float pixel_height) {
    if (!is_safe_font_path(path)) {
        return ascii::Result::fail(ascii::ErrorCode::INVALID_ARGUMENT, "Invalid or unsafe font path");
    }
    
    if (!validate_font_file(path)) {
        return ascii::Result::fail(ascii::ErrorCode::FONT_ERROR, "Invalid font file: " + path);
    }
    
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return ascii::Result::fail(ascii::ErrorCode::FILE_NOT_FOUND, "Cannot open font file: " + path);
    }
    
    size_t fsize = file.tellg();
    file.seekg(0);
    
    if (fsize == 0) {
        return ascii::Result::fail(ascii::ErrorCode::INVALID_FORMAT, "Font file is empty: " + path);
    }
    
    std::vector<uint8_t> data(fsize);
    if (!file.read(reinterpret_cast<char*>(data.data()), fsize)) {
        return ascii::Result::fail(ascii::ErrorCode::FILE_NOT_FOUND, "Failed to read font file: " + path);
    }
    
    return load_from_memory(data.data(), fsize, pixel_height);
}

Result FontLoader::load_from_memory(const uint8_t* data, size_t size, float pixel_height) {
    if (!std::isfinite(pixel_height) || pixel_height <= 0.0f || pixel_height > 1024.0f) {
        return Result::fail(ErrorCode::INVALID_ARGUMENT, "Font pixel height must be finite, > 0, and <= 1024");
    }
    if (!validate_font_data(data, size)) {
        return ascii::Result::fail(ascii::ErrorCode::FONT_ERROR, "Invalid font data");
    }
    
    std::vector<uint8_t> font_data(data, data + size);
    auto font_info = std::make_unique<FontInfoImpl>();
    const int offset = stbtt_GetFontOffsetForIndex(font_data.data(), 0);
    if (offset < 0 || !stbtt_InitFont(&font_info->info, font_data.data(), offset)) {
        return ascii::Result::fail(ascii::ErrorCode::FONT_ERROR, "Failed to initialize font");
    }
    
    const float scale = stbtt_ScaleForPixelHeight(&font_info->info, pixel_height);
    
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&font_info->info, &ascent, &descent, &line_gap);
    const double line_height = (ascent - descent + line_gap) * scale;

    int advance, lsb;
    stbtt_GetCodepointHMetrics(&font_info->info, 'M', &advance, &lsb);
    const double max_advance = advance * scale;
    if (!std::isfinite(scale) || scale <= 0.0f ||
        !std::isfinite(line_height) || line_height < 0.0 || line_height > std::numeric_limits<int>::max() ||
        !std::isfinite(max_advance) || max_advance < 0.0 || max_advance > std::numeric_limits<int>::max()) {
        return Result::fail(ErrorCode::FONT_ERROR, "Font metrics exceed the supported rendering range");
    }

    font_data_ = std::move(font_data);
    font_info_ = std::move(font_info);
    pixel_height_ = pixel_height;
    scale_ = scale;
    ascent_ = ascent;
    descent_ = descent;
    line_gap_ = line_gap;
    line_height_ = static_cast<int>(line_height);
    max_advance_ = static_cast<int>(max_advance);
    cache_.clear();
    
    loaded_ = true;
    return ascii::Result::ok();
}

GlyphBitmap FontLoader::render_glyph(uint32_t codepoint) const {
    if (!loaded_) return GlyphBitmap();
    
    auto it = cache_.find(codepoint);
    if (it != cache_.end()) return it->second;
    
    int advance, lsb;
    stbtt_GetCodepointHMetrics(&font_info_->info, codepoint, &advance, &lsb);
    
    int x0, y0, x1, y1;
    stbtt_GetCodepointBitmapBox(&font_info_->info, codepoint, scale_, scale_, &x0, &y0, &x1, &y1);
    
    int w = x1 - x0;
    int h = y1 - y0;
    if (w < 0 || h < 0 || static_cast<uint64_t>(w) * static_cast<uint64_t>(h) > 1000000ull) {
        return GlyphBitmap();
    }
    
    GlyphBitmap bitmap;
    bitmap.width = w;
    bitmap.height = h;
    bitmap.advance = static_cast<int>(advance * scale_);
    bitmap.bearing_x = x0;
    bitmap.bearing_y = y0;
    bitmap.pixels.resize(checked_image_size(w, h), 0);
    
    stbtt_MakeCodepointBitmap(&font_info_->info, bitmap.pixels.data(), w, h, w, scale_, scale_, codepoint);
    
    cache_[codepoint] = bitmap;
    return bitmap;
}

bool FontLoader::has_glyph(uint32_t codepoint) const {
    if (!loaded_) return false;
    return stbtt_FindGlyphIndex(&font_info_->info, codepoint) != 0;
}

bool FontLoader::validate_font_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    if (size == 0 || size > 10 * 1024 * 1024) {
        return false;
    }
    
    file.seekg(0);
    uint8_t bytes[4];
    if (!file.read(reinterpret_cast<char*>(bytes), 4)) {
        return false;
    }
    
    uint32_t signature = (static_cast<uint32_t>(bytes[0]) << 24) |
                         (static_cast<uint32_t>(bytes[1]) << 16) |
                         (static_cast<uint32_t>(bytes[2]) << 8) |
                         static_cast<uint32_t>(bytes[3]);
    
    return (signature == 0x00010000) ||
           (signature == 0x74727565) ||
           (signature == 0x4F54544F) ||
           (signature == 0x74746366);  // TrueType Collection (ttcf), used by macOS Menlo.
}

bool FontLoader::validate_font_data(const uint8_t* data, size_t size) {
    if (!data || size < 12) return false;
    
    if (size > 10 * 1024 * 1024) return false;
    
    uint32_t signature = (static_cast<uint32_t>(data[0]) << 24) |
                         (static_cast<uint32_t>(data[1]) << 16) |
                         (static_cast<uint32_t>(data[2]) << 8) |
                         static_cast<uint32_t>(data[3]);
    
    return (signature == 0x00010000) ||
           (signature == 0x74727565) ||
           (signature == 0x4F54544F) ||
           (signature == 0x74746366);  // TrueType Collection (ttcf), used by macOS Menlo.
}

Result FontLoader::load_system_fallback(float pixel_height) {
    std::string font_path = find_system_monospace_font();
    
    if (font_path.empty()) {
        return Result::fail(ErrorCode::FONT_ERROR, "No system monospace font found");
    }
    
    return load(font_path, pixel_height);
}

}
