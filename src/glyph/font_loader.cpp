#include "font_loader.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#include <cmath>
#include <fstream>
#include <algorithm>
#include <cstdlib>

namespace ascii {

struct FontInfoImpl {
    stbtt_fontinfo info;
};

static bool has_path_traversal(const std::string& path) {
    if (path.find("..") != std::string::npos) return true;
    if (path.find('\0') != std::string::npos) return true;
    return false;
}

static bool is_safe_font_path(const std::string& path) {
    if (path.empty()) return false;
    if (has_path_traversal(path)) return false;
    
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

std::vector<float> GlyphBitmap::orientation_histogram(int bins) const {
    std::vector<float> hist(bins, 0.0f);
    if (width < 3 || height < 3) return hist;
    constexpr float kPi = 3.14159265358979323846f;
    
    std::vector<float> blurred(pixels.size());
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    sum += pixels[(y + dy) * width + (x + dx)] / 255.0f;
                }
            }
            blurred[y * width + x] = sum / 9.0f;
        }
    }
    
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            float gx = -blurred[(y-1)*width + (x-1)] + blurred[(y-1)*width + (x+1)]
                     - 2*blurred[y*width + (x-1)] + 2*blurred[y*width + (x+1)]
                     - blurred[(y+1)*width + (x-1)] + blurred[(y+1)*width + (x+1)];
            float gy = -blurred[(y-1)*width + (x-1)] - 2*blurred[(y-1)*width + x] - blurred[(y-1)*width + (x+1)]
                     + blurred[(y+1)*width + (x-1)] + 2*blurred[(y+1)*width + x] + blurred[(y+1)*width + (x+1)];
            
            float mag = std::sqrt(gx*gx + gy*gy);
            if (mag > 0.05f) {
                float angle = std::atan2(gy, gx);
                float normalized = (angle + kPi) / (2.0f * kPi);
                int bin = static_cast<int>(normalized * bins) % bins;
                hist[bin] += mag;
            }
        }
    }
    
    float total = 0.0f;
    for (float v : hist) total += v;
    if (total > 0.0f) {
        for (float& v : hist) v /= total;
    }
    
    return hist;
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
    
    font_data_.resize(fsize);
    if (!file.read(reinterpret_cast<char*>(font_data_.data()), fsize)) {
        return ascii::Result::fail(ascii::ErrorCode::FILE_NOT_FOUND, "Failed to read font file: " + path);
    }
    
    return load_from_memory(font_data_.data(), fsize, pixel_height);
}

Result FontLoader::load_from_memory(const uint8_t* data, size_t size, float pixel_height) {
    if (!validate_font_data(data, size)) {
        return ascii::Result::fail(ascii::ErrorCode::FONT_ERROR, "Invalid font data");
    }
    
    if (!stbtt_InitFont(&font_info_->info, data, stbtt_GetFontOffsetForIndex(data, 0))) {
        return ascii::Result::fail(ascii::ErrorCode::FONT_ERROR, "Failed to initialize font");
    }
    
    pixel_height_ = pixel_height;
    scale_ = stbtt_ScaleForPixelHeight(&font_info_->info, pixel_height);
    
    int ascent, descent, line_gap;
    stbtt_GetFontVMetrics(&font_info_->info, &ascent, &descent, &line_gap);
    ascent_ = ascent;
    descent_ = descent;
    line_gap_ = line_gap;
    line_height_ = static_cast<int>((ascent - descent + line_gap) * scale_);
    
    int advance, lsb;
    stbtt_GetCodepointHMetrics(&font_info_->info, 'M', &advance, &lsb);
    max_advance_ = static_cast<int>(advance * scale_);
    
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
    
    GlyphBitmap bitmap;
    bitmap.width = w;
    bitmap.height = h;
    bitmap.advance = static_cast<int>(advance * scale_);
    bitmap.bearing_x = x0;
    bitmap.bearing_y = y0;
    bitmap.pixels.resize(w * h, 0);
    
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
           (signature == 0x4F54544F);
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
           (signature == 0x4F54544F);
}

Result FontLoader::load_system_fallback(float pixel_height) {
    std::string font_path = find_system_monospace_font();
    
    if (font_path.empty()) {
        return Result::fail(ErrorCode::FONT_ERROR, "No system monospace font found");
    }
    
    return load(font_path, pixel_height);
}

}
