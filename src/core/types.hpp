#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>

namespace ascii {

enum class ErrorCode {
    SUCCESS = 0,
    FILE_NOT_FOUND,
    INVALID_FORMAT,
    MEMORY_ERROR,
    PROCESSING_ERROR,
    FONT_ERROR,
    INVALID_ARGUMENT,
    DEVICE_ERROR
};

struct Result {
    ErrorCode error = ErrorCode::SUCCESS;
    std::string message;
    
    bool success() const { return error == ErrorCode::SUCCESS; }
    bool failure() const { return error != ErrorCode::SUCCESS; }
    
    static Result ok() { return {ErrorCode::SUCCESS, ""}; }
    static Result fail(ErrorCode code, const std::string& msg) { return {code, msg}; }
};

struct Size {
    int width = 0;
    int height = 0;
    
    bool operator==(const Size& other) const {
        return width == other.width && height == other.height;
    }
    bool operator!=(const Size& other) const { return !(*this == other); }
    int area() const { return width * height; }
};

struct Color {
    uint8_t r = 0, g = 0, b = 0, a = 255;
    
    Color() = default;
    Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) : r(r), g(g), b(b), a(a) {}
    
    static Color from_float(float rf, float gf, float bf, float af = 1.0f) {
        return Color(
            static_cast<uint8_t>(std::clamp(rf * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(gf * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(bf * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(af * 255.0f, 0.0f, 255.0f))
        );
    }
    
    float luminance() const {
        return (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255.0f;
    }
};

class FrameBuffer {
public:
    FrameBuffer() = default;
    FrameBuffer(int w, int h) : width_(w), height_(h), data_(w * h * 4, 0) {}
    FrameBuffer(int w, int h, const Color& fill) : width_(w), height_(h), data_(w * h * 4) {
        for (int i = 0; i < w * h; ++i) {
            set_pixel(i % w, i / w, fill);
        }
    }
    
    int width() const { return width_; }
    int height() const { return height_; }
    Size size() const { return {width_, height_}; }
    bool empty() const { return width_ == 0 || height_ == 0; }
    size_t byte_size() const { return data_.size(); }
    const uint8_t* data() const { return data_.data(); }
    uint8_t* data() { return data_.data(); }
    
    Color get_pixel(int x, int y) const {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return Color();
        const size_t idx = (y * width_ + x) * 4;
        return Color(data_[idx], data_[idx+1], data_[idx+2], data_[idx+3]);
    }
    
    void set_pixel(int x, int y, const Color& c) {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return;
        const size_t idx = (y * width_ + x) * 4;
        data_[idx] = c.r;
        data_[idx+1] = c.g;
        data_[idx+2] = c.b;
        data_[idx+3] = c.a;
    }
    
    void fill(const Color& c) {
        for (int y = 0; y < height_; ++y) {
            for (int x = 0; x < width_; ++x) {
                set_pixel(x, y, c);
            }
        }
    }
    
    void clear() {
        std::fill(data_.begin(), data_.end(), 0);
    }
    
private:
    int width_ = 0;
    int height_ = 0;
    std::vector<uint8_t> data_;
};

class FloatImage {
public:
    FloatImage() = default;
    FloatImage(int w, int h) : width_(w), height_(h), data_(w * h, 0.0f) {}
    FloatImage(int w, int h, float fill) : width_(w), height_(h), data_(w * h, fill) {}
    
    int width() const { return width_; }
    int height() const { return height_; }
    Size size() const { return {width_, height_}; }
    bool empty() const { return width_ == 0 || height_ == 0; }
    size_t size_in_elements() const { return data_.size(); }
    const float* data() const { return data_.data(); }
    float* data() { return data_.data(); }
    
    float get(int x, int y) const {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return 0.0f;
        return data_[y * width_ + x];
    }
    
    void set(int x, int y, float v) {
        if (x < 0 || x >= width_ || y < 0 || y >= height_) return;
        data_[y * width_ + x] = v;
    }
    
    float get_clamped(int x, int y) const {
        x = std::clamp(x, 0, width_ - 1);
        y = std::clamp(y, 0, height_ - 1);
        return data_[y * width_ + x];
    }
    
    void fill(float v) {
        std::fill(data_.begin(), data_.end(), v);
    }
    
    void clear() {
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    static FloatImage from_rgba(const uint8_t* rgba, int w, int h) {
        FloatImage img(w, h);
        for (int i = 0; i < w * h; ++i) {
            float r = rgba[i * 4] / 255.0f;
            float g = rgba[i * 4 + 1] / 255.0f;
            float b = rgba[i * 4 + 2] / 255.0f;
            img.data_[i] = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        }
        return img;
    }
    
private:
    int width_ = 0;
    int height_ = 0;
    std::vector<float> data_;
};

struct GradientData {
    FloatImage magnitude;
    FloatImage orientation;
    FloatImage gx;
    FloatImage gy;
    
    bool empty() const { return magnitude.empty(); }
    Size size() const { return magnitude.size(); }
};

struct EdgeData {
    FloatImage magnitude;
    FloatImage orientation;
    std::vector<bool> edge_mask;
    
    bool empty() const { return magnitude.empty(); }
    Size size() const { return magnitude.size(); }
    
    bool is_edge(int x, int y) const {
        if (x < 0 || x >= magnitude.width() || y < 0 || y >= magnitude.height()) return false;
        return edge_mask[y * magnitude.width() + x];
    }
};

struct CellStats {
    float mean_luminance = 0.0f;
    float luminance_variance = 0.0f;
    float local_contrast = 0.0f;
    float mean_gx = 0.0f;
    float mean_gy = 0.0f;
    float cell_orientation = 0.0f;
    float edge_strength = 0.0f;
    float edge_strength_max = 0.0f;
    float edge_occupancy = 0.0f;
    bool is_edge_cell = false;
    
    float structure_coherence = 0.0f;
    float dominant_orientation = 0.0f;
    float orientation_histogram[8] = {0};
    float frequency_signature[8] = {0};
    float texture_signature[8] = {0};
    int adaptive_level = 0;
    
    float mean_r = 0.0f;
    float mean_g = 0.0f;
    float mean_b = 0.0f;
};

}
