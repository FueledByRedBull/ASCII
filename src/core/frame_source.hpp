#pragma once

#include "core/types.hpp"
#include <string>
#include <memory>
#include <vector>
#include <cstddef>
#ifdef ASCII_USE_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace ascii {

enum class FrameReadStatus {
    Frame,
    End,
    Error,
    Paused
};

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual bool open(const std::string& uri) = 0;
    virtual FrameReadStatus read_next(FrameBuffer& out) = 0;
    bool read(FrameBuffer& out) { return read_next(out) == FrameReadStatus::Frame; }
    virtual double fps() const = 0;
    virtual Size frame_size() const = 0;
    virtual bool is_open() const = 0;
    virtual void reset() = 0;
    // Set before open; callers may lower the default 100-megapixel decode limit.
    void set_pixel_limit(uint64_t pixels) { pixel_limit_ = std::min<uint64_t>(pixels, 100000000ull); }

protected:
    uint64_t pixel_limit_ = 100000000ull;
#ifdef ASCII_USE_OPENCV
    bool convert_mat_to_framebuffer(const cv::Mat& mat, FrameBuffer& out);
#endif
};

class VideoFileSource : public FrameSource {
public:
    VideoFileSource();
    ~VideoFileSource() override;
    
    bool open(const std::string& uri) override;
    FrameReadStatus read_next(FrameBuffer& out) override;
    double fps() const override;
    Size frame_size() const override;
    bool is_open() const override;
    void reset() override;
    
private:
#ifdef ASCII_USE_OPENCV
    cv::VideoCapture cap_;
#else
    struct Impl;
    std::unique_ptr<Impl> impl_;
#endif
    double fps_ = 30.0;
    Size size_;
};

class WebcamSource : public FrameSource {
public:
    WebcamSource(int index = 0);
    ~WebcamSource() override;
    
    bool open(const std::string& uri) override;
    FrameReadStatus read_next(FrameBuffer& out) override;
    double fps() const override;
    Size frame_size() const override;
    bool is_open() const override;
    void reset() override;
    
private:
#ifdef ASCII_USE_OPENCV
    cv::VideoCapture cap_;
#else
    bool opened_ = false;
#endif
    int index_;
    double fps_ = 30.0;
    Size size_;
};

class ImageSource : public FrameSource {
public:
    ImageSource();
    ~ImageSource() override;
    
    bool open(const std::string& uri) override;
    FrameReadStatus read_next(FrameBuffer& out) override;
    double fps() const override;
    Size frame_size() const override;
    bool is_open() const override;
    void reset() override;
    
private:
#ifdef ASCII_USE_OPENCV
    cv::Mat image_;
#else
    FrameBuffer image_buffer_;
    bool loaded_ = false;
#endif
    Size size_;
    bool sent_ = false;
};

class ImageSequenceSource : public FrameSource {
public:
    ImageSequenceSource();
    ~ImageSequenceSource() override;
    
    bool open(const std::string& uri) override;
    FrameReadStatus read_next(FrameBuffer& out) override;
    double fps() const override;
    Size frame_size() const override;
    bool is_open() const override;
    void reset() override;
    const std::vector<std::string>& input_files() const { return files_; }
    
private:
    std::vector<std::string> files_;
    std::size_t current_index_ = 0;
    Size size_;
    double fps_ = 30.0;
};

class PipeSource : public FrameSource {
public:
    PipeSource();
    ~PipeSource() override;
    
    bool open(const std::string& uri) override;
    FrameReadStatus read_next(FrameBuffer& out) override;
    double fps() const override;
    Size frame_size() const override;
    bool is_open() const override;
    void reset() override;
    
private:
    void restore_stdin_mode();
    int original_stdin_mode_ = -1;
    bool opened_ = false;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 3;
    double fps_ = 30.0;
};

std::unique_ptr<FrameSource> create_source(const std::string& uri);
FrameReadStatus read_frame_if_ready(FrameSource& source, bool paused, FrameBuffer& out);

}
