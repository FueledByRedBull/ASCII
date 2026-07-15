#pragma once

#include "core/types.hpp"
#include <string>
#include <memory>

extern "C" {
struct AVFormatContext;
struct AVCodecContext;
struct AVStream;
struct AVFrame;
struct AVPacket;
struct SwsContext;
}

namespace ascii {

class VideoEncoder {
public:
    struct Config {
        int width = 640;
        int height = 480;
        int fps = 30;
        int bitrate = 2000000;
        std::string codec = "libx264";
        std::string preset = "medium";
    };
    
    VideoEncoder();
    ~VideoEncoder();
    
    bool open(const std::string& filename, const Config& config);
    bool close();
    bool write_frame(const FrameBuffer& frame);
    bool is_open() const { return format_ctx_ != nullptr; }
    bool last_frame_was_written() const { return last_frame_written_; }
    bool is_still_image_target() const { return output_is_still_image_; }
    const std::string& last_error() const { return last_error_; }
    
private:
    bool init_codec();
    bool write_header();
    bool write_trailer();
    void set_error(const std::string& message);
    
    Config config_;
    AVFormatContext* format_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int64_t pts_ = 0;
    std::string output_filename_;
    std::string temporary_filename_;
    bool output_is_gif_ = false;
    bool output_is_still_image_ = false;
    bool wrote_still_image_frame_ = false;
    bool last_frame_written_ = false;
    bool header_written_ = false;
    bool failed_ = false;
    int source_width_ = 0;
    int source_height_ = 0;
    std::string last_error_;
};

}
