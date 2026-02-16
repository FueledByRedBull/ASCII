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
    void close();
    bool write_frame(const FrameBuffer& frame);
    bool is_open() const { return format_ctx_ != nullptr; }
    
private:
    bool init_codec();
    bool write_header();
    bool write_trailer();
    
    Config config_;
    AVFormatContext* format_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVStream* stream_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* pkt_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int64_t pts_ = 0;
    bool output_is_gif_ = false;
};

}
