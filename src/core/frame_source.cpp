#include "frame_source.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCAL
#include "stb_image.h"

#ifdef ASCII_USE_OPENCV
#include <opencv2/opencv.hpp>
#else
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

namespace ascii {

#ifdef ASCII_USE_OPENCV
void FrameSource::convert_mat_to_framebuffer(const cv::Mat& mat, FrameBuffer& out) {
    if (mat.empty()) return;

    cv::Mat rgb_mat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGRA2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_GRAY2RGB);
    } else {
        return;
    }

    if (rgb_mat.empty()) return;

    int w = rgb_mat.cols;
    int h = rgb_mat.rows;

    if (out.width() != w || out.height() != h) {
        out = FrameBuffer(w, h);
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3b pixel = rgb_mat.at<cv::Vec3b>(y, x);
            out.set_pixel(x, y, Color(pixel[0], pixel[1], pixel[2], 255));
        }
    }
}
#endif

namespace {

std::string wildcard_to_regex(const std::string& pattern) {
    std::string regex = "^";
    for (char c : pattern) {
        switch (c) {
            case '*': regex += ".*"; break;
            case '?': regex += "."; break;
            case '.': regex += "\\."; break;
            case '\\': regex += "\\\\"; break;
            case '+': case '^': case '$': case '(': case ')':
            case '[': case ']': case '{': case '}': case '|':
                regex += '\\';
                regex += c;
                break;
            default:
                regex += c;
                break;
        }
    }
    regex += "$";
    return regex;
}

std::string to_lower_copy(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::vector<std::string> split(const std::string& s, char delim) {
    std::vector<std::string> parts;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, delim)) {
        parts.push_back(part);
    }
    return parts;
}

bool is_numeric(const std::string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

bool check_image_extension(const std::string& path) {
    std::string lower = to_lower_copy(path);
    return lower.ends_with(".png") || lower.ends_with(".jpg") ||
           lower.ends_with(".jpeg") || lower.ends_with(".bmp") ||
           lower.ends_with(".gif") || lower.ends_with(".tiff") ||
           lower.ends_with(".webp");
}

#ifndef ASCII_USE_OPENCV
AVCodecID image_codec_from_extension(const std::string& path) {
    std::string lower = to_lower_copy(path);
    if (lower.ends_with(".png")) return AV_CODEC_ID_PNG;
    if (lower.ends_with(".jpg") || lower.ends_with(".jpeg")) return AV_CODEC_ID_MJPEG;
    if (lower.ends_with(".bmp")) return AV_CODEC_ID_BMP;
    if (lower.ends_with(".gif")) return AV_CODEC_ID_GIF;
    if (lower.ends_with(".tiff")) return AV_CODEC_ID_TIFF;
    if (lower.ends_with(".webp")) return AV_CODEC_ID_WEBP;
    return AV_CODEC_ID_NONE;
}
#endif

#ifndef ASCII_USE_OPENCV
struct FFmpegDecoder {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    const AVCodec* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* rgb_frame = nullptr;
    SwsContext* sws_ctx = nullptr;
    int stream_idx = -1;
    bool eof = false;
    std::vector<uint8_t> rgb_buffer;

    void close() {
        if (sws_ctx) sws_freeContext(sws_ctx);
        if (rgb_frame) av_frame_free(&rgb_frame);
        if (frame) av_frame_free(&frame);
        if (packet) av_packet_free(&packet);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        if (format_ctx) avformat_close_input(&format_ctx);

        sws_ctx = nullptr;
        rgb_frame = nullptr;
        frame = nullptr;
        packet = nullptr;
        codec_ctx = nullptr;
        format_ctx = nullptr;
        stream_idx = -1;
        eof = false;
        rgb_buffer.clear();
    }
};

bool ensure_rgb_pipeline(FFmpegDecoder& dec, int width, int height, AVPixelFormat src_fmt) {
    if (width <= 0 || height <= 0 || !dec.rgb_frame) {
        return false;
    }

    if (dec.sws_ctx) {
        sws_freeContext(dec.sws_ctx);
        dec.sws_ctx = nullptr;
    }

    int rgb_size = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
    if (rgb_size <= 0) {
        return false;
    }

    dec.rgb_buffer.resize(static_cast<size_t>(rgb_size));
    if (av_image_fill_arrays(dec.rgb_frame->data, dec.rgb_frame->linesize, dec.rgb_buffer.data(),
                             AV_PIX_FMT_RGB24, width, height, 1) < 0) {
        dec.rgb_buffer.clear();
        return false;
    }

    dec.sws_ctx = sws_getContext(width, height, src_fmt,
                                 width, height, AV_PIX_FMT_RGB24,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!dec.sws_ctx) {
        dec.rgb_buffer.clear();
        return false;
    }

    return true;
}

bool init_video_decoder(const std::string& uri, FFmpegDecoder& dec, Size& size, double& fps) {
    dec.close();

    const AVInputFormat* input_fmt = nullptr;
    if (check_image_extension(uri)) {
        input_fmt = av_find_input_format("image2");
    }

    AVDictionary* open_opts = nullptr;
    av_dict_set(&open_opts, "probesize", "5000000", 0);
    av_dict_set(&open_opts, "analyzeduration", "5000000", 0);
    int open_ret = avformat_open_input(&dec.format_ctx, uri.c_str(), input_fmt, &open_opts);
    av_dict_free(&open_opts);
    if (open_ret < 0) {
        return false;
    }
    avformat_find_stream_info(dec.format_ctx, nullptr);

    dec.stream_idx = av_find_best_stream(dec.format_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (dec.stream_idx < 0) {
        for (unsigned int i = 0; i < dec.format_ctx->nb_streams; ++i) {
            AVStream* s = dec.format_ctx->streams[i];
            if (s && s->codecpar && s->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                dec.stream_idx = static_cast<int>(i);
                break;
            }
        }
    }
    if (dec.stream_idx < 0 && check_image_extension(uri) && dec.format_ctx->nb_streams > 0) {
        dec.stream_idx = 0;
    }
    if (dec.stream_idx < 0) {
        dec.close();
        return false;
    }

    AVStream* stream = dec.format_ctx->streams[dec.stream_idx];
    AVCodecID codec_id = stream->codecpar ? stream->codecpar->codec_id : AV_CODEC_ID_NONE;
    if (codec_id == AV_CODEC_ID_NONE && check_image_extension(uri)) {
        codec_id = image_codec_from_extension(uri);
    }
    dec.codec = avcodec_find_decoder(codec_id);
    if (!dec.codec) {
        dec.close();
        return false;
    }

    dec.codec_ctx = avcodec_alloc_context3(dec.codec);
    if (!dec.codec_ctx) {
        dec.close();
        return false;
    }
    if (avcodec_parameters_to_context(dec.codec_ctx, stream->codecpar) < 0) {
        dec.close();
        return false;
    }
    if (dec.codec_ctx->codec_id == AV_CODEC_ID_NONE && codec_id != AV_CODEC_ID_NONE) {
        dec.codec_ctx->codec_id = codec_id;
    }
    if (avcodec_open2(dec.codec_ctx, dec.codec, nullptr) < 0) {
        dec.close();
        return false;
    }

    int stream_w = stream->codecpar ? stream->codecpar->width : 0;
    int stream_h = stream->codecpar ? stream->codecpar->height : 0;
    size.width = std::max(dec.codec_ctx->width, stream_w);
    size.height = std::max(dec.codec_ctx->height, stream_h);

    AVRational fr = stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate;
    if (fr.num > 0 && fr.den > 0) {
        fps = av_q2d(fr);
    } else {
        fps = 30.0;
    }
    if (fps <= 0.0) fps = 30.0;

    dec.packet = av_packet_alloc();
    dec.frame = av_frame_alloc();
    dec.rgb_frame = av_frame_alloc();
    if (!dec.packet || !dec.frame || !dec.rgb_frame) {
        dec.close();
        return false;
    }

    if (size.width > 0 && size.height > 0) {
        if (!ensure_rgb_pipeline(dec, size.width, size.height, dec.codec_ctx->pix_fmt)) {
            dec.close();
            return false;
        }
    }

    dec.eof = false;
    return true;
}

void copy_rgb_frame_to_buffer(const AVFrame* rgb, int width, int height, FrameBuffer& out) {
    if (out.width() != width || out.height() != height) {
        out = FrameBuffer(width, height);
    }

    for (int y = 0; y < height; ++y) {
        const uint8_t* row = rgb->data[0] + static_cast<size_t>(y) * rgb->linesize[0];
        for (int x = 0; x < width; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * 3;
            out.set_pixel(x, y, Color(px[0], px[1], px[2], 255));
        }
    }
}

bool decode_next_frame(FFmpegDecoder& dec, Size& size, FrameBuffer& out) {
    while (true) {
        int recv = avcodec_receive_frame(dec.codec_ctx, dec.frame);
        if (recv == 0) {
            int frame_w = dec.frame->width;
            int frame_h = dec.frame->height;
            AVPixelFormat src_fmt = static_cast<AVPixelFormat>(dec.frame->format);
            if (frame_w <= 0 || frame_h <= 0) {
                av_frame_unref(dec.frame);
                return false;
            }
            if (!dec.sws_ctx || size.width != frame_w || size.height != frame_h) {
                if (!ensure_rgb_pipeline(dec, frame_w, frame_h, src_fmt)) {
                    av_frame_unref(dec.frame);
                    return false;
                }
            }
            size.width = frame_w;
            size.height = frame_h;
            sws_scale(dec.sws_ctx,
                      dec.frame->data, dec.frame->linesize,
                      0, frame_h,
                      dec.rgb_frame->data, dec.rgb_frame->linesize);
            copy_rgb_frame_to_buffer(dec.rgb_frame, frame_w, frame_h, out);
            av_frame_unref(dec.frame);
            return true;
        }

        if (recv != AVERROR(EAGAIN) && recv != AVERROR_EOF) {
            return false;
        }
        if (dec.eof && recv == AVERROR_EOF) {
            return false;
        }

        bool fed_decoder = false;
        while (!fed_decoder) {
            int read_ret = av_read_frame(dec.format_ctx, dec.packet);
            if (read_ret < 0) {
                dec.eof = true;
                int flush_ret = avcodec_send_packet(dec.codec_ctx, nullptr);
                if (flush_ret < 0 && flush_ret != AVERROR_EOF) {
                    return false;
                }
                fed_decoder = true;
                continue;
            }

            if (dec.packet->stream_index == dec.stream_idx) {
                int send_ret = avcodec_send_packet(dec.codec_ctx, dec.packet);
                av_packet_unref(dec.packet);
                if (send_ret < 0 && send_ret != AVERROR(EAGAIN)) {
                    return false;
                }
                fed_decoder = true;
            } else {
                av_packet_unref(dec.packet);
            }
        }
    }
}

bool decode_first_frame(const std::string& uri, FrameBuffer& out, Size& size) {
    FFmpegDecoder dec;
    double fps = 30.0;
    if (!init_video_decoder(uri, dec, size, fps)) {
        return false;
    }
    bool ok = decode_next_frame(dec, size, out);
    dec.close();
    return ok;
}

bool decode_image_file_direct(const std::string& uri, FrameBuffer& out, Size& size) {
    int w = 0, h = 0, channels = 0;
    unsigned char* data = stbi_load(uri.c_str(), &w, &h, &channels, 3);  // Force RGB
    if (!data) {
        return false;
    }

    if (w <= 0 || h <= 0) {
        stbi_image_free(data);
        return false;
    }

    size.width = w;
    size.height = h;
    out = FrameBuffer(w, h);

    for (int y = 0; y < h; ++y) {
        const unsigned char* row = data + static_cast<size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            const unsigned char* px = row + static_cast<size_t>(x) * 3;
            out.set_pixel(x, y, Color(px[0], px[1], px[2], 255));
        }
    }

    stbi_image_free(data);
    return true;
}
#endif

}  // namespace

#ifndef ASCII_USE_OPENCV
struct VideoFileSource::Impl {
    FFmpegDecoder decoder;
    bool opened = false;
};
#endif

VideoFileSource::VideoFileSource() {
#ifndef ASCII_USE_OPENCV
    impl_ = std::make_unique<Impl>();
#endif
}

VideoFileSource::~VideoFileSource() {
#ifndef ASCII_USE_OPENCV
    if (impl_) {
        impl_->decoder.close();
        impl_->opened = false;
    }
#endif
}

bool VideoFileSource::open(const std::string& uri) {
#ifdef ASCII_USE_OPENCV
    cap_.open(uri);
    if (!cap_.isOpened()) return false;

    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0) fps_ = 30.0;

    size_.width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    size_.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    return true;
#else
    if (!impl_) impl_ = std::make_unique<Impl>();

    if (!init_video_decoder(uri, impl_->decoder, size_, fps_)) {
        impl_->opened = false;
        return false;
    }
    impl_->opened = true;
    return true;
#endif
}

bool VideoFileSource::read(FrameBuffer& out) {
#ifdef ASCII_USE_OPENCV
    cv::Mat frame;
    if (!cap_.read(frame)) return false;
    convert_mat_to_framebuffer(frame, out);
    return true;
#else
    if (!impl_ || !impl_->opened) return false;
    return decode_next_frame(impl_->decoder, size_, out);
#endif
}

double VideoFileSource::fps() const { return fps_; }
Size VideoFileSource::frame_size() const { return size_; }

bool VideoFileSource::is_open() const {
#ifdef ASCII_USE_OPENCV
    return cap_.isOpened();
#else
    return impl_ && impl_->opened;
#endif
}

void VideoFileSource::reset() {
#ifdef ASCII_USE_OPENCV
    cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
#else
    if (!impl_ || !impl_->opened) return;
    av_seek_frame(impl_->decoder.format_ctx, impl_->decoder.stream_idx, 0, AVSEEK_FLAG_BACKWARD);
    avcodec_flush_buffers(impl_->decoder.codec_ctx);
    impl_->decoder.eof = false;
#endif
}

WebcamSource::WebcamSource(int index) : index_(index) {}
WebcamSource::~WebcamSource() = default;

bool WebcamSource::open(const std::string& uri) {
    int idx = index_;
    try {
        idx = std::stoi(uri);
    } catch (...) {
        if (uri == "webcam" || uri.empty()) idx = 0;
    }

#ifdef ASCII_USE_OPENCV
    cap_.open(idx);
    if (!cap_.isOpened()) return false;

    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0) fps_ = 30.0;

    size_.width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    size_.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    index_ = idx;
    return true;
#else
    (void)idx;
    opened_ = false;
    size_ = {};
    fps_ = 30.0;
    return false;
#endif
}

bool WebcamSource::read(FrameBuffer& out) {
#ifdef ASCII_USE_OPENCV
    cv::Mat frame;
    if (!cap_.read(frame)) return false;
    convert_mat_to_framebuffer(frame, out);
    return true;
#else
    (void)out;
    return false;
#endif
}

double WebcamSource::fps() const { return fps_; }
Size WebcamSource::frame_size() const { return size_; }

bool WebcamSource::is_open() const {
#ifdef ASCII_USE_OPENCV
    return cap_.isOpened();
#else
    return opened_;
#endif
}

void WebcamSource::reset() {}

ImageSource::ImageSource() = default;
ImageSource::~ImageSource() = default;

bool ImageSource::open(const std::string& uri) {
#ifdef ASCII_USE_OPENCV
    image_ = cv::imread(uri, cv::IMREAD_COLOR);
    if (image_.empty()) return false;
    size_.width = image_.cols;
    size_.height = image_.rows;
    sent_ = false;
    return true;
#else
    size_ = {};
    image_buffer_ = FrameBuffer();
    loaded_ = false;
    if (check_image_extension(uri)) {
        loaded_ = decode_image_file_direct(uri, image_buffer_, size_);
    }
    if (!loaded_) {
        loaded_ = decode_first_frame(uri, image_buffer_, size_);
    }
    sent_ = false;
    return loaded_;
#endif
}

bool ImageSource::read(FrameBuffer& out) {
#ifdef ASCII_USE_OPENCV
    if (sent_ || image_.empty()) return false;
    convert_mat_to_framebuffer(image_, out);
    sent_ = true;
    return true;
#else
    if (sent_ || !loaded_) return false;
    out = image_buffer_;
    sent_ = true;
    return true;
#endif
}

double ImageSource::fps() const { return 0.0; }
Size ImageSource::frame_size() const { return size_; }

bool ImageSource::is_open() const {
#ifdef ASCII_USE_OPENCV
    return !image_.empty();
#else
    return loaded_;
#endif
}

void ImageSource::reset() {
    sent_ = false;
}

ImageSequenceSource::ImageSequenceSource() = default;
ImageSequenceSource::~ImageSequenceSource() = default;

bool ImageSequenceSource::open(const std::string& uri) {
    files_.clear();
    current_index_ = 0;
    size_ = {};

    if (uri.empty()) return false;

    namespace fs = std::filesystem;
    fs::path path(uri);
    fs::path directory = path.has_parent_path() ? path.parent_path() : fs::path(".");
    std::string pattern = path.filename().string();

    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        return false;
    }

    bool has_wildcard = pattern.find('*') != std::string::npos || pattern.find('?') != std::string::npos;
    std::regex matcher;
    if (has_wildcard) {
        matcher = std::regex(wildcard_to_regex(pattern), std::regex::icase);
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (!has_wildcard || std::regex_match(name, matcher)) {
            files_.push_back(entry.path().string());
        }
    }

    std::sort(files_.begin(), files_.end());
    if (files_.empty()) return false;

#ifdef ASCII_USE_OPENCV
    cv::Mat first = cv::imread(files_[0], cv::IMREAD_COLOR);
    if (first.empty()) {
        files_.clear();
        return false;
    }
    size_.width = first.cols;
    size_.height = first.rows;
#else
    FrameBuffer first;
    bool ok = decode_image_file_direct(files_[0], first, size_);
    if (!ok) {
        ok = decode_first_frame(files_[0], first, size_);
    }
    if (!ok) {
            files_.clear();
            return false;
    }
#endif
    return true;
}

bool ImageSequenceSource::read(FrameBuffer& out) {
    while (current_index_ < files_.size()) {
#ifdef ASCII_USE_OPENCV
        cv::Mat image = cv::imread(files_[current_index_], cv::IMREAD_COLOR);
        ++current_index_;
        if (image.empty()) continue;
        convert_mat_to_framebuffer(image, out);
        return true;
#else
        Size decoded_size{};
        bool ok = decode_image_file_direct(files_[current_index_], out, decoded_size);
        if (!ok) ok = decode_first_frame(files_[current_index_], out, decoded_size);
        ++current_index_;
        if (!ok) continue;
        if (size_.width == 0 || size_.height == 0) {
            size_ = decoded_size;
        }
        return true;
#endif
    }
    return false;
}

double ImageSequenceSource::fps() const { return fps_; }
Size ImageSequenceSource::frame_size() const { return size_; }
bool ImageSequenceSource::is_open() const { return !files_.empty(); }

void ImageSequenceSource::reset() {
    current_index_ = 0;
}

PipeSource::PipeSource() = default;
PipeSource::~PipeSource() = default;

bool PipeSource::open(const std::string& uri) {
    opened_ = false;
    width_ = 0;
    height_ = 0;
    channels_ = 3;
    fps_ = 30.0;

    if (uri.rfind("pipe:", 0) != 0) return false;

    std::string spec = uri.substr(5);
    auto parts = split(spec, ':');
    if (parts.empty()) return false;

    std::string size_part = parts[0];
    size_t x_pos = size_part.find('x');
    if (x_pos == std::string::npos) return false;

    try {
        width_ = std::stoi(size_part.substr(0, x_pos));
        height_ = std::stoi(size_part.substr(x_pos + 1));
    } catch (...) {
        return false;
    }

    if (width_ <= 0 || height_ <= 0) return false;

    if (parts.size() >= 2) {
        std::string fmt = to_lower_copy(parts[1]);
        if (fmt == "rgb") {
            channels_ = 3;
        } else if (fmt == "rgba") {
            channels_ = 4;
        } else {
            return false;
        }
    }

    if (parts.size() >= 3) {
        try {
            fps_ = std::stod(parts[2]);
        } catch (...) {
            return false;
        }
        if (fps_ <= 0.0) return false;
    }

    opened_ = true;
    return true;
}

bool PipeSource::read(FrameBuffer& out) {
    if (!opened_) return false;

    size_t frame_bytes = static_cast<size_t>(width_) * height_ * channels_;
    std::vector<uint8_t> buffer(frame_bytes);
    std::cin.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(frame_bytes));
    if (static_cast<size_t>(std::cin.gcount()) != frame_bytes) {
        return false;
    }

    if (out.width() != width_ || out.height() != height_) {
        out = FrameBuffer(width_, height_);
    }

    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            size_t idx = static_cast<size_t>(y * width_ + x) * channels_;
            if (channels_ == 3) {
                out.set_pixel(x, y, Color(buffer[idx], buffer[idx + 1], buffer[idx + 2], 255));
            } else {
                out.set_pixel(x, y, Color(buffer[idx], buffer[idx + 1], buffer[idx + 2], buffer[idx + 3]));
            }
        }
    }

    return true;
}

double PipeSource::fps() const { return fps_; }
Size PipeSource::frame_size() const { return {width_, height_}; }
bool PipeSource::is_open() const { return opened_; }

void PipeSource::reset() {}

std::unique_ptr<FrameSource> create_source(const std::string& uri) {
    if (uri.rfind("pipe:", 0) == 0) {
        return std::make_unique<PipeSource>();
    }

    if (uri.find('*') != std::string::npos || uri.find('?') != std::string::npos) {
        return std::make_unique<ImageSequenceSource>();
    }

    if (uri == "webcam" || uri.find("/dev/video") == 0 || is_numeric(uri)) {
        return std::make_unique<WebcamSource>();
    }

    if (check_image_extension(uri)) {
        return std::make_unique<ImageSource>();
    }

#ifdef ASCII_USE_OPENCV
    cv::VideoCapture test(uri);
    bool is_video = test.isOpened();
    test.release();
    if (is_video) {
        return std::make_unique<VideoFileSource>();
    }

    cv::Mat img = cv::imread(uri);
    if (!img.empty()) {
        img.release();
        return std::make_unique<ImageSource>();
    }
#endif

    return std::make_unique<VideoFileSource>();
}

}  // namespace ascii
