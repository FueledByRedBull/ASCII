#include "frame_source.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <regex>
#include <sstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef ASCII_USE_OPENCV
#include <opencv2/opencv.hpp>
#else
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/imgutils.h>
#include <libavutil/parseutils.h>
#include <libswscale/swscale.h>
}
#endif

namespace ascii {

#ifdef ASCII_USE_OPENCV
bool FrameSource::convert_mat_to_framebuffer(const cv::Mat& mat, FrameBuffer& out) {
    if (mat.empty() || static_cast<uint64_t>(mat.cols) * mat.rows > pixel_limit_) return false;

    cv::Mat rgb_mat;
    if (mat.channels() == 3) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGR2RGB);
    } else if (mat.channels() == 4) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_BGRA2RGB);
    } else if (mat.channels() == 1) {
        cv::cvtColor(mat, rgb_mat, cv::COLOR_GRAY2RGB);
    } else {
        return false;
    }

    if (rgb_mat.empty()) return false;

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
    return true;
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

bool is_gif_extension(const std::string& path) {
    std::string lower = to_lower_copy(path);
    return lower.ends_with(".gif");
}

bool should_force_image2(const std::string& path) {
    std::string lower = to_lower_copy(path);
    // Keep animated GIF on normal demux path, otherwise frames collapse to still-image behavior.
    return lower.ends_with(".png") || lower.ends_with(".jpg") ||
           lower.ends_with(".jpeg") || lower.ends_with(".bmp") ||
           lower.ends_with(".tiff") || lower.ends_with(".webp");
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

bool valid_source_dimensions(int width, int height, uint64_t pixel_limit, int channels = 4) {
    return width > 0 && height > 0 && channels > 0 &&
           static_cast<uint64_t>(width) * static_cast<uint64_t>(height) <= pixel_limit &&
           static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * channels <=
               std::numeric_limits<size_t>::max();
}

#ifndef ASCII_USE_OPENCV
struct FFmpegDecoder {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    const AVCodec* codec = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* rgba_frame = nullptr;
    SwsContext* sws_ctx = nullptr;
    int stream_idx = -1;
    bool eof = false;
    bool input_error = false;
    int64_t expected_frames = 0;
    int64_t decoded_frames = 0;
    uint64_t pixel_limit = 100000000ull;
    AVPixelFormat sws_format = AV_PIX_FMT_NONE;
    AVRational time_base{0, 1};
    int64_t expected_duration_us = 0;
    int64_t expected_end_us = AV_NOPTS_VALUE;
    int64_t decoded_end_us = AV_NOPTS_VALUE;
    bool decoded_end_known = false;

    ~FFmpegDecoder() { close(); }

    void close() {
        if (sws_ctx) sws_freeContext(sws_ctx);
        if (rgba_frame) av_frame_free(&rgba_frame);
        if (frame) av_frame_free(&frame);
        if (packet) av_packet_free(&packet);
        if (codec_ctx) avcodec_free_context(&codec_ctx);
        if (format_ctx) avformat_close_input(&format_ctx);

        sws_ctx = nullptr;
        rgba_frame = nullptr;
        frame = nullptr;
        packet = nullptr;
        codec_ctx = nullptr;
        format_ctx = nullptr;
        stream_idx = -1;
        eof = false;
        input_error = false;
        expected_frames = 0;
        decoded_frames = 0;
        sws_format = AV_PIX_FMT_NONE;
        time_base = {0, 1};
        expected_duration_us = 0;
        expected_end_us = AV_NOPTS_VALUE;
        decoded_end_us = AV_NOPTS_VALUE;
        decoded_end_known = false;
    }
};

bool ensure_rgba_pipeline(FFmpegDecoder& dec, int width, int height, AVPixelFormat src_fmt) {
    if (!valid_source_dimensions(width, height, dec.pixel_limit) || width > INT_MAX / 4 || !dec.rgba_frame) {
        return false;
    }

    if (dec.sws_ctx) {
        sws_freeContext(dec.sws_ctx);
        dec.sws_ctx = nullptr;
    }

    av_frame_unref(dec.rgba_frame);
    dec.rgba_frame->format = AV_PIX_FMT_RGBA;
    dec.rgba_frame->width = width;
    dec.rgba_frame->height = height;
    // swscale requires aligned rows and padding beyond the logical image.
    if (av_frame_get_buffer(dec.rgba_frame, 0) < 0) {
        return false;
    }

    dec.sws_ctx = sws_getContext(width, height, src_fmt,
                                 width, height, AV_PIX_FMT_RGBA,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
    if (!dec.sws_ctx) {
        av_frame_unref(dec.rgba_frame);
        return false;
    }

    dec.sws_format = src_fmt;
    return true;
}

bool init_video_decoder(const std::string& uri, FFmpegDecoder& dec, Size& size, double& fps,
                        uint64_t pixel_limit) {
    dec.close();
    dec.pixel_limit = pixel_limit;
    if (pixel_limit == 0) return false;

    const AVInputFormat* input_fmt = nullptr;
    if (should_force_image2(uri)) {
        input_fmt = av_find_input_format("image2");
    }

    AVDictionary* open_opts = nullptr;
    av_dict_set(&open_opts, "probesize", "5000000", 0);
    av_dict_set(&open_opts, "analyzeduration", "5000000", 0);
    int open_ret = avformat_open_input(&dec.format_ctx, uri.c_str(), input_fmt, &open_opts);
    av_dict_free(&open_opts);
    if (open_ret < 0) {
        dec.close();
        return false;
    }
    std::vector<AVDictionary*> probe_options(dec.format_ctx->nb_streams, nullptr);
    for (unsigned i = 0; i < dec.format_ctx->nb_streams; ++i) {
        const auto* parameters = dec.format_ctx->streams[i]->codecpar;
        if (parameters && parameters->codec_type == AVMEDIA_TYPE_VIDEO) {
            if (parameters->width > 0 && parameters->height > 0 &&
                !valid_source_dimensions(parameters->width, parameters->height, pixel_limit)) {
                for (auto& options : probe_options) av_dict_free(&options);
                dec.close();
                return false;
            }
            av_dict_set_int(&probe_options[i], "max_pixels", static_cast<int64_t>(pixel_limit), 0);
        }
    }
    const int info_result = avformat_find_stream_info(dec.format_ctx, probe_options.data());
    for (auto& options : probe_options) av_dict_free(&options);
    if (info_result < 0) {
        dec.close();
        return false;
    }

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
    if (!stream->codecpar ||
        !valid_source_dimensions(stream->codecpar->width, stream->codecpar->height, pixel_limit)) {
        dec.close();
        return false;
    }
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
    dec.codec_ctx->max_pixels = static_cast<int64_t>(pixel_limit);
    if (avcodec_open2(dec.codec_ctx, dec.codec, nullptr) < 0) {
        dec.close();
        return false;
    }

    int stream_w = stream->codecpar ? stream->codecpar->width : 0;
    int stream_h = stream->codecpar ? stream->codecpar->height : 0;
    size.width = std::max(dec.codec_ctx->width, stream_w);
    size.height = std::max(dec.codec_ctx->height, stream_h);
    if (!valid_source_dimensions(size.width, size.height, pixel_limit)) {
        dec.close();
        return false;
    }

    AVRational fr = stream->avg_frame_rate.num > 0 ? stream->avg_frame_rate : stream->r_frame_rate;
    if (fr.num > 0 && fr.den > 0) {
        fps = av_q2d(fr);
    } else {
        fps = 30.0;
    }
    if (!std::isfinite(fps) || fps <= 0.0) fps = 30.0;
    dec.expected_frames = std::max<int64_t>(0, stream->nb_frames);
    dec.time_base = stream->time_base;
    if (stream->duration > 0) {
        dec.expected_duration_us = av_rescale_q(stream->duration, stream->time_base, AV_TIME_BASE_Q);
        if (stream->start_time != AV_NOPTS_VALUE) {
            const int64_t start = av_rescale_q(stream->start_time, stream->time_base, AV_TIME_BASE_Q);
            if (dec.expected_duration_us > 0 && start <= INT64_MAX - dec.expected_duration_us) {
                dec.expected_end_us = start + dec.expected_duration_us;
            }
        }
    } else if (std::strstr(dec.format_ctx->iformat->name, "matroska")) {
        // Matroska's per-track DURATION tag records the ending timestamp, including any start offset.
        if (const auto* duration = av_dict_get(stream->metadata, "DURATION", nullptr, 0)) {
            int64_t end = 0;
            if (av_parse_time(&end, duration->value, 1) >= 0 && end > 0) dec.expected_end_us = end;
        }
    }

    dec.packet = av_packet_alloc();
    dec.frame = av_frame_alloc();
    dec.rgba_frame = av_frame_alloc();
    if (!dec.packet || !dec.frame || !dec.rgba_frame) {
        dec.close();
        return false;
    }

    dec.eof = false;
    return true;
}

void copy_rgba_frame_to_buffer(const AVFrame* rgba, int width, int height, FrameBuffer& out) {
    if (out.width() != width || out.height() != height) {
        out = FrameBuffer(width, height);
    }

    const size_t row_bytes = static_cast<size_t>(width) * 4;
    for (int y = 0; y < height; ++y) {
        const uint8_t* row = rgba->data[0] + static_cast<size_t>(y) * rgba->linesize[0];
        uint8_t* destination = out.data() + static_cast<size_t>(y) * row_bytes;
        std::memcpy(destination, row, row_bytes);
        // Decoded media has always been opaque, including sources with alpha.
        for (size_t x = 3; x < row_bytes; x += 4) destination[x] = 255;
    }
}

FrameReadStatus decode_next_frame(FFmpegDecoder& dec, Size& size, FrameBuffer& out) {
    while (true) {
        int recv = avcodec_receive_frame(dec.codec_ctx, dec.frame);
        if (recv == 0) {
            int frame_w = dec.frame->width;
            int frame_h = dec.frame->height;
            AVPixelFormat src_fmt = static_cast<AVPixelFormat>(dec.frame->format);
            if (!valid_source_dimensions(frame_w, frame_h, dec.pixel_limit)) {
                av_frame_unref(dec.frame);
                return FrameReadStatus::Error;
            }
            if (!dec.sws_ctx || size.width != frame_w || size.height != frame_h || dec.sws_format != src_fmt) {
                if (!ensure_rgba_pipeline(dec, frame_w, frame_h, src_fmt)) {
                    av_frame_unref(dec.frame);
                    return FrameReadStatus::Error;
                }
            }
            size.width = frame_w;
            size.height = frame_h;
            const int converted_rows = sws_scale(dec.sws_ctx,
                      dec.frame->data, dec.frame->linesize,
                      0, frame_h,
                      dec.rgba_frame->data, dec.rgba_frame->linesize);
            if (converted_rows != frame_h) {
                av_frame_unref(dec.frame);
                return FrameReadStatus::Error;
            }
            copy_rgba_frame_to_buffer(dec.rgba_frame, frame_w, frame_h, out);
            if (dec.frame->best_effort_timestamp != AV_NOPTS_VALUE) {
                const int64_t timestamp = av_rescale_q(dec.frame->best_effort_timestamp, dec.time_base, AV_TIME_BASE_Q);
                const int64_t duration = av_rescale_q(dec.frame->duration, dec.time_base, AV_TIME_BASE_Q);
                if (dec.decoded_frames == 0 && dec.expected_end_us == AV_NOPTS_VALUE &&
                    dec.expected_duration_us > 0 && timestamp <= INT64_MAX - dec.expected_duration_us) {
                    dec.expected_end_us = timestamp + dec.expected_duration_us;
                }
                const bool known_duration = duration > 0 && timestamp <= INT64_MAX - duration;
                const int64_t end = known_duration ? timestamp + duration : timestamp;
                if (dec.decoded_end_us == AV_NOPTS_VALUE || end >= dec.decoded_end_us) {
                    dec.decoded_end_us = end;
                    dec.decoded_end_known = known_duration;
                }
            }
            av_frame_unref(dec.frame);
            ++dec.decoded_frames;
            return FrameReadStatus::Frame;
        }

        if (recv != AVERROR(EAGAIN) && recv != AVERROR_EOF) {
            return FrameReadStatus::Error;
        }
        if (dec.eof && recv == AVERROR_EOF) {
            const int64_t tolerance = std::max<int64_t>(1, av_rescale_q(1, dec.time_base, AV_TIME_BASE_Q));
            const bool missing_frames = dec.expected_frames > 0 && dec.decoded_frames < dec.expected_frames;
            const bool missing_time = dec.decoded_end_known && dec.expected_end_us != AV_NOPTS_VALUE &&
                                      dec.decoded_end_us < dec.expected_end_us &&
                                      static_cast<uint64_t>(dec.expected_end_us) - static_cast<uint64_t>(dec.decoded_end_us) >
                                          static_cast<uint64_t>(tolerance);
            const bool incomplete = missing_frames || missing_time;
            return (dec.input_error || incomplete) ? FrameReadStatus::Error : FrameReadStatus::End;
        }

        bool fed_decoder = false;
        while (!fed_decoder) {
            int read_ret = av_read_frame(dec.format_ctx, dec.packet);
            if (read_ret < 0) {
                if (read_ret != AVERROR_EOF) {
                    dec.input_error = true;
                } else if (dec.format_ctx->pb && dec.format_ctx->pb->error < 0) {
                    dec.input_error = true;
                }
                dec.eof = true;
                int flush_ret = avcodec_send_packet(dec.codec_ctx, nullptr);
                if (flush_ret < 0 && flush_ret != AVERROR_EOF) {
                    return FrameReadStatus::Error;
                }
                fed_decoder = true;
                continue;
            }

            if (dec.packet->stream_index == dec.stream_idx) {
                int send_ret = avcodec_send_packet(dec.codec_ctx, dec.packet);
                av_packet_unref(dec.packet);
                if (send_ret < 0 && send_ret != AVERROR(EAGAIN)) {
                    return FrameReadStatus::Error;
                }
                fed_decoder = true;
            } else {
                av_packet_unref(dec.packet);
            }
        }
    }
}

bool decode_first_frame(const std::string& uri, FrameBuffer& out, Size& size, uint64_t pixel_limit) {
    FFmpegDecoder dec;
    double fps = 30.0;
    if (!init_video_decoder(uri, dec, size, fps, pixel_limit)) {
        return false;
    }
    bool ok = decode_next_frame(dec, size, out) == FrameReadStatus::Frame;
    dec.close();
    return ok;
}

bool decode_image_file_direct(const std::string& uri, FrameBuffer& out, Size& size, uint64_t pixel_limit) {
    int w = 0, h = 0, channels = 0;
    if (!stbi_info(uri.c_str(), &w, &h, &channels) || !valid_source_dimensions(w, h, pixel_limit)) return false;
    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> data(
        stbi_load(uri.c_str(), &w, &h, &channels, 3), stbi_image_free);
    if (!data) {
        return false;
    }

    if (!valid_source_dimensions(w, h, pixel_limit)) return false;

    size.width = w;
    size.height = h;
    out = FrameBuffer(w, h);

    for (int y = 0; y < h; ++y) {
        const unsigned char* row = data.get() + static_cast<size_t>(y) * w * 3;
        for (int x = 0; x < w; ++x) {
            const unsigned char* px = row + static_cast<size_t>(x) * 3;
            out.set_pixel(x, y, Color(px[0], px[1], px[2], 255));
        }
    }

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
    size_ = {};
    fps_ = 30.0;
#ifdef ASCII_USE_OPENCV
    cap_.open(uri);
    if (!cap_.isOpened()) return false;

    fps_ = cap_.get(cv::CAP_PROP_FPS);
    if (fps_ <= 0) fps_ = 30.0;

    size_.width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    size_.height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (!valid_source_dimensions(size_.width, size_.height, pixel_limit_)) {
        cap_.release();
        size_ = {};
        return false;
    }
    return true;
#else
    if (!impl_) impl_ = std::make_unique<Impl>();

    if (!init_video_decoder(uri, impl_->decoder, size_, fps_, pixel_limit_)) {
        impl_->opened = false;
        size_ = {};
        return false;
    }
    impl_->opened = true;
    return true;
#endif
}

FrameReadStatus VideoFileSource::read_next(FrameBuffer& out) {
#ifdef ASCII_USE_OPENCV
    cv::Mat frame;
    if (!cap_.read(frame)) return FrameReadStatus::End;
    return convert_mat_to_framebuffer(frame, out) ? FrameReadStatus::Frame : FrameReadStatus::Error;
#else
    if (!impl_ || !impl_->opened) return FrameReadStatus::Error;
    impl_->decoder.pixel_limit = pixel_limit_;
    impl_->decoder.codec_ctx->max_pixels = static_cast<int64_t>(pixel_limit_);
    const auto status = decode_next_frame(impl_->decoder, size_, out);
    if (status == FrameReadStatus::Error) {
        impl_->decoder.close();
        impl_->opened = false;
    }
    return status;
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
    if (av_seek_frame(impl_->decoder.format_ctx, impl_->decoder.stream_idx, 0, AVSEEK_FLAG_BACKWARD) < 0) {
        impl_->decoder.close();
        impl_->opened = false;
        return;
    }
    avcodec_flush_buffers(impl_->decoder.codec_ctx);
    impl_->decoder.eof = false;
    impl_->decoder.input_error = false;
    impl_->decoder.decoded_frames = 0;
    impl_->decoder.decoded_end_us = AV_NOPTS_VALUE;
    impl_->decoder.decoded_end_known = false;
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
    if (!valid_source_dimensions(size_.width, size_.height, pixel_limit_)) {
        cap_.release();
        size_ = {};
        return false;
    }
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

FrameReadStatus WebcamSource::read_next(FrameBuffer& out) {
#ifdef ASCII_USE_OPENCV
    cv::Mat frame;
    if (!cap_.read(frame)) return FrameReadStatus::Error;
    return convert_mat_to_framebuffer(frame, out) ? FrameReadStatus::Frame : FrameReadStatus::Error;
#else
    (void)out;
    return FrameReadStatus::Error;
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
    size_ = {};
    sent_ = false;
#ifdef ASCII_USE_OPENCV
    image_.release();
    int width = 0, height = 0, channels = 0;
    if (stbi_info(uri.c_str(), &width, &height, &channels) &&
        !valid_source_dimensions(width, height, pixel_limit_)) return false;
    image_ = cv::imread(uri, cv::IMREAD_COLOR);
    if (image_.empty()) return false;
    if (!valid_source_dimensions(image_.cols, image_.rows, pixel_limit_)) {
        image_.release();
        return false;
    }
    size_.width = image_.cols;
    size_.height = image_.rows;
    sent_ = false;
    return true;
#else
    size_ = {};
    image_buffer_ = FrameBuffer();
    loaded_ = false;
    if (check_image_extension(uri)) {
        loaded_ = decode_image_file_direct(uri, image_buffer_, size_, pixel_limit_);
    }
    if (!loaded_) {
        loaded_ = decode_first_frame(uri, image_buffer_, size_, pixel_limit_);
    }
    if (!loaded_) { size_ = {}; image_buffer_ = {}; }
    sent_ = false;
    return loaded_;
#endif
}

FrameReadStatus ImageSource::read_next(FrameBuffer& out) {
    if (!valid_source_dimensions(size_.width, size_.height, pixel_limit_)) return FrameReadStatus::Error;
#ifdef ASCII_USE_OPENCV
    if (sent_) return FrameReadStatus::End;
    if (image_.empty()) return FrameReadStatus::Error;
    if (!convert_mat_to_framebuffer(image_, out)) return FrameReadStatus::Error;
    sent_ = true;
    return FrameReadStatus::Frame;
#else
    if (sent_) return FrameReadStatus::End;
    if (!loaded_) return FrameReadStatus::Error;
    out = image_buffer_;
    sent_ = true;
    return FrameReadStatus::Frame;
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
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (!entry.is_regular_file()) continue;
            std::string name = entry.path().filename().string();
            if (std::regex_match(name, matcher)) files_.push_back(entry.path().string());
        }
    } else {
        if (!fs::is_regular_file(path)) return false;
        files_.push_back(path.string());
    }

    std::sort(files_.begin(), files_.end());
    if (files_.empty()) return false;

    ImageSource first;
    first.set_pixel_limit(pixel_limit_);
    if (!first.open(files_[0])) {
        files_.clear();
        return false;
    }
    size_ = first.frame_size();
    return true;
}

FrameReadStatus ImageSequenceSource::read_next(FrameBuffer& out) {
    if (current_index_ >= files_.size()) return FrameReadStatus::End;
    ImageSource image;
    image.set_pixel_limit(pixel_limit_);
    const auto& path = files_[current_index_++];
    if (!image.open(path) || image.read_next(out) != FrameReadStatus::Frame) return FrameReadStatus::Error;
    size_ = image.frame_size();
    return FrameReadStatus::Frame;
}

double ImageSequenceSource::fps() const { return fps_; }
Size ImageSequenceSource::frame_size() const { return size_; }
bool ImageSequenceSource::is_open() const { return !files_.empty(); }

void ImageSequenceSource::reset() {
    current_index_ = 0;
}

PipeSource::PipeSource() = default;
PipeSource::~PipeSource() { restore_stdin_mode(); }

void PipeSource::restore_stdin_mode() {
#ifdef _WIN32
    if (original_stdin_mode_ != -1) _setmode(_fileno(stdin), original_stdin_mode_);
#endif
    original_stdin_mode_ = -1;
}

bool PipeSource::open(const std::string& uri) {
    restore_stdin_mode();
    opened_ = false;
    width_ = 0;
    height_ = 0;
    channels_ = 3;
    fps_ = 30.0;

    if (uri.rfind("pipe:", 0) != 0) return false;

    std::string spec = uri.substr(5);
    auto parts = split(spec, ':');
    if (parts.empty() || parts.size() > 3 || spec.empty() || spec.back() == ':') return false;

    std::string size_part = parts[0];
    size_t x_pos = size_part.find('x');
    if (x_pos == std::string::npos) return false;

    const auto width = std::from_chars(size_part.data(), size_part.data() + x_pos, width_);
    const auto height = std::from_chars(size_part.data() + x_pos + 1, size_part.data() + size_part.size(), height_);
    if (width.ec != std::errc{} || width.ptr != size_part.data() + x_pos ||
        height.ec != std::errc{} || height.ptr != size_part.data() + size_part.size()) return false;

    if (!valid_source_dimensions(width_, height_, pixel_limit_, channels_)) return false;

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
        const auto rate = std::from_chars(parts[2].data(), parts[2].data() + parts[2].size(), fps_);
        if (rate.ec != std::errc{} || rate.ptr != parts[2].data() + parts[2].size()) return false;
        if (!std::isfinite(fps_) || fps_ <= 0.0 || fps_ > 120.0) return false;
    }

#ifdef _WIN32
    original_stdin_mode_ = _setmode(_fileno(stdin), _O_BINARY);
    if (original_stdin_mode_ == -1) return false;
#endif
    opened_ = true;
    return true;
}

FrameReadStatus PipeSource::read_next(FrameBuffer& out) {
    if (!opened_) return FrameReadStatus::Error;
    if (!valid_source_dimensions(width_, height_, pixel_limit_, channels_)) return FrameReadStatus::Error;

    size_t frame_bytes = static_cast<size_t>(width_) * height_ * channels_;
    std::vector<uint8_t> buffer(frame_bytes);
    std::cin.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(frame_bytes));
    if (static_cast<size_t>(std::cin.gcount()) != frame_bytes) {
        return std::cin.gcount() == 0 && std::cin.eof() && !std::cin.bad()
            ? FrameReadStatus::End : FrameReadStatus::Error;
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

    return FrameReadStatus::Frame;
}

double PipeSource::fps() const { return fps_; }
Size PipeSource::frame_size() const { return {width_, height_}; }
bool PipeSource::is_open() const { return opened_; }

void PipeSource::reset() {}

FrameReadStatus read_frame_if_ready(FrameSource& source, bool paused, FrameBuffer& out) {
    if (paused) return FrameReadStatus::Paused;
    return source.read_next(out);
}

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

    if (is_gif_extension(uri)) {
        // Animated GIF should be treated as video stream.
        return std::make_unique<VideoFileSource>();
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
