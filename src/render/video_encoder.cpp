#include "video_encoder.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libswscale/swscale.h>
}

namespace ascii {

namespace {

bool ends_with_ci(const std::string& value, const std::string& suffix) {
    if (suffix.size() > value.size()) {
        return false;
    }
    size_t offset = value.size() - suffix.size();
    for (size_t i = 0; i < suffix.size(); ++i) {
        unsigned char a = static_cast<unsigned char>(value[offset + i]);
        unsigned char b = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(a) != std::tolower(b)) {
            return false;
        }
    }
    return true;
}

AVPixelFormat choose_pixel_format(const AVCodec* codec, bool gif_output) {
    const AVPixelFormat fallback = gif_output ? AV_PIX_FMT_RGB8 : AV_PIX_FMT_YUV420P;
    if (!codec) {
        return fallback;
    }

    const void* raw_formats = nullptr;
    int num_formats = 0;
    const int ret = avcodec_get_supported_config(
        nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &raw_formats, &num_formats);
    if (ret < 0 || !raw_formats || num_formats <= 0) {
        return fallback;
    }

    const auto* pix_fmts = static_cast<const AVPixelFormat*>(raw_formats);
    auto has_format = [&](AVPixelFormat fmt) -> bool {
        for (int i = 0; i < num_formats; ++i) {
            if (pix_fmts[i] == fmt) {
                return true;
            }
        }
        return false;
    };

    if (gif_output) {
        const AVPixelFormat preferred[] = {
            AV_PIX_FMT_RGB8,
            AV_PIX_FMT_BGR8,
            AV_PIX_FMT_PAL8
        };
        for (AVPixelFormat pf : preferred) {
            if (has_format(pf)) {
                return pf;
            }
        }
    } else if (has_format(AV_PIX_FMT_YUV420P)) {
        return AV_PIX_FMT_YUV420P;
    }

    return pix_fmts[0];
}

bool contains_ci(const char* s, const char* needle) {
    if (!s || !needle) {
        return false;
    }
    std::string hay(s);
    std::string ndl(needle);
    std::transform(hay.begin(), hay.end(), hay.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(ndl.begin(), ndl.end(), ndl.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return hay.find(ndl) != std::string::npos;
}

void push_codec_candidate(std::vector<const AVCodec*>& out, const AVCodec* codec) {
    if (!codec) {
        return;
    }
    for (const AVCodec* existing : out) {
        if (existing && codec->name && existing->name &&
            std::strcmp(existing->name, codec->name) == 0) {
            return;
        }
    }
    out.push_back(codec);
}

}  // namespace

VideoEncoder::VideoEncoder() = default;

VideoEncoder::~VideoEncoder() {
    close();
}

bool VideoEncoder::open(const std::string& filename, const Config& config) {
    config_ = config;
    output_is_gif_ = ends_with_ci(filename, ".gif");
    
    int ret = avformat_alloc_output_context2(&format_ctx_, nullptr, nullptr, filename.c_str());
    if (ret < 0 || !format_ctx_) {
        return false;
    }
    
    if (!init_codec()) {
        close();
        return false;
    }
    
    if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&format_ctx_->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            close();
            return false;
        }
    }
    
    return write_header();
}

void VideoEncoder::close() {
    if (format_ctx_) {
        write_trailer();
        
        if (codec_ctx_) {
            avcodec_free_context(&codec_ctx_);
        }
        
        if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
            avio_closep(&format_ctx_->pb);
        }
        
        avformat_free_context(format_ctx_);
        format_ctx_ = nullptr;
    }
    
    if (frame_) {
        av_frame_free(&frame_);
    }
    
    if (pkt_) {
        av_packet_free(&pkt_);
    }
    
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    
    pts_ = 0;
    output_is_gif_ = false;
}

bool VideoEncoder::write_frame(const FrameBuffer& frame) {
    if (!is_open() || !frame_) return false;

    if (av_frame_make_writable(frame_) < 0) {
        return false;
    }
    
    if (!sws_ctx_) {
        sws_ctx_ = sws_getContext(
            frame.width(), frame.height(), AV_PIX_FMT_RGBA,
            config_.width, config_.height, codec_ctx_->pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        if (!sws_ctx_) return false;
    }
    
    const uint8_t* src_data[1] = { frame.data() };
    int src_linesize[1] = { frame.width() * 4 };
    
    sws_scale(sws_ctx_, src_data, src_linesize, 0, frame.height(),
              frame_->data, frame_->linesize);
    
    frame_->pts = pts_++;
    
    int ret = avcodec_send_frame(codec_ctx_, frame_);
    if (ret < 0) return false;
    
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx_, pkt_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) return false;
        
        av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
        pkt_->stream_index = stream_->index;
        
        ret = av_interleaved_write_frame(format_ctx_, pkt_);
        if (ret < 0) return false;
    }
    
    return true;
}

bool VideoEncoder::init_codec() {
    std::vector<const AVCodec*> candidates;
    if (output_is_gif_) {
        push_codec_candidate(candidates, avcodec_find_encoder(AV_CODEC_ID_GIF));
    } else {
        push_codec_candidate(candidates, avcodec_find_encoder_by_name(config_.codec.c_str()));
        push_codec_candidate(candidates, avcodec_find_encoder_by_name("libx264"));
        push_codec_candidate(candidates, avcodec_find_encoder_by_name("libopenh264"));
        push_codec_candidate(candidates, avcodec_find_encoder_by_name("mpeg4"));
        push_codec_candidate(candidates, avcodec_find_encoder(AV_CODEC_ID_MPEG4));
        push_codec_candidate(candidates, avcodec_find_encoder(AV_CODEC_ID_H264));
    }
    if (candidates.empty()) {
        return false;
    }

    const AVCodec* opened_codec = nullptr;
    for (const AVCodec* codec : candidates) {
        if (!codec) {
            continue;
        }
#ifdef _WIN32
        // `h264_mf` can fail under STA PowerShell with:
        // "COM must not be in STA mode".
        // Prefer software encoders for robust CLI behavior.
        if (contains_ci(codec->name, "_mf")) {
            continue;
        }
#endif

        codec_ctx_ = avcodec_alloc_context3(codec);
        if (!codec_ctx_) {
            continue;
        }

        codec_ctx_->width = config_.width;
        codec_ctx_->height = config_.height;
        codec_ctx_->time_base = {1, config_.fps};
        codec_ctx_->framerate = {config_.fps, 1};
        codec_ctx_->pix_fmt = choose_pixel_format(codec, output_is_gif_);
        codec_ctx_->gop_size = std::max(1, config_.fps);
        if (!output_is_gif_) {
            codec_ctx_->bit_rate = config_.bitrate;
        }

        if (format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
            codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        }

        if (!output_is_gif_ && codec->name && std::strcmp(codec->name, "libx264") == 0) {
            av_opt_set(codec_ctx_->priv_data, "preset", config_.preset.c_str(), 0);
        }

        int ret = avcodec_open2(codec_ctx_, codec, nullptr);
        if (ret >= 0) {
            opened_codec = codec;
            break;
        }
        avcodec_free_context(&codec_ctx_);
    }
    if (!opened_codec || !codec_ctx_) {
        return false;
    }
    
    stream_ = avformat_new_stream(format_ctx_, nullptr);
    if (!stream_) return false;
    
    stream_->time_base = codec_ctx_->time_base;
    int ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
    if (ret < 0) return false;
    
    frame_ = av_frame_alloc();
    if (!frame_) return false;
    
    frame_->format = codec_ctx_->pix_fmt;
    frame_->width = codec_ctx_->width;
    frame_->height = codec_ctx_->height;
    
    ret = av_frame_get_buffer(frame_, 0);
    if (ret < 0) return false;
    
    pkt_ = av_packet_alloc();
    if (!pkt_) return false;
    
    return true;
}

bool VideoEncoder::write_header() {
    return avformat_write_header(format_ctx_, nullptr) >= 0;
}

bool VideoEncoder::write_trailer() {
    if (!format_ctx_) return false;
    
    if (codec_ctx_) {
        avcodec_send_frame(codec_ctx_, nullptr);
        
        while (true) {
            int ret = avcodec_receive_packet(codec_ctx_, pkt_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) return false;
            
            av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
            pkt_->stream_index = stream_->index;
            ret = av_interleaved_write_frame(format_ctx_, pkt_);
            if (ret < 0) return false;
        }
    }
    
    return av_write_trailer(format_ctx_) >= 0;
}

}
