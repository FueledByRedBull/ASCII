#include "video_encoder.hpp"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdio>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/version.h>
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

bool is_still_image_output(const std::string& filename) {
    return ends_with_ci(filename, ".png") ||
           ends_with_ci(filename, ".jpg") ||
           ends_with_ci(filename, ".jpeg") ||
           ends_with_ci(filename, ".bmp") ||
           ends_with_ci(filename, ".webp") ||
           ends_with_ci(filename, ".tif") ||
           ends_with_ci(filename, ".tiff");
}

AVCodecID preferred_still_codec_id(const std::string& filename) {
    if (ends_with_ci(filename, ".png")) return AV_CODEC_ID_PNG;
    if (ends_with_ci(filename, ".jpg") || ends_with_ci(filename, ".jpeg")) return AV_CODEC_ID_MJPEG;
    if (ends_with_ci(filename, ".bmp")) return AV_CODEC_ID_BMP;
    if (ends_with_ci(filename, ".webp")) return AV_CODEC_ID_WEBP;
    if (ends_with_ci(filename, ".tif") || ends_with_ci(filename, ".tiff")) return AV_CODEC_ID_TIFF;
    return AV_CODEC_ID_NONE;
}

AVPixelFormat choose_pixel_format(const AVCodec* codec, bool gif_output, bool still_image_output) {
    const AVPixelFormat fallback = gif_output
        ? AV_PIX_FMT_RGB8
        : (still_image_output ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_YUV420P);
    if (!codec) {
        return fallback;
    }

    int num_formats = 0;
#if LIBAVUTIL_VERSION_MAJOR >= 59
    const void* raw_formats = nullptr;
    const int ret = avcodec_get_supported_config(
        nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT, 0, &raw_formats, &num_formats);
    if (ret < 0 || !raw_formats || num_formats <= 0) {
        return fallback;
    }

    const auto* pix_fmts = static_cast<const AVPixelFormat*>(raw_formats);
#else
    const AVPixelFormat* pix_fmts = codec->pix_fmts;
    if (!pix_fmts) {
        return fallback;
    }
    while (pix_fmts[num_formats] != AV_PIX_FMT_NONE) {
        ++num_formats;
    }
    if (num_formats <= 0) {
        return fallback;
    }
#endif
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
    } else if (still_image_output) {
        if (codec->id == AV_CODEC_ID_MJPEG) {
            const AVPixelFormat preferred_jpeg[] = {
                AV_PIX_FMT_YUVJ444P,
                AV_PIX_FMT_YUVJ422P,
                AV_PIX_FMT_YUVJ420P,
                AV_PIX_FMT_YUV444P,
                AV_PIX_FMT_YUV422P,
                AV_PIX_FMT_YUV420P
            };
            for (AVPixelFormat pf : preferred_jpeg) {
                if (has_format(pf)) {
                    return pf;
                }
            }
        } else {
            const AVPixelFormat preferred[] = {
                AV_PIX_FMT_RGBA,
                AV_PIX_FMT_RGB24,
                AV_PIX_FMT_BGRA,
                AV_PIX_FMT_BGR24,
                AV_PIX_FMT_YUV420P
            };
            for (AVPixelFormat pf : preferred) {
                if (has_format(pf)) {
                    return pf;
                }
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

std::string ffmpeg_error_string(int errnum) {
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    if (av_strerror(errnum, buf, sizeof(buf)) == 0) {
        return std::string(buf);
    }
    return "unknown ffmpeg error";
}

}  // namespace

VideoEncoder::VideoEncoder() = default;

VideoEncoder::~VideoEncoder() {
    close();
}

bool VideoEncoder::open(const std::string& filename, const Config& config) {
    close();
    last_error_.clear();
    config_ = config;
    output_filename_ = filename;
    output_is_gif_ = ends_with_ci(filename, ".gif");
    output_is_still_image_ = is_still_image_output(filename);
    wrote_still_image_frame_ = false;
    
    int ret = avformat_alloc_output_context2(&format_ctx_, nullptr, nullptr, filename.c_str());
    if (ret < 0 || !format_ctx_) {
        if (ret < 0) {
            set_error("Failed to allocate output context: " + ffmpeg_error_string(ret));
        } else {
            set_error("Failed to allocate output context");
        }
        return false;
    }
    
    if (!init_codec()) {
        close();
        return false;
    }
    
    if (!(format_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&format_ctx_->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            set_error("Failed to open output file: " + ffmpeg_error_string(ret));
            close();
            return false;
        }
    }
    
    if (!write_header()) {
        close();
        return false;
    }
    return true;
}

bool VideoEncoder::close() {
    bool ok = true;
    if (format_ctx_) {
        if (!write_trailer()) {
            ok = false;
        }
        
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
    output_filename_.clear();
    output_is_gif_ = false;
    output_is_still_image_ = false;
    wrote_still_image_frame_ = false;
    return ok;
}

bool VideoEncoder::write_frame(const FrameBuffer& frame) {
    last_frame_written_ = false;
    if (!is_open() || !frame_) {
        set_error("Encoder is not open");
        return false;
    }
    if (output_is_still_image_ && wrote_still_image_frame_) {
        return true;
    }

    if (av_frame_make_writable(frame_) < 0) {
        set_error("Failed to make output frame writable");
        return false;
    }
    
    if (!sws_ctx_) {
        sws_ctx_ = sws_getContext(
            frame.width(), frame.height(), AV_PIX_FMT_RGBA,
            config_.width, config_.height, codec_ctx_->pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr
        );
        if (!sws_ctx_) {
            set_error("Failed to create scaling context");
            return false;
        }
    }
    
    const uint8_t* src_data[1] = { frame.data() };
    int src_linesize[1] = { frame.width() * 4 };
    
    sws_scale(sws_ctx_, src_data, src_linesize, 0, frame.height(),
              frame_->data, frame_->linesize);
    
    frame_->pts = pts_++;
    
    int ret = avcodec_send_frame(codec_ctx_, frame_);
    if (ret < 0) {
        set_error("Failed to send frame to encoder: " + ffmpeg_error_string(ret));
        return false;
    }
    
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx_, pkt_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) {
            set_error("Failed to receive encoded packet: " + ffmpeg_error_string(ret));
            return false;
        }
        
        av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
        pkt_->stream_index = stream_->index;
        
        ret = av_interleaved_write_frame(format_ctx_, pkt_);
        if (ret < 0) {
            set_error("Failed to write encoded packet: " + ffmpeg_error_string(ret));
            return false;
        }
    }

    if (output_is_still_image_) {
        wrote_still_image_frame_ = true;
    }
    last_frame_written_ = true;
    
    return true;
}

bool VideoEncoder::init_codec() {
    std::vector<const AVCodec*> candidates;
    if (output_is_gif_) {
        push_codec_candidate(candidates, avcodec_find_encoder(AV_CODEC_ID_GIF));
    } else if (output_is_still_image_) {
        const AVCodecID preferred_id = preferred_still_codec_id(output_filename_);
        if (preferred_id != AV_CODEC_ID_NONE) {
            push_codec_candidate(candidates, avcodec_find_encoder(preferred_id));
            // For explicit still-image extensions (e.g. .png), avoid falling back
            // to unrelated codecs such as MJPEG which can trigger range/pixfmt warnings.
            if (candidates.empty()) {
                set_error("No encoder available for requested still-image extension");
                return false;
            }
        } else if (candidates.empty()) {
            const AVCodecID guessed = av_guess_codec(
                format_ctx_->oformat, nullptr, output_filename_.c_str(), nullptr, AVMEDIA_TYPE_VIDEO);
            if (guessed != AV_CODEC_ID_NONE) {
                push_codec_candidate(candidates, avcodec_find_encoder(guessed));
            }
        }
        if (candidates.empty()) {
            const AVCodecID muxer_default = format_ctx_->oformat->video_codec;
            if (muxer_default != AV_CODEC_ID_NONE) {
                push_codec_candidate(candidates, avcodec_find_encoder(muxer_default));
            }
        }
    } else {
        push_codec_candidate(candidates, avcodec_find_encoder_by_name(config_.codec.c_str()));
        push_codec_candidate(candidates, avcodec_find_encoder_by_name("libx264"));
        push_codec_candidate(candidates, avcodec_find_encoder_by_name("libopenh264"));
        push_codec_candidate(candidates, avcodec_find_encoder_by_name("mpeg4"));
        push_codec_candidate(candidates, avcodec_find_encoder(AV_CODEC_ID_MPEG4));
        push_codec_candidate(candidates, avcodec_find_encoder(AV_CODEC_ID_H264));
    }
    if (candidates.empty()) {
        set_error("No encoder candidates available for output format");
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
        codec_ctx_->pix_fmt = choose_pixel_format(codec, output_is_gif_, output_is_still_image_);
        codec_ctx_->gop_size = std::max(1, config_.fps);
        if (!output_is_gif_ && !output_is_still_image_) {
            codec_ctx_->bit_rate = config_.bitrate;
        }
        if (codec->id == AV_CODEC_ID_MJPEG) {
            codec_ctx_->color_range = AVCOL_RANGE_JPEG;
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
        set_error("Failed to open encoder '" + std::string(codec->name ? codec->name : "unknown") +
                  "': " + ffmpeg_error_string(ret));
        avcodec_free_context(&codec_ctx_);
    }
    if (!opened_codec || !codec_ctx_) {
        if (last_error_.empty()) {
            set_error("Failed to open any compatible encoder");
        }
        return false;
    }
    
    stream_ = avformat_new_stream(format_ctx_, nullptr);
    if (!stream_) {
        set_error("Failed to create output stream");
        return false;
    }
    
    stream_->time_base = codec_ctx_->time_base;
    int ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
    if (ret < 0) {
        set_error("Failed to copy codec parameters: " + ffmpeg_error_string(ret));
        return false;
    }
    
    frame_ = av_frame_alloc();
    if (!frame_) {
        set_error("Failed to allocate output frame");
        return false;
    }
    
    frame_->format = codec_ctx_->pix_fmt;
    frame_->width = codec_ctx_->width;
    frame_->height = codec_ctx_->height;
    
    ret = av_frame_get_buffer(frame_, 0);
    if (ret < 0) {
        set_error("Failed to allocate frame buffer: " + ffmpeg_error_string(ret));
        return false;
    }
    
    pkt_ = av_packet_alloc();
    if (!pkt_) {
        set_error("Failed to allocate packet");
        return false;
    }
    
    return true;
}

bool VideoEncoder::write_header() {
    AVDictionary* options = nullptr;
    if (output_is_still_image_) {
        av_dict_set(&options, "update", "1", 0);
    }
    const int ret = avformat_write_header(format_ctx_, options ? &options : nullptr);
    av_dict_free(&options);
    if (ret < 0) {
        set_error("Failed to write output header: " + ffmpeg_error_string(ret));
        return false;
    }
    return true;
}

bool VideoEncoder::write_trailer() {
    if (!format_ctx_) return false;
    
    if (codec_ctx_) {
        avcodec_send_frame(codec_ctx_, nullptr);
        
        while (true) {
            int ret = avcodec_receive_packet(codec_ctx_, pkt_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                set_error("Failed to flush encoder packet: " + ffmpeg_error_string(ret));
                return false;
            }
            
            av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
            pkt_->stream_index = stream_->index;
            ret = av_interleaved_write_frame(format_ctx_, pkt_);
            if (ret < 0) {
                set_error("Failed to write trailer packet: " + ffmpeg_error_string(ret));
                return false;
            }
        }
    }
    
    const int ret = av_write_trailer(format_ctx_);
    if (ret < 0) {
        set_error("Failed to finalize output file: " + ffmpeg_error_string(ret));
        return false;
    }
    return true;
}

void VideoEncoder::set_error(const std::string& message) {
    if (!message.empty()) {
        last_error_ = message;
    }
}

}
