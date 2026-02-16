#include "video_encoder.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace ascii {

VideoEncoder::VideoEncoder() = default;

VideoEncoder::~VideoEncoder() {
    close();
}

bool VideoEncoder::open(const std::string& filename, const Config& config) {
    config_ = config;
    
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
}

bool VideoEncoder::write_frame(const FrameBuffer& frame) {
    if (!is_open() || !frame_) return false;
    
    if (!sws_ctx_) {
        sws_ctx_ = sws_getContext(
            frame.width(), frame.height(), AV_PIX_FMT_RGBA,
            config_.width, config_.height, AV_PIX_FMT_YUV420P,
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
    const AVCodec* codec = avcodec_find_encoder_by_name(config_.codec.c_str());
    if (!codec) {
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    }
    if (!codec) return false;
    
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) return false;
    
    codec_ctx_->width = config_.width;
    codec_ctx_->height = config_.height;
    codec_ctx_->time_base = {1, config_.fps};
    codec_ctx_->framerate = {config_.fps, 1};
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx_->bit_rate = config_.bitrate;
    codec_ctx_->gop_size = config_.fps;
    
    if (format_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    
    if (config_.codec == "libx264") {
        av_opt_set(codec_ctx_->priv_data, "preset", config_.preset.c_str(), 0);
    }
    
    int ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) return false;
    
    stream_ = avformat_new_stream(format_ctx_, nullptr);
    if (!stream_) return false;
    
    stream_->time_base = codec_ctx_->time_base;
    ret = avcodec_parameters_from_context(stream_->codecpar, codec_ctx_);
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
