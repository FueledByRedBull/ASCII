#include "audio_player.hpp"
#include <SDL2/SDL.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
}

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstring>
#include <vector>

namespace ascii {

AudioPlayer::AudioPlayer() {
    SDL_Init(SDL_INIT_AUDIO);
}

AudioPlayer::~AudioPlayer() {
    close();
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

bool AudioPlayer::open(const std::string& filename) {
    close();
    
    if (!load_audio(filename)) {
        return false;
    }
    
    SDL_AudioSpec desired{};
    desired.freq = sample_rate_;
    desired.format = AUDIO_S16SYS;
    desired.channels = static_cast<Uint8>(channels_);
    desired.samples = 4096;
    desired.callback = audio_callback;
    desired.userdata = this;
    
    device_id_ = SDL_OpenAudioDevice(nullptr, 0, &desired, nullptr, 0);
    if (device_id_ <= 0) {
        close();
        return false;
    }
    
    return true;
}

bool AudioPlayer::load_audio(const std::string& filename) {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext* codec_ctx = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    SwrContext* swr_ctx = nullptr;
    std::vector<uint8_t> pcm_buffer;
    
    bool success = false;
    int stream_idx = -1;
    AVStream* audio_stream = nullptr;
    const AVCodec* codec = nullptr;
    int out_sample_rate = 48000;
    int in_sample_rate = 48000;
    int in_channels = 2;
    int out_channels = 2;
    AVChannelLayout out_ch_layout{};
    AVChannelLayout in_ch_layout{};
    
    if (avformat_open_input(&format_ctx, filename.c_str(), nullptr, nullptr) < 0) {
        goto cleanup;
    }
    
    if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
        goto cleanup;
    }
    
    stream_idx = av_find_best_stream(format_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (stream_idx < 0) {
        goto cleanup;
    }
    
    audio_stream = format_ctx->streams[stream_idx];
    codec = avcodec_find_decoder(audio_stream->codecpar->codec_id);
    if (!codec) {
        goto cleanup;
    }
    
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        goto cleanup;
    }
    
    if (avcodec_parameters_to_context(codec_ctx, audio_stream->codecpar) < 0) {
        goto cleanup;
    }
    
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        goto cleanup;
    }
    
    out_sample_rate = codec_ctx->sample_rate > 0 ? codec_ctx->sample_rate : 48000;
    in_sample_rate = codec_ctx->sample_rate > 0 ? codec_ctx->sample_rate : out_sample_rate;
    in_channels = codec_ctx->ch_layout.nb_channels > 0 ? codec_ctx->ch_layout.nb_channels : 2;
    out_channels = in_channels > 1 ? 2 : 1;

    if (codec_ctx->ch_layout.nb_channels > 0) {
        if (av_channel_layout_copy(&in_ch_layout, &codec_ctx->ch_layout) < 0) {
            goto cleanup;
        }
    } else {
        av_channel_layout_default(&in_ch_layout, in_channels);
    }
    av_channel_layout_default(&out_ch_layout, out_channels);

    if (swr_alloc_set_opts2(
            &swr_ctx,
            &out_ch_layout,
            AV_SAMPLE_FMT_S16,
            out_sample_rate,
            &in_ch_layout,
            codec_ctx->sample_fmt,
            in_sample_rate,
            0,
            nullptr) < 0) {
        goto cleanup;
    }
    if (!swr_ctx || swr_init(swr_ctx) < 0) {
        goto cleanup;
    }
    
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    if (!packet || !frame) {
        goto cleanup;
    }
    
    {
        auto append_frame = [&](AVFrame* decoded) -> bool {
            int64_t out_samples_64 = av_rescale_rnd(
                swr_get_delay(swr_ctx, in_sample_rate) + decoded->nb_samples,
                out_sample_rate,
                in_sample_rate,
                AV_ROUND_UP
            );
            int out_samples = static_cast<int>(std::clamp<int64_t>(out_samples_64, 0, INT_MAX));
            
            uint8_t* out_data = nullptr;
            int out_linesize = 0;
            if (av_samples_alloc(&out_data, &out_linesize, out_channels, out_samples, AV_SAMPLE_FMT_S16, 0) < 0) {
                return false;
            }
            
            int converted = swr_convert(
                swr_ctx,
                &out_data,
                out_samples,
                const_cast<const uint8_t**>(decoded->extended_data),
                decoded->nb_samples
            );
            if (converted < 0) {
                av_freep(&out_data);
                return false;
            }
            
            int converted_bytes = av_samples_get_buffer_size(&out_linesize, out_channels, converted, AV_SAMPLE_FMT_S16, 1);
            if (converted_bytes < 0) {
                av_freep(&out_data);
                return false;
            }
            
            pcm_buffer.insert(pcm_buffer.end(), out_data, out_data + converted_bytes);
            av_freep(&out_data);
            return true;
        };
        
        while (av_read_frame(format_ctx, packet) >= 0) {
            if (packet->stream_index != stream_idx) {
                av_packet_unref(packet);
                continue;
            }
            
            if (avcodec_send_packet(codec_ctx, packet) < 0) {
                av_packet_unref(packet);
                goto cleanup;
            }
            av_packet_unref(packet);
            
            while (true) {
                int ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) goto cleanup;
                if (!append_frame(frame)) goto cleanup;
                av_frame_unref(frame);
            }
        }
        
        if (avcodec_send_packet(codec_ctx, nullptr) >= 0) {
            while (true) {
                int ret = avcodec_receive_frame(codec_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
                if (ret < 0) goto cleanup;
                if (!append_frame(frame)) goto cleanup;
                av_frame_unref(frame);
            }
        }
    }
    
    if (pcm_buffer.empty()) {
        goto cleanup;
    }
    
    audio_data_ = static_cast<uint8_t*>(SDL_malloc(pcm_buffer.size()));
    if (!audio_data_) {
        goto cleanup;
    }
    std::memcpy(audio_data_, pcm_buffer.data(), pcm_buffer.size());
    audio_len_ = static_cast<uint32_t>(pcm_buffer.size());
    audio_pos_ = 0;
    
    sample_rate_ = out_sample_rate;
    channels_ = out_channels;
    bytes_per_sample_ = 2;
    duration_ = static_cast<double>(audio_len_) / (sample_rate_ * channels_ * bytes_per_sample_);
    
    success = true;
    
cleanup:
    if (!success && audio_data_) {
        SDL_free(audio_data_);
        audio_data_ = nullptr;
    }
    if (swr_ctx) swr_free(&swr_ctx);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
    av_channel_layout_uninit(&in_ch_layout);
    av_channel_layout_uninit(&out_ch_layout);
    
    return success;
}

void AudioPlayer::close() {
    if (device_id_ > 0) {
        SDL_CloseAudioDevice(device_id_);
        device_id_ = 0;
    }
    
    if (audio_data_) {
        SDL_free(audio_data_);
        audio_data_ = nullptr;
    }
    
    audio_len_ = 0;
    audio_pos_ = 0;
    playing_ = false;
    duration_ = 0.0;
    sample_rate_ = 0;
    bytes_per_sample_ = 0;
}

void AudioPlayer::play() {
    if (device_id_ > 0) {
        SDL_PauseAudioDevice(device_id_, 0);
        playing_ = true;
    }
}

void AudioPlayer::pause() {
    if (device_id_ > 0) {
        SDL_PauseAudioDevice(device_id_, 1);
        playing_ = false;
    }
}

void AudioPlayer::stop() {
    pause();
    audio_pos_ = 0;
}

bool AudioPlayer::is_playing() const {
    return playing_;
}

double AudioPlayer::position() const {
    if (sample_rate_ <= 0 || bytes_per_sample_ <= 0 || channels_ <= 0) return 0.0;
    return static_cast<double>(audio_pos_) / (sample_rate_ * channels_ * bytes_per_sample_);
}

double AudioPlayer::duration() const {
    return duration_;
}

void AudioPlayer::seek(double seconds) {
    if (!audio_data_ || sample_rate_ <= 0 || bytes_per_sample_ <= 0 || channels_ <= 0) return;
    
    uint32_t target_pos = static_cast<uint32_t>(seconds * sample_rate_ * channels_ * bytes_per_sample_);
    audio_pos_ = std::min(target_pos, audio_len_);
}

void AudioPlayer::audio_callback(void* userdata, uint8_t* stream, int len) {
    AudioPlayer* player = static_cast<AudioPlayer*>(userdata);
    
    if (!player->audio_data_ || player->audio_pos_ >= player->audio_len_) {
        std::memset(stream, 0, len);
        player->playing_ = false;
        return;
    }
    
    uint32_t remaining = player->audio_len_ - player->audio_pos_;
    uint32_t to_copy = std::min(static_cast<uint32_t>(len), remaining);
    
    std::memcpy(stream, player->audio_data_ + player->audio_pos_, to_copy);
    
    if (to_copy < static_cast<uint32_t>(len)) {
        std::memset(stream + to_copy, 0, len - to_copy);
        player->playing_ = false;
    }
    
    player->audio_pos_ += to_copy;
}

void AudioPlayer::sync_to_frame(int frame_number, double fps, double max_drift) {
    if (!audio_data_ || sample_rate_ <= 0 || bytes_per_sample_ <= 0 || channels_ <= 0 || fps <= 0.0) return;
    
    double expected_pos = frame_number / fps;
    double current_pos = position();
    double drift = std::abs(current_pos - expected_pos);
    
    if (drift > max_drift) {
        seek(expected_pos);
    }
}

}
