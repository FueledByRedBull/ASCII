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
#include <cmath>
#include <cstring>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ascii {

AudioPlayer::AudioPlayer() {
    audio_initialized_ = SDL_Init(SDL_INIT_AUDIO) == 0;
}

AudioPlayer::~AudioPlayer() {
    close();
    if (audio_initialized_) SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

void AudioPlayer::set_memory_limit(size_t bytes) {
    memory_limit_ = std::min(bytes, size_t{256} * 1024 * 1024);
}

bool AudioPlayer::open(const std::string& filename) {
    close();
    if (!audio_initialized_ || memory_limit_ == 0) return false;

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
    if (device_id_ == 0) {
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

    try {
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
            const size_t frame_bytes = static_cast<size_t>(out_channels) * 2;
            auto convert_frame = [&](const AVFrame* decoded) -> int {
                const int input_samples = decoded ? decoded->nb_samples : 0;
                if (input_samples < 0) return -1;
                const int out_samples = swr_get_out_samples(swr_ctx, input_samples);
                if (out_samples < 0) return -1;
                if (out_samples == 0) return 0;
                if (static_cast<size_t>(out_samples) > (memory_limit_ - pcm_buffer.size()) / frame_bytes) return -1;

                const size_t previous_size = pcm_buffer.size();
                const size_t required = previous_size + static_cast<size_t>(out_samples) * frame_bytes;
                if (required > pcm_buffer.capacity()) {
                    const size_t capacity = std::min(memory_limit_, std::max(required,
                        std::max<size_t>(4096, pcm_buffer.capacity() * 2)));
                    pcm_buffer.reserve(capacity);
                }
                pcm_buffer.resize(required);
                uint8_t* output = pcm_buffer.data() + previous_size;
                const int converted = swr_convert(swr_ctx, &output, out_samples,
                    decoded ? const_cast<const uint8_t**>(decoded->extended_data) : nullptr, input_samples);
                if (converted < 0) return -1;
                pcm_buffer.resize(previous_size + static_cast<size_t>(converted) * frame_bytes);
                return converted;
            };
            const auto drain_decoder = [&](bool flushing) {
                while (true) {
                    const int result = avcodec_receive_frame(codec_ctx, frame);
                    if (result == AVERROR_EOF) return flushing;
                    if (result == AVERROR(EAGAIN)) return !flushing;
                    if (result < 0 || convert_frame(frame) < 0) return false;
                    av_frame_unref(frame);
                }
            };
            int read_status = 0;
            while ((read_status = av_read_frame(format_ctx, packet)) >= 0) {
                if (packet->stream_index != stream_idx) {
                    av_packet_unref(packet);
                    continue;
                }
                const int sent = avcodec_send_packet(codec_ctx, packet);
                av_packet_unref(packet);
                if (sent < 0 || !drain_decoder(false)) goto cleanup;
            }
            if (read_status != AVERROR_EOF) goto cleanup;
            if (format_ctx->pb && format_ctx->pb->error < 0 && format_ctx->pb->error != AVERROR_EOF) goto cleanup;
            if (avcodec_send_packet(codec_ctx, nullptr) < 0 || !drain_decoder(true)) goto cleanup;
            int converted = 0;
            do {
                converted = convert_frame(nullptr);
                if (converted < 0) goto cleanup;
            } while (converted > 0);
        }
        if (pcm_buffer.empty()) {
            goto cleanup;
        }

        audio_len_ = static_cast<uint32_t>(pcm_buffer.size());
        audio_data_ = std::move(pcm_buffer);
        audio_pos_ = 0;

        sample_rate_ = out_sample_rate;
        channels_ = out_channels;
        bytes_per_sample_ = 2;
        duration_ = static_cast<double>(audio_len_) / sample_rate_ / channels_ / bytes_per_sample_;

        success = true;
    } catch (const std::bad_alloc&) {
        success = false;
    } catch (const std::length_error&) {
        success = false;
    }

cleanup:
    if (swr_ctx) swr_free(&swr_ctx);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (format_ctx) avformat_close_input(&format_ctx);
    av_channel_layout_uninit(&in_ch_layout);
    av_channel_layout_uninit(&out_ch_layout);
    if (!success) close();

    return success;
}

void AudioPlayer::close() {
    if (device_id_ > 0) {
        SDL_CloseAudioDevice(device_id_);
        device_id_ = 0;
    }

    std::vector<uint8_t>().swap(audio_data_);

    audio_len_ = 0;
    audio_pos_ = 0;
    playing_ = false;
    duration_ = 0.0;
    sample_rate_ = 0;
    bytes_per_sample_ = 0;
}

void AudioPlayer::play() {
    if (device_id_ > 0) {
        SDL_LockAudioDevice(device_id_);
        const bool start = audio_pos_ < audio_len_;
        playing_ = start;
        SDL_UnlockAudioDevice(device_id_);
        SDL_PauseAudioDevice(device_id_, start ? 0 : 1);
    }
}

void AudioPlayer::pause() {
    if (device_id_ > 0) {
        SDL_PauseAudioDevice(device_id_, 1);
        SDL_LockAudioDevice(device_id_);
        playing_ = false;
        SDL_UnlockAudioDevice(device_id_);
    }
}

void AudioPlayer::stop() {
    pause();
    if (device_id_ > 0) SDL_LockAudioDevice(device_id_);
    audio_pos_ = 0;
    if (device_id_ > 0) SDL_UnlockAudioDevice(device_id_);
}

bool AudioPlayer::is_playing() const {
    if (device_id_ > 0) SDL_LockAudioDevice(device_id_);
    const bool playing = playing_;
    if (device_id_ > 0) SDL_UnlockAudioDevice(device_id_);
    return playing;
}

double AudioPlayer::position() const {
    if (sample_rate_ <= 0 || bytes_per_sample_ <= 0 || channels_ <= 0) return 0.0;
    if (device_id_ > 0) SDL_LockAudioDevice(device_id_);
    const double position = static_cast<double>(audio_pos_) / sample_rate_ / channels_ / bytes_per_sample_;
    if (device_id_ > 0) SDL_UnlockAudioDevice(device_id_);
    return position;
}

double AudioPlayer::duration() const {
    return duration_;
}

void AudioPlayer::seek(double seconds) {
    if (!std::isfinite(seconds) || audio_data_.empty() || sample_rate_ <= 0 ||
        bytes_per_sample_ <= 0 || channels_ <= 0) return;
    const uint32_t frame_bytes = static_cast<uint32_t>(channels_ * bytes_per_sample_);
    const double clamped = std::clamp(seconds, 0.0, duration_);
    const uint32_t frames = clamped >= duration_ ? audio_len_ / frame_bytes :
        static_cast<uint32_t>(std::floor(clamped * sample_rate_));
    if (device_id_ > 0) SDL_LockAudioDevice(device_id_);
    audio_pos_ = std::min(frames, audio_len_ / frame_bytes) * frame_bytes;
    if (audio_pos_ == audio_len_) playing_ = false;
    if (device_id_ > 0) SDL_UnlockAudioDevice(device_id_);
}

void AudioPlayer::audio_callback(void* userdata, uint8_t* stream, int len) {
    AudioPlayer* player = static_cast<AudioPlayer*>(userdata);
    if (len <= 0) return;
    if (!player->playing_ || player->audio_data_.empty() || player->audio_pos_ >= player->audio_len_) {
        std::memset(stream, 0, len);
        player->playing_ = false;
        return;
    }

    uint32_t remaining = player->audio_len_ - player->audio_pos_;
    uint32_t to_copy = std::min(static_cast<uint32_t>(len), remaining);

    std::memcpy(stream, player->audio_data_.data() + player->audio_pos_, to_copy);

    if (to_copy < static_cast<uint32_t>(len)) {
        std::memset(stream + to_copy, 0, len - to_copy);
        player->playing_ = false;
    }

    player->audio_pos_ += to_copy;
    if (player->audio_pos_ == player->audio_len_) player->playing_ = false;
}

void AudioPlayer::sync_to_frame(int frame_number, double fps, double max_drift) {
    if (audio_data_.empty() || sample_rate_ <= 0 || bytes_per_sample_ <= 0 || channels_ <= 0 ||
        !std::isfinite(fps) || fps <= 0.0 || !std::isfinite(max_drift) || max_drift < 0.0) return;

    double expected_pos = frame_number / fps;
    double current_pos = position();
    double drift = std::abs(current_pos - expected_pos);

    if (drift > max_drift) {
        seek(expected_pos);
    }
}

}
