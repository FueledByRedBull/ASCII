#pragma once

#include <string>
#include <cstdint>
#include <cstddef>
#include <vector>

namespace ascii {

class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();
    AudioPlayer(const AudioPlayer&) = delete;
    AudioPlayer& operator=(const AudioPlayer&) = delete;

    // Applies to subsequent opens; never exceeds the 256 MiB decoded PCM cap.
    void set_memory_limit(size_t bytes);
    
    bool open(const std::string& filename);
    void close();
    void play();
    void pause();
    void stop();
    
    bool is_playing() const;
    double position() const;
    double duration() const;
    void seek(double seconds);
    
    void sync_to_frame(int frame_number, double fps, double max_drift = 0.1);
    
private:
    bool load_audio(const std::string& filename);
    static void audio_callback(void* userdata, uint8_t* stream, int len);
    
    // Public operations must be serialized by the caller; SDL callback access is locked.
    std::vector<uint8_t> audio_data_;
    uint32_t audio_len_ = 0;
    uint32_t audio_pos_ = 0;
    
    uint32_t device_id_ = 0;
    bool audio_initialized_ = false;
    bool playing_ = false;
    double duration_ = 0.0;
    int bytes_per_sample_ = 0;
    int sample_rate_ = 0;
    int channels_ = 2;
    size_t memory_limit_ = 256u * 1024u * 1024u;
};

}
