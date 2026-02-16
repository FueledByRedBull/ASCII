#pragma once

#include <string>
#include <cstdint>

struct SDL_AudioSpec;

namespace ascii {

class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();
    
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
    
    uint8_t* audio_data_ = nullptr;
    uint32_t audio_len_ = 0;
    uint32_t audio_pos_ = 0;
    
    SDL_AudioSpec* spec_ = nullptr;
    int device_id_ = 0;
    bool playing_ = false;
    double duration_ = 0.0;
    int bytes_per_sample_ = 0;
    int sample_rate_ = 0;
    int channels_ = 2;
};

}
