#include "audio/audio_player.hpp"
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>

namespace {

int failures = 0;
constexpr int sample_rate = 48000;
constexpr int frame_count = 4800;
constexpr size_t pcm_bytes = frame_count * 4;

void check(bool condition, const std::string& description) {
    std::cout << (condition ? "PASS " : "FAIL ") << description << '\n';
    if (!condition) ++failures;
}

void write_wav(const std::filesystem::path& path) {
    std::ofstream file(path, std::ios::binary);
    const auto little_endian = [&](uint32_t value, int bytes) {
        for (int i = 0; i < bytes; ++i) file.put(static_cast<char>(value >> (i * 8)));
    };
    file.write("RIFF", 4);
    little_endian(36 + pcm_bytes, 4);
    file.write("WAVEfmt ", 8);
    little_endian(16, 4);
    little_endian(1, 2);
    little_endian(2, 2);
    little_endian(sample_rate, 4);
    little_endian(sample_rate * 4, 4);
    little_endian(4, 2);
    little_endian(16, 2);
    file.write("data", 4);
    little_endian(pcm_bytes, 4);
    for (int i = 0; i < frame_count; ++i) {
        const uint16_t sample = static_cast<uint16_t>((i % 64) * 128);
        little_endian(sample, 2);
        little_endian(sample, 2);
    }
    if (!file) throw std::runtime_error("Failed to create WAV fixture");
}

void test_playback(const std::string& path) {
    ascii::AudioPlayer player;
    check(!std::is_copy_constructible_v<ascii::AudioPlayer> &&
          !std::is_copy_assignable_v<ascii::AudioPlayer>, "audio device owner is noncopyable");
    const bool opened = player.open(path);
    check(opened, "generated WAV opens through dummy audio device");
    if (!opened) return;
    check(std::abs(player.duration() - .1) < 1e-9, "decoded duration equals WAV sample count");
    check(player.position() == 0.0 && !player.is_playing(), "new audio starts paused at zero");

    player.seek(1.25 / sample_rate);
    check(std::abs(player.position() - 1.0 / sample_rate) < 1e-12, "seek aligns to a complete stereo sample frame");
    player.seek(-.05);
    check(player.position() == 0.0, "negative seek clamps to start");
    player.seek(.04);
    const double saved = player.position();
    player.seek(std::numeric_limits<double>::quiet_NaN());
    check(player.position() == saved, "nonfinite seek preserves position");
    player.seek(.04);
    player.seek(std::numeric_limits<double>::infinity());
    check(player.position() == saved, "infinite seek preserves position");
    player.seek(10.0);
    check(player.position() == player.duration(), "seek beyond duration clamps to end");
    player.play();
    check(!player.is_playing(), "playing at EOF remains stopped");
    player.stop();
    check(player.position() == 0.0 && !player.is_playing(), "stop resets position and playback state");

    player.play();
    bool aligned = true;
    for (int i = 0; i < 32; ++i) {
        player.seek((i % 4) * .01);
        const double position = player.position();
        aligned &= position >= 0.0 && position <= player.duration() &&
                   std::abs(position * sample_rate - std::round(position * sample_rate)) < 1e-6;
        (void)player.is_playing();
        SDL_Delay(1);
    }
    player.pause();
    check(aligned && !player.is_playing(), "controls preserve sample alignment during dummy callbacks");
    player.seek(.04);
    player.sync_to_frame(10, std::numeric_limits<double>::infinity(), .001);
    check(player.position() == saved, "nonfinite sync rate does not move playback");

    player.stop();
    player.play();
    const Uint64 started = SDL_GetTicks64();
    while (player.is_playing() && SDL_GetTicks64() - started < 2000) SDL_Delay(2);
    check(!player.is_playing() && player.position() == player.duration(), "dummy playback reaches EOF");
    player.close();
    check(player.position() == 0.0 && player.duration() == 0.0 && !player.is_playing(),
          "close releases playback state");
}

void test_budget(const std::string& path) {
    ascii::AudioPlayer player;
    player.set_memory_limit(pcm_bytes - 4);
    check(!player.open(path), "decoded PCM exceeding budget is rejected");
    check(player.duration() == 0.0 && player.position() == 0.0 && !player.is_playing(),
          "budget rejection leaves a closed player");
    player.set_memory_limit(pcm_bytes);
    check(player.open(path), "decoded PCM at budget boundary opens");
    player.close();
    player.set_memory_limit(0);
    check(!player.open(path), "zero PCM budget rejects nonempty audio");
}

}  // namespace

int main() {
    SDL_SetMainReady();
    if (SDL_setenv("SDL_AUDIODRIVER", "dummy", 1) != 0) return 2;
    const std::filesystem::path path = "audio_regression.wav";
    write_wav(path);
    test_playback(path.string());
    test_budget(path.string());
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::cout << "FAILURES=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}
