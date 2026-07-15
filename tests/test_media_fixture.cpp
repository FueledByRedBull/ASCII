#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "../src/core/frame_source.hpp"
#include "../src/core/replay.hpp"
#include "../src/render/video_encoder.hpp"

using namespace ascii;

namespace {

bool generate_video(const std::filesystem::path& path, int frame_count,
                    int width = 32, int height = 24, int fps = 12) {
    VideoEncoder::Config config;
    config.width = width;
    config.height = height;
    config.fps = fps;
    config.bitrate = width >= 1920 ? 4000000 : 300000;
    VideoEncoder encoder;
    if (!encoder.open(path.string(), config)) {
        std::cerr << encoder.last_error() << "\n";
        return false;
    }
    for (int frame_index = 0; frame_index < frame_count; ++frame_index) {
        FrameBuffer frame(config.width, config.height);
        for (int y = 0; y < config.height; ++y) {
            for (int x = 0; x < config.width; ++x) {
                frame.set_pixel(x, y, Color(
                    static_cast<uint8_t>((x * 7 + frame_index * 31) % 256),
                    static_cast<uint8_t>((y * 11 + frame_index * 17) % 256),
                    static_cast<uint8_t>((x + y + frame_index * 47) % 256)));
            }
        }
        if (!encoder.write_frame(frame)) {
            std::cerr << encoder.last_error() << "\n";
            return false;
        }
    }
    return encoder.close();
}

bool verify_video(const std::filesystem::path& path, int expected_frames, double expected_fps) {
    VideoFileSource source;
    if (!source.open(path.string())) {
        std::cerr << "Failed to open video fixture: " << path << "\n";
        return false;
    }
    const double actual_fps = source.fps();
    // Short Matroska streams can expose the millisecond muxer time base as
    // 30.303 FPS instead of the nominal 30 FPS on some FFmpeg versions.
    if (std::abs(actual_fps - expected_fps) > 0.5) {
        std::cerr << "Unexpected frame rate: expected " << expected_fps
                  << ", got " << actual_fps << "\n";
        return false;
    }
    FrameBuffer frame;
    int count = 0;
    while (true) {
        const auto status = source.read_next(frame);
        if (status == FrameReadStatus::End) break;
        if (status != FrameReadStatus::Frame || frame.empty()) {
            std::cerr << "Video decode failed after " << count << " frames\n";
            return false;
        }
        ++count;
    }
    if (count != expected_frames) {
        std::cerr << "Unexpected frame count: expected " << expected_frames
                  << ", got " << count << "\n";
        return false;
    }
    return true;
}

bool verify_image(const std::filesystem::path& path) {
    ImageSource source;
    if (!source.open(path.string())) return false;
    FrameBuffer frame;
    return source.read_next(frame) == FrameReadStatus::Frame &&
           !frame.empty() && source.read_next(frame) == FrameReadStatus::End;
}

bool verify_text_frames(const std::filesystem::path& base, int expected_frames) {
    const std::string stem = base.stem().string();
    for (int i = 0; i < expected_frames; ++i) {
        const auto frame = base.parent_path() / (stem + "_" + std::to_string(i) + ".txt");
        std::ifstream input(frame, std::ios::binary);
        if (!input || input.peek() == std::ifstream::traits_type::eof()) return false;
    }
    const auto extra = base.parent_path() / (stem + "_" + std::to_string(expected_frames) + ".txt");
    return !std::filesystem::exists(extra);
}

bool truncate_file(const std::filesystem::path& source, const std::filesystem::path& target) {
    std::ifstream input(source, std::ios::binary | std::ios::ate);
    if (!input) return false;
    const auto size = input.tellg();
    if (size < 16) return false;
    const auto keep = static_cast<std::streamoff>(size) * 3 / 4;
    input.seekg(0);
    std::ofstream output(target, std::ios::binary | std::ios::trunc);
    if (!output) return false;
    char buffer[4096];
    std::streamoff remaining = keep;
    while (remaining > 0) {
        const auto chunk = static_cast<std::streamsize>(std::min<std::streamoff>(remaining, sizeof(buffer)));
        input.read(buffer, chunk);
        const auto read = input.gcount();
        if (read <= 0) return false;
        output.write(buffer, read);
        remaining -= read;
    }
    return output.good();
}

ASCIICell replay_cell(uint32_t codepoint) {
    ASCIICell result;
    result.codepoint = codepoint;
    result.fg_r = 255;
    result.fg_g = 255;
    result.fg_b = 255;
    return result;
}

bool generate_golden_replay(const std::filesystem::path& path) {
    const std::vector<ASCIICell> first{
        replay_cell('A'), replay_cell(0x2500),
        replay_cell('C'), replay_cell('D')};
    std::vector<ASCIICell> second = first;
    second[1] = replay_cell('Q');
    second[2] = replay_cell(0x1F680);
    ReplayWriter writer;
    return writer.open(path.string(), 2, 2, 12, "golden01") &&
           writer.write_frame(0, first) &&
           writer.write_frame_delta(1, second, first) &&
           writer.close();
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return input ? std::string(std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>()) : std::string{};
}

bool verify_golden_replay_text(const std::filesystem::path& base) {
    const std::string stem = base.stem().string();
    const auto first = base.parent_path() / (stem + "_0.txt");
    const auto second = base.parent_path() / (stem + "_1.txt");
    return read_text(first) == "A\xE2\x94\x80\nCD\n" &&
           read_text(second) == "AQ\n\xF0\x9F\x9A\x80" "D\n";
}

bool verify_encoder_partial_failure(const std::filesystem::path& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    VideoEncoder::Config config;
    config.width = 16;
    config.height = 16;
    config.fps = 12;
    VideoEncoder encoder;
    if (!encoder.open(path.string(), config)) return false;
    const bool write_ok = encoder.write_frame(FrameBuffer{});
    const bool close_ok = encoder.close();
    return !write_ok && !close_ok && !std::filesystem::exists(path);
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 3) return 2;
    const std::string command = argv[1];
    const std::filesystem::path path = argv[2];
    bool ok = false;
    if (command == "generate" && (argc == 4 || argc == 7)) {
        ok = argc == 4
            ? generate_video(path, std::stoi(argv[3]))
            : generate_video(path, std::stoi(argv[3]), std::stoi(argv[4]),
                             std::stoi(argv[5]), std::stoi(argv[6]));
    } else if (command == "verify-video" && argc == 5) {
        ok = verify_video(path, std::stoi(argv[3]), std::stod(argv[4]));
    } else if (command == "verify-image" && argc == 3) {
        ok = verify_image(path);
    } else if (command == "verify-text" && argc == 4) {
        ok = verify_text_frames(path, std::stoi(argv[3]));
    } else if (command == "truncate" && argc == 4) {
        ok = truncate_file(path, argv[3]);
    } else if (command == "generate-replay" && argc == 3) {
        ok = generate_golden_replay(path);
    } else if (command == "verify-replay-text" && argc == 3) {
        ok = verify_golden_replay_text(path);
    } else if (command == "verify-encoder-partial-failure" && argc == 3) {
        ok = verify_encoder_partial_failure(path);
    }
    if (!ok) std::cerr << "Media fixture command failed: " << command << "\n";
    return ok ? 0 : 1;
}
