#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "../src/core/replay.hpp"

using namespace ascii;

namespace {

ASCIICell cell(uint32_t codepoint, uint8_t r, uint8_t g, uint8_t b) {
    ASCIICell c;
    c.codepoint = codepoint;
    c.fg_r = r;
    c.fg_g = g;
    c.fg_b = b;
    return c;
}

std::filesystem::path temp_replay_path(const char* name) {
    return std::filesystem::temp_directory_path() / name;
}

void test_full_frame_round_trip() {
    auto path = temp_replay_path("ascii_engine_replay_full.areplay");
    std::filesystem::remove(path);

    ReplayWriter writer;
    assert(writer.open(path.string(), 2, 2, 12, "deadbeef"));
    std::vector<ASCIICell> input{
        cell('A', 255, 0, 0),
        cell('B', 0, 255, 0),
        cell('C', 0, 0, 255),
        cell('D', 255, 255, 255),
    };
    assert(writer.write_frame(0, input));
    writer.close();

    ReplayReader reader;
    assert(reader.open(path.string()));
    assert(reader.cols() == 2);
    assert(reader.rows() == 2);
    assert(reader.header().fps == 12);
    assert(reader.frame_count() == 1);
    assert(reader.indexed_frame_count() == 1);
    assert(reader.config_hash() == "deadbeef");

    std::vector<ASCIICell> output;
    assert(reader.read_frame(0, output));
    assert(output.size() == input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        assert(output[i].codepoint == input[i].codepoint);
        assert(output[i].fg_r == input[i].fg_r);
        assert(output[i].fg_g == input[i].fg_g);
        assert(output[i].fg_b == input[i].fg_b);
    }
    reader.close();
    std::filesystem::remove(path);
}

void test_delta_frame_round_trip() {
    auto path = temp_replay_path("ascii_engine_replay_delta.areplay");
    std::filesystem::remove(path);

    std::vector<ASCIICell> frame0{
        cell('A', 1, 2, 3),
        cell('B', 4, 5, 6),
        cell('C', 7, 8, 9),
        cell('D', 10, 11, 12),
    };
    std::vector<ASCIICell> frame1 = frame0;
    frame1[2] = cell('Z', 20, 21, 22);

    ReplayWriter writer;
    assert(writer.open(path.string(), 2, 2, 30, "12345678"));
    assert(writer.write_frame(0, frame0));
    assert(writer.write_frame_delta(1, frame1, frame0));
    writer.close();

    ReplayReader reader;
    assert(reader.open(path.string()));
    std::vector<ASCIICell> output0;
    std::vector<ASCIICell> output1;
    assert(reader.read_frame(0, output0));
    assert(reader.read_frame(1, output1));
    assert(output0[2].codepoint == static_cast<uint32_t>('C'));
    assert(output1[0].codepoint == static_cast<uint32_t>('A'));
    assert(output1[2].codepoint == static_cast<uint32_t>('Z'));
    assert(output1[2].fg_r == 20);
    assert(output1[2].fg_g == 21);
    assert(output1[2].fg_b == 22);
    reader.close();
    std::filesystem::remove(path);
}

std::vector<char> read_all_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    assert(in);
    return std::vector<char>(
        std::istreambuf_iterator<char>(in),
        std::istreambuf_iterator<char>());
}

void write_deterministic_fixture(const std::filesystem::path& path) {
    std::vector<ASCIICell> frame0{
        cell('A', 1, 2, 3),
        cell('B', 4, 5, 6),
        cell('C', 7, 8, 9),
        cell('D', 10, 11, 12),
    };
    std::vector<ASCIICell> frame1 = frame0;
    frame1[1] = cell('Q', 40, 41, 42);

    ReplayWriter writer;
    assert(writer.open(path.string(), 2, 2, 24, "abcd1234"));
    assert(writer.write_frame(0, frame0));
    assert(writer.write_frame_delta(1, frame1, frame0));
    writer.close();
}

void test_replay_bytes_are_deterministic() {
    auto path_a = temp_replay_path("ascii_engine_replay_deterministic_a.areplay");
    auto path_b = temp_replay_path("ascii_engine_replay_deterministic_b.areplay");
    std::filesystem::remove(path_a);
    std::filesystem::remove(path_b);

    write_deterministic_fixture(path_a);
    write_deterministic_fixture(path_b);

    auto a = read_all_bytes(path_a);
    auto b = read_all_bytes(path_b);
    assert(!a.empty());
    assert(a == b);

    std::filesystem::remove(path_a);
    std::filesystem::remove(path_b);
}

}  // namespace

int main() {
    test_full_frame_round_trip();
    test_delta_frame_round_trip();
    test_replay_bytes_are_deterministic();
    std::cout << "Replay tests passed\n";
    return 0;
}
