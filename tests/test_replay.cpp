#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstddef>
#include <exception>

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
    assert(writer.close());

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
    assert(writer.close());

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
    assert(writer.close());
}

void patch_u32(const std::filesystem::path& path, std::streamoff offset, uint32_t value) {
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    assert(io);
    io.seekp(offset);
    io.write(reinterpret_cast<const char*>(&value), sizeof(value));
    assert(io);
}

void patch_byte(const std::filesystem::path& path, std::streamoff offset) {
    std::fstream io(path, std::ios::binary | std::ios::in | std::ios::out);
    assert(io);
    io.seekg(offset);
    char value = 0;
    io.read(&value, 1);
    value ^= static_cast<char>(0x5A);
    io.seekp(offset);
    io.write(&value, 1);
    assert(io);
}

void test_unicode_random_access_and_validation() {
    auto path = temp_replay_path("ascii_engine_replay_validation.areplay");
    std::filesystem::remove(path);

    std::vector<ASCIICell> frame0{
        cell(0x1F642, 1, 2, 3), cell('B', 4, 5, 6),
        cell('C', 7, 8, 9), cell('D', 10, 11, 12),
    };
    std::vector<ASCIICell> frame1 = frame0;
    frame1[3] = cell(0x1F680, 20, 21, 22);

    ReplayWriter writer;
    assert(writer.open(path.string(), 2, 2, 30, "unicode1"));
    assert(writer.write_frame(0, frame0));
    assert(writer.write_frame_delta(1, frame1, frame0));
    assert(writer.close());

    ReplayReader reader;
    assert(reader.open(path.string()));
    std::vector<ASCIICell> output;
    assert(reader.read_frame(1, output));
    assert(output[0].codepoint == 0x1F642);
    assert(output[3].codepoint == 0x1F680);
    assert(reader.read_frame(0, output));
    assert(output[3].codepoint == static_cast<uint32_t>('D'));
    reader.close();

    auto invalid_version = temp_replay_path("ascii_engine_replay_invalid_version.areplay");
    std::filesystem::copy_file(path, invalid_version, std::filesystem::copy_options::overwrite_existing);
    patch_u32(invalid_version, offsetof(ReplayHeader, version), 2);
    assert(!reader.open(invalid_version.string()));

    auto invalid_dimensions = temp_replay_path("ascii_engine_replay_invalid_dimensions.areplay");
    std::filesystem::copy_file(path, invalid_dimensions, std::filesystem::copy_options::overwrite_existing);
    patch_u32(invalid_dimensions, offsetof(ReplayHeader, cols), REPLAY_MAX_COLS + 1);
    assert(!reader.open(invalid_dimensions.string()));

    auto invalid_reserved = temp_replay_path("ascii_engine_replay_invalid_reserved.areplay");
    std::filesystem::copy_file(path, invalid_reserved, std::filesystem::copy_options::overwrite_existing);
    patch_u32(invalid_reserved, offsetof(ReplayHeader, reserved), 1);
    assert(!reader.open(invalid_reserved.string()));

    auto invalid_flags = temp_replay_path("ascii_engine_replay_invalid_flags.areplay");
    std::filesystem::copy_file(path, invalid_flags, std::filesystem::copy_options::overwrite_existing);
    patch_u32(invalid_flags, sizeof(ReplayHeader) + offsetof(ReplayFrameHeader, flags), 0x80000000u);
    assert(!reader.open(invalid_flags.string()));

    auto invalid_size = temp_replay_path("ascii_engine_replay_invalid_size.areplay");
    std::filesystem::copy_file(path, invalid_size, std::filesystem::copy_options::overwrite_existing);
    patch_u32(invalid_size, sizeof(ReplayHeader) + offsetof(ReplayFrameHeader, data_size), 0xFFFFFFFFu);
    assert(!reader.open(invalid_size.string()));

    auto corrupt_payload = temp_replay_path("ascii_engine_replay_corrupt_payload.areplay");
    std::filesystem::copy_file(path, corrupt_payload, std::filesystem::copy_options::overwrite_existing);
    patch_byte(corrupt_payload, sizeof(ReplayHeader) + sizeof(ReplayFrameHeader) + 2);
    assert(reader.open(corrupt_payload.string()));
    assert(!reader.read_frame(0, output));
    reader.close();

    auto invalid_output = temp_replay_path("ascii_engine_replay_invalid_unicode.areplay");
    std::filesystem::remove(invalid_output);
    ReplayWriter invalid_writer;
    assert(invalid_writer.open(invalid_output.string(), 1, 1, 30, "invalid1"));
    assert(!invalid_writer.write_frame(0, {cell(0x110000, 0, 0, 0)}));
    assert(!invalid_writer.close());
    assert(!std::filesystem::exists(invalid_output));

    std::filesystem::remove(path);
    std::filesystem::remove(invalid_version);
    std::filesystem::remove(invalid_dimensions);
    std::filesystem::remove(invalid_reserved);
    std::filesystem::remove(invalid_flags);
    std::filesystem::remove(invalid_size);
    std::filesystem::remove(corrupt_payload);
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

void test_delta_rejects_stale_previous_frame() {
    const auto path = temp_replay_path("ascii_engine_replay_stale.areplay");
    const std::vector<ASCIICell> first{cell('A', 1, 2, 3)};
    const std::vector<ASCIICell> next{cell('B', 4, 5, 6)};
    ReplayWriter writer;
    assert(writer.open(path.string(), 1, 1, 30, "stale001"));
    assert(writer.write_frame(0, first));
    assert(!writer.write_frame_delta(1, next, next));
    assert(!writer.close());
    assert(!std::filesystem::exists(path));
}

void test_unfinished_and_empty_writers_preserve_destination() {
    const auto path = temp_replay_path("ascii_engine_replay_preserve.areplay");
    {
        std::ofstream out(path, std::ios::binary);
        out << "existing destination";
    }
    const auto before = read_all_bytes(path);
    {
        ReplayWriter writer;
        assert(writer.open(path.string(), 1, 1, 30, "abort001"));
        assert(writer.write_frame(0, {cell('A', 1, 2, 3)}));
    }
    assert(read_all_bytes(path) == before);
    ReplayWriter empty;
    assert(empty.open(path.string(), 1, 1, 30, "empty001"));
    assert(!empty.close());
    assert(read_all_bytes(path) == before);
    std::filesystem::remove(path);
}

void test_reader_memory_budget() {
    const auto path = temp_replay_path("ascii_engine_replay_budget.areplay");
    ReplayWriter writer;
    assert(writer.open(path.string(), 2, 2, 30, "budget01"));
    assert(writer.write_frame(0, std::vector<ASCIICell>(4, cell('A', 1, 2, 3))));
    assert(writer.close());
    ReplayReader reader;
    assert(!reader.open(path.string(), 32));
    assert(!reader.is_open());
    assert(reader.open(path.string(), 4096));
    std::vector<ASCIICell> cells;
    assert(reader.read_frame(0, cells) && cells.size() == 4);
    assert(!reader.open(path.string(), 32));
    assert(reader.open(path.string(), 4096));
    reader.close();
    std::filesystem::remove(path);
}

void test_reader_failed_index_clears_state() {
    const auto path = temp_replay_path("ascii_engine_replay_index_failure.areplay");
    ReplayWriter writer;
    assert(writer.open(path.string(), 2, 2, 30, "index001"));
    const std::vector<ASCIICell> frame(4, cell('A', 1, 2, 3));
    assert(writer.write_frame(0, frame));
    assert(writer.write_frame(1, frame));
    assert(writer.close());
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);

    ReplayReader reader;
    assert(!reader.open(path.string(), 4096));
    assert(!reader.is_open());
    assert(reader.indexed_frame_count() == 0);
    std::vector<ASCIICell> cells;
    assert(!reader.read_frame(0, cells));

    assert(writer.open(path.string(), 2, 2, 30, "index002"));
    assert(writer.write_frame(0, frame));
    assert(writer.close());
    assert(reader.open(path.string(), 4096));
    assert(reader.read_frame(0, cells) && cells.size() == 4);
    reader.close();
    assert(reader.indexed_frame_count() == 0);
    std::filesystem::remove(path);
}

void test_reader_reset_after_rejected_header() {
    const auto path = temp_replay_path("ascii_engine_replay_rejected_reset.areplay");
    write_deterministic_fixture(path);
    patch_u32(path, offsetof(ReplayHeader, cols), UINT32_MAX);
    patch_u32(path, offsetof(ReplayHeader, rows), UINT32_MAX);
    ReplayReader reader;
    assert(!reader.open(path.string()));
    assert(!reader.is_open());
    bool reset_threw = false;
    try {
        reader.reset_decode_state();
    } catch (const std::exception& error) {
        std::cerr << "Closed reader reset threw: " << error.what() << '\n';
        reset_threw = true;
    }
    assert(!reset_threw);
    assert(reader.header().cols == UINT32_MAX);
    assert(reader.header().rows == UINT32_MAX);
    assert(reader.indexed_frame_count() == 0);

    write_deterministic_fixture(path);
    assert(reader.open(path.string()));
    std::vector<ASCIICell> cells;
    assert(reader.read_frame(1, cells) && cells[1].codepoint == 'Q');
    reader.reset_decode_state();
    assert(reader.read_frame(0, cells) && cells[1].codepoint == 'B');
    reader.close();
    reader.reset_decode_state();
    assert(!reader.is_open());
    std::filesystem::remove(path);
}

}  // namespace

int main() {
    test_full_frame_round_trip();
    test_delta_frame_round_trip();
    test_replay_bytes_are_deterministic();
    test_unicode_random_access_and_validation();
    test_delta_rejects_stale_previous_frame();
    test_unfinished_and_empty_writers_preserve_destination();
    test_reader_memory_budget();
    test_reader_failed_index_clears_state();
    test_reader_reset_after_rejected_header();
    std::cout << "Replay tests passed\n";
    return 0;
}
