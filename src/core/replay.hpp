#pragma once

#include "core/types.hpp"
#include "render/terminal_renderer.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace ascii {

#pragma pack(push, 1)
struct ReplayHeader {
    char magic[8] = {'A', 'R', 'E', 'P', 'L', 'A', 'Y', '\0'};
    uint32_t version = 1;
    uint32_t cols = 0;
    uint32_t rows = 0;
    uint32_t frame_count = 0;
    uint32_t fps = 30;
    char config_hash[9] = {0};
    uint32_t reserved[4] = {0};
};

struct ReplayFrameHeader {
    uint32_t frame_index = 0;
    uint32_t data_size = 0;
    uint32_t changed_cells = 0;
    uint32_t flags = 0;
};

struct ReplayCellData {
    uint32_t glyph_index = 0;
    uint8_t fg_r = 0, fg_g = 0, fg_b = 0;
    uint8_t bg_r = 0, bg_g = 0, bg_b = 0;
};
#pragma pack(pop)

constexpr uint32_t REPLAY_FRAME_FULL = 1u << 0;
constexpr uint32_t REPLAY_FRAME_DELTA = 1u << 1;

class ReplayWriter {
public:
    ReplayWriter();
    ~ReplayWriter();
    
    bool open(const std::string& path, int cols, int rows, int fps, const std::string& config_hash);
    bool write_frame(uint32_t frame_index, const std::vector<ASCIICell>& cells);
    bool write_frame_delta(uint32_t frame_index, const std::vector<ASCIICell>& cells, 
                           const std::vector<ASCIICell>& prev_cells);
    void close();
    
    uint32_t frame_count() const { return frame_count_; }
    bool is_open() const { return file_ != nullptr; }
    
private:
    FILE* file_ = nullptr;
    uint32_t frame_count_ = 0;
    int cols_ = 0;
    int rows_ = 0;
    std::vector<uint8_t> compress_buffer_;
    std::vector<ASCIICell> last_cells_;
    
    bool write_header(const std::string& config_hash);
    bool write_compressed_block(const void* data, size_t size);
};

class ReplayReader {
public:
    ReplayReader();
    ~ReplayReader();
    
    bool open(const std::string& path);
    bool read_frame(uint32_t frame_index, std::vector<ASCIICell>& cells);
    bool seek_frame(uint32_t frame_index);
    void close();
    
    const ReplayHeader& header() const { return header_; }
    uint32_t frame_count() const { return header_.frame_count; }
    int cols() const { return static_cast<int>(header_.cols); }
    int rows() const { return static_cast<int>(header_.rows); }
    std::string config_hash() const { return std::string(header_.config_hash, 8); }
    bool is_open() const { return file_ != nullptr; }
    
private:
    FILE* file_ = nullptr;
    ReplayHeader header_;
    std::vector<uint8_t> decompress_buffer_;
    std::vector<ASCIICell> last_cells_;
    std::vector<uint64_t> frame_offsets_;
    
    bool read_header();
    bool read_compressed_block(std::vector<uint8_t>& out, size_t expected_size);
    bool build_frame_index();
};

}
