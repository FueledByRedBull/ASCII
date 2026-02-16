#include "core/replay.hpp"
#include <zstd.h>
#include <cstddef>
#include <cstring>
#include <algorithm>

namespace ascii {

constexpr size_t COMPRESS_BUFFER_SIZE = 256 * 1024;
constexpr int ZSTD_COMPRESSION_LEVEL = 3;

ReplayWriter::ReplayWriter() {
    compress_buffer_.resize(COMPRESS_BUFFER_SIZE);
}

ReplayWriter::~ReplayWriter() {
    close();
}

bool ReplayWriter::open(const std::string& path, int cols, int rows, int fps, const std::string& config_hash) {
    close();
    
    cols_ = cols;
    rows_ = rows;
    frame_count_ = 0;
    last_cells_.clear();
    last_cells_.resize(cols * rows);
    
    file_ = fopen(path.c_str(), "wb");
    if (!file_) return false;
    
    ReplayHeader hdr;
    hdr.cols = static_cast<uint32_t>(cols);
    hdr.rows = static_cast<uint32_t>(rows);
    hdr.fps = static_cast<uint32_t>(fps);
    std::strncpy(hdr.config_hash, config_hash.c_str(), 8);
    hdr.config_hash[8] = '\0';
    
    if (fwrite(&hdr, sizeof(hdr), 1, file_) != 1) {
        fclose(file_);
        file_ = nullptr;
        return false;
    }
    
    return true;
}

bool ReplayWriter::write_frame(uint32_t frame_index, const std::vector<ASCIICell>& cells) {
    if (!file_ || cells.size() != static_cast<size_t>(cols_ * rows_)) {
        return false;
    }
    
    std::vector<ReplayCellData> cell_data(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        cell_data[i].glyph_index = static_cast<uint16_t>(cells[i].codepoint);
        cell_data[i].fg_r = cells[i].fg_r;
        cell_data[i].fg_g = cells[i].fg_g;
        cell_data[i].fg_b = cells[i].fg_b;
        cell_data[i].bg_r = cells[i].bg_r;
        cell_data[i].bg_g = cells[i].bg_g;
        cell_data[i].bg_b = cells[i].bg_b;
    }
    
    size_t src_size = cell_data.size() * sizeof(ReplayCellData);
    size_t bound = ZSTD_compressBound(src_size);
    if (bound > compress_buffer_.size()) {
        compress_buffer_.resize(bound);
    }
    
    size_t compressed_size = ZSTD_compress(
        compress_buffer_.data(), compress_buffer_.size(),
        cell_data.data(), src_size,
        ZSTD_COMPRESSION_LEVEL
    );
    
    if (ZSTD_isError(compressed_size)) {
        return false;
    }
    
    ReplayFrameHeader frame_hdr;
    frame_hdr.frame_index = frame_index;
    frame_hdr.data_size = static_cast<uint32_t>(compressed_size);
    frame_hdr.changed_cells = static_cast<uint32_t>(cells.size());
    frame_hdr.flags = REPLAY_FRAME_FULL;
    
    if (fwrite(&frame_hdr, sizeof(frame_hdr), 1, file_) != 1) {
        return false;
    }
    
    if (fwrite(compress_buffer_.data(), 1, compressed_size, file_) != compressed_size) {
        return false;
    }
    
    last_cells_ = cells;
    frame_count_++;
    
    ReplayHeader hdr_update;
    hdr_update.frame_count = frame_count_;
    fseek(file_, offsetof(ReplayHeader, frame_count), SEEK_SET);
    if (fwrite(&hdr_update.frame_count, sizeof(hdr_update.frame_count), 1, file_) != 1) {
        return false;
    }
    fseek(file_, 0, SEEK_END);
    
    return true;
}

bool ReplayWriter::write_frame_delta(uint32_t frame_index, const std::vector<ASCIICell>& cells,
                                     const std::vector<ASCIICell>& prev_cells) {
    if (!file_ || cells.size() != static_cast<size_t>(cols_ * rows_) || prev_cells.size() != cells.size()) {
        return false;
    }
    
    std::vector<std::pair<uint32_t, ReplayCellData>> changes;
    changes.reserve(cells.size() / 4);
    
    for (size_t i = 0; i < cells.size(); ++i) {
        const auto& curr = cells[i];
        const auto& prev = prev_cells[i];
        
        if (curr.codepoint != prev.codepoint ||
            curr.fg_r != prev.fg_r || curr.fg_g != prev.fg_g || curr.fg_b != prev.fg_b ||
            curr.bg_r != prev.bg_r || curr.bg_g != prev.bg_g || curr.bg_b != prev.bg_b) {
            
            ReplayCellData cd;
            cd.glyph_index = static_cast<uint16_t>(curr.codepoint);
            cd.fg_r = curr.fg_r;
            cd.fg_g = curr.fg_g;
            cd.fg_b = curr.fg_b;
            cd.bg_r = curr.bg_r;
            cd.bg_g = curr.bg_g;
            cd.bg_b = curr.bg_b;
            changes.emplace_back(static_cast<uint32_t>(i), cd);
        }
    }
    
    size_t src_size = changes.size() * (sizeof(uint32_t) + sizeof(ReplayCellData));
    if (src_size > compress_buffer_.size()) {
        compress_buffer_.resize(src_size * 2);
    }
    
    uint8_t* ptr = compress_buffer_.data();
    for (const auto& [idx, cd] : changes) {
        std::memcpy(ptr, &idx, sizeof(idx));
        ptr += sizeof(idx);
        std::memcpy(ptr, &cd, sizeof(cd));
        ptr += sizeof(cd);
    }
    
    size_t bound = ZSTD_compressBound(src_size);
    std::vector<uint8_t> compressed(bound);
    
    size_t compressed_size = ZSTD_compress(
        compressed.data(), compressed.size(),
        compress_buffer_.data(), src_size,
        ZSTD_COMPRESSION_LEVEL
    );
    
    if (ZSTD_isError(compressed_size)) {
        return false;
    }
    
    ReplayFrameHeader frame_hdr;
    frame_hdr.frame_index = frame_index;
    frame_hdr.data_size = static_cast<uint32_t>(compressed_size);
    frame_hdr.changed_cells = static_cast<uint32_t>(changes.size());
    frame_hdr.flags = REPLAY_FRAME_DELTA;
    
    if (fwrite(&frame_hdr, sizeof(frame_hdr), 1, file_) != 1) {
        return false;
    }
    
    if (fwrite(compressed.data(), 1, compressed_size, file_) != compressed_size) {
        return false;
    }
    
    last_cells_ = cells;
    frame_count_++;

    ReplayHeader hdr_update;
    hdr_update.frame_count = frame_count_;
    fseek(file_, offsetof(ReplayHeader, frame_count), SEEK_SET);
    if (fwrite(&hdr_update.frame_count, sizeof(hdr_update.frame_count), 1, file_) != 1) {
        return false;
    }
    fseek(file_, 0, SEEK_END);
    
    return true;
}

void ReplayWriter::close() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
    frame_count_ = 0;
}

ReplayReader::ReplayReader() {
    decompress_buffer_.resize(COMPRESS_BUFFER_SIZE);
}

ReplayReader::~ReplayReader() {
    close();
}

bool ReplayReader::open(const std::string& path) {
    close();
    
    file_ = fopen(path.c_str(), "rb");
    if (!file_) return false;
    
    if (!read_header()) {
        fclose(file_);
        file_ = nullptr;
        return false;
    }
    
    last_cells_.resize(header_.cols * header_.rows);
    
    if (!build_frame_index()) {
        fclose(file_);
        file_ = nullptr;
        return false;
    }
    
    return true;
}

bool ReplayReader::read_header() {
    if (fread(&header_, sizeof(header_), 1, file_) != 1) {
        return false;
    }
    
    if (std::memcmp(header_.magic, "AREPLAY", 7) != 0) {
        return false;
    }
    
    return true;
}

bool ReplayReader::build_frame_index() {
    frame_offsets_.clear();
    
    fseek(file_, sizeof(ReplayHeader), SEEK_SET);
    
    ReplayFrameHeader frame_hdr;
    while (fread(&frame_hdr, sizeof(frame_hdr), 1, file_) == 1) {
        frame_offsets_.push_back(static_cast<uint64_t>(ftell(file_) - sizeof(frame_hdr)));
        
        if (fseek(file_, frame_hdr.data_size, SEEK_CUR) != 0) {
            break;
        }
    }
    
    fseek(file_, sizeof(ReplayHeader), SEEK_SET);
    return true;
}

bool ReplayReader::read_frame(uint32_t frame_index, std::vector<ASCIICell>& cells) {
    if (!file_ || frame_index >= frame_offsets_.size()) {
        return false;
    }
    
    if (fseek(file_, static_cast<long>(frame_offsets_[frame_index]), SEEK_SET) != 0) {
        return false;
    }
    
    ReplayFrameHeader frame_hdr;
    if (fread(&frame_hdr, sizeof(frame_hdr), 1, file_) != 1) {
        return false;
    }
    
    std::vector<uint8_t> compressed(frame_hdr.data_size);
    if (fread(compressed.data(), 1, frame_hdr.data_size, file_) != frame_hdr.data_size) {
        return false;
    }
    
    const uint32_t total_cells = header_.cols * header_.rows;
    bool is_delta = false;
    if ((frame_hdr.flags & (REPLAY_FRAME_FULL | REPLAY_FRAME_DELTA)) != 0) {
        is_delta = (frame_hdr.flags & REPLAY_FRAME_DELTA) != 0;
    } else {
        // Backward compatibility for old files without explicit frame flags.
        is_delta = (frame_hdr.changed_cells != total_cells);
    }
    
    if (is_delta) {
        size_t src_size = static_cast<size_t>(frame_hdr.changed_cells) * (sizeof(uint32_t) + sizeof(ReplayCellData));
        if (src_size > decompress_buffer_.size()) {
            decompress_buffer_.resize(src_size);
        }
        
        size_t result = ZSTD_decompress(
            decompress_buffer_.data(), decompress_buffer_.size(),
            compressed.data(), compressed.size()
        );
        
        if (ZSTD_isError(result)) {
            return false;
        }
        if (result != src_size) {
            return false;
        }
        
        if (last_cells_.size() != total_cells) {
            last_cells_.assign(total_cells, ASCIICell{});
        }
        cells = last_cells_;
        
        const uint8_t* ptr = decompress_buffer_.data();
        for (uint32_t i = 0; i < frame_hdr.changed_cells; ++i) {
            uint32_t idx;
            std::memcpy(&idx, ptr, sizeof(idx));
            ptr += sizeof(idx);
            
            ReplayCellData cd;
            std::memcpy(&cd, ptr, sizeof(cd));
            ptr += sizeof(cd);
            
            if (idx < cells.size()) {
                cells[idx].codepoint = cd.glyph_index;
                cells[idx].fg_r = cd.fg_r;
                cells[idx].fg_g = cd.fg_g;
                cells[idx].fg_b = cd.fg_b;
                cells[idx].bg_r = cd.bg_r;
                cells[idx].bg_g = cd.bg_g;
                cells[idx].bg_b = cd.bg_b;
            }
        }
    } else {
        size_t src_size = static_cast<size_t>(header_.cols) * header_.rows * sizeof(ReplayCellData);
        if (src_size > decompress_buffer_.size()) {
            decompress_buffer_.resize(src_size);
        }
        
        size_t result = ZSTD_decompress(
            decompress_buffer_.data(), decompress_buffer_.size(),
            compressed.data(), compressed.size()
        );
        
        if (ZSTD_isError(result)) {
            return false;
        }
        if (result != src_size) {
            return false;
        }
        
        cells.resize(header_.cols * header_.rows);
        const ReplayCellData* cell_data = reinterpret_cast<const ReplayCellData*>(decompress_buffer_.data());
        
        for (size_t i = 0; i < cells.size(); ++i) {
            cells[i].codepoint = cell_data[i].glyph_index;
            cells[i].fg_r = cell_data[i].fg_r;
            cells[i].fg_g = cell_data[i].fg_g;
            cells[i].fg_b = cell_data[i].fg_b;
            cells[i].bg_r = cell_data[i].bg_r;
            cells[i].bg_g = cell_data[i].bg_g;
            cells[i].bg_b = cell_data[i].bg_b;
        }
    }
    
    last_cells_ = cells;
    return true;
}

bool ReplayReader::seek_frame(uint32_t frame_index) {
    if (!file_ || frame_index >= frame_offsets_.size()) {
        return false;
    }
    
    return fseek(file_, static_cast<long>(frame_offsets_[frame_index]), SEEK_SET) == 0;
}

void ReplayReader::close() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
    frame_offsets_.clear();
}

}
