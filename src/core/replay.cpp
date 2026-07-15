#include "core/replay.hpp"
#include <zstd.h>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>

#ifdef _WIN32
#include <windows.h>
#endif

namespace ascii {

constexpr size_t COMPRESS_BUFFER_SIZE = 256 * 1024;
constexpr int ZSTD_COMPRESSION_LEVEL = 3;

bool checked_multiply(size_t a, size_t b, size_t& result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
    result = a * b;
    return true;
}

bool file_seek(FILE* file, int64_t offset, int origin) {
#ifdef _WIN32
    return _fseeki64(file, offset, origin) == 0;
#else
    return fseeko(file, static_cast<off_t>(offset), origin) == 0;
#endif
}

int64_t file_tell(FILE* file) {
#ifdef _WIN32
    return _ftelli64(file);
#else
    return static_cast<int64_t>(ftello(file));
#endif
}

bool replace_file(const std::string& from, const std::string& to) {
#ifdef _WIN32
    return MoveFileExA(from.c_str(), to.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

std::string temporary_replay_path(const std::string& target) {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    return target + ".tmp-" + std::to_string(nonce);
}

bool valid_codepoint(uint32_t codepoint) {
    return codepoint <= 0x10FFFFu && !(codepoint >= 0xD800u && codepoint <= 0xDFFFu);
}

ReplayWriter::ReplayWriter() {
    compress_buffer_.resize(COMPRESS_BUFFER_SIZE);
}

ReplayWriter::~ReplayWriter() {
    close();
}

bool ReplayWriter::open(const std::string& path, int cols, int rows, int fps, const std::string& config_hash) {
    close();
    if (path.empty() || cols <= 0 || rows <= 0 || fps <= 0 ||
        cols > static_cast<int>(REPLAY_MAX_COLS) || rows > static_cast<int>(REPLAY_MAX_ROWS) ||
        fps > static_cast<int>(REPLAY_MAX_FPS) ||
        static_cast<uint64_t>(cols) * static_cast<uint64_t>(rows) > REPLAY_MAX_CELLS) {
        return false;
    }

    cols_ = cols;
    rows_ = rows;
    frame_count_ = 0;
    last_cells_.clear();
    last_cells_.resize(static_cast<size_t>(cols) * rows);
    target_path_ = path;
    temporary_path_ = temporary_replay_path(path);
    failed_ = false;

    file_ = fopen(temporary_path_.c_str(), "wb");
    if (!file_) return false;
    
    ReplayHeader hdr;
    hdr.cols = static_cast<uint32_t>(cols);
    hdr.rows = static_cast<uint32_t>(rows);
    hdr.fps = static_cast<uint32_t>(fps);
    std::strncpy(hdr.config_hash, config_hash.c_str(), 8);
    hdr.config_hash[8] = '\0';
    
    if (fwrite(&hdr, sizeof(hdr), 1, file_) != 1) {
        failed_ = true;
        close();
        return false;
    }
    
    return true;
}

bool ReplayWriter::write_frame(uint32_t frame_index, const std::vector<ASCIICell>& cells) {
    if (!file_ || frame_index != frame_count_ || cells.size() != static_cast<size_t>(cols_ * rows_)) {
        failed_ = true;
        return false;
    }
    
    std::vector<ReplayCellData> cell_data(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        if (!valid_codepoint(cells[i].codepoint)) { failed_ = true; return false; }
        cell_data[i].glyph_index = cells[i].codepoint;
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
        failed_ = true;
        return false;
    }
    
    ReplayFrameHeader frame_hdr;
    frame_hdr.frame_index = frame_index;
    frame_hdr.data_size = static_cast<uint32_t>(compressed_size);
    frame_hdr.changed_cells = static_cast<uint32_t>(cells.size());
    frame_hdr.flags = REPLAY_FRAME_FULL;
    
    if (fwrite(&frame_hdr, sizeof(frame_hdr), 1, file_) != 1) {
        failed_ = true;
        return false;
    }
    
    if (fwrite(compress_buffer_.data(), 1, compressed_size, file_) != compressed_size) {
        failed_ = true;
        return false;
    }
    
    last_cells_ = cells;
    frame_count_++;
    
    ReplayHeader hdr_update;
    hdr_update.frame_count = frame_count_;
    if (!file_seek(file_, offsetof(ReplayHeader, frame_count), SEEK_SET)) { failed_ = true; return false; }
    if (fwrite(&hdr_update.frame_count, sizeof(hdr_update.frame_count), 1, file_) != 1) {
        failed_ = true;
        return false;
    }
    if (!file_seek(file_, 0, SEEK_END)) { failed_ = true; return false; }
    
    return true;
}

bool ReplayWriter::write_frame_delta(uint32_t frame_index, const std::vector<ASCIICell>& cells,
                                     const std::vector<ASCIICell>& prev_cells) {
    if (!file_ || frame_count_ == 0 || frame_index != frame_count_ ||
        cells.size() != static_cast<size_t>(cols_ * rows_) || prev_cells.size() != cells.size()) {
        failed_ = true;
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
            
            if (!valid_codepoint(curr.codepoint)) { failed_ = true; return false; }
            ReplayCellData cd;
            cd.glyph_index = curr.codepoint;
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
        failed_ = true;
        return false;
    }
    
    ReplayFrameHeader frame_hdr;
    frame_hdr.frame_index = frame_index;
    frame_hdr.data_size = static_cast<uint32_t>(compressed_size);
    frame_hdr.changed_cells = static_cast<uint32_t>(changes.size());
    frame_hdr.flags = REPLAY_FRAME_DELTA;
    
    if (fwrite(&frame_hdr, sizeof(frame_hdr), 1, file_) != 1) {
        failed_ = true;
        return false;
    }
    
    if (fwrite(compressed.data(), 1, compressed_size, file_) != compressed_size) {
        failed_ = true;
        return false;
    }
    
    last_cells_ = cells;
    frame_count_++;

    ReplayHeader hdr_update;
    hdr_update.frame_count = frame_count_;
    if (!file_seek(file_, offsetof(ReplayHeader, frame_count), SEEK_SET)) { failed_ = true; return false; }
    if (fwrite(&hdr_update.frame_count, sizeof(hdr_update.frame_count), 1, file_) != 1) {
        failed_ = true;
        return false;
    }
    if (!file_seek(file_, 0, SEEK_END)) { failed_ = true; return false; }
    
    return true;
}

bool ReplayWriter::close() {
    bool success = !failed_;
    if (file_) {
        if (std::fflush(file_) != 0) success = false;
        if (fclose(file_) != 0) success = false;
        file_ = nullptr;
    }
    if (!temporary_path_.empty()) {
        if (success && frame_count_ > 0) {
            success = replace_file(temporary_path_, target_path_);
        }
        if (!success || frame_count_ == 0) {
            std::error_code ec;
            std::filesystem::remove(temporary_path_, ec);
        }
    }
    frame_count_ = 0;
    target_path_.clear();
    temporary_path_.clear();
    failed_ = false;
    return success;
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
    const uint64_t total_cells = static_cast<uint64_t>(header_.cols) * header_.rows;
    if (header_.version != REPLAY_VERSION ||
        header_.cols == 0 || header_.cols > REPLAY_MAX_COLS ||
        header_.rows == 0 || header_.rows > REPLAY_MAX_ROWS ||
        total_cells > REPLAY_MAX_CELLS ||
        header_.fps == 0 || header_.fps > REPLAY_MAX_FPS) {
        fclose(file_);
        file_ = nullptr;
        return false;
    }
    
    last_cells_.resize(static_cast<size_t>(total_cells));
    
    if (!build_frame_index()) {
        fclose(file_);
        file_ = nullptr;
        return false;
    }
    if (frame_offsets_.size() != header_.frame_count) {
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
    
    static constexpr char expected_magic[8] = {'A', 'R', 'E', 'P', 'L', 'A', 'Y', '\0'};
    if (std::memcmp(header_.magic, expected_magic, sizeof(expected_magic)) != 0) {
        return false;
    }
    
    return true;
}

bool ReplayReader::build_frame_index() {
    frame_offsets_.clear();
    
    if (!file_seek(file_, 0, SEEK_END)) return false;
    const int64_t file_size = file_tell(file_);
    if (file_size < static_cast<int64_t>(sizeof(ReplayHeader)) ||
        !file_seek(file_, sizeof(ReplayHeader), SEEK_SET)) return false;
    
    ReplayFrameHeader frame_hdr;
    const uint32_t total_cells = header_.cols * header_.rows;
    for (uint32_t expected_index = 0; expected_index < header_.frame_count; ++expected_index) {
        const int64_t offset = file_tell(file_);
        if (offset < 0 || offset > file_size - static_cast<int64_t>(sizeof(frame_hdr)) ||
            fread(&frame_hdr, sizeof(frame_hdr), 1, file_) != 1) return false;
        const uint32_t known_flags = REPLAY_FRAME_FULL | REPLAY_FRAME_DELTA;
        const bool full = frame_hdr.flags == REPLAY_FRAME_FULL;
        const bool delta = frame_hdr.flags == REPLAY_FRAME_DELTA;
        if ((frame_hdr.flags & ~known_flags) != 0 || (!full && !delta) ||
            frame_hdr.frame_index != expected_index ||
            (expected_index == 0 && !full) ||
            (full && frame_hdr.changed_cells != total_cells) ||
            (delta && frame_hdr.changed_cells > total_cells) ||
            frame_hdr.data_size == 0) return false;
        size_t decoded_size = 0;
        const size_t record_size = delta
            ? sizeof(uint32_t) + sizeof(ReplayCellData)
            : sizeof(ReplayCellData);
        const size_t records = delta ? frame_hdr.changed_cells : total_cells;
        if (!checked_multiply(records, record_size, decoded_size) ||
            frame_hdr.data_size > ZSTD_compressBound(decoded_size)) return false;
        const int64_t data_end = file_tell(file_) + static_cast<int64_t>(frame_hdr.data_size);
        if (data_end < 0 || data_end > file_size) return false;
        frame_offsets_.push_back(static_cast<uint64_t>(offset));
        if (!file_seek(file_, data_end, SEEK_SET)) return false;
    }
    for (uint32_t reserved : header_.reserved) {
        if (reserved != 0) return false;
    }
    if (file_tell(file_) != file_size) return false;
    return file_seek(file_, sizeof(ReplayHeader), SEEK_SET);
}

bool ReplayReader::read_frame(uint32_t frame_index, std::vector<ASCIICell>& cells) {
    if (!file_ || frame_index >= frame_offsets_.size()) {
        return false;
    }
    
    if (decoded_frame_index_ + 1 != static_cast<int64_t>(frame_index)) {
        reset_decode_state();
        std::vector<ASCIICell> intermediate;
        for (uint32_t i = 0; i < frame_index; ++i) {
            if (!decode_indexed_frame(i, intermediate)) return false;
            decoded_frame_index_ = i;
        }
    }
    if (!decode_indexed_frame(frame_index, cells)) return false;
    decoded_frame_index_ = frame_index;
    return true;
}

bool ReplayReader::decode_indexed_frame(uint32_t frame_index, std::vector<ASCIICell>& cells) {
    if (!file_seek(file_, static_cast<int64_t>(frame_offsets_[frame_index]), SEEK_SET)) {
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
    const bool is_delta = frame_hdr.flags == REPLAY_FRAME_DELTA;
    
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
        std::vector<bool> seen(cells.size(), false);
        for (uint32_t i = 0; i < frame_hdr.changed_cells; ++i) {
            uint32_t idx;
            std::memcpy(&idx, ptr, sizeof(idx));
            ptr += sizeof(idx);
            
            ReplayCellData cd;
            std::memcpy(&cd, ptr, sizeof(cd));
            ptr += sizeof(cd);
            
            if (idx >= cells.size() || seen[idx] || !valid_codepoint(cd.glyph_index)) return false;
            seen[idx] = true;
            cells[idx].codepoint = cd.glyph_index;
            cells[idx].fg_r = cd.fg_r;
            cells[idx].fg_g = cd.fg_g;
            cells[idx].fg_b = cd.fg_b;
            cells[idx].bg_r = cd.bg_r;
            cells[idx].bg_g = cd.bg_g;
            cells[idx].bg_b = cd.bg_b;
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
        for (size_t i = 0; i < cells.size(); ++i) {
            ReplayCellData cell_data;
            std::memcpy(&cell_data,
                        decompress_buffer_.data() + i * sizeof(ReplayCellData),
                        sizeof(cell_data));
            if (!valid_codepoint(cell_data.glyph_index)) return false;
            cells[i].codepoint = cell_data.glyph_index;
            cells[i].fg_r = cell_data.fg_r;
            cells[i].fg_g = cell_data.fg_g;
            cells[i].fg_b = cell_data.fg_b;
            cells[i].bg_r = cell_data.bg_r;
            cells[i].bg_g = cell_data.bg_g;
            cells[i].bg_b = cell_data.bg_b;
        }
    }
    
    last_cells_ = cells;
    return true;
}

bool ReplayReader::seek_frame(uint32_t frame_index) {
    if (!file_ || frame_index >= frame_offsets_.size()) {
        return false;
    }
    
    reset_decode_state();
    return file_seek(file_, static_cast<int64_t>(frame_offsets_[frame_index]), SEEK_SET);
}

void ReplayReader::close() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
    frame_offsets_.clear();
    decoded_frame_index_ = -1;
}

void ReplayReader::reset_decode_state() {
    last_cells_.assign(static_cast<size_t>(header_.cols) * header_.rows, ASCIICell{});
    decoded_frame_index_ = -1;
}

}
