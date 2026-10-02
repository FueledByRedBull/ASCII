#pragma once

#include "core/types.hpp"
#include "core/color_space.hpp"
#include "terminal/terminal.hpp"
#include <vector>
#include <cstdint>

namespace ascii {

struct BlockCell {
    uint32_t codepoint = 0x0020;
    uint8_t fg_r = 255, fg_g = 255, fg_b = 255;
    uint8_t bg_r = 0, bg_g = 0, bg_b = 0;
};

class BlockRenderer {
public:
    struct Config {
        bool use_half_blocks = true;
        bool use_quarter_blocks = true;
        bool use_eighth_blocks = false;
        int color_quantization_levels = 16;  // Nonpositive disables; positive values are clamped to 2-256.
    };
    
    BlockRenderer() = default;
    explicit BlockRenderer(const Config& config);
    
    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }
    
    void set_grid_size(int cols, int rows);
    
    struct CellData {
        float mean_r = 0.0f;
        float mean_g = 0.0f;
        float mean_b = 0.0f;
        float mean_luminance = 0.0f;
        
        float top_left_lum = 0.0f;
        float top_right_lum = 0.0f;
        float bottom_left_lum = 0.0f;
        float bottom_right_lum = 0.0f;
        
        float top_left_r = 0.0f, top_left_g = 0.0f, top_left_b = 0.0f;
        float top_right_r = 0.0f, top_right_g = 0.0f, top_right_b = 0.0f;
        float bottom_left_r = 0.0f, bottom_left_g = 0.0f, bottom_left_b = 0.0f;
        float bottom_right_r = 0.0f, bottom_right_g = 0.0f, bottom_right_b = 0.0f;
        
        bool is_edge_cell = false;
    };

    CellData analyze_cell(const FloatImage& luminance,
                          const FrameBuffer& color_buffer,
                          int cell_x,
                          int cell_y,
                          int cell_width,
                          int cell_height,
                          const CellStats& stats) const;
    
    BlockCell render_cell(const CellData& data) const;
    
    std::vector<BlockCell> render_frame(const std::vector<CellData>& cells) const;
    // Uses bounded deterministic OKLab clustering; the existing API name is retained.
    void spectral_quantize_frame(std::vector<BlockCell>& cells, int palette_size,
                                 int max_samples, int iterations) const;
    
    std::string render_to_ansi(const std::vector<BlockCell>& cells, ColorMode mode,
                                const std::vector<BlockCell>* prev_cells = nullptr) const;
    
    int cols() const { return cols_; }
    int rows() const { return rows_; }
    
private:
    Config config_;
    int cols_ = 80;
    int rows_ = 24;
    
    void quantize_colors(uint8_t& r, uint8_t& g, uint8_t& b) const;
    
    std::string codepoint_to_utf8(uint32_t cp) const;
};

}
