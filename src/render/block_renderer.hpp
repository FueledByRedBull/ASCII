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
        int color_quantization_levels = 16;
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
    
    BlockCell render_cell(const CellData& data) const;
    
    std::vector<BlockCell> render_frame(const std::vector<CellData>& cells) const;
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
    mutable std::vector<BlockCell> prev_cells_;
    
    static constexpr uint32_t BLOCK_FULL = 0x2588;
    static constexpr uint32_t BLOCK_DARK = 0x2593;
    static constexpr uint32_t BLOCK_MEDIUM = 0x2592;
    static constexpr uint32_t BLOCK_LIGHT = 0x2591;
    static constexpr uint32_t BLOCK_UPPER = 0x2580;
    static constexpr uint32_t BLOCK_LOWER = 0x2584;
    static constexpr uint32_t BLOCK_LEFT = 0x258C;
    static constexpr uint32_t BLOCK_RIGHT = 0x2590;
    static constexpr uint32_t BLOCK_QUARTER_LL = 0x2596;
    static constexpr uint32_t BLOCK_QUARTER_LR = 0x2597;
    static constexpr uint32_t BLOCK_QUARTER_UL = 0x2598;
    static constexpr uint32_t BLOCK_QUARTER_UR = 0x259D;
    static constexpr uint32_t BLOCK_QUARTER_LEFT = 0x2599;
    static constexpr uint32_t BLOCK_QUARTER_RIGHT = 0x259F;
    static constexpr uint32_t BLOCK_SPACE = 0x0020;
    
    struct ColorPair {
        LinearColor fg;
        LinearColor bg;
        float error;
    };
    
    ColorPair find_best_color_pair(const CellData& data, float coverage) const;
    
    float compute_color_error(const LinearColor& c1, const LinearColor& target) const;
    
    uint32_t select_block_glyph(float coverage) const;
    
    uint32_t select_half_block(float top_lum, float bottom_lum) const;
    
    uint32_t select_quarter_block(const CellData& data) const;
    
    void quantize_colors(uint8_t& r, uint8_t& g, uint8_t& b) const;
    
    std::string codepoint_to_utf8(uint32_t cp) const;
};

}
