#include "block_renderer.hpp"
#include "mapping/color_mapper.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <array>
#include <limits>
#include <vector>

namespace ascii {

BlockRenderer::BlockRenderer(const Config& config) : config_(config) {}

void BlockRenderer::set_grid_size(int cols, int rows) {
    cols_ = cols;
    rows_ = rows;
}

BlockRenderer::CellData BlockRenderer::analyze_cell(const FloatImage& luminance,
                                                    const FrameBuffer& color_buffer,
                                                    int cell_x,
                                                    int cell_y,
                                                    int cell_width,
                                                    int cell_height,
                                                    const CellStats& stats) const {
    CellData data;
    data.mean_r = stats.mean_r;
    data.mean_g = stats.mean_g;
    data.mean_b = stats.mean_b;
    data.mean_luminance = stats.mean_luminance;
    data.is_edge_cell = stats.is_edge_cell;

    const int px0 = cell_x * cell_width;
    const int py0 = cell_y * cell_height;
    const int px1 = std::min(px0 + cell_width, luminance.width());
    const int py1 = std::min(py0 + cell_height, luminance.height());
    const int pmx = (px0 + px1) / 2;
    const int pmy = (py0 + py1) / 2;

    auto accumulate_quad = [&](int sx0, int sy0, int sx1, int sy1,
                               float& out_lum, float& out_r, float& out_g, float& out_b) {
        double sum_l = 0.0;
        double sum_r = 0.0;
        double sum_g = 0.0;
        double sum_b = 0.0;
        int count = 0;

        for (int yy = sy0; yy < sy1; ++yy) {
            for (int xx = sx0; xx < sx1; ++xx) {
                sum_l += luminance.get(xx, yy);
                Color c = color_buffer.get_pixel(xx, yy);
                LinearColor lc = ColorSpace::srgb_to_linear(c.r, c.g, c.b);
                sum_r += lc.r;
                sum_g += lc.g;
                sum_b += lc.b;
                ++count;
            }
        }

        if (count > 0) {
            const float inv = 1.0f / static_cast<float>(count);
            out_lum = static_cast<float>(sum_l) * inv;
            out_r = static_cast<float>(sum_r) * inv;
            out_g = static_cast<float>(sum_g) * inv;
            out_b = static_cast<float>(sum_b) * inv;
        } else {
            out_lum = data.mean_luminance;
            out_r = data.mean_r;
            out_g = data.mean_g;
            out_b = data.mean_b;
        }
    };

    accumulate_quad(px0, py0, pmx, pmy,
                    data.top_left_lum, data.top_left_r, data.top_left_g, data.top_left_b);
    accumulate_quad(pmx, py0, px1, pmy,
                    data.top_right_lum, data.top_right_r, data.top_right_g, data.top_right_b);
    accumulate_quad(px0, pmy, pmx, py1,
                    data.bottom_left_lum, data.bottom_left_r, data.bottom_left_g, data.bottom_left_b);
    accumulate_quad(pmx, pmy, px1, py1,
                    data.bottom_right_lum, data.bottom_right_r, data.bottom_right_g, data.bottom_right_b);

    return data;
}

void BlockRenderer::quantize_colors(uint8_t& r, uint8_t& g, uint8_t& b) const {
    if (config_.color_quantization_levels <= 0) return;
    
    int levels = std::clamp(config_.color_quantization_levels, 2, 256);
    float step = 255.0f / (levels - 1);
    
    auto quantize = [levels, step](uint8_t v) -> uint8_t {
        int idx = static_cast<int>(std::round(v / step));
        idx = std::clamp(idx, 0, levels - 1);
        return static_cast<uint8_t>(idx * step);
    };
    
    r = quantize(r);
    g = quantize(g);
    b = quantize(b);
}

std::string BlockRenderer::codepoint_to_utf8(uint32_t cp) const {
    if (cp < 0x20 || (cp >= 0x7F && cp <= 0x9F) ||
        (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
        cp = 0xFFFD;
    }
    std::string result;
    if (cp < 0x80) {
        result += static_cast<char>(cp);
    } else if (cp < 0x800) {
        result += static_cast<char>(0xC0 | (cp >> 6));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        result += static_cast<char>(0xE0 | (cp >> 12));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        result += static_cast<char>(0xF0 | (cp >> 18));
        result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        result += static_cast<char>(0x80 | (cp & 0x3F));
    }
    return result;
}

BlockCell BlockRenderer::render_cell(const CellData& data) const {
    // Bits describe top-left, top-right, bottom-left, and bottom-right coverage.
    static constexpr std::array<uint32_t, 16> glyphs = {
        0x0020, 0x2598, 0x259D, 0x2580, 0x2596, 0x258C, 0x259E, 0x259B,
        0x2597, 0x259A, 0x2590, 0x259C, 0x2584, 0x2599, 0x259F, 0x2588
    };
    const std::array<LinearColor, 4> colors = {{
        {data.top_left_r, data.top_left_g, data.top_left_b},
        {data.top_right_r, data.top_right_g, data.top_right_b},
        {data.bottom_left_r, data.bottom_left_g, data.bottom_left_b},
        {data.bottom_right_r, data.bottom_right_g, data.bottom_right_b}
    }};
    BlockCell best;
    float best_error = std::numeric_limits<float>::max();
    for (unsigned mask = 0; mask < glyphs.size(); ++mask) {
        const bool half = mask == 3 || mask == 5 || mask == 10 || mask == 12;
        if (half && !config_.use_half_blocks) continue;
        if (mask != 0 && mask != 15 && !half && !config_.use_quarter_blocks) continue;
        LinearColor fg, bg;
        int foreground_count = 0;
        for (unsigned q = 0; q < colors.size(); ++q) {
            if (mask & (1u << q)) {
                fg = fg + colors[q];
                ++foreground_count;
            } else {
                bg = bg + colors[q];
            }
        }
        if (foreground_count) fg = fg * (1.0f / foreground_count);
        if (foreground_count < 4) bg = bg * (1.0f / (4 - foreground_count));
        if (foreground_count == 0) fg = bg;
        if (foreground_count == 4) bg = fg;

        BlockCell candidate;
        candidate.codepoint = glyphs[mask];
        ColorSpace::linear_to_srgb(fg, candidate.fg_r, candidate.fg_g, candidate.fg_b);
        ColorSpace::linear_to_srgb(bg, candidate.bg_r, candidate.bg_g, candidate.bg_b);
        quantize_colors(candidate.fg_r, candidate.fg_g, candidate.fg_b);
        quantize_colors(candidate.bg_r, candidate.bg_g, candidate.bg_b);
        fg = ColorSpace::srgb_to_linear(candidate.fg_r, candidate.fg_g, candidate.fg_b);
        bg = ColorSpace::srgb_to_linear(candidate.bg_r, candidate.bg_g, candidate.bg_b);
        float error = 0.0f;
        for (unsigned q = 0; q < colors.size(); ++q) {
            const auto difference = colors[q] - ((mask & (1u << q)) ? fg : bg);
            error += difference.r * difference.r + difference.g * difference.g + difference.b * difference.b;
        }
        if (error < best_error) {
            best_error = error;
            best = candidate;
        }
    }
    return best;
}

std::vector<BlockCell> BlockRenderer::render_frame(const std::vector<CellData>& cells) const {
    std::vector<BlockCell> result(cells.size());
    
    for (size_t i = 0; i < cells.size(); ++i) {
        result[i] = render_cell(cells[i]);
    }
    
    return result;
}

void BlockRenderer::spectral_quantize_frame(std::vector<BlockCell>& cells, int palette_size,
                                            int max_samples, int iterations) const {
    if (palette_size <= 1 || cells.empty()) return;
    palette_size = std::clamp(palette_size, 2, 32);
    max_samples = std::clamp(max_samples, 8, 2048);
    iterations = std::clamp(iterations, 1, 64);
    const auto& first = cells.front();
    if (std::all_of(cells.begin(), cells.end(), [&](const BlockCell& cell) {
        return cell.fg_r == first.fg_r && cell.fg_g == first.fg_g && cell.fg_b == first.fg_b &&
               cell.bg_r == first.fg_r && cell.bg_g == first.fg_g && cell.bg_b == first.fg_b;
    })) return;

    const size_t total_colors = cells.size() * 2;
    const size_t sample_count = std::min(total_colors, static_cast<size_t>(max_samples));
    std::vector<OKLab> samples;
    samples.reserve(sample_count);
    for (size_t i = 0; i < sample_count; ++i) {
        const size_t index = i * (total_colors - 1) / (sample_count - 1);
        const auto& cell = cells[index / 2];
        samples.push_back(index % 2 == 0
            ? ColorSpace::srgb_to_oklab(cell.fg_r, cell.fg_g, cell.fg_b)
            : ColorSpace::srgb_to_oklab(cell.bg_r, cell.bg_g, cell.bg_b));
    }
    const auto distance_squared = [](const OKLab& a, const OKLab& b) {
        const float dL = a.L - b.L, da = a.a - b.a, db = a.b - b.b;
        return dL * dL + da * da + db * db;
    };

    std::vector<OKLab> centers{samples.front()};
    std::vector<float> nearest(samples.size(), std::numeric_limits<float>::max());
    while (centers.size() < static_cast<size_t>(palette_size)) {
        size_t farthest = 0;
        for (size_t i = 0; i < samples.size(); ++i) {
            nearest[i] = std::min(nearest[i], distance_squared(samples[i], centers.back()));
            if (nearest[i] > nearest[farthest]) farthest = i;
        }
        if (nearest[farthest] <= 1e-12f) break;
        centers.push_back(samples[farthest]);
    }
    const auto nearest_center = [&](const OKLab& sample) {
        size_t best = 0;
        for (size_t k = 1; k < centers.size(); ++k) {
            if (distance_squared(sample, centers[k]) < distance_squared(sample, centers[best])) best = k;
        }
        return best;
    };

    std::vector<size_t> assignments(samples.size(), centers.size());
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<OKLab> sums(centers.size());
        std::vector<int> counts(centers.size(), 0);
        bool changed = false;
        for (size_t i = 0; i < samples.size(); ++i) {
            const size_t k = nearest_center(samples[i]);
            changed |= assignments[i] != k;
            assignments[i] = k;
            sums[k].L += samples[i].L;
            sums[k].a += samples[i].a;
            sums[k].b += samples[i].b;
            ++counts[k];
        }
        for (size_t k = 0; k < centers.size(); ++k) {
            if (counts[k] > 0) {
                const float inv = 1.0f / counts[k];
                centers[k] = {sums[k].L * inv, sums[k].a * inv, sums[k].b * inv};
            }
        }
        if (!changed) break;
    }

    const auto quantize = [&](uint8_t& r, uint8_t& g, uint8_t& b) {
        const size_t k = nearest_center(ColorSpace::srgb_to_oklab(r, g, b));
        ColorSpace::oklab_to_srgb(centers[k], r, g, b);
    };
    for (auto& cell : cells) {
        quantize(cell.fg_r, cell.fg_g, cell.fg_b);
        quantize(cell.bg_r, cell.bg_g, cell.bg_b);
    }
}

std::string BlockRenderer::render_to_ansi(const std::vector<BlockCell>& cells, ColorMode mode,
                                           const std::vector<BlockCell>* prev_cells) const {
    std::ostringstream out;
    
    uint8_t last_fg_r = 255, last_fg_g = 255, last_fg_b = 255;
    uint8_t last_bg_r = 0, last_bg_g = 0, last_bg_b = 0;
    bool need_reset = true;
    
    for (int y = 0; y < rows_; ++y) {
        for (int x = 0; x < cols_; ++x) {
            int idx = y * cols_ + x;
            if (idx >= static_cast<int>(cells.size())) break;
            
            const BlockCell& cell = cells[idx];
            
            bool skip = false;
            if (mode != ColorMode::None && prev_cells && idx < static_cast<int>(prev_cells->size())) {
                const BlockCell& prev = (*prev_cells)[idx];
                skip = (cell.codepoint == prev.codepoint &&
                        cell.fg_r == prev.fg_r && cell.fg_g == prev.fg_g && cell.fg_b == prev.fg_b &&
                        cell.bg_r == prev.bg_r && cell.bg_g == prev.bg_g && cell.bg_b == prev.bg_b);
            }
            
            if (skip) {
                out << "\033[1C";
                continue;
            }
            
            bool fg_changed = (cell.fg_r != last_fg_r || cell.fg_g != last_fg_g || cell.fg_b != last_fg_b);
            bool bg_changed = (cell.bg_r != last_bg_r || cell.bg_g != last_bg_g || cell.bg_b != last_bg_b);
            
            if (need_reset || fg_changed || bg_changed) {
                switch (mode) {
                    case ColorMode::None:
                        break;
                        
                    case ColorMode::Ansi16: {
                        uint8_t fg_idx = ColorMapper::find_nearest_16_oklab(cell.fg_r, cell.fg_g, cell.fg_b);
                        uint8_t bg_idx = ColorMapper::find_nearest_16_oklab(cell.bg_r, cell.bg_g, cell.bg_b);
                        int fg_code = (fg_idx < 8) ? (30 + fg_idx) : (90 + (fg_idx - 8));
                        int bg_code = (bg_idx < 8) ? (40 + bg_idx) : (100 + (bg_idx - 8));
                        out << "\033[" << fg_code << ";" << bg_code << "m";
                        break;
                    }
                        
                    case ColorMode::Ansi256: {
                        uint8_t fg_idx = ColorMapper::find_nearest_256_oklab(cell.fg_r, cell.fg_g, cell.fg_b);
                        uint8_t bg_idx = ColorMapper::find_nearest_256_oklab(cell.bg_r, cell.bg_g, cell.bg_b);
                        out << "\033[38;5;" << static_cast<int>(fg_idx) << ";48;5;" << static_cast<int>(bg_idx) << "m";
                        break;
                    }
                        
                    case ColorMode::Truecolor:
                    case ColorMode::BlockArt:
                    default:
                        out << "\033[38;2;" << static_cast<int>(cell.fg_r) << ";" 
                            << static_cast<int>(cell.fg_g) << ";" << static_cast<int>(cell.fg_b)
                            << ";48;2;" << static_cast<int>(cell.bg_r) << ";"
                            << static_cast<int>(cell.bg_g) << ";" << static_cast<int>(cell.bg_b) << "m";
                        break;
                }
                
                last_fg_r = cell.fg_r;
                last_fg_g = cell.fg_g;
                last_fg_b = cell.fg_b;
                last_bg_r = cell.bg_r;
                last_bg_g = cell.bg_g;
                last_bg_b = cell.bg_b;
                need_reset = false;
            }
            
            out << codepoint_to_utf8(cell.codepoint);
        }
        
        if (mode != ColorMode::None) out << "\033[0m";
        out << '\n';
        need_reset = true;
        last_fg_r = 255; last_fg_g = 255; last_fg_b = 255;
        last_bg_r = 0; last_bg_g = 0; last_bg_b = 0;
    }
    
    return out.str();
}

}
