#include "block_renderer.hpp"
#include "mapping/color_mapper.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <limits>
#include <tuple>
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

BlockRenderer::ColorPair BlockRenderer::find_best_color_pair(const CellData& data, float coverage) const {
    ColorPair result;
    
    LinearColor cell_color(data.mean_r, data.mean_g, data.mean_b);
    
    result.fg = cell_color;
    result.bg = LinearColor(0.0f, 0.0f, 0.0f);
    
    if (coverage > 0.9f) {
        result.bg = result.fg;
        result.error = 0.0f;
        return result;
    }
    
    if (coverage < 0.1f) {
        result.fg = result.bg;
        result.error = 0.0f;
        return result;
    }
    
    float target_fg_r = data.mean_r / coverage;
    float target_fg_g = data.mean_g / coverage;
    float target_fg_b = data.mean_b / coverage;
    
    if (data.mean_r > coverage * 0.5f && data.mean_g > coverage * 0.5f && data.mean_b > coverage * 0.5f) {
        result.fg = LinearColor(
            std::clamp(target_fg_r, 0.0f, 1.0f),
            std::clamp(target_fg_g, 0.0f, 1.0f),
            std::clamp(target_fg_b, 0.0f, 1.0f)
        );
    }
    
    result.error = compute_color_error(result.fg * coverage + result.bg * (1.0f - coverage), cell_color);
    
    return result;
}

float BlockRenderer::compute_color_error(const LinearColor& c1, const LinearColor& c2) const {
    OKLab lab1 = ColorSpace::to_oklab(c1);
    OKLab lab2 = ColorSpace::to_oklab(c2);
    return OKLab::distance(lab1, lab2);
}

uint32_t BlockRenderer::select_block_glyph(float coverage) const {
    if (coverage < 0.0625f) return BLOCK_SPACE;
    if (coverage < 0.1875f) return BLOCK_LIGHT;
    if (coverage < 0.4375f) return BLOCK_MEDIUM;
    if (coverage < 0.6875f) return BLOCK_DARK;
    return BLOCK_FULL;
}

uint32_t BlockRenderer::select_half_block(float top_lum, float bottom_lum) const {
    float diff = top_lum - bottom_lum;
    
    if (std::abs(diff) < 0.05f) {
        float avg = (top_lum + bottom_lum) / 2.0f;
        return select_block_glyph(avg);
    }
    
    if (diff > 0.0f) {
        return BLOCK_UPPER;
    } else {
        return BLOCK_LOWER;
    }
}

uint32_t BlockRenderer::select_quarter_block(const CellData& data) const {
    if (!config_.use_quarter_blocks) {
        return select_half_block(
            (data.top_left_lum + data.top_right_lum) / 2.0f,
            (data.bottom_left_lum + data.bottom_right_lum) / 2.0f
        );
    }
    
    float tl = data.top_left_lum;
    float tr = data.top_right_lum;
    float bl = data.bottom_left_lum;
    float br = data.bottom_right_lum;
    
    float avg = (tl + tr + bl + br) / 4.0f;
    
    bool tl_high = tl > avg;
    bool tr_high = tr > avg;
    bool bl_high = bl > avg;
    bool br_high = br > avg;
    
    int high_count = (tl_high ? 1 : 0) + (tr_high ? 1 : 0) + (bl_high ? 1 : 0) + (br_high ? 1 : 0);
    
    if (high_count == 0 || high_count == 4) {
        return select_block_glyph(avg);
    }
    
    if (high_count == 1) {
        if (tl_high) return BLOCK_QUARTER_UL;
        if (tr_high) return BLOCK_QUARTER_UR;
        if (bl_high) return BLOCK_QUARTER_LL;
        if (br_high) return BLOCK_QUARTER_LR;
    }
    
    if (high_count == 3) {
        if (!tl_high) return BLOCK_QUARTER_LR;
        if (!tr_high) return BLOCK_QUARTER_LL;
        if (!bl_high) return BLOCK_QUARTER_UR;
        if (!br_high) return BLOCK_QUARTER_UL;
    }
    
    if (tl_high && bl_high) return BLOCK_QUARTER_LEFT;
    if (tr_high && br_high) return BLOCK_QUARTER_RIGHT;
    
    return select_half_block((tl + tr) / 2.0f, (bl + br) / 2.0f);
}

void BlockRenderer::quantize_colors(uint8_t& r, uint8_t& g, uint8_t& b) const {
    if (config_.color_quantization_levels <= 0) return;
    
    int levels = config_.color_quantization_levels;
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
    BlockCell result;

    auto to_srgb8 = [](float lr, float lg, float lb, uint8_t& r, uint8_t& g, uint8_t& b) {
        ColorSpace::linear_to_srgb({lr, lg, lb}, r, g, b);
    };
    
    if (config_.use_quarter_blocks && 
        std::abs(data.top_left_lum - data.top_right_lum) > 0.1f &&
        std::abs(data.top_left_lum - data.bottom_left_lum) > 0.1f) {
        
        result.codepoint = select_quarter_block(data);
        
        float fg_r = 0.0f, fg_g = 0.0f, fg_b = 0.0f;
        float bg_r = 0.0f, bg_g = 0.0f, bg_b = 0.0f;
        int fg_count = 0, bg_count = 0;
        
        float avg = (data.top_left_lum + data.top_right_lum + 
                    data.bottom_left_lum + data.bottom_right_lum) / 4.0f;
        
        auto add_color = [&](float r, float g, float b, float lum) {
            if (lum > avg) {
                fg_r += r; fg_g += g; fg_b += b;
                fg_count++;
            } else {
                bg_r += r; bg_g += g; bg_b += b;
                bg_count++;
            }
        };
        
        add_color(data.top_left_r, data.top_left_g, data.top_left_b, data.top_left_lum);
        add_color(data.top_right_r, data.top_right_g, data.top_right_b, data.top_right_lum);
        add_color(data.bottom_left_r, data.bottom_left_g, data.bottom_left_b, data.bottom_left_lum);
        add_color(data.bottom_right_r, data.bottom_right_g, data.bottom_right_b, data.bottom_right_lum);
        
        if (fg_count > 0) {
            to_srgb8(fg_r / fg_count, fg_g / fg_count, fg_b / fg_count, result.fg_r, result.fg_g, result.fg_b);
        }
        
        if (bg_count > 0) {
            to_srgb8(bg_r / bg_count, bg_g / bg_count, bg_b / bg_count, result.bg_r, result.bg_g, result.bg_b);
        }
    } else if (config_.use_half_blocks) {
        float top_avg = (data.top_left_lum + data.top_right_lum) / 2.0f;
        float bottom_avg = (data.bottom_left_lum + data.bottom_right_lum) / 2.0f;
        
        result.codepoint = select_half_block(top_avg, bottom_avg);
        
        float top_r = (data.top_left_r + data.top_right_r) / 2.0f;
        float top_g = (data.top_left_g + data.top_right_g) / 2.0f;
        float top_b = (data.top_left_b + data.top_right_b) / 2.0f;
        
        float bottom_r = (data.bottom_left_r + data.bottom_right_r) / 2.0f;
        float bottom_g = (data.bottom_left_g + data.bottom_right_g) / 2.0f;
        float bottom_b = (data.bottom_left_b + data.bottom_right_b) / 2.0f;
        
        if (result.codepoint == BLOCK_UPPER) {
            to_srgb8(top_r, top_g, top_b, result.fg_r, result.fg_g, result.fg_b);
            to_srgb8(bottom_r, bottom_g, bottom_b, result.bg_r, result.bg_g, result.bg_b);
        } else if (result.codepoint == BLOCK_LOWER) {
            to_srgb8(bottom_r, bottom_g, bottom_b, result.fg_r, result.fg_g, result.fg_b);
            to_srgb8(top_r, top_g, top_b, result.bg_r, result.bg_g, result.bg_b);
        } else {
            to_srgb8(data.mean_r, data.mean_g, data.mean_b, result.fg_r, result.fg_g, result.fg_b);
            result.bg_r = result.fg_r;
            result.bg_g = result.fg_g;
            result.bg_b = result.fg_b;
        }
    } else {
        result.codepoint = select_block_glyph(data.mean_luminance);
        
        to_srgb8(data.mean_r, data.mean_g, data.mean_b, result.fg_r, result.fg_g, result.fg_b);
        
        if (result.codepoint == BLOCK_SPACE) {
            result.bg_r = result.bg_g = result.bg_b = 0;
        } else if (result.codepoint == BLOCK_FULL) {
            result.bg_r = result.fg_r;
            result.bg_g = result.fg_g;
            result.bg_b = result.fg_b;
        } else {
            result.bg_r = result.bg_g = result.bg_b = 0;
        }
    }
    
    quantize_colors(result.fg_r, result.fg_g, result.fg_b);
    quantize_colors(result.bg_r, result.bg_g, result.bg_b);
    
    return result;
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

    struct ColorPoint {
        float r = 0.0f, g = 0.0f, b = 0.0f;
    };
    std::vector<ColorPoint> samples;
    samples.reserve(static_cast<size_t>(max_samples));

    int stride = std::max(1, static_cast<int>((cells.size() * 2) / max_samples));
    int counter = 0;
    for (const auto& c : cells) {
        if (counter % stride == 0) {
            samples.push_back({c.fg_r / 255.0f, c.fg_g / 255.0f, c.fg_b / 255.0f});
            if (static_cast<int>(samples.size()) >= max_samples) break;
            samples.push_back({c.bg_r / 255.0f, c.bg_g / 255.0f, c.bg_b / 255.0f});
            if (static_cast<int>(samples.size()) >= max_samples) break;
        }
        counter++;
    }

    int n = static_cast<int>(samples.size());
    if (n < palette_size) return;

    const float sigma2 = 2.0f * 0.20f * 0.20f;
    std::vector<float> w(static_cast<size_t>(n) * n, 0.0f);
    std::vector<float> d(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) {
            float dr = samples[i].r - samples[j].r;
            float dg = samples[i].g - samples[j].g;
            float db = samples[i].b - samples[j].b;
            float dist2 = dr * dr + dg * dg + db * db;
            float a = std::exp(-dist2 / sigma2);
            w[static_cast<size_t>(i) * n + j] = a;
            w[static_cast<size_t>(j) * n + i] = a;
            d[i] += a;
            if (i != j) d[j] += a;
        }
    }

    std::vector<float> principal(n, 1.0f / std::sqrt(static_cast<float>(n)));
    std::vector<float> fiedler(n, 0.0f);
    for (int i = 0; i < n; ++i) {
        float v = (samples[i].r + samples[i].g + samples[i].b) / 3.0f;
        fiedler[i] = v - 0.5f;
    }

    auto mat_vec = [&](const std::vector<float>& x, std::vector<float>& y) {
        std::fill(y.begin(), y.end(), 0.0f);
        for (int i = 0; i < n; ++i) {
            float di = 1.0f / std::sqrt(std::max(d[i], 1e-6f));
            float sum = 0.0f;
            for (int j = 0; j < n; ++j) {
                float dj = 1.0f / std::sqrt(std::max(d[j], 1e-6f));
                sum += (di * w[static_cast<size_t>(i) * n + j] * dj) * x[j];
            }
            y[i] = sum;
        }
    };

    std::vector<float> tmp(n, 0.0f);
    for (int it = 0; it < 18; ++it) {
        mat_vec(principal, tmp);
        float norm = 0.0f;
        for (float v : tmp) norm += v * v;
        norm = std::sqrt(std::max(norm, 1e-8f));
        for (int i = 0; i < n; ++i) principal[i] = tmp[i] / norm;
    }

    for (int it = 0; it < 24; ++it) {
        mat_vec(fiedler, tmp);
        float dot = 0.0f;
        for (int i = 0; i < n; ++i) dot += tmp[i] * principal[i];
        for (int i = 0; i < n; ++i) tmp[i] -= dot * principal[i];
        float norm = 0.0f;
        for (float v : tmp) norm += v * v;
        norm = std::sqrt(std::max(norm, 1e-8f));
        for (int i = 0; i < n; ++i) fiedler[i] = tmp[i] / norm;
    }

    float min_e = *std::min_element(fiedler.begin(), fiedler.end());
    float max_e = *std::max_element(fiedler.begin(), fiedler.end());
    if (max_e - min_e < 1e-6f) return;

    std::vector<float> center_e(palette_size, 0.0f);
    std::vector<ColorPoint> center_c(palette_size);
    for (int k = 0; k < palette_size; ++k) {
        float t = (k + 0.5f) / palette_size;
        center_e[k] = min_e + t * (max_e - min_e);
    }
    std::vector<int> assign(n, 0);

    for (int it = 0; it < iterations; ++it) {
        for (int i = 0; i < n; ++i) {
            int best = 0;
            float best_dist = std::numeric_limits<float>::max();
            for (int k = 0; k < palette_size; ++k) {
                float dist = std::abs(fiedler[i] - center_e[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best = k;
                }
            }
            assign[i] = best;
        }

        std::vector<float> sum_e(palette_size, 0.0f);
        std::vector<float> sum_r(palette_size, 0.0f);
        std::vector<float> sum_g(palette_size, 0.0f);
        std::vector<float> sum_b(palette_size, 0.0f);
        std::vector<int> count(palette_size, 0);
        for (int i = 0; i < n; ++i) {
            int k = assign[i];
            sum_e[k] += fiedler[i];
            sum_r[k] += samples[i].r;
            sum_g[k] += samples[i].g;
            sum_b[k] += samples[i].b;
            count[k]++;
        }
        for (int k = 0; k < palette_size; ++k) {
            if (count[k] > 0) {
                float inv = 1.0f / count[k];
                center_e[k] = sum_e[k] * inv;
                center_c[k] = {sum_r[k] * inv, sum_g[k] * inv, sum_b[k] * inv};
            }
        }
    }

    auto nearest_palette = [&](uint8_t r8, uint8_t g8, uint8_t b8) -> std::tuple<uint8_t, uint8_t, uint8_t> {
        float r = r8 / 255.0f;
        float g = g8 / 255.0f;
        float b = b8 / 255.0f;
        int best = 0;
        float best_dist = std::numeric_limits<float>::max();
        for (int k = 0; k < palette_size; ++k) {
            float dr = r - center_c[k].r;
            float dg = g - center_c[k].g;
            float db = b - center_c[k].b;
            float dist = dr * dr + dg * dg + db * db;
            if (dist < best_dist) {
                best_dist = dist;
                best = k;
            }
        }
        return {
            static_cast<uint8_t>(std::clamp(center_c[best].r * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(center_c[best].g * 255.0f, 0.0f, 255.0f)),
            static_cast<uint8_t>(std::clamp(center_c[best].b * 255.0f, 0.0f, 255.0f))
        };
    };

    for (auto& c : cells) {
        auto [fr, fg, fb] = nearest_palette(c.fg_r, c.fg_g, c.fg_b);
        auto [br, bg, bb] = nearest_palette(c.bg_r, c.bg_g, c.bg_b);
        c.fg_r = fr; c.fg_g = fg; c.fg_b = fb;
        c.bg_r = br; c.bg_g = bg; c.bg_b = bb;
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
            if (prev_cells && idx < static_cast<int>(prev_cells->size())) {
                const BlockCell& prev = (*prev_cells)[idx];
                skip = (cell.codepoint == prev.codepoint &&
                        cell.fg_r == prev.fg_r && cell.fg_g == prev.fg_g && cell.fg_b == prev.fg_b &&
                        cell.bg_r == prev.bg_r && cell.bg_g == prev.bg_g && cell.bg_b == prev.bg_b);
            }
            
            if (skip) {
                out << " ";
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
        
        out << "\033[0m\n";
        need_reset = true;
        last_fg_r = 255; last_fg_g = 255; last_fg_b = 255;
        last_bg_r = 0; last_bg_g = 0; last_bg_b = 0;
    }
    
    return out.str();
}

}
