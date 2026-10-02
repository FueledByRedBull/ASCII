#include "core/cell_stats.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <utility>

// Full separable DCT and scalar Gabor references retain every transform
// coefficient and tap. Production may omit unused work or use independent SIMD
// lanes, but must preserve these descriptor values and reduction order.
namespace reference {
using ascii::FloatImage;
constexpr int kFreqBins = 8;
constexpr int kTextureBins = 8;
constexpr float kPi = 3.14159265358979323846f;

struct DCTBasis {
    static constexpr int W = 8;
    static constexpr int H = 16;
    float cos_w[W][W] = {};
    float cos_h[H][H] = {};
    float alpha_w[W] = {};
    float alpha_h[H] = {};
};

const DCTBasis& dct_basis() {
    static const DCTBasis basis = [] {
        DCTBasis b{};
        for (int u = 0; u < DCTBasis::W; ++u) {
            b.alpha_w[u] = (u == 0) ? std::sqrt(1.0f / DCTBasis::W) : std::sqrt(2.0f / DCTBasis::W);
            for (int x = 0; x < DCTBasis::W; ++x) {
                b.cos_w[u][x] = std::cos((kPi * (2.0f * x + 1.0f) * u) / (2.0f * DCTBasis::W));
            }
        }
        for (int v = 0; v < DCTBasis::H; ++v) {
            b.alpha_h[v] = (v == 0) ? std::sqrt(1.0f / DCTBasis::H) : std::sqrt(2.0f / DCTBasis::H);
            for (int y = 0; y < DCTBasis::H; ++y) {
                b.cos_h[v][y] = std::cos((kPi * (2.0f * y + 1.0f) * v) / (2.0f * DCTBasis::H));
            }
        }
        return b;
    }();
    return basis;
}

struct GaborBank {
    static constexpr int Radius = 2;
    static constexpr int Size = 2 * Radius + 1;
    static constexpr int Orientations = 4;
    static constexpr int Frequencies = 2;
    float kernel[Frequencies][Orientations][Size][Size] = {};
};

const GaborBank& gabor_bank() {
    static const GaborBank bank = [] {
        GaborBank g{};
        constexpr float kSigma = 1.3f;
        constexpr float kGamma = 0.6f;
        constexpr std::array<float, GaborBank::Orientations> kAngles = {
            0.0f, 0.25f * kPi, 0.5f * kPi, 0.75f * kPi
        };
        constexpr std::array<float, GaborBank::Frequencies> kLambdas = {3.2f, 6.4f};

        for (int fi = 0; fi < GaborBank::Frequencies; ++fi) {
            const float lambda = kLambdas[fi];
            for (int oi = 0; oi < GaborBank::Orientations; ++oi) {
                const float theta = kAngles[oi];
                const float ct = std::cos(theta);
                const float st = std::sin(theta);
                for (int ky = -GaborBank::Radius; ky <= GaborBank::Radius; ++ky) {
                    for (int kx = -GaborBank::Radius; kx <= GaborBank::Radius; ++kx) {
                        const float xr = kx * ct + ky * st;
                        const float yr = -kx * st + ky * ct;
                        const float gauss = std::exp(-(xr * xr + (kGamma * kGamma) * yr * yr) / (2.0f * kSigma * kSigma));
                        const float carrier = std::cos((2.0f * kPi * xr) / lambda);
                        g.kernel[fi][oi][ky + GaborBank::Radius][kx + GaborBank::Radius] = gauss * carrier;
                    }
                }
            }
        }
        return g;
    }();
    return bank;
}

void compute_cell_frequency_signature(const FloatImage& img,
                                      int x0, int y0, int x1, int y1,
                                      float out[kFreqBins]) {
    for (int i = 0; i < kFreqBins; ++i) out[i] = 0.0f;
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);

    constexpr int W = DCTBasis::W;
    constexpr int H = DCTBasis::H;
    const DCTBasis& basis = dct_basis();
    float sample[H][W] = {};
    for (int sy = 0; sy < H; ++sy) {
        for (int sx = 0; sx < W; ++sx) {
            int px = x0 + (sx * w) / W;
            int py = y0 + (sy * h) / H;
            px = std::clamp(px, x0, x1 - 1);
            py = std::clamp(py, y0, y1 - 1);
            sample[sy][sx] = img.get(px, py);
        }
    }

    float row_dct[H][W] = {};
    for (int yy = 0; yy < H; ++yy) {
        for (int u = 0; u < W; ++u) {
            float sum = 0.0f;
            for (int xx = 0; xx < W; ++xx) {
                sum += sample[yy][xx] * basis.cos_w[u][xx];
            }
            row_dct[yy][u] = sum;
        }
    }

    float dct[H][W] = {};
    for (int v = 0; v < H; ++v) {
        for (int u = 0; u < W; ++u) {
            float sum = 0.0f;
            for (int yy = 0; yy < H; ++yy) {
                sum += row_dct[yy][u] * basis.cos_h[v][yy];
            }
            dct[v][u] = basis.alpha_w[u] * basis.alpha_h[v] * sum;
        }
    }

    static constexpr std::array<std::pair<int, int>, kFreqBins> kZigZag = {
        std::pair<int, int>{1, 0}, {0, 1}, {2, 0}, {1, 1},
        {0, 2}, {3, 0}, {2, 1}, {0, 3}
    };
    for (int i = 0; i < kFreqBins; ++i) {
        out[i] = dct[kZigZag[i].second][kZigZag[i].first];
    }

    float norm = 0.0f;
    for (int i = 0; i < kFreqBins; ++i) norm += out[i] * out[i];
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < kFreqBins; ++i) out[i] /= norm;
    }
}

void compute_cell_texture_signature(const FloatImage& img,
                                    int x0, int y0, int x1, int y1,
                                    float out[kTextureBins]) {
    for (int i = 0; i < kTextureBins; ++i) out[i] = 0.0f;
    int w = std::max(1, x1 - x0);
    int h = std::max(1, y1 - y0);
    if (w < 5 || h < 5) return;

    constexpr int kRadius = GaborBank::Radius;
    const GaborBank& bank = gabor_bank();
    const float* src = img.data();
    const int stride = img.width();

    for (int fi = 0; fi < GaborBank::Frequencies; ++fi) {
        for (int oi = 0; oi < GaborBank::Orientations; ++oi) {
            float energy = 0.0f;
            int samples = 0;

            for (int y = y0 + kRadius; y < y1 - kRadius; ++y) {
                for (int x = x0 + kRadius; x < x1 - kRadius; ++x) {
                    float resp = 0.0f;
                    const float* row_base = src + static_cast<size_t>(y) * stride + x;
                    for (int ky = -kRadius; ky <= kRadius; ++ky) {
                        const float* src_row = row_base + ky * stride;
                        const float* kernel_row = bank.kernel[fi][oi][ky + kRadius];
                        for (int kx = -kRadius; kx <= kRadius; ++kx) {
                            resp += kernel_row[kx + kRadius] * src_row[kx];
                        }
                    }
                    energy += std::abs(resp);
                    samples++;
                }
            }
            out[fi * GaborBank::Orientations + oi] = samples > 0 ? (energy / samples) : 0.0f;
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < kTextureBins; ++i) norm += out[i] * out[i];
    norm = std::sqrt(norm);
    if (norm > 1e-6f) {
        for (int i = 0; i < kTextureBins; ++i) out[i] /= norm;
    }
}

}

int main() {
    using namespace ascii;
    int checked_cells = 0, frequency_failures = 0, texture_failures = 0;
    for (int width : {1,4,5,7,8,9,16,31,32}) {
        for (int height : {1,4,5,8,16,17,32,64}) {
            for (int pattern = 0; pattern < 7; ++pattern) {
                FloatImage image(width * 3 + 1, height * 3 + 1);
                for (int y = 0; y < image.height(); ++y) {
                    for (int x = 0; x < image.width(); ++x) {
                        float value = 0;
                        if (pattern == 1) value = .25f;
                        if (pattern == 2) value = 1.0f;
                        if (pattern == 3) value = static_cast<float>(x) / image.width();
                        if (pattern == 4) value = (x+y)%2 ? 1.0f : 0.0f;
                        if (pattern == 5) value = ((x*127+y*73+x*y*29)%251)/250.0f;
                        if (pattern == 6) value = (x+y)%2 ? 1e-7f : 0.0f;
                        image.set(x,y,value);
                    }
                }
                CellStatsAggregator::Config config;
                config.cell_width = width;
                config.cell_height = height;
                config.enable_orientation_histogram = false;
                const auto actual = CellStatsAggregator(config).compute(image, {});
                for (int row = 0; row < 4; ++row) {
                    for (int col = 0; col < 4; ++col) {
                        const int x0 = col*width, y0 = row*height;
                        const int x1 = std::min(x0+width,image.width());
                        const int y1 = std::min(y0+height,image.height());
                        float frequency[8], texture[8];
                        reference::compute_cell_frequency_signature(image,x0,y0,x1,y1,frequency);
                        reference::compute_cell_texture_signature(image,x0,y0,x1,y1,texture);
                        const auto& cell = actual[row*4+col];
                        const bool frequency_equal = std::memcmp(frequency,cell.frequency_signature,sizeof(frequency)) == 0;
                        const bool texture_equal = std::memcmp(texture,cell.texture_signature,sizeof(texture)) == 0;
                        frequency_failures += !frequency_equal;
                        texture_failures += !texture_equal;
                        if ((!frequency_equal || !texture_equal) && frequency_failures+texture_failures < 8)
                            std::cerr << "Descriptor mismatch: cell " << width << 'x' << height << " pattern " << pattern
                                      << " grid position " << col << ',' << row << '\n';
                        ++checked_cells;
                    }
                }
            }
        }
    }
    std::cout << "Compared " << checked_cells << " cells: DCT bitwise mismatches=" << frequency_failures
              << ", Gabor bitwise mismatches=" << texture_failures << '\n';
    return frequency_failures || texture_failures;
}