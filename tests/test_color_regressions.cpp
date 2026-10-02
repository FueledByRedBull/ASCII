#include "mapping/bilateral_grid.hpp"
#include "mapping/color_mapper.hpp"
#include <cmath>
#include <iostream>

using namespace ascii;

int main() {
    int failures = 0;
    const auto check = [&](bool condition, const char* message) {
        if (!condition) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    // Static callers must not depend on a mapper having been constructed.
    check(ColorMapper::find_nearest_16_oklab(255, 0, 0) == 9,
          "Static palette lookup did not find exact red");
    check(ColorMapper::find_nearest_256_oklab(255, 255, 255) == 15,
          "Static palette lookup did not find exact white");

    BilateralGrid::Config config;
    config.enabled = true;
    config.spatial_bins = 2;
    config.range_bins = 4;
    config.spatial_sigma = 1.0f;
    config.range_sigma = 0.0f;
    BilateralGrid grid(config);
    std::vector<CellStats> cells(2);
    cells[0].mean_luminance = 0.0f;
    cells[0].mean_r = 1.0f;
    cells[1].mean_luminance = 0.25f;
    cells[1].mean_b = 1.0f;
    grid.build(cells, 2, 1);
    const auto result = grid.sample(0, 0, 1.0f / 6.0f);
    // The two range slices have different weights. Slice sums and weights
    // together before normalization; the common kernel factor cancels.
    const float neighbor = std::exp(-0.5f);
    const float expected_red = (1.0f + neighbor) / (1.0f + 2.0f * neighbor);
    check(std::abs(result.r - expected_red) < 1e-5f &&
          std::abs(result.b - (1.0f - expected_red)) < 1e-5f,
          "Bilateral slice does not normalize interpolated weights");
    check(result.has_support && !grid.sample(0, 0, 1.0f).has_support,
          "Bilateral slice cannot distinguish empty support from black");
    config.enabled = false;
    grid.set_config(config);
    check(!grid.valid(), "Changed bilateral configuration retains stale samples");

    Ditherer::Config dither_config;
    dither_config.enabled = true;
    dither_config.use_blue_noise_halftone = true;
    Ditherer dither(dither_config);
    dither.begin_frame(16, 1);
    ColorMapper truecolor(ColorMode::Truecolor);
    truecolor.set_ditherer(&dither);
    bool exact_truecolor = true;
    for (int x = 0; x < 16; ++x) {
        const auto color = truecolor.map_with_dither(x, 0, 1, 128.0f / 255, 128.0f / 255, 128.0f / 255, false);
        exact_truecolor = exact_truecolor && color.r == 128 && color.g == 128 && color.b == 128;
    }
    check(exact_truecolor, "Truecolor uses palette dithering despite independent cell rendering");
    return failures ? 1 : 0;
}
