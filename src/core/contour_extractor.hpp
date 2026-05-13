#pragma once

#include "core/types.hpp"

#include <vector>

namespace ascii {

class ContourExtractor {
public:
    struct Config {
        bool enabled = true;
        float min_occupancy = 0.055f;
        int min_pixels = 4;
        float dominance_ratio = 1.35f;
        float intersection_ratio = 1.25f;
        float dog_sigma_inner = 0.8f;
        float dog_sigma_outer = 1.6f;
    };

    ContourExtractor() = default;
    explicit ContourExtractor(const Config& config) : config_(config) {}

    void set_config(const Config& config) { config_ = config; }
    const Config& config() const { return config_; }

    void apply(const FloatImage& luminance,
               const EdgeData& edges,
               int cell_width,
               int cell_height,
               int grid_cols,
               int grid_rows,
               std::vector<CellStats>& cells) const;

private:
    Config config_;
};

}  // namespace ascii
