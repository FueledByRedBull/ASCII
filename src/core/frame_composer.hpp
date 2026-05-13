#pragma once

#include "core/config.hpp"
#include "core/motion.hpp"
#include "core/pipeline.hpp"
#include "core/temporal.hpp"
#include "mapping/bilateral_grid.hpp"
#include "mapping/char_selector.hpp"
#include "mapping/color_mapper.hpp"
#include "render/block_renderer.hpp"
#include "render/terminal_renderer.hpp"

#include <vector>

namespace ascii {

class FrameComposer {
public:
    struct Context {
        TemporalSmoother& smoother;
        CharSelector& selector;
        ColorMapper& color_mapper;
        BilateralGrid& bilateral_grid;
        MotionEstimator& motion;
        BlockRenderer& block_renderer;
        const Config& config;
        ColorMode color_mode = ColorMode::Truecolor;
        float edge_threshold = 0.1f;
    };

    struct Output {
        std::vector<ASCIICell> cells;
        std::vector<BlockCell> block_cells;
    };

    Output compose(const Pipeline::Result& result, Context& context);

private:
    bool smoother_initialized_ = false;
    int smoother_cols_ = 0;
    int smoother_rows_ = 0;
};

}  // namespace ascii
