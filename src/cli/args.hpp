#pragma once

#include "terminal/terminal.hpp"
#include <string>

namespace ascii {

struct Args {
    std::string input;
    std::string output;
    std::string font_path;
    std::string char_set = "basic";
    std::string profile;
    std::string debug_mode;
    std::string config_path;
    std::string replay_path;
    ColorMode color_mode = ColorMode::Truecolor;
    bool color_mode_set = false;
    
    int fps = 30;
    int cols = 0;
    int rows = 0;
    int cell_width = 8;
    int cell_height = 16;
    
    float edge_threshold = 0.1f;
    float blur_sigma = 1.0f;
    float temporal_alpha = 0.3f;
    int motion_solve_divisor = 0;
    int motion_max_reuse_frames = -1;
    float motion_reuse_scene_threshold = -1.0f;
    float motion_reuse_confidence_decay = -1.0f;
    int motion_phase_interval = 0;
    float motion_phase_scene_trigger = -1.0f;
    float motion_still_scene_threshold = -1.0f;
    
    std::string scale_mode = "fit";
    bool no_audio = false;
    bool use_hysteresis = true;
    bool use_orientation_matching = true;
    bool use_simple_orientation = false;
    
    bool profile_live = false;
    bool strict_memory = false;
    bool fast_mode = false;
    
    bool show_help = false;
};

Args parse_args(int argc, char* argv[]);
void print_help(const char* prog);

}
