#pragma once

#include "core/types.hpp"
#include "terminal/terminal.hpp"
#include <string>
#include <optional>
#include <cstdint>

namespace ascii {

constexpr int CONFIG_VERSION = 1;

struct ConfigInput {
    std::string source;
    std::string mode = "file";
};

struct ConfigOutput {
    std::string target;
    std::string mode = "terminal";
    std::string replay_path;
};

struct ConfigGrid {
    int cols = 0;
    int rows = 0;
    int cell_width = 8;
    int cell_height = 16;
    float char_aspect = 2.0f;
    std::string scale_mode = "fit";
    bool quad_tree_adaptive = false;
    int quad_tree_max_depth = 2;
    float quad_tree_variance_threshold = 0.01f;
};

struct ConfigEdge {
    float low_threshold = 0.05f;
    float high_threshold = 0.1f;
    float blur_sigma = 1.0f;
    bool use_hysteresis = true;
    bool multi_scale = true;
    float scale_sigma_0 = 0.8f;
    float scale_sigma_1 = 1.6f;
    bool adaptive_scale_selection = true;
    float scale_variance_floor = 0.0005f;
    float scale_variance_ceil = 0.02f;
    bool use_anisotropic_diffusion = false;
    int diffusion_iterations = 4;
    float diffusion_kappa = 0.08f;
    float diffusion_lambda = 0.2f;
    std::string adaptive_mode = "hybrid";
    int tile_size = 16;
    float dark_scene_floor = 0.02f;
    float global_percentile = 0.7f;
};

struct ConfigTemporal {
    float alpha = 0.3f;
    float transition_penalty = 0.15f;
    float edge_enter_threshold = 0.08f;
    float edge_exit_threshold = 0.04f;
    int motion_cap_pixels = 6;
    bool use_wavelet_flicker = true;
    float wavelet_strength = 0.45f;
    int wavelet_window = 8;
    bool use_phase_correlation = true;
    int phase_search_radius = 6;
    float phase_blend = 0.28f;
};

struct ConfigSelector {
    std::string char_set = "basic";
    std::string mode = "histogram";
    float weight_brightness = 0.45f;
    float weight_orientation = 0.40f;
    float weight_contrast = 0.15f;
    float weight_frequency = 0.20f;
    float weight_texture = 0.15f;
    bool enable_frequency_matching = true;
    bool enable_gabor_texture = true;
    bool use_simple_orientation = false;
};

struct ConfigColor {
    ColorMode mode = ColorMode::Truecolor;
    std::string quantization = "oklab";
    float dither_error_clamp = 0.12f;
    bool use_blue_noise_halftone = false;
    float halftone_strength = 0.24f;
    int halftone_cell_size = 6;
    bool use_bilateral_grid = false;
    int bilateral_spatial_bins = 32;
    int bilateral_range_bins = 16;
    float bilateral_spatial_sigma = 2.0f;
    float bilateral_range_sigma = 0.15f;
    int block_spectral_palette = 0;
    int block_spectral_samples = 64;
    int block_spectral_iterations = 8;
};

struct ConfigDebug {
    bool enabled = false;
    std::string mode;
    bool profile_live = false;
    bool strict_memory = false;
};

struct Config {
    int version = CONFIG_VERSION;
    ConfigInput input;
    ConfigOutput output;
    ConfigGrid grid;
    ConfigEdge edge;
    ConfigTemporal temporal;
    ConfigSelector selector;
    ConfigColor color;
    ConfigDebug debug;
    std::string profile;
    
    std::string config_path;
    std::string font_path;
    int fps = 30;
    bool no_audio = false;
    
    std::string compute_hash() const;
    bool validate(std::string& error) const;
    
    static Config defaults();
    static std::optional<Config> load(const std::string& path);
    static std::optional<Config> load_default();
    static std::string default_config_path();
    static std::string default_config_dir();
};

Config merge_config(Config base, const Config& override);
Config apply_cli_overrides(Config config, const struct Args& args);
void apply_content_profile(Config& config);

}
