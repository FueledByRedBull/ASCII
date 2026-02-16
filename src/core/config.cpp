#include "core/config.hpp"
#include "cli/args.hpp"
#include <toml.hpp>

#include <filesystem>
#include <sstream>
#include <iomanip>
#include <functional>
#include <cstring>

#ifdef _WIN32
    #include <shlobj.h>
#else
    #include <unistd.h>
    #include <pwd.h>
#endif

namespace ascii {

namespace {

std::string get_home_dir() {
#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_PROFILE, nullptr, 0, path))) {
        return std::string(path);
    }
    const char* userprofile = std::getenv("USERPROFILE");
    if (userprofile) return std::string(userprofile);
    const char* homedrive = std::getenv("HOMEDRIVE");
    const char* homepath = std::getenv("HOMEPATH");
    if (homedrive && homepath) {
        return std::string(homedrive) + std::string(homepath);
    }
    return ".";
#else
    const char* home = std::getenv("HOME");
    if (home) return std::string(home);
    struct passwd* pw = getpwuid(getuid());
    if (pw) return std::string(pw->pw_dir);
    return ".";
#endif
}

std::string get_app_data_dir() {
#ifdef _WIN32
    char path[MAX_PATH];
    if (SUCCEEDED(SHGetFolderPathA(nullptr, CSIDL_APPDATA, nullptr, 0, path))) {
        return std::string(path);
    }
    const char* appdata = std::getenv("APPDATA");
    if (appdata) return std::string(appdata);
    return get_home_dir();
#elif defined(__APPLE__)
    return get_home_dir() + "/Library/Application Support";
#else
    const char* xdg_config = std::getenv("XDG_CONFIG_HOME");
    if (xdg_config) return std::string(xdg_config);
    return get_home_dir() + "/.config";
#endif
}

uint32_t hash_combine(uint32_t a, uint32_t b) {
    a ^= b + 0x9e3779b9 + (a << 6) + (a >> 2);
    return a;
}

uint32_t hash_string(const std::string& s) {
    uint32_t h = 0;
    for (char c : s) {
        h = hash_combine(h, static_cast<uint32_t>(c));
    }
    return h;
}

uint32_t hash_float(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(f));
    return u;
}

uint32_t hash_int(int i) {
    return static_cast<uint32_t>(i);
}

}

Config Config::defaults() {
    Config cfg;
    cfg.version = CONFIG_VERSION;
    return cfg;
}

std::string Config::default_config_dir() {
    return get_app_data_dir() + "/ascii-engine";
}

std::string Config::default_config_path() {
    return default_config_dir() + "/config.toml";
}

bool Config::validate(std::string& error) const {
    if (grid.cols < 0 || grid.cols > 1000) {
        error = "grid.cols must be between 0 and 1000";
        return false;
    }
    if (grid.rows < 0 || grid.rows > 500) {
        error = "grid.rows must be between 0 and 500";
        return false;
    }
    if (grid.cell_width < 1 || grid.cell_width > 32) {
        error = "grid.cell_width must be between 1 and 32";
        return false;
    }
    if (grid.cell_height < 1 || grid.cell_height > 64) {
        error = "grid.cell_height must be between 1 and 64";
        return false;
    }
    if (grid.quad_tree_max_depth < 0 || grid.quad_tree_max_depth > 6) {
        error = "grid.quad_tree_max_depth must be between 0 and 6";
        return false;
    }
    if (grid.quad_tree_variance_threshold < 0.0f || grid.quad_tree_variance_threshold > 1.0f) {
        error = "grid.quad_tree_variance_threshold must be between 0.0 and 1.0";
        return false;
    }
    if (fps < 1 || fps > 120) {
        error = "fps must be between 1 and 120";
        return false;
    }
    if (edge.blur_sigma < 0.1f || edge.blur_sigma > 10.0f) {
        error = "edge.blur_sigma must be between 0.1 and 10.0";
        return false;
    }
    if (edge.low_threshold < 0.0f || edge.low_threshold > 1.0f) {
        error = "edge.low_threshold must be between 0.0 and 1.0";
        return false;
    }
    if (edge.high_threshold < 0.0f || edge.high_threshold > 1.0f) {
        error = "edge.high_threshold must be between 0.0 and 1.0";
        return false;
    }
    if (edge.low_threshold > edge.high_threshold) {
        error = "edge.low_threshold cannot exceed edge.high_threshold";
        return false;
    }
    if (edge.scale_sigma_0 <= 0.0f || edge.scale_sigma_1 <= 0.0f) {
        error = "edge.scale_sigma_0 and edge.scale_sigma_1 must be > 0";
        return false;
    }
    if (edge.scale_variance_floor < 0.0f || edge.scale_variance_ceil <= edge.scale_variance_floor) {
        error = "edge.scale_variance_floor must be >= 0 and < edge.scale_variance_ceil";
        return false;
    }
    if (edge.diffusion_iterations < 0 || edge.diffusion_iterations > 20) {
        error = "edge.diffusion_iterations must be between 0 and 20";
        return false;
    }
    if (edge.diffusion_kappa <= 0.0f || edge.diffusion_kappa > 2.0f) {
        error = "edge.diffusion_kappa must be > 0 and <= 2";
        return false;
    }
    if (edge.diffusion_lambda <= 0.0f || edge.diffusion_lambda > 0.25f) {
        error = "edge.diffusion_lambda must be > 0 and <= 0.25";
        return false;
    }
    if (edge.tile_size < 4 || edge.tile_size > 256) {
        error = "edge.tile_size must be between 4 and 256";
        return false;
    }
    if (edge.dark_scene_floor < 0.0f || edge.dark_scene_floor > 1.0f) {
        error = "edge.dark_scene_floor must be between 0.0 and 1.0";
        return false;
    }
    if (edge.global_percentile <= 0.0f || edge.global_percentile >= 1.0f) {
        error = "edge.global_percentile must be between 0.0 and 1.0 (exclusive)";
        return false;
    }
    if (temporal.alpha < 0.0f || temporal.alpha > 1.0f) {
        error = "temporal.alpha must be between 0.0 and 1.0";
        return false;
    }
    if (temporal.transition_penalty < 0.0f || temporal.transition_penalty > 1.0f) {
        error = "temporal.transition_penalty must be between 0.0 and 1.0";
        return false;
    }
    if (temporal.motion_solve_divisor < 1 || temporal.motion_solve_divisor > 8) {
        error = "temporal.motion_solve_divisor must be between 1 and 8";
        return false;
    }
    if (temporal.motion_max_reuse_frames < 0 || temporal.motion_max_reuse_frames > 32) {
        error = "temporal.motion_max_reuse_frames must be between 0 and 32";
        return false;
    }
    if (temporal.motion_reuse_scene_threshold < 0.0f || temporal.motion_reuse_scene_threshold > 1.0f) {
        error = "temporal.motion_reuse_scene_threshold must be between 0.0 and 1.0";
        return false;
    }
    if (temporal.motion_reuse_confidence_decay < 0.0f || temporal.motion_reuse_confidence_decay > 1.0f) {
        error = "temporal.motion_reuse_confidence_decay must be between 0.0 and 1.0";
        return false;
    }
    if (temporal.motion_still_scene_threshold < 0.0f || temporal.motion_still_scene_threshold > 1.0f) {
        error = "temporal.motion_still_scene_threshold must be between 0.0 and 1.0";
        return false;
    }
    if (temporal.wavelet_strength < 0.0f || temporal.wavelet_strength > 1.0f) {
        error = "temporal.wavelet_strength must be between 0.0 and 1.0";
        return false;
    }
    if (temporal.wavelet_window < 2 || temporal.wavelet_window > 8) {
        error = "temporal.wavelet_window must be between 2 and 8";
        return false;
    }
    if (temporal.phase_search_radius < 1 || temporal.phase_search_radius > 32) {
        error = "temporal.phase_search_radius must be between 1 and 32";
        return false;
    }
    if (temporal.phase_blend < 0.0f || temporal.phase_blend > 1.0f) {
        error = "temporal.phase_blend must be between 0.0 and 1.0";
        return false;
    }
    if (temporal.motion_phase_interval < 1 || temporal.motion_phase_interval > 64) {
        error = "temporal.motion_phase_interval must be between 1 and 64";
        return false;
    }
    if (temporal.motion_phase_scene_trigger < 0.0f || temporal.motion_phase_scene_trigger > 1.0f) {
        error = "temporal.motion_phase_scene_trigger must be between 0.0 and 1.0";
        return false;
    }
    if (selector.weight_brightness < 0.0f || selector.weight_orientation < 0.0f ||
        selector.weight_contrast < 0.0f || selector.weight_frequency < 0.0f ||
        selector.weight_texture < 0.0f) {
        error = "selector weights must be non-negative";
        return false;
    }
    float total_weight = selector.weight_brightness + selector.weight_orientation +
                         selector.weight_contrast + selector.weight_frequency + selector.weight_texture;
    if (total_weight < 0.001f) {
        error = "selector weights must sum to a positive value";
        return false;
    }
    if (grid.scale_mode != "fit" && grid.scale_mode != "fill" && grid.scale_mode != "stretch") {
        error = "grid.scale_mode must be 'fit', 'fill', or 'stretch'";
        return false;
    }
    if (edge.adaptive_mode != "global" && edge.adaptive_mode != "local" && edge.adaptive_mode != "hybrid") {
        error = "edge.adaptive_mode must be 'global', 'local', or 'hybrid'";
        return false;
    }
    if (selector.mode != "simple" && selector.mode != "histogram") {
        error = "selector.mode must be 'simple' or 'histogram'";
        return false;
    }
    if (!profile.empty() && profile != "natural" && profile != "anime" && profile != "ui") {
        error = "profile must be '', 'natural', 'anime', or 'ui'";
        return false;
    }
    if (color.dither_error_clamp < 0.0f || color.dither_error_clamp > 1.0f) {
        error = "color.dither_error_clamp must be between 0.0 and 1.0";
        return false;
    }
    if (color.halftone_strength < 0.0f || color.halftone_strength > 1.0f) {
        error = "color.halftone_strength must be between 0.0 and 1.0";
        return false;
    }
    if (color.halftone_cell_size < 2 || color.halftone_cell_size > 16) {
        error = "color.halftone_cell_size must be between 2 and 16";
        return false;
    }
    if (color.bilateral_spatial_bins < 4 || color.bilateral_spatial_bins > 256) {
        error = "color.bilateral_spatial_bins must be between 4 and 256";
        return false;
    }
    if (color.bilateral_range_bins < 4 || color.bilateral_range_bins > 64) {
        error = "color.bilateral_range_bins must be between 4 and 64";
        return false;
    }
    if (color.bilateral_spatial_sigma <= 0.0f || color.bilateral_spatial_sigma > 16.0f) {
        error = "color.bilateral_spatial_sigma must be > 0 and <= 16";
        return false;
    }
    if (color.bilateral_range_sigma <= 0.0f || color.bilateral_range_sigma > 1.0f) {
        error = "color.bilateral_range_sigma must be > 0 and <= 1";
        return false;
    }
    if (color.block_spectral_palette < 0 || color.block_spectral_palette > 32) {
        error = "color.block_spectral_palette must be between 0 and 32";
        return false;
    }
    if (color.block_spectral_samples < 8 || color.block_spectral_samples > 1024) {
        error = "color.block_spectral_samples must be between 8 and 1024";
        return false;
    }
    if (color.block_spectral_iterations < 1 || color.block_spectral_iterations > 64) {
        error = "color.block_spectral_iterations must be between 1 and 64";
        return false;
    }
    return true;
}

std::string Config::compute_hash() const {
    uint32_t h = 0;
    
    h = hash_combine(h, hash_int(version));
    h = hash_combine(h, hash_string(input.source));
    h = hash_combine(h, hash_string(input.mode));
    h = hash_combine(h, hash_string(output.target));
    h = hash_combine(h, hash_string(output.mode));
    h = hash_combine(h, hash_int(grid.cols));
    h = hash_combine(h, hash_int(grid.rows));
    h = hash_combine(h, hash_int(grid.cell_width));
    h = hash_combine(h, hash_int(grid.cell_height));
    h = hash_combine(h, hash_float(grid.char_aspect));
    h = hash_combine(h, hash_string(grid.scale_mode));
    h = hash_combine(h, hash_int(static_cast<int>(grid.quad_tree_adaptive)));
    h = hash_combine(h, hash_int(grid.quad_tree_max_depth));
    h = hash_combine(h, hash_float(grid.quad_tree_variance_threshold));
    h = hash_combine(h, hash_float(edge.low_threshold));
    h = hash_combine(h, hash_float(edge.high_threshold));
    h = hash_combine(h, hash_float(edge.blur_sigma));
    h = hash_combine(h, hash_int(static_cast<int>(edge.use_hysteresis)));
    h = hash_combine(h, hash_int(static_cast<int>(edge.multi_scale)));
    h = hash_combine(h, hash_float(edge.scale_sigma_0));
    h = hash_combine(h, hash_float(edge.scale_sigma_1));
    h = hash_combine(h, hash_int(static_cast<int>(edge.adaptive_scale_selection)));
    h = hash_combine(h, hash_float(edge.scale_variance_floor));
    h = hash_combine(h, hash_float(edge.scale_variance_ceil));
    h = hash_combine(h, hash_int(static_cast<int>(edge.use_anisotropic_diffusion)));
    h = hash_combine(h, hash_int(edge.diffusion_iterations));
    h = hash_combine(h, hash_float(edge.diffusion_kappa));
    h = hash_combine(h, hash_float(edge.diffusion_lambda));
    h = hash_combine(h, hash_string(edge.adaptive_mode));
    h = hash_combine(h, hash_int(edge.tile_size));
    h = hash_combine(h, hash_float(edge.dark_scene_floor));
    h = hash_combine(h, hash_float(edge.global_percentile));
    h = hash_combine(h, hash_float(temporal.alpha));
    h = hash_combine(h, hash_float(temporal.transition_penalty));
    h = hash_combine(h, hash_float(temporal.edge_enter_threshold));
    h = hash_combine(h, hash_float(temporal.edge_exit_threshold));
    h = hash_combine(h, hash_int(temporal.motion_cap_pixels));
    h = hash_combine(h, hash_int(temporal.motion_solve_divisor));
    h = hash_combine(h, hash_int(temporal.motion_max_reuse_frames));
    h = hash_combine(h, hash_float(temporal.motion_reuse_scene_threshold));
    h = hash_combine(h, hash_float(temporal.motion_reuse_confidence_decay));
    h = hash_combine(h, hash_float(temporal.motion_still_scene_threshold));
    h = hash_combine(h, hash_int(static_cast<int>(temporal.use_wavelet_flicker)));
    h = hash_combine(h, hash_float(temporal.wavelet_strength));
    h = hash_combine(h, hash_int(temporal.wavelet_window));
    h = hash_combine(h, hash_int(static_cast<int>(temporal.use_phase_correlation)));
    h = hash_combine(h, hash_int(temporal.phase_search_radius));
    h = hash_combine(h, hash_float(temporal.phase_blend));
    h = hash_combine(h, hash_int(temporal.motion_phase_interval));
    h = hash_combine(h, hash_float(temporal.motion_phase_scene_trigger));
    h = hash_combine(h, hash_string(selector.char_set));
    h = hash_combine(h, hash_string(selector.mode));
    h = hash_combine(h, hash_float(selector.weight_brightness));
    h = hash_combine(h, hash_float(selector.weight_orientation));
    h = hash_combine(h, hash_float(selector.weight_contrast));
    h = hash_combine(h, hash_float(selector.weight_frequency));
    h = hash_combine(h, hash_float(selector.weight_texture));
    h = hash_combine(h, hash_int(static_cast<int>(selector.enable_frequency_matching)));
    h = hash_combine(h, hash_int(static_cast<int>(selector.enable_gabor_texture)));
    h = hash_combine(h, hash_int(static_cast<int>(color.mode)));
    h = hash_combine(h, hash_string(color.quantization));
    h = hash_combine(h, hash_float(color.dither_error_clamp));
    h = hash_combine(h, hash_int(static_cast<int>(color.use_blue_noise_halftone)));
    h = hash_combine(h, hash_float(color.halftone_strength));
    h = hash_combine(h, hash_int(color.halftone_cell_size));
    h = hash_combine(h, hash_int(static_cast<int>(color.use_bilateral_grid)));
    h = hash_combine(h, hash_int(color.bilateral_spatial_bins));
    h = hash_combine(h, hash_int(color.bilateral_range_bins));
    h = hash_combine(h, hash_float(color.bilateral_spatial_sigma));
    h = hash_combine(h, hash_float(color.bilateral_range_sigma));
    h = hash_combine(h, hash_int(color.block_spectral_palette));
    h = hash_combine(h, hash_int(color.block_spectral_samples));
    h = hash_combine(h, hash_int(color.block_spectral_iterations));
    h = hash_combine(h, hash_string(profile));
    h = hash_combine(h, hash_int(fps));
    h = hash_combine(h, hash_string(font_path));
    
    std::ostringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(8) << h;
    return ss.str();
}

std::optional<Config> Config::load(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(std::filesystem::path(path), ec) || ec) {
        return std::nullopt;
    }
    
    try {
        auto tbl = toml::parse_file(path);
        
        Config cfg = defaults();
        cfg.config_path = path;
        
        if (auto v = tbl["config_version"].value<int>()) {
            if (*v != CONFIG_VERSION) {
                return std::nullopt;
            }
        }
        
        if (auto input = tbl["input"]) {
            if (auto v = input["source"].value<std::string>()) cfg.input.source = *v;
            if (auto v = input["mode"].value<std::string>()) cfg.input.mode = *v;
        }
        
        if (auto output = tbl["output"]) {
            if (auto v = output["target"].value<std::string>()) cfg.output.target = *v;
            if (auto v = output["mode"].value<std::string>()) cfg.output.mode = *v;
            if (auto v = output["replay_path"].value<std::string>()) cfg.output.replay_path = *v;
        }
        
        if (auto grid = tbl["grid"]) {
            if (auto v = grid["cols"].value<int>()) cfg.grid.cols = *v;
            if (auto v = grid["rows"].value<int>()) cfg.grid.rows = *v;
            if (auto v = grid["cell_width"].value<int>()) cfg.grid.cell_width = *v;
            if (auto v = grid["cell_height"].value<int>()) cfg.grid.cell_height = *v;
            if (auto v = grid["char_aspect"].value<double>()) cfg.grid.char_aspect = static_cast<float>(*v);
            if (auto v = grid["scale_mode"].value<std::string>()) cfg.grid.scale_mode = *v;
            if (auto v = grid["quad_tree_adaptive"].value<bool>()) cfg.grid.quad_tree_adaptive = *v;
            if (auto v = grid["quad_tree_max_depth"].value<int>()) cfg.grid.quad_tree_max_depth = *v;
            if (auto v = grid["quad_tree_variance_threshold"].value<double>()) cfg.grid.quad_tree_variance_threshold = static_cast<float>(*v);
        }
        
        if (auto edge = tbl["edge"]) {
            if (auto v = edge["low_threshold"].value<double>()) cfg.edge.low_threshold = static_cast<float>(*v);
            if (auto v = edge["high_threshold"].value<double>()) cfg.edge.high_threshold = static_cast<float>(*v);
            if (auto v = edge["blur_sigma"].value<double>()) cfg.edge.blur_sigma = static_cast<float>(*v);
            if (auto v = edge["use_hysteresis"].value<bool>()) cfg.edge.use_hysteresis = *v;
            if (auto v = edge["multi_scale"].value<bool>()) cfg.edge.multi_scale = *v;
            if (auto v = edge["scale_sigma_0"].value<double>()) cfg.edge.scale_sigma_0 = static_cast<float>(*v);
            if (auto v = edge["scale_sigma_1"].value<double>()) cfg.edge.scale_sigma_1 = static_cast<float>(*v);
            if (auto v = edge["adaptive_scale_selection"].value<bool>()) cfg.edge.adaptive_scale_selection = *v;
            if (auto v = edge["scale_variance_floor"].value<double>()) cfg.edge.scale_variance_floor = static_cast<float>(*v);
            if (auto v = edge["scale_variance_ceil"].value<double>()) cfg.edge.scale_variance_ceil = static_cast<float>(*v);
            if (auto v = edge["use_anisotropic_diffusion"].value<bool>()) cfg.edge.use_anisotropic_diffusion = *v;
            if (auto v = edge["diffusion_iterations"].value<int>()) cfg.edge.diffusion_iterations = *v;
            if (auto v = edge["diffusion_kappa"].value<double>()) cfg.edge.diffusion_kappa = static_cast<float>(*v);
            if (auto v = edge["diffusion_lambda"].value<double>()) cfg.edge.diffusion_lambda = static_cast<float>(*v);
            if (auto v = edge["adaptive_mode"].value<std::string>()) cfg.edge.adaptive_mode = *v;
            if (auto v = edge["tile_size"].value<int>()) cfg.edge.tile_size = *v;
            if (auto v = edge["dark_scene_floor"].value<double>()) cfg.edge.dark_scene_floor = static_cast<float>(*v);
            if (auto v = edge["global_percentile"].value<double>()) cfg.edge.global_percentile = static_cast<float>(*v);
        }
        
        if (auto temporal = tbl["temporal"]) {
            if (auto v = temporal["alpha"].value<double>()) cfg.temporal.alpha = static_cast<float>(*v);
            if (auto v = temporal["transition_penalty"].value<double>()) cfg.temporal.transition_penalty = static_cast<float>(*v);
            if (auto v = temporal["edge_enter_threshold"].value<double>()) cfg.temporal.edge_enter_threshold = static_cast<float>(*v);
            if (auto v = temporal["edge_exit_threshold"].value<double>()) cfg.temporal.edge_exit_threshold = static_cast<float>(*v);
            if (auto v = temporal["motion_cap_pixels"].value<int>()) cfg.temporal.motion_cap_pixels = *v;
            if (auto v = temporal["motion_solve_divisor"].value<int>()) cfg.temporal.motion_solve_divisor = *v;
            if (auto v = temporal["motion_max_reuse_frames"].value<int>()) cfg.temporal.motion_max_reuse_frames = *v;
            if (auto v = temporal["motion_reuse_scene_threshold"].value<double>()) cfg.temporal.motion_reuse_scene_threshold = static_cast<float>(*v);
            if (auto v = temporal["motion_reuse_confidence_decay"].value<double>()) cfg.temporal.motion_reuse_confidence_decay = static_cast<float>(*v);
            if (auto v = temporal["motion_still_scene_threshold"].value<double>()) cfg.temporal.motion_still_scene_threshold = static_cast<float>(*v);
            if (auto v = temporal["use_wavelet_flicker"].value<bool>()) cfg.temporal.use_wavelet_flicker = *v;
            if (auto v = temporal["wavelet_strength"].value<double>()) cfg.temporal.wavelet_strength = static_cast<float>(*v);
            if (auto v = temporal["wavelet_window"].value<int>()) cfg.temporal.wavelet_window = *v;
            if (auto v = temporal["use_phase_correlation"].value<bool>()) cfg.temporal.use_phase_correlation = *v;
            if (auto v = temporal["phase_search_radius"].value<int>()) cfg.temporal.phase_search_radius = *v;
            if (auto v = temporal["phase_blend"].value<double>()) cfg.temporal.phase_blend = static_cast<float>(*v);
            if (auto v = temporal["motion_phase_interval"].value<int>()) cfg.temporal.motion_phase_interval = *v;
            if (auto v = temporal["motion_phase_scene_trigger"].value<double>()) cfg.temporal.motion_phase_scene_trigger = static_cast<float>(*v);
        }
        
        if (auto selector = tbl["selector"]) {
            if (auto v = selector["char_set"].value<std::string>()) cfg.selector.char_set = *v;
            if (auto v = selector["mode"].value<std::string>()) cfg.selector.mode = *v;
            if (auto v = selector["weight_brightness"].value<double>()) cfg.selector.weight_brightness = static_cast<float>(*v);
            if (auto v = selector["weight_orientation"].value<double>()) cfg.selector.weight_orientation = static_cast<float>(*v);
            if (auto v = selector["weight_contrast"].value<double>()) cfg.selector.weight_contrast = static_cast<float>(*v);
            if (auto v = selector["weight_frequency"].value<double>()) cfg.selector.weight_frequency = static_cast<float>(*v);
            if (auto v = selector["weight_texture"].value<double>()) cfg.selector.weight_texture = static_cast<float>(*v);
            if (auto v = selector["enable_frequency_matching"].value<bool>()) cfg.selector.enable_frequency_matching = *v;
            if (auto v = selector["enable_gabor_texture"].value<bool>()) cfg.selector.enable_gabor_texture = *v;
            if (auto v = selector["use_simple_orientation"].value<bool>()) cfg.selector.use_simple_orientation = *v;
        }
        
        if (auto color = tbl["color"]) {
            if (auto v = color["mode"].value<std::string>()) {
                if (*v == "none") cfg.color.mode = ColorMode::None;
                else if (*v == "ansi16") cfg.color.mode = ColorMode::Ansi16;
                else if (*v == "ansi256") cfg.color.mode = ColorMode::Ansi256;
                else if (*v == "truecolor") cfg.color.mode = ColorMode::Truecolor;
                else if (*v == "blockart") cfg.color.mode = ColorMode::BlockArt;
            }
            if (auto v = color["quantization"].value<std::string>()) cfg.color.quantization = *v;
            if (auto v = color["dither_error_clamp"].value<double>()) cfg.color.dither_error_clamp = static_cast<float>(*v);
            if (auto v = color["use_blue_noise_halftone"].value<bool>()) cfg.color.use_blue_noise_halftone = *v;
            if (auto v = color["halftone_strength"].value<double>()) cfg.color.halftone_strength = static_cast<float>(*v);
            if (auto v = color["halftone_cell_size"].value<int>()) cfg.color.halftone_cell_size = *v;
            if (auto v = color["use_bilateral_grid"].value<bool>()) cfg.color.use_bilateral_grid = *v;
            if (auto v = color["bilateral_spatial_bins"].value<int>()) cfg.color.bilateral_spatial_bins = *v;
            if (auto v = color["bilateral_range_bins"].value<int>()) cfg.color.bilateral_range_bins = *v;
            if (auto v = color["bilateral_spatial_sigma"].value<double>()) cfg.color.bilateral_spatial_sigma = static_cast<float>(*v);
            if (auto v = color["bilateral_range_sigma"].value<double>()) cfg.color.bilateral_range_sigma = static_cast<float>(*v);
            if (auto v = color["block_spectral_palette"].value<int>()) cfg.color.block_spectral_palette = *v;
            if (auto v = color["block_spectral_samples"].value<int>()) cfg.color.block_spectral_samples = *v;
            if (auto v = color["block_spectral_iterations"].value<int>()) cfg.color.block_spectral_iterations = *v;
        }
        
        if (auto debug = tbl["debug"]) {
            if (auto v = debug["enabled"].value<bool>()) cfg.debug.enabled = *v;
            if (auto v = debug["mode"].value<std::string>()) cfg.debug.mode = *v;
            if (auto v = debug["profile_live"].value<bool>()) cfg.debug.profile_live = *v;
            if (auto v = debug["strict_memory"].value<bool>()) cfg.debug.strict_memory = *v;
        }
        if (auto v = tbl["profile"].value<std::string>()) cfg.profile = *v;
        
        if (auto v = tbl["font_path"].value<std::string>()) cfg.font_path = *v;
        if (auto v = tbl["fps"].value<int>()) cfg.fps = *v;
        if (auto v = tbl["no_audio"].value<bool>()) cfg.no_audio = *v;
        
        std::string error;
        if (!cfg.validate(error)) {
            return std::nullopt;
        }
        
        return cfg;
    } catch (const toml::parse_error&) {
        return std::nullopt;
    }
}

std::optional<Config> Config::load_default() {
    std::string path = default_config_path();
    return load(path);
}

Config merge_config(Config base, const Config& override) {
    Config result = base;
    
    if (!override.input.source.empty()) result.input.source = override.input.source;
    if (!override.input.mode.empty()) result.input.mode = override.input.mode;
    if (!override.output.target.empty()) result.output.target = override.output.target;
    if (!override.output.mode.empty()) result.output.mode = override.output.mode;
    if (!override.output.replay_path.empty()) result.output.replay_path = override.output.replay_path;
    
    if (override.grid.cols != 0) result.grid.cols = override.grid.cols;
    if (override.grid.rows != 0) result.grid.rows = override.grid.rows;
    if (override.grid.cell_width != Config::defaults().grid.cell_width) 
        result.grid.cell_width = override.grid.cell_width;
    if (override.grid.cell_height != Config::defaults().grid.cell_height) 
        result.grid.cell_height = override.grid.cell_height;
    if (override.grid.char_aspect != Config::defaults().grid.char_aspect)
        result.grid.char_aspect = override.grid.char_aspect;
    if (!override.grid.scale_mode.empty()) result.grid.scale_mode = override.grid.scale_mode;
    result.grid.quad_tree_adaptive = override.grid.quad_tree_adaptive;
    if (override.grid.quad_tree_max_depth != Config::defaults().grid.quad_tree_max_depth)
        result.grid.quad_tree_max_depth = override.grid.quad_tree_max_depth;
    if (override.grid.quad_tree_variance_threshold != Config::defaults().grid.quad_tree_variance_threshold)
        result.grid.quad_tree_variance_threshold = override.grid.quad_tree_variance_threshold;
    
    if (override.edge.low_threshold != Config::defaults().edge.low_threshold)
        result.edge.low_threshold = override.edge.low_threshold;
    if (override.edge.high_threshold != Config::defaults().edge.high_threshold)
        result.edge.high_threshold = override.edge.high_threshold;
    if (override.edge.blur_sigma != Config::defaults().edge.blur_sigma)
        result.edge.blur_sigma = override.edge.blur_sigma;
    result.edge.use_hysteresis = override.edge.use_hysteresis;
    result.edge.multi_scale = override.edge.multi_scale;
    if (override.edge.scale_sigma_0 != Config::defaults().edge.scale_sigma_0)
        result.edge.scale_sigma_0 = override.edge.scale_sigma_0;
    if (override.edge.scale_sigma_1 != Config::defaults().edge.scale_sigma_1)
        result.edge.scale_sigma_1 = override.edge.scale_sigma_1;
    result.edge.adaptive_scale_selection = override.edge.adaptive_scale_selection;
    if (override.edge.scale_variance_floor != Config::defaults().edge.scale_variance_floor)
        result.edge.scale_variance_floor = override.edge.scale_variance_floor;
    if (override.edge.scale_variance_ceil != Config::defaults().edge.scale_variance_ceil)
        result.edge.scale_variance_ceil = override.edge.scale_variance_ceil;
    result.edge.use_anisotropic_diffusion = override.edge.use_anisotropic_diffusion;
    if (override.edge.diffusion_iterations != Config::defaults().edge.diffusion_iterations)
        result.edge.diffusion_iterations = override.edge.diffusion_iterations;
    if (override.edge.diffusion_kappa != Config::defaults().edge.diffusion_kappa)
        result.edge.diffusion_kappa = override.edge.diffusion_kappa;
    if (override.edge.diffusion_lambda != Config::defaults().edge.diffusion_lambda)
        result.edge.diffusion_lambda = override.edge.diffusion_lambda;
    if (!override.edge.adaptive_mode.empty()) result.edge.adaptive_mode = override.edge.adaptive_mode;
    if (override.edge.tile_size != Config::defaults().edge.tile_size)
        result.edge.tile_size = override.edge.tile_size;
    if (override.edge.dark_scene_floor != Config::defaults().edge.dark_scene_floor)
        result.edge.dark_scene_floor = override.edge.dark_scene_floor;
    if (override.edge.global_percentile != Config::defaults().edge.global_percentile)
        result.edge.global_percentile = override.edge.global_percentile;
    
    if (override.temporal.alpha != Config::defaults().temporal.alpha)
        result.temporal.alpha = override.temporal.alpha;
    if (override.temporal.transition_penalty != Config::defaults().temporal.transition_penalty)
        result.temporal.transition_penalty = override.temporal.transition_penalty;
    if (override.temporal.edge_enter_threshold != Config::defaults().temporal.edge_enter_threshold)
        result.temporal.edge_enter_threshold = override.temporal.edge_enter_threshold;
    if (override.temporal.edge_exit_threshold != Config::defaults().temporal.edge_exit_threshold)
        result.temporal.edge_exit_threshold = override.temporal.edge_exit_threshold;
    if (override.temporal.motion_cap_pixels != Config::defaults().temporal.motion_cap_pixels)
        result.temporal.motion_cap_pixels = override.temporal.motion_cap_pixels;
    if (override.temporal.motion_solve_divisor != Config::defaults().temporal.motion_solve_divisor)
        result.temporal.motion_solve_divisor = override.temporal.motion_solve_divisor;
    if (override.temporal.motion_max_reuse_frames != Config::defaults().temporal.motion_max_reuse_frames)
        result.temporal.motion_max_reuse_frames = override.temporal.motion_max_reuse_frames;
    if (override.temporal.motion_reuse_scene_threshold != Config::defaults().temporal.motion_reuse_scene_threshold)
        result.temporal.motion_reuse_scene_threshold = override.temporal.motion_reuse_scene_threshold;
    if (override.temporal.motion_reuse_confidence_decay != Config::defaults().temporal.motion_reuse_confidence_decay)
        result.temporal.motion_reuse_confidence_decay = override.temporal.motion_reuse_confidence_decay;
    if (override.temporal.motion_still_scene_threshold != Config::defaults().temporal.motion_still_scene_threshold)
        result.temporal.motion_still_scene_threshold = override.temporal.motion_still_scene_threshold;
    result.temporal.use_wavelet_flicker = override.temporal.use_wavelet_flicker;
    if (override.temporal.wavelet_strength != Config::defaults().temporal.wavelet_strength)
        result.temporal.wavelet_strength = override.temporal.wavelet_strength;
    if (override.temporal.wavelet_window != Config::defaults().temporal.wavelet_window)
        result.temporal.wavelet_window = override.temporal.wavelet_window;
    result.temporal.use_phase_correlation = override.temporal.use_phase_correlation;
    if (override.temporal.phase_search_radius != Config::defaults().temporal.phase_search_radius)
        result.temporal.phase_search_radius = override.temporal.phase_search_radius;
    if (override.temporal.phase_blend != Config::defaults().temporal.phase_blend)
        result.temporal.phase_blend = override.temporal.phase_blend;
    if (override.temporal.motion_phase_interval != Config::defaults().temporal.motion_phase_interval)
        result.temporal.motion_phase_interval = override.temporal.motion_phase_interval;
    if (override.temporal.motion_phase_scene_trigger != Config::defaults().temporal.motion_phase_scene_trigger)
        result.temporal.motion_phase_scene_trigger = override.temporal.motion_phase_scene_trigger;
    
    if (!override.selector.char_set.empty()) result.selector.char_set = override.selector.char_set;
    if (!override.selector.mode.empty()) result.selector.mode = override.selector.mode;
    if (override.selector.weight_brightness != Config::defaults().selector.weight_brightness)
        result.selector.weight_brightness = override.selector.weight_brightness;
    if (override.selector.weight_orientation != Config::defaults().selector.weight_orientation)
        result.selector.weight_orientation = override.selector.weight_orientation;
    if (override.selector.weight_contrast != Config::defaults().selector.weight_contrast)
        result.selector.weight_contrast = override.selector.weight_contrast;
    if (override.selector.weight_frequency != Config::defaults().selector.weight_frequency)
        result.selector.weight_frequency = override.selector.weight_frequency;
    if (override.selector.weight_texture != Config::defaults().selector.weight_texture)
        result.selector.weight_texture = override.selector.weight_texture;
    result.selector.enable_frequency_matching = override.selector.enable_frequency_matching;
    result.selector.enable_gabor_texture = override.selector.enable_gabor_texture;
    result.selector.use_simple_orientation = override.selector.use_simple_orientation;
    
    if (override.color.mode != Config::defaults().color.mode)
        result.color.mode = override.color.mode;
    if (!override.color.quantization.empty()) result.color.quantization = override.color.quantization;
    if (override.color.dither_error_clamp != Config::defaults().color.dither_error_clamp)
        result.color.dither_error_clamp = override.color.dither_error_clamp;
    result.color.use_blue_noise_halftone = override.color.use_blue_noise_halftone;
    if (override.color.halftone_strength != Config::defaults().color.halftone_strength)
        result.color.halftone_strength = override.color.halftone_strength;
    if (override.color.halftone_cell_size != Config::defaults().color.halftone_cell_size)
        result.color.halftone_cell_size = override.color.halftone_cell_size;
    result.color.use_bilateral_grid = override.color.use_bilateral_grid;
    if (override.color.bilateral_spatial_bins != Config::defaults().color.bilateral_spatial_bins)
        result.color.bilateral_spatial_bins = override.color.bilateral_spatial_bins;
    if (override.color.bilateral_range_bins != Config::defaults().color.bilateral_range_bins)
        result.color.bilateral_range_bins = override.color.bilateral_range_bins;
    if (override.color.bilateral_spatial_sigma != Config::defaults().color.bilateral_spatial_sigma)
        result.color.bilateral_spatial_sigma = override.color.bilateral_spatial_sigma;
    if (override.color.bilateral_range_sigma != Config::defaults().color.bilateral_range_sigma)
        result.color.bilateral_range_sigma = override.color.bilateral_range_sigma;
    if (override.color.block_spectral_palette != Config::defaults().color.block_spectral_palette)
        result.color.block_spectral_palette = override.color.block_spectral_palette;
    if (override.color.block_spectral_samples != Config::defaults().color.block_spectral_samples)
        result.color.block_spectral_samples = override.color.block_spectral_samples;
    if (override.color.block_spectral_iterations != Config::defaults().color.block_spectral_iterations)
        result.color.block_spectral_iterations = override.color.block_spectral_iterations;
    
    result.debug = override.debug;
    if (!override.profile.empty()) result.profile = override.profile;
    
    if (!override.font_path.empty()) result.font_path = override.font_path;
    if (override.fps != Config::defaults().fps) result.fps = override.fps;
    result.no_audio = override.no_audio;
    
    return result;
}

Config apply_cli_overrides(Config config, const Args& args) {
    if (!args.input.empty()) config.input.source = args.input;
    if (!args.output.empty()) config.output.target = args.output;
    if (!args.replay_path.empty()) config.output.replay_path = args.replay_path;
    if (!args.font_path.empty()) config.font_path = args.font_path;
    if (!args.char_set.empty()) config.selector.char_set = args.char_set;
    if (!args.profile.empty()) config.profile = args.profile;
    if (!args.debug_mode.empty()) {
        config.debug.enabled = true;
        config.debug.mode = args.debug_mode;
    }
    
    if (args.cols > 0) config.grid.cols = args.cols;
    if (args.rows > 0) config.grid.rows = args.rows;
    if (args.cell_width != Config::defaults().grid.cell_width) 
        config.grid.cell_width = args.cell_width;
    if (args.cell_height != Config::defaults().grid.cell_height) 
        config.grid.cell_height = args.cell_height;
    
    if (args.edge_threshold != Config::defaults().edge.high_threshold) {
        config.edge.high_threshold = args.edge_threshold;
        config.edge.low_threshold = args.edge_threshold * 0.5f;
    }
    if (args.blur_sigma != Config::defaults().edge.blur_sigma)
        config.edge.blur_sigma = args.blur_sigma;
    config.edge.use_hysteresis = args.use_hysteresis;
    
    if (args.temporal_alpha != Config::defaults().temporal.alpha)
        config.temporal.alpha = args.temporal_alpha;
    if (args.motion_solve_divisor > 0)
        config.temporal.motion_solve_divisor = args.motion_solve_divisor;
    if (args.motion_max_reuse_frames >= 0)
        config.temporal.motion_max_reuse_frames = args.motion_max_reuse_frames;
    if (args.motion_reuse_scene_threshold >= 0.0f)
        config.temporal.motion_reuse_scene_threshold = args.motion_reuse_scene_threshold;
    if (args.motion_reuse_confidence_decay >= 0.0f)
        config.temporal.motion_reuse_confidence_decay = args.motion_reuse_confidence_decay;
    if (args.motion_phase_interval > 0)
        config.temporal.motion_phase_interval = args.motion_phase_interval;
    if (args.motion_phase_scene_trigger >= 0.0f)
        config.temporal.motion_phase_scene_trigger = args.motion_phase_scene_trigger;
    if (args.motion_still_scene_threshold >= 0.0f)
        config.temporal.motion_still_scene_threshold = args.motion_still_scene_threshold;
    
    if (!args.scale_mode.empty()) config.grid.scale_mode = args.scale_mode;
    
    if (args.color_mode_set)
        config.color.mode = args.color_mode;
    
    config.selector.use_simple_orientation = args.use_simple_orientation;

    if (args.fps != Config::defaults().fps) config.fps = args.fps;
    config.no_audio = args.no_audio;
    config.debug.profile_live = args.profile_live;
    config.debug.strict_memory = args.strict_memory;
    if (args.fast_mode) {
        config.selector.mode = "simple";
        config.selector.use_simple_orientation = true;
        config.selector.enable_frequency_matching = false;
        config.selector.enable_gabor_texture = false;

        config.edge.multi_scale = false;
        config.edge.adaptive_scale_selection = false;
        config.edge.use_anisotropic_diffusion = false;

        config.temporal.use_phase_correlation = false;
        config.temporal.use_wavelet_flicker = false;
        config.temporal.motion_cap_pixels = 0;

        config.grid.quad_tree_adaptive = false;

        config.color.use_bilateral_grid = false;
        config.color.block_spectral_palette = 0;
    }

    return config;
}

void apply_content_profile(Config& config) {
    if (config.profile.empty()) {
        return;
    }

    if (config.profile == "natural") {
        config.edge.low_threshold = 0.05f;
        config.edge.high_threshold = 0.10f;
        config.temporal.wavelet_strength = 0.45f;
        config.temporal.wavelet_window = 8;
        config.temporal.phase_search_radius = 6;
        config.temporal.phase_blend = 0.28f;
        config.color.halftone_strength = 0.24f;
        config.color.halftone_cell_size = 6;
        return;
    }

    if (config.profile == "anime") {
        config.selector.weight_brightness = 0.36f;
        config.selector.weight_orientation = 0.50f;
        config.selector.weight_contrast = 0.14f;
        config.selector.weight_frequency = 0.12f;
        config.selector.weight_texture = 0.08f;
        config.edge.low_threshold = 0.04f;
        config.edge.high_threshold = 0.085f;
        config.temporal.wavelet_strength = 0.56f;
        config.temporal.wavelet_window = 8;
        config.temporal.phase_search_radius = 5;
        config.temporal.phase_blend = 0.20f;
        config.color.halftone_strength = 0.16f;
        config.color.halftone_cell_size = 8;
        return;
    }

    if (config.profile == "ui") {
        config.selector.weight_brightness = 0.30f;
        config.selector.weight_orientation = 0.38f;
        config.selector.weight_contrast = 0.22f;
        config.selector.weight_frequency = 0.24f;
        config.selector.weight_texture = 0.10f;
        config.edge.low_threshold = 0.06f;
        config.edge.high_threshold = 0.12f;
        config.temporal.wavelet_strength = 0.34f;
        config.temporal.wavelet_window = 8;
        config.temporal.phase_search_radius = 4;
        config.temporal.phase_blend = 0.18f;
        config.color.use_blue_noise_halftone = false;
        config.color.halftone_strength = 0.18f;
        config.color.halftone_cell_size = 6;
        return;
    }
}

}
