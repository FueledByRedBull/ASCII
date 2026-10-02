#include "core/config.hpp"
#include "cli/args.hpp"
#include <toml.hpp>

#include <filesystem>
#include <sstream>
#include <iomanip>
#include <functional>
#include <cstring>
#include <cmath>
#include <type_traits>

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
    if (input.mode != "file") {
        error = "input.mode must be 'file'; the source URI selects the input type";
        return false;
    }
    if (output.mode != "terminal") {
        error = "output.mode must be 'terminal'; the output target selects the output type";
        return false;
    }
    if (color.quantization != "oklab") {
        error = "color.quantization must be 'oklab'";
        return false;
    }
    for (float value : {
             grid.char_aspect, grid.quad_tree_variance_threshold,
             edge.low_threshold, edge.high_threshold, edge.blur_sigma,
             edge.scale_sigma_0, edge.scale_sigma_1, edge.scale_variance_floor, edge.scale_variance_ceil,
             edge.diffusion_kappa, edge.diffusion_lambda, edge.dark_scene_floor, edge.global_percentile,
             edge.contour_min_occupancy, edge.contour_dominance_ratio, edge.contour_intersection_ratio,
             edge.contour_dog_sigma_inner, edge.contour_dog_sigma_outer,
             temporal.alpha, temporal.transition_penalty, temporal.edge_enter_threshold, temporal.edge_exit_threshold,
             temporal.motion_reuse_scene_threshold, temporal.motion_reuse_confidence_decay,
             temporal.motion_still_scene_threshold, temporal.wavelet_strength, temporal.phase_blend,
             temporal.motion_phase_scene_trigger, selector.weight_brightness, selector.weight_orientation,
             selector.weight_contrast, selector.weight_frequency, selector.weight_texture,
             color.dither_error_clamp, color.halftone_strength,
             color.bilateral_spatial_sigma, color.bilateral_range_sigma}) {
        if (!std::isfinite(value)) {
            error = "floating-point configuration values must be finite";
            return false;
        }
    }
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
    if (grid.char_aspect < 1.0f / 32.0f || grid.char_aspect > 64.0f) {
        error = "grid.char_aspect must be between 1/32 and 64";
        return false;
    }
    const uint64_t pixel_width = static_cast<uint64_t>(grid.cols) * grid.cell_width;
    const uint64_t pixel_height = static_cast<uint64_t>(grid.rows) * grid.cell_height;
    if (pixel_width != 0 && pixel_height > 100000000ull / pixel_width) {
        error = "configured output grid exceeds the 100 megapixel safety cap";
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
    if (edge.scale_sigma_0 <= 0.0f || edge.scale_sigma_0 > 10.0f ||
        edge.scale_sigma_1 <= 0.0f || edge.scale_sigma_1 > 10.0f) {
        error = "edge.scale_sigma_0 and edge.scale_sigma_1 must be > 0 and <= 10";
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
    if (edge.contour_min_occupancy < 0.0f || edge.contour_min_occupancy > 1.0f) {
        error = "edge.contour_min_occupancy must be between 0.0 and 1.0";
        return false;
    }
    if (edge.contour_min_pixels < 0 || edge.contour_min_pixels > 4096) {
        error = "edge.contour_min_pixels must be between 0 and 4096";
        return false;
    }
    if (edge.contour_dominance_ratio < 1.0f || edge.contour_intersection_ratio < 1.0f) {
        error = "edge contour ratios must be >= 1.0";
        return false;
    }
    if (edge.contour_dog_sigma_inner < 0.1f || edge.contour_dog_sigma_outer > 10.0f ||
        edge.contour_dog_sigma_inner >= edge.contour_dog_sigma_outer) {
        error = "edge contour DoG sigmas must be between 0.1 and 10 with inner < outer";
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
    if (temporal.edge_exit_threshold < 0.0f || temporal.edge_enter_threshold > 1.0f ||
        temporal.edge_exit_threshold > temporal.edge_enter_threshold) {
        error = "temporal edge thresholds must be between 0 and 1 with exit <= enter";
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
    if (temporal.motion_cap_pixels < 0 || temporal.motion_cap_pixels > 256) {
        error = "temporal.motion_cap_pixels must be between 0 and 256";
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
    if (!std::isfinite(total_weight) || total_weight < 0.001f) {
        error = "selector weights must sum to a finite positive value";
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
    if (selector.char_set != "basic" && selector.char_set != "traditional" && selector.char_set != "blocks" &&
        selector.char_set != "line-art") {
        error = "selector.char_set must be 'basic', 'traditional', 'blocks', or 'line-art'";
        return false;
    }
    if (!debug.mode.empty() && debug.mode != "grayscale" && debug.mode != "edges" &&
        debug.mode != "orientation") {
        error = "debug.mode must be '', 'grayscale', 'edges', or 'orientation'";
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
    h = hash_combine(h, hash_int(static_cast<int>(edge.contours_enabled)));
    h = hash_combine(h, hash_float(edge.contour_min_occupancy));
    h = hash_combine(h, hash_int(edge.contour_min_pixels));
    h = hash_combine(h, hash_float(edge.contour_dominance_ratio));
    h = hash_combine(h, hash_float(edge.contour_intersection_ratio));
    h = hash_combine(h, hash_float(edge.contour_dog_sigma_inner));
    h = hash_combine(h, hash_float(edge.contour_dog_sigma_outer));
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
    h = hash_combine(h, hash_int(static_cast<int>(selector.use_orientation_matching)));
    h = hash_combine(h, hash_int(static_cast<int>(selector.use_simple_orientation)));
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
    h = hash_combine(h, hash_int(static_cast<int>(debug.enabled)));
    h = hash_combine(h, hash_string(debug.mode));
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
        bool valid = true;
        const auto read = [&]<typename T>(toml::node_view<toml::node> node, T& destination) {
            if (!node) return;
            // TOML permits scalar coercions; application booleans and integers require their declared type.
            if constexpr (std::is_same_v<T, bool>) {
                if (!node.is_boolean()) { valid = false; return; }
            } else if constexpr (std::is_same_v<T, int>) {
                if (!node.is_integer()) { valid = false; return; }
            }
            if (auto value = node.template value<T>()) destination = *value;
            else valid = false;
        };

        read(tbl["profile"], cfg.profile);
        apply_content_profile(cfg);
        
        read(tbl["config_version"], cfg.version);
        if (!valid || cfg.version != CONFIG_VERSION) return std::nullopt;
        
        if (auto input = tbl["input"]) {
            if (!input.is_table()) return std::nullopt;
            read(input["source"], cfg.input.source);
            read(input["mode"], cfg.input.mode);
        }
        
        if (auto output = tbl["output"]) {
            if (!output.is_table()) return std::nullopt;
            read(output["target"], cfg.output.target);
            read(output["mode"], cfg.output.mode);
            read(output["replay_path"], cfg.output.replay_path);
        }
        
        if (auto grid = tbl["grid"]) {
            if (!grid.is_table()) return std::nullopt;
            read(grid["cols"], cfg.grid.cols);
            read(grid["rows"], cfg.grid.rows);
            read(grid["cell_width"], cfg.grid.cell_width);
            read(grid["cell_height"], cfg.grid.cell_height);
            read(grid["char_aspect"], cfg.grid.char_aspect);
            read(grid["scale_mode"], cfg.grid.scale_mode);
            read(grid["quad_tree_adaptive"], cfg.grid.quad_tree_adaptive);
            read(grid["quad_tree_max_depth"], cfg.grid.quad_tree_max_depth);
            read(grid["quad_tree_variance_threshold"], cfg.grid.quad_tree_variance_threshold);
        }
        
        if (auto edge = tbl["edge"]) {
            if (!edge.is_table()) return std::nullopt;
            read(edge["low_threshold"], cfg.edge.low_threshold);
            read(edge["high_threshold"], cfg.edge.high_threshold);
            read(edge["blur_sigma"], cfg.edge.blur_sigma);
            read(edge["use_hysteresis"], cfg.edge.use_hysteresis);
            read(edge["multi_scale"], cfg.edge.multi_scale);
            read(edge["scale_sigma_0"], cfg.edge.scale_sigma_0);
            read(edge["scale_sigma_1"], cfg.edge.scale_sigma_1);
            read(edge["adaptive_scale_selection"], cfg.edge.adaptive_scale_selection);
            read(edge["scale_variance_floor"], cfg.edge.scale_variance_floor);
            read(edge["scale_variance_ceil"], cfg.edge.scale_variance_ceil);
            read(edge["use_anisotropic_diffusion"], cfg.edge.use_anisotropic_diffusion);
            read(edge["diffusion_iterations"], cfg.edge.diffusion_iterations);
            read(edge["diffusion_kappa"], cfg.edge.diffusion_kappa);
            read(edge["diffusion_lambda"], cfg.edge.diffusion_lambda);
            read(edge["adaptive_mode"], cfg.edge.adaptive_mode);
            read(edge["tile_size"], cfg.edge.tile_size);
            read(edge["dark_scene_floor"], cfg.edge.dark_scene_floor);
            read(edge["global_percentile"], cfg.edge.global_percentile);
            read(edge["contours_enabled"], cfg.edge.contours_enabled);
            read(edge["contour_min_occupancy"], cfg.edge.contour_min_occupancy);
            read(edge["contour_min_pixels"], cfg.edge.contour_min_pixels);
            read(edge["contour_dominance_ratio"], cfg.edge.contour_dominance_ratio);
            read(edge["contour_intersection_ratio"], cfg.edge.contour_intersection_ratio);
            read(edge["contour_dog_sigma_inner"], cfg.edge.contour_dog_sigma_inner);
            read(edge["contour_dog_sigma_outer"], cfg.edge.contour_dog_sigma_outer);
        }
        
        if (auto temporal = tbl["temporal"]) {
            if (!temporal.is_table()) return std::nullopt;
            read(temporal["alpha"], cfg.temporal.alpha);
            read(temporal["transition_penalty"], cfg.temporal.transition_penalty);
            read(temporal["edge_enter_threshold"], cfg.temporal.edge_enter_threshold);
            read(temporal["edge_exit_threshold"], cfg.temporal.edge_exit_threshold);
            read(temporal["motion_cap_pixels"], cfg.temporal.motion_cap_pixels);
            read(temporal["motion_solve_divisor"], cfg.temporal.motion_solve_divisor);
            read(temporal["motion_max_reuse_frames"], cfg.temporal.motion_max_reuse_frames);
            read(temporal["motion_reuse_scene_threshold"], cfg.temporal.motion_reuse_scene_threshold);
            read(temporal["motion_reuse_confidence_decay"], cfg.temporal.motion_reuse_confidence_decay);
            read(temporal["motion_still_scene_threshold"], cfg.temporal.motion_still_scene_threshold);
            read(temporal["use_wavelet_flicker"], cfg.temporal.use_wavelet_flicker);
            read(temporal["wavelet_strength"], cfg.temporal.wavelet_strength);
            read(temporal["wavelet_window"], cfg.temporal.wavelet_window);
            read(temporal["use_phase_correlation"], cfg.temporal.use_phase_correlation);
            read(temporal["phase_search_radius"], cfg.temporal.phase_search_radius);
            read(temporal["phase_blend"], cfg.temporal.phase_blend);
            read(temporal["motion_phase_interval"], cfg.temporal.motion_phase_interval);
            read(temporal["motion_phase_scene_trigger"], cfg.temporal.motion_phase_scene_trigger);
        }
        
        if (auto selector = tbl["selector"]) {
            if (!selector.is_table()) return std::nullopt;
            read(selector["char_set"], cfg.selector.char_set);
            read(selector["mode"], cfg.selector.mode);
            read(selector["weight_brightness"], cfg.selector.weight_brightness);
            read(selector["weight_orientation"], cfg.selector.weight_orientation);
            read(selector["weight_contrast"], cfg.selector.weight_contrast);
            read(selector["weight_frequency"], cfg.selector.weight_frequency);
            read(selector["weight_texture"], cfg.selector.weight_texture);
            read(selector["use_orientation_matching"], cfg.selector.use_orientation_matching);
            read(selector["enable_frequency_matching"], cfg.selector.enable_frequency_matching);
            read(selector["enable_gabor_texture"], cfg.selector.enable_gabor_texture);
            read(selector["use_simple_orientation"], cfg.selector.use_simple_orientation);
        }
        
        if (auto color = tbl["color"]) {
            if (!color.is_table()) return std::nullopt;
            if (color["mode"] && !color["mode"].is_string()) return std::nullopt;
            if (auto v = color["mode"].value<std::string>()) {
                if (*v == "none") cfg.color.mode = ColorMode::None;
                else if (*v == "ansi16") cfg.color.mode = ColorMode::Ansi16;
                else if (*v == "ansi256") cfg.color.mode = ColorMode::Ansi256;
                else if (*v == "truecolor") cfg.color.mode = ColorMode::Truecolor;
                else if (*v == "blockart") cfg.color.mode = ColorMode::BlockArt;
                else return std::nullopt;
            }
            read(color["quantization"], cfg.color.quantization);
            read(color["dither_error_clamp"], cfg.color.dither_error_clamp);
            read(color["use_blue_noise_halftone"], cfg.color.use_blue_noise_halftone);
            read(color["halftone_strength"], cfg.color.halftone_strength);
            read(color["halftone_cell_size"], cfg.color.halftone_cell_size);
            read(color["use_bilateral_grid"], cfg.color.use_bilateral_grid);
            read(color["bilateral_spatial_bins"], cfg.color.bilateral_spatial_bins);
            read(color["bilateral_range_bins"], cfg.color.bilateral_range_bins);
            read(color["bilateral_spatial_sigma"], cfg.color.bilateral_spatial_sigma);
            read(color["bilateral_range_sigma"], cfg.color.bilateral_range_sigma);
            read(color["block_spectral_palette"], cfg.color.block_spectral_palette);
            read(color["block_spectral_samples"], cfg.color.block_spectral_samples);
            read(color["block_spectral_iterations"], cfg.color.block_spectral_iterations);
        }
        
        if (auto debug = tbl["debug"]) {
            if (!debug.is_table()) return std::nullopt;
            read(debug["enabled"], cfg.debug.enabled);
            read(debug["mode"], cfg.debug.mode);
            read(debug["profile_live"], cfg.debug.profile_live);
            read(debug["strict_memory"], cfg.debug.strict_memory);
        }
        read(tbl["font_path"], cfg.font_path);
        read(tbl["fps"], cfg.fps);
        read(tbl["no_audio"], cfg.no_audio);
        
        std::string error;
        if (!valid || !cfg.validate(error)) {
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

Config apply_cli_overrides(Config config, const Args& args) {
    if (!args.input.empty()) config.input.source = args.input;
    if (!args.output.empty()) config.output.target = args.output;
    if (!args.replay_path.empty()) config.output.replay_path = args.replay_path;
    if (args.font_path_set) config.font_path = args.font_path;
    if (args.char_set_set) config.selector.char_set = args.char_set;
    if (!args.profile.empty()) config.profile = args.profile;
    if (!args.debug_mode.empty()) {
        config.debug.enabled = true;
        config.debug.mode = args.debug_mode;
    }
    
    if (args.cols_set) config.grid.cols = args.cols;
    if (args.rows_set) config.grid.rows = args.rows;
    if (args.cell_width != Config::defaults().grid.cell_width) 
        config.grid.cell_width = args.cell_width;
    if (args.cell_height != Config::defaults().grid.cell_height) 
        config.grid.cell_height = args.cell_height;
    
    if (args.edge_threshold_set) {
        config.edge.high_threshold = args.edge_threshold;
        config.edge.low_threshold = args.edge_threshold * 0.5f;
    }
    if (args.blur_sigma_set)
        config.edge.blur_sigma = args.blur_sigma;
    if (args.use_hysteresis_set)
        config.edge.use_hysteresis = args.use_hysteresis;
    if (args.contours_enabled_set)
        config.edge.contours_enabled = args.contours_enabled;
    if (args.contour_threshold >= 0.0f)
        config.edge.contour_min_occupancy = args.contour_threshold;
    
    if (args.temporal_alpha_set)
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
    
    if (args.scale_mode_set) config.grid.scale_mode = args.scale_mode;
    
    if (args.color_mode_set)
        config.color.mode = args.color_mode;
    
    if (args.fps_set) config.fps = args.fps;
    if (args.no_audio_set) config.no_audio = args.no_audio;
    if (args.profile_live_set) config.debug.profile_live = args.profile_live;
    if (args.strict_memory_set) config.debug.strict_memory = args.strict_memory;
    if (args.fast_mode) {
        config.selector.mode = "simple";
        config.selector.use_simple_orientation = config.selector.use_orientation_matching ||
                                                 config.selector.use_simple_orientation;
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

    if (args.orientation_mode_set) {
        config.selector.use_orientation_matching = args.use_orientation_matching;
        config.selector.use_simple_orientation = args.use_simple_orientation;
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
