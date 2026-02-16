#include "core/types.hpp"
#include "core/config.hpp"
#include "core/frame_source.hpp"
#include "core/pipeline.hpp"
#include "core/motion.hpp"
#include "core/replay.hpp"
#include "core/temporal.hpp"
#include "core/color_space.hpp"
#include "glyph/font_loader.hpp"
#include "glyph/glyph_cache.hpp"
#include "glyph/char_sets.hpp"
#include "mapping/char_selector.hpp"
#include "mapping/color_mapper.hpp"
#include "mapping/bilateral_grid.hpp"
#include "render/dither.hpp"
#include "render/terminal_renderer.hpp"
#include "render/block_renderer.hpp"
#include "render/bitmap_renderer.hpp"
#include "render/video_encoder.hpp"
#include "audio/audio_player.hpp"
#include "terminal/terminal.hpp"
#include "cli/args.hpp"

#include <chrono>
#include <thread>
#include <iostream>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <conio.h>
#else
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace ascii {
namespace input {

#ifdef _WIN32
static DWORD original_mode = 0;
static HANDLE h_stdin = nullptr;
static bool console_modified = false;

void setup_nonblocking_stdin() {
    h_stdin = GetStdHandle(STD_INPUT_HANDLE);
    if (h_stdin == INVALID_HANDLE_VALUE) return;
    GetConsoleMode(h_stdin, &original_mode);
    SetConsoleMode(h_stdin, original_mode & ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT));
    console_modified = true;
}

void restore_stdin() {
    if (console_modified && h_stdin != nullptr) {
        SetConsoleMode(h_stdin, original_mode);
        console_modified = false;
    }
}

int read_key() {
    if (_kbhit()) {
        return _getch();
    }
    return -1;
}
#else
static termios original_termios;
static bool termios_modified = false;

void setup_nonblocking_stdin() {
    tcgetattr(STDIN_FILENO, &original_termios);
    termios new_termios = original_termios;
    new_termios.c_lflag &= ~(ICANON | ECHO);
    new_termios.c_cc[VMIN] = 0;
    new_termios.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
    termios_modified = true;
}

void restore_stdin() {
    if (termios_modified) {
        tcsetattr(STDIN_FILENO, TCSANOW, &original_termios);
        termios_modified = false;
    }
}

int read_key() {
    timeval tv = {0, 0};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(STDIN_FILENO, &fds);
    
    if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0) {
        unsigned char c;
        if (read(STDIN_FILENO, &c, 1) == 1) {
            return c;
        }
    }
    return -1;
}
#endif

std::string codepoint_to_utf8(uint32_t cp) {
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

bool ends_with_txt(const std::string& path) {
    if (path.size() < 4) return false;
    std::string ext = path.substr(path.size() - 4);
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext == ".txt";
}

void write_ascii_to_file(const std::string& path, const std::vector<ascii::ASCIICell>& cells, int cols, int rows) {
    std::ofstream out(path);
    if (!out) return;
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            if (idx < static_cast<int>(cells.size())) {
                out << codepoint_to_utf8(cells[idx].codepoint);
            }
        }
        out << '\n';
    }
}

char luminance_to_ascii(float lum) {
    static const char ramp[] = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$";
    int len = sizeof(ramp) - 1;
    int idx = static_cast<int>(lum * (len - 1));
    if (idx < 0) idx = 0;
    if (idx >= len) idx = len - 1;
    return ramp[idx];
}

void write_ascii_to_file_simple(const std::string& path, const std::vector<float>& luminance, int cols, int rows) {
    std::ofstream out(path);
    if (!out) return;
    
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int idx = y * cols + x;
            if (idx < static_cast<int>(luminance.size())) {
                out << luminance_to_ascii(luminance[idx]);
            }
        }
        out << '\n';
    }
}

}
}

namespace {

void build_scene_signature(const ascii::FrameBuffer& frame, std::vector<float>& signature) {
    signature.clear();
    const int w = frame.width();
    const int h = frame.height();
    if (w <= 0 || h <= 0 || frame.empty()) {
        return;
    }

    const int stride = std::max(1, std::min(w, h) / 48);
    const uint8_t* src = frame.data();
    const int samples_x = (w + stride - 1) / stride;
    const int samples_y = (h + stride - 1) / stride;
    signature.reserve(static_cast<size_t>(samples_x) * samples_y);

    for (int y = 0; y < h; y += stride) {
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(w) * 4;
        for (int x = 0; x < w; x += stride) {
            const size_t idx = row + static_cast<size_t>(x) * 4;
            // Fast luma in [0,1], BT.709 coefficients.
            float lum = (0.2126f * src[idx + 0] +
                         0.7152f * src[idx + 1] +
                         0.0722f * src[idx + 2]) / 255.0f;
            signature.push_back(lum);
        }
    }
}

float signature_scene_change(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) {
        return 1.0f;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return static_cast<float>(sum / static_cast<double>(a.size()));
}

}  // namespace

int main(int argc, char* argv[]) {
    ascii::Args args = ascii::parse_args(argc, argv);
    
    if (args.show_help) {
        ascii::print_help(argv[0]);
        return 0;
    }
    
    ascii::Terminal terminal;
    auto term_info = terminal.get_info();

    ascii::Config config = ascii::Config::defaults();
    if (!args.config_path.empty()) {
        auto loaded = ascii::Config::load(args.config_path);
        if (!loaded) {
            std::cerr << "Error: Failed to load config file: " << args.config_path << "\n";
            return 1;
        }
        config = ascii::merge_config(config, *loaded);
    } else {
        if (auto loaded_default = ascii::Config::load_default()) {
            config = ascii::merge_config(config, *loaded_default);
        }
    }
    if (!args.profile.empty()) {
        config.profile = args.profile;
    }
    ascii::apply_content_profile(config);
    config = ascii::apply_cli_overrides(config, args);
    if (!args.replay_path.empty()) {
        config.output.replay_path = args.replay_path;
    }
    if (!config.output.target.empty() &&
        config.color.mode == ascii::ColorMode::None &&
        !args.color_mode_set) {
        config.color.mode = ascii::ColorMode::Truecolor;
    }

    if (config.input.source.empty()) {
        std::cerr << "Error: No input specified\n";
        ascii::print_help(argv[0]);
        return 1;
    }

    std::string config_error;
    if (!config.validate(config_error)) {
        std::cerr << "Error: Invalid config: " << config_error << "\n";
        return 1;
    }

    int cols = config.grid.cols > 0 ? config.grid.cols : term_info.cols;
    int rows = config.grid.rows > 0 ? config.grid.rows : term_info.rows;

    ascii::ColorMode color_mode = config.color.mode;
    ascii::ColorSpace::init();
    
    ascii::FontLoader font_loader;
    if (!config.font_path.empty()) {
        auto font_result = font_loader.load(config.font_path, static_cast<float>(config.grid.cell_height));
        if (!font_result.success()) {
            std::cerr << "Warning: Failed to load font: " << config.font_path << " - " << font_result.message << "\n";
        }
    }
    
    if (!font_loader.is_loaded()) {
        auto fallback_result = font_loader.load_system_fallback(static_cast<float>(config.grid.cell_height));
        if (!fallback_result.success()) {
            std::cerr << "Warning: " << fallback_result.message << " - character rendering may be limited\n";
        }
    }
    
    ascii::GlyphCache glyph_cache;
    auto codepoints = ascii::CharSet::get_set(config.selector.char_set);
    if (!glyph_cache.initialize(&font_loader, codepoints, config.grid.cell_width, config.grid.cell_height)) {
        std::cerr << "Warning: Failed to initialize glyph cache\n";
    }
    
    ascii::Pipeline::Config pipeline_cfg;
    pipeline_cfg.target_cols = cols;
    pipeline_cfg.target_rows = rows;
    pipeline_cfg.cell_width = config.grid.cell_width;
    pipeline_cfg.cell_height = config.grid.cell_height;
    pipeline_cfg.char_aspect = config.grid.char_aspect;
    pipeline_cfg.blur_sigma = config.edge.blur_sigma;
    pipeline_cfg.edge_low = config.edge.low_threshold;
    pipeline_cfg.edge_high = config.edge.high_threshold;
    pipeline_cfg.use_hysteresis = config.edge.use_hysteresis;
    pipeline_cfg.scale_mode = config.grid.scale_mode;
    pipeline_cfg.quad_tree_adaptive = config.grid.quad_tree_adaptive;
    pipeline_cfg.quad_tree_max_depth = config.grid.quad_tree_max_depth;
    pipeline_cfg.quad_tree_variance_threshold = config.grid.quad_tree_variance_threshold;
    pipeline_cfg.multi_scale = config.edge.multi_scale;
    pipeline_cfg.scale_sigma_0 = config.edge.scale_sigma_0;
    pipeline_cfg.scale_sigma_1 = config.edge.scale_sigma_1;
    pipeline_cfg.adaptive_scale_selection = config.edge.adaptive_scale_selection;
    pipeline_cfg.scale_variance_floor = config.edge.scale_variance_floor;
    pipeline_cfg.scale_variance_ceil = config.edge.scale_variance_ceil;
    pipeline_cfg.use_anisotropic_diffusion = config.edge.use_anisotropic_diffusion;
    pipeline_cfg.diffusion_iterations = config.edge.diffusion_iterations;
    pipeline_cfg.diffusion_kappa = config.edge.diffusion_kappa;
    pipeline_cfg.diffusion_lambda = config.edge.diffusion_lambda;
    pipeline_cfg.adaptive_mode = config.edge.adaptive_mode;
    pipeline_cfg.tile_size = config.edge.tile_size;
    pipeline_cfg.dark_scene_floor = config.edge.dark_scene_floor;
    pipeline_cfg.global_percentile = config.edge.global_percentile;
    pipeline_cfg.enable_orientation_histogram = !(config.selector.use_simple_orientation || config.selector.mode == "simple");
    pipeline_cfg.enable_frequency_signature = config.selector.enable_frequency_matching;
    pipeline_cfg.enable_texture_signature = config.selector.enable_gabor_texture;
    
    ascii::Pipeline pipeline(pipeline_cfg);
    
    ascii::TemporalSmoother::Config temporal_cfg;
    temporal_cfg.alpha = config.temporal.alpha;
    temporal_cfg.transition_penalty = config.temporal.transition_penalty;
    temporal_cfg.edge_enter_threshold = config.temporal.edge_enter_threshold;
    temporal_cfg.edge_exit_threshold = config.temporal.edge_exit_threshold;
    temporal_cfg.use_wavelet_flicker = config.temporal.use_wavelet_flicker;
    temporal_cfg.wavelet_strength = config.temporal.wavelet_strength;
    temporal_cfg.wavelet_window = config.temporal.wavelet_window;
    ascii::TemporalSmoother smoother(temporal_cfg);
    
    ascii::CharSelector::Config selector_cfg;
    selector_cfg.edge_threshold = config.edge.high_threshold;
    selector_cfg.use_orientation_matching = (config.selector.mode == "histogram");
    selector_cfg.use_simple_orientation = config.selector.use_simple_orientation || config.selector.mode == "simple";
    selector_cfg.use_unified_loss = !selector_cfg.use_simple_orientation;
    selector_cfg.loss_weights.brightness = config.selector.weight_brightness;
    selector_cfg.loss_weights.orientation = config.selector.weight_orientation;
    selector_cfg.loss_weights.contrast = config.selector.weight_contrast;
    selector_cfg.loss_weights.frequency = config.selector.weight_frequency;
    selector_cfg.loss_weights.texture = config.selector.weight_texture;
    selector_cfg.enable_frequency_matching = config.selector.enable_frequency_matching;
    selector_cfg.enable_texture_matching = config.selector.enable_gabor_texture;
    selector_cfg.transition_penalty = config.temporal.transition_penalty;
    ascii::CharSelector selector(selector_cfg);
    selector.set_cache(&glyph_cache);
    
    ascii::Ditherer::Config dither_cfg;
    dither_cfg.enabled = (color_mode == ascii::ColorMode::Ansi16 || color_mode == ascii::ColorMode::Ansi256);
    dither_cfg.error_clamp = config.color.dither_error_clamp;
    dither_cfg.use_blue_noise_halftone = config.color.use_blue_noise_halftone;
    dither_cfg.halftone_strength = config.color.halftone_strength;
    dither_cfg.halftone_cell_size = config.color.halftone_cell_size;
    ascii::Ditherer ditherer(dither_cfg);

    ascii::ColorMapper color_mapper(color_mode);
    color_mapper.set_ditherer(&ditherer);

    ascii::BilateralGrid::Config bilateral_cfg;
    bilateral_cfg.enabled = config.color.use_bilateral_grid;
    bilateral_cfg.spatial_bins = config.color.bilateral_spatial_bins;
    bilateral_cfg.range_bins = config.color.bilateral_range_bins;
    bilateral_cfg.spatial_sigma = config.color.bilateral_spatial_sigma;
    bilateral_cfg.range_sigma = config.color.bilateral_range_sigma;
    ascii::BilateralGrid bilateral_grid(bilateral_cfg);

    ascii::MotionEstimator::Config motion_cfg;
    motion_cfg.motion_cap = static_cast<float>(config.temporal.motion_cap_pixels);
    motion_cfg.solve_divisor = config.temporal.motion_solve_divisor;
    motion_cfg.max_reuse_frames = config.temporal.motion_max_reuse_frames;
    motion_cfg.reuse_scene_threshold = config.temporal.motion_reuse_scene_threshold;
    motion_cfg.reuse_confidence_decay = config.temporal.motion_reuse_confidence_decay;
    motion_cfg.still_scene_threshold = config.temporal.motion_still_scene_threshold;
    motion_cfg.use_phase_correlation = config.temporal.use_phase_correlation;
    motion_cfg.phase_search_radius = config.temporal.phase_search_radius;
    motion_cfg.phase_blend = config.temporal.phase_blend;
    motion_cfg.phase_interval = config.temporal.motion_phase_interval;
    motion_cfg.phase_scene_trigger = config.temporal.motion_phase_scene_trigger;
    ascii::MotionEstimator motion(motion_cfg);
    
    auto source = ascii::create_source(config.input.source);
    if (!source->open(config.input.source)) {
        std::cerr << "Error: Failed to open input: " << config.input.source << "\n";
        return 1;
    }
    
    ascii::AudioPlayer audio;
    bool has_audio = false;
    if (!config.no_audio && config.output.target.empty()) {
        has_audio = audio.open(config.input.source);
    }
    
    ascii::VideoEncoder video_encoder;
    ascii::BitmapRenderer bitmap_renderer;
    bitmap_renderer.set_cache(&glyph_cache);
    bitmap_renderer.set_cell_size(config.grid.cell_width, config.grid.cell_height);
    
    const std::string output_target = config.output.target;
    bool has_output = !output_target.empty() && !ascii::input::ends_with_txt(output_target);
    if (has_output) {
        ascii::VideoEncoder::Config enc_cfg;
        enc_cfg.width = cols * config.grid.cell_width;
        enc_cfg.height = rows * config.grid.cell_height;
        enc_cfg.fps = config.fps;
        if (!video_encoder.open(output_target, enc_cfg)) {
            std::cerr << "Warning: Failed to open video output: " << output_target << "\n";
            has_output = false;
        }
    }
    
    ascii::TerminalRenderer term_renderer(terminal, color_mode);
    term_renderer.set_grid_size(cols, rows);
    
    ascii::BlockRenderer block_renderer;
    block_renderer.set_grid_size(cols, rows);
    
    ascii::ReplayWriter replay_writer;
    std::vector<ascii::ASCIICell> replay_prev_cells;
    bool replay_enabled = false;
    if (!config.output.replay_path.empty()) {
        replay_enabled = replay_writer.open(config.output.replay_path, cols, rows, config.fps, config.compute_hash());
        if (!replay_enabled) {
            std::cerr << "Warning: Failed to open replay output: " << config.output.replay_path << "\n";
        }
    }
    
    double source_fps = source->fps();
    bool is_single_image = (source_fps == 0.0);
    
    if (!has_output && !is_single_image) {
        terminal.enter_alt_screen();
        terminal.hide_cursor();
        terminal.clear_screen();
        ascii::input::setup_nonblocking_stdin();
    }
    
    if (has_audio) {
        audio.play();
    }
    
    double target_fps = config.fps > 0 ? config.fps : (source_fps > 0.0 ? source_fps : 30.0);
    auto frame_duration = std::chrono::duration<double>(1.0 / target_fps);
    auto session_start = std::chrono::steady_clock::now();
    
    ascii::FrameBuffer frame;
    bool initialized = false;
    int frame_count = 0;
    double processing_seconds_total = 0.0;
    double stage_pipeline_seconds = 0.0;
    double stage_motion_seconds = 0.0;
    double stage_select_seconds = 0.0;
    double stage_render_seconds = 0.0;
    double stage_encode_seconds = 0.0;
    std::atomic<bool> paused{false};
    std::atomic<bool> running{true};
    float edge_threshold = config.edge.high_threshold;
    ascii::FloatImage prev_luminance;
    bool have_prev_luminance = false;
    ascii::Pipeline::Result cached_pipeline_result;
    bool have_cached_pipeline_result = false;
    bool cached_pipeline_has_color_buffer = false;
    int pipeline_reuse_frames = 0;
    std::vector<ascii::CellStats> cached_cell_stats;
    bool have_cached_cell_stats = false;
    int cell_stats_reuse_frames = 0;
    std::vector<float> prev_scene_signature;
    std::vector<float> curr_scene_signature;

    auto invalidate_pipeline_cache = [&]() {
        have_cached_pipeline_result = false;
        cached_pipeline_has_color_buffer = false;
        pipeline_reuse_frames = 0;
        have_cached_cell_stats = false;
        cell_stats_reuse_frames = 0;
    };
    
    while (running && source->read(frame)) {
        auto start = std::chrono::high_resolution_clock::now();
        
        if (!has_output) {
            int key = ascii::input::read_key();
            if (key == 'q' || key == 27) {
                running = false;
                break;
            } else if (key == ' ') {
                paused = !paused;
                if (paused) {
                    audio.pause();
                } else if (has_audio) {
                    audio.play();
                }
            } else if (key == 'c') {
                int mode_int = static_cast<int>(color_mode);
                mode_int = (mode_int + 1) % 5;
                color_mode = static_cast<ascii::ColorMode>(mode_int);
                color_mapper.set_mode(color_mode);
                term_renderer.set_color_mode(color_mode);
                auto dc = ditherer.config();
                dc.enabled = (color_mode == ascii::ColorMode::Ansi16 || color_mode == ascii::ColorMode::Ansi256);
                ditherer.set_config(dc);
                invalidate_pipeline_cache();
            } else if (key == '+' || key == '=') {
                edge_threshold = std::min(1.0f, edge_threshold + 0.05f);
                pipeline_cfg.edge_low = edge_threshold * 0.5f;
                pipeline_cfg.edge_high = edge_threshold;
                pipeline.set_config(pipeline_cfg);
                invalidate_pipeline_cache();
            } else if (key == '-') {
                edge_threshold = std::max(0.0f, edge_threshold - 0.05f);
                pipeline_cfg.edge_low = edge_threshold * 0.5f;
                pipeline_cfg.edge_high = edge_threshold;
                pipeline.set_config(pipeline_cfg);
                invalidate_pipeline_cache();
            }
        }
        
        if (paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        auto stage_start = std::chrono::high_resolution_clock::now();
        ascii::Pipeline::Result computed_result;
        const ascii::Pipeline::Result* result_ptr = nullptr;
        curr_scene_signature.clear();
        build_scene_signature(frame, curr_scene_signature);
        float scene_change = 1.0f;
        if (!prev_scene_signature.empty() && prev_scene_signature.size() == curr_scene_signature.size()) {
            scene_change = signature_scene_change(prev_scene_signature, curr_scene_signature);
        }

        const bool allow_mixed_block_mode = config.grid.quad_tree_adaptive &&
                                            color_mode != ascii::ColorMode::BlockArt;
        const bool need_color_buffer = (color_mode == ascii::ColorMode::BlockArt) ||
                                       allow_mixed_block_mode;
        const bool need_color_stats = need_color_buffer ||
                                      color_mode != ascii::ColorMode::None ||
                                      config.color.use_bilateral_grid;

        bool reused_pipeline = false;
        const int pipeline_reuse_limit = std::max(0, config.temporal.motion_max_reuse_frames);
        const float pipeline_still_thresh = std::max(0.0f, config.temporal.motion_still_scene_threshold);

        if (have_cached_pipeline_result &&
            !prev_scene_signature.empty() &&
            pipeline_reuse_limit > 0) {
            const bool cached_color_ok = !need_color_buffer || cached_pipeline_has_color_buffer;
            if (scene_change < pipeline_still_thresh &&
                pipeline_reuse_frames < pipeline_reuse_limit &&
                cached_color_ok) {
                reused_pipeline = true;
                result_ptr = &cached_pipeline_result;
                ++pipeline_reuse_frames;
                if (have_cached_cell_stats) {
                    cell_stats_reuse_frames = std::min(cell_stats_reuse_frames + 1, pipeline_reuse_limit);
                }
            }
        }

        if (!reused_pipeline) {
            const int cell_stats_reuse_limit = std::max(0, config.temporal.motion_max_reuse_frames);
            const float cell_stats_reuse_thresh =
                std::max(0.0f, config.temporal.motion_still_scene_threshold * 0.8f);
            const bool reuse_cell_stats =
                have_cached_cell_stats &&
                !prev_scene_signature.empty() &&
                cell_stats_reuse_limit > 0 &&
                scene_change < cell_stats_reuse_thresh &&
                cell_stats_reuse_frames < cell_stats_reuse_limit;

            ascii::Pipeline::ProcessOptions process_options;
            process_options.need_color_buffer = need_color_buffer;
            process_options.need_color_stats = need_color_stats;
            if (reuse_cell_stats) {
                process_options.reuse_cell_stats = &cached_cell_stats;
            }

            computed_result = pipeline.process(frame, process_options);
            cached_pipeline_result = computed_result;
            have_cached_pipeline_result = true;
            cached_pipeline_has_color_buffer = need_color_buffer;
            pipeline_reuse_frames = 0;
            result_ptr = &cached_pipeline_result;

            if (reuse_cell_stats) {
                ++cell_stats_reuse_frames;
            } else {
                cached_cell_stats = cached_pipeline_result.cell_stats;
                have_cached_cell_stats = true;
                cell_stats_reuse_frames = 0;
            }
        }
        prev_scene_signature = curr_scene_signature;

        const ascii::Pipeline::Result& result = *result_ptr;
        stage_pipeline_seconds += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        stage_start = std::chrono::high_resolution_clock::now();
        if (config.color.use_bilateral_grid) {
            bilateral_grid.build(result.cell_stats, result.grid_cols, result.grid_rows);
        }

        if (config.temporal.motion_cap_pixels > 0 &&
            have_prev_luminance &&
            prev_luminance.width() == result.luminance.width() &&
            prev_luminance.height() == result.luminance.height()) {
            motion.compute_flow(prev_luminance, result.luminance);
        } else {
            motion.reset();
        }
        stage_motion_seconds += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();

        ditherer.begin_frame(result.grid_cols, result.grid_rows);
        
        if (has_audio && !paused) {
            audio.sync_to_frame(frame_count, target_fps);
        }
        
        if (!config.debug.mode.empty() && !has_output) {
            if (config.debug.mode == "grayscale") {
                std::vector<ascii::ASCIICell> debug_cells(result.grid_cols * result.grid_rows);
                for (int i = 0; i < static_cast<int>(debug_cells.size()); ++i) {
                    int x = i % result.grid_cols;
                    int y = i / result.grid_cols;
                    int px0 = x * config.grid.cell_width;
                    int py0 = y * config.grid.cell_height;
                    float sum = 0;
                    int cnt = 0;
                    for (int dy = 0; dy < config.grid.cell_height && py0 + dy < result.luminance.height(); ++dy) {
                        for (int dx = 0; dx < config.grid.cell_width && px0 + dx < result.luminance.width(); ++dx) {
                            sum += result.luminance.get(px0 + dx, py0 + dy);
                            cnt++;
                        }
                    }
                    float lum = cnt > 0 ? sum / cnt : 0;
                    auto& cell = debug_cells[i];
                    cell.codepoint = static_cast<uint32_t>(' ');
                    cell.fg_r = cell.fg_g = cell.fg_b = static_cast<uint8_t>(lum * 255);
                }
                term_renderer.render(debug_cells);
                terminal.flush();
            } else if (config.debug.mode == "edges") {
                std::vector<ascii::ASCIICell> debug_cells(result.grid_cols * result.grid_rows);
                for (int i = 0; i < static_cast<int>(debug_cells.size()); ++i) {
                    const auto& stats = result.cell_stats[i];
                    auto& cell = debug_cells[i];
                    cell.codepoint = static_cast<uint32_t>(stats.is_edge_cell ? '#' : ' ');
                    cell.fg_r = cell.fg_g = cell.fg_b = static_cast<uint8_t>(stats.edge_strength * 255 * 4);
                }
                term_renderer.render(debug_cells);
                terminal.flush();
            } else if (config.debug.mode == "orientation") {
                constexpr float kPi = 3.14159265358979323846f;
                std::vector<ascii::ASCIICell> debug_cells(result.grid_cols * result.grid_rows);
                for (int i = 0; i < static_cast<int>(debug_cells.size()); ++i) {
                    const auto& stats = result.cell_stats[i];
                    auto& cell = debug_cells[i];
                    float angle = stats.cell_orientation;
                    float hue = (angle + kPi) / (2.0f * kPi);
                    uint8_t r = static_cast<uint8_t>(std::abs(std::sin(hue * 6.28f)) * 255);
                    uint8_t g = static_cast<uint8_t>(std::abs(std::sin((hue + 0.33f) * 6.28f)) * 255);
                    uint8_t b = static_cast<uint8_t>(std::abs(std::sin((hue + 0.66f) * 6.28f)) * 255);
                    cell.codepoint = static_cast<uint32_t>(stats.is_edge_cell ? 'O' : '.');
                    cell.fg_r = r; cell.fg_g = g; cell.fg_b = b;
                }
                term_renderer.render(debug_cells);
                terminal.flush();
            }
            frame_count++;
            continue;
        }
        
        stage_start = std::chrono::high_resolution_clock::now();
        if (!initialized) {
            smoother.initialize(result.grid_cols, result.grid_rows);
            initialized = true;
        }
        
        std::vector<ascii::ASCIICell> cells(result.grid_cols * result.grid_rows);
        std::vector<ascii::BlockCell> block_cells;
        const bool allow_mixed_block = config.grid.quad_tree_adaptive &&
                                       color_mode != ascii::ColorMode::BlockArt;
        if (color_mode == ascii::ColorMode::BlockArt || allow_mixed_block) {
            block_cells.resize(cells.size());
        }
        
        for (int i = 0; i < static_cast<int>(result.cell_stats.size()); ++i) {
            const auto& stats = result.cell_stats[i];
            int cell_x = i % result.grid_cols;
            int cell_y = i / result.grid_cols;
            
            float smoothed_lum = smoother.smooth_luminance(i, stats.mean_luminance);
            float smoothed_edge = smoother.smooth_edge_strength(i, stats.edge_strength);
            float smoothed_coh = smoother.smooth_coherence(i, stats.structure_coherence);
            
            ascii::CellStats effective_stats = stats;
            effective_stats.mean_luminance = smoothed_lum;
            effective_stats.edge_strength = smoothed_edge;
            effective_stats.structure_coherence = smoothed_coh;

            if (bilateral_grid.valid()) {
                auto smooth_rgb = bilateral_grid.sample(cell_x, cell_y, smoothed_lum);
                effective_stats.mean_r = smooth_rgb.r;
                effective_stats.mean_g = smooth_rgb.g;
                effective_stats.mean_b = smooth_rgb.b;
            }

            float adaptive_edge_margin = 0.02f * static_cast<float>(effective_stats.adaptive_level);
            float adaptive_edge_threshold = std::clamp(edge_threshold - adaptive_edge_margin, 0.0f, 1.0f);

            bool edge_candidate = (stats.edge_occupancy >= adaptive_edge_threshold) ||
                                  (smoothed_edge >= adaptive_edge_threshold) ||
                                  (smoothed_coh >= std::max(0.05f, 0.2f - 0.04f * effective_stats.adaptive_level));
            smoother.update_edge_state(i, edge_candidate);
            effective_stats.is_edge_cell = smoother.get_edge_state(i);
            
            if (motion.has_motion()) {
                float dx = 0.0f, dy = 0.0f;
                motion.get_motion_for_cell(
                    cell_x * config.grid.cell_width,
                    cell_y * config.grid.cell_height,
                    config.grid.cell_width,
                    config.grid.cell_height,
                    dx, dy
                );
                smoother.set_motion_offset(i,
                    dx / static_cast<float>(std::max(1, config.grid.cell_width)),
                    dy / static_cast<float>(std::max(1, config.grid.cell_height)));
            } else {
                smoother.set_motion_offset(i, 0.0f, 0.0f);
            }
            
            bool use_block_cell = (color_mode == ascii::ColorMode::BlockArt) ||
                                  (allow_mixed_block && effective_stats.adaptive_level >= 2);

            if (use_block_cell) {
                ascii::BlockRenderer::CellData block_data = block_renderer.analyze_cell(
                    result.luminance,
                    result.color_buffer,
                    cell_x,
                    cell_y,
                    config.grid.cell_width,
                    config.grid.cell_height,
                    effective_stats);
                
                auto block_result = block_renderer.render_cell(block_data);
                float block_score = std::clamp(1.0f - std::abs(block_data.top_left_lum - block_data.bottom_right_lum), 0.0f, 1.0f);
                uint32_t final_cp = block_result.codepoint;
                if (smoother.should_change_glyph(i, block_result.codepoint, block_score)) {
                    final_cp = block_result.codepoint;
                    smoother.update_glyph(i, block_result.codepoint, block_score);
                } else {
                    final_cp = smoother.frame_state()[i].last_glyph;
                }
                block_result.codepoint = final_cp;
                block_cells[i] = block_result;
                cells[i].codepoint = final_cp;

                if (color_mode == ascii::ColorMode::BlockArt) {
                    cells[i].fg_r = block_result.fg_r;
                    cells[i].fg_g = block_result.fg_g;
                    cells[i].fg_b = block_result.fg_b;
                    cells[i].bg_r = block_result.bg_r;
                    cells[i].bg_g = block_result.bg_g;
                    cells[i].bg_b = block_result.bg_b;
                } else {
                    int row_dir = (cell_y % 2 == 0) ? 1 : -1;
                    auto mapped = color_mapper.map_with_dither(
                        cell_x, cell_y, row_dir,
                        block_result.fg_r / 255.0f,
                        block_result.fg_g / 255.0f,
                        block_result.fg_b / 255.0f,
                        effective_stats.is_edge_cell
                    );
                    cells[i].fg_r = mapped.r;
                    cells[i].fg_g = mapped.g;
                    cells[i].fg_b = mapped.b;
                    cells[i].bg_r = 0;
                    cells[i].bg_g = 0;
                    cells[i].bg_b = 0;
                }
            } else {
                auto selection = selector.select(effective_stats, smoother, i);
                
                if (selector.config().use_unified_loss) {
                    float transition_cost = selector.compute_transition_cost(
                        smoother.frame_state()[i].last_glyph, selection.codepoint);
                    
                    if (smoother.should_change_glyph_with_loss(i, selection.codepoint, selection.loss, transition_cost)) {
                        cells[i].codepoint = selection.codepoint;
                        smoother.update_glyph_with_loss(i, selection.codepoint, selection.score, selection.loss + transition_cost);
                    } else {
                        cells[i].codepoint = smoother.frame_state()[i].last_glyph;
                    }
                } else {
                    if (smoother.should_change_glyph(i, selection.codepoint, selection.score)) {
                        cells[i].codepoint = selection.codepoint;
                        smoother.update_glyph(i, selection.codepoint, selection.score);
                    } else {
                        cells[i].codepoint = smoother.frame_state()[i].last_glyph;
                    }
                }
                
                uint8_t sr = 0, sg = 0, sb = 0;
                ascii::ColorSpace::linear_to_srgb({effective_stats.mean_r, effective_stats.mean_g, effective_stats.mean_b}, sr, sg, sb);
                int row_dir = (cell_y % 2 == 0) ? 1 : -1;
                auto color = color_mapper.map_with_dither(
                    cell_x, cell_y, row_dir,
                    sr / 255.0f, sg / 255.0f, sb / 255.0f,
                    effective_stats.is_edge_cell
                );
                
                cells[i].fg_r = color.r;
                cells[i].fg_g = color.g;
                cells[i].fg_b = color.b;
            }
        }

        if (color_mode == ascii::ColorMode::BlockArt &&
            !block_cells.empty() &&
            config.color.block_spectral_palette > 1) {
            block_renderer.spectral_quantize_frame(
                block_cells,
                config.color.block_spectral_palette,
                config.color.block_spectral_samples,
                config.color.block_spectral_iterations
            );
            for (size_t i = 0; i < cells.size() && i < block_cells.size(); ++i) {
                cells[i].fg_r = block_cells[i].fg_r;
                cells[i].fg_g = block_cells[i].fg_g;
                cells[i].fg_b = block_cells[i].fg_b;
                cells[i].bg_r = block_cells[i].bg_r;
                cells[i].bg_g = block_cells[i].bg_g;
                cells[i].bg_b = block_cells[i].bg_b;
            }
        }
        stage_select_seconds += std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - stage_start).count();
        
        if (has_output) {
            auto encode_start = std::chrono::high_resolution_clock::now();
            std::vector<uint32_t> cps(cells.size());
            for (size_t i = 0; i < cells.size(); ++i) {
                cps[i] = cells[i].codepoint;
            }
            
            auto bitmap = bitmap_renderer.render(cps, result.grid_cols, result.grid_rows);
            video_encoder.write_frame(bitmap);
            stage_encode_seconds += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - encode_start).count();
        }

        if (replay_enabled) {
            bool replay_ok = false;
            if (replay_prev_cells.empty()) {
                replay_ok = replay_writer.write_frame(static_cast<uint32_t>(frame_count), cells);
            } else {
                replay_ok = replay_writer.write_frame_delta(
                    static_cast<uint32_t>(frame_count), cells, replay_prev_cells);
            }
            if (!replay_ok) {
                std::cerr << "Warning: Replay write failed at frame " << frame_count << "\n";
                replay_enabled = false;
            } else {
                replay_prev_cells = cells;
            }
        }
        
        if (!output_target.empty() && ascii::input::ends_with_txt(output_target)) {
            std::string frame_path = output_target;
            if (!is_single_image) {
                size_t dot = output_target.rfind('.');
                if (dot != std::string::npos) {
                    frame_path = output_target.substr(0, dot) + "_" + std::to_string(frame_count) + ".txt";
                } else {
                    frame_path = output_target + "_" + std::to_string(frame_count) + ".txt";
                }
            }
            ascii::input::write_ascii_to_file(frame_path, cells, result.grid_cols, result.grid_rows);
        }
        
        if (!has_output) {
            auto render_start = std::chrono::high_resolution_clock::now();
            term_renderer.render(cells);
            terminal.flush();
            if (is_single_image) {
                terminal.write("\n");
            }
            stage_render_seconds += std::chrono::duration<double>(
                std::chrono::high_resolution_clock::now() - render_start).count();
        }
        
        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        processing_seconds_total += std::chrono::duration<double>(elapsed).count();
        if (config.debug.profile_live) {
            double ms = std::chrono::duration<double, std::milli>(elapsed).count();
            std::cerr << "{\"frame\":" << frame_count
                      << ",\"ms\":" << ms
                      << ",\"fps\":" << (ms > 0.0 ? 1000.0 / ms : 0.0)
                      << "}\n";
        }
        auto remaining = frame_duration - elapsed;
        if (remaining > std::chrono::nanoseconds(0)) {
            std::this_thread::sleep_for(remaining);
        }
        prev_luminance = result.luminance;
        have_prev_luminance = true;
        
        frame_count++;
    }
    
    if (!has_output) {
        ascii::input::restore_stdin();
        if (is_single_image) {
            terminal.write("\n\nPress Enter to exit...");
            terminal.flush();
            std::cin.get();
        }
        terminal.show_cursor();
        terminal.exit_alt_screen();
    }
    
    video_encoder.close();
    replay_writer.close();
    audio.close();

    auto session_end = std::chrono::steady_clock::now();
    double wall_seconds = std::chrono::duration<double>(session_end - session_start).count();
    if (frame_count > 0 && wall_seconds > 0.0) {
        double effective_fps = static_cast<double>(frame_count) / wall_seconds;
        double processing_fps = static_cast<double>(frame_count) / std::max(processing_seconds_total, 1e-9);
        double known_stage_seconds = stage_pipeline_seconds + stage_motion_seconds +
                                     stage_select_seconds + stage_render_seconds +
                                     stage_encode_seconds;
        double stage_misc_seconds = std::max(0.0, processing_seconds_total - known_stage_seconds);
        auto stage_pct = [&](double seconds) -> double {
            return (processing_seconds_total > 0.0)
                ? (100.0 * seconds / processing_seconds_total)
                : 0.0;
        };
        std::cerr << std::fixed << std::setprecision(2)
                  << "[PERF] frames=" << frame_count
                  << ", wall_s=" << wall_seconds
                  << ", effective_fps=" << effective_fps
                  << ", processing_fps=" << processing_fps
                  << "\n";
        std::cerr << std::fixed << std::setprecision(2)
                  << "[PERF_STAGES] pipeline_s=" << stage_pipeline_seconds
                  << ", motion_s=" << stage_motion_seconds
                  << ", select_s=" << stage_select_seconds
                  << ", render_s=" << stage_render_seconds
                  << ", encode_s=" << stage_encode_seconds
                  << ", misc_s=" << stage_misc_seconds
                  << "\n";
        std::cerr << std::fixed << std::setprecision(1)
                  << "[PERF_STAGES_PCT] pipeline=" << stage_pct(stage_pipeline_seconds) << "%"
                  << ", motion=" << stage_pct(stage_motion_seconds) << "%"
                  << ", select=" << stage_pct(stage_select_seconds) << "%"
                  << ", render=" << stage_pct(stage_render_seconds) << "%"
                  << ", encode=" << stage_pct(stage_encode_seconds) << "%"
                  << ", misc=" << stage_pct(stage_misc_seconds) << "%"
                  << "\n";
    } else {
        std::cerr << "[PERF] no frames processed.\n";
    }
    
    return 0;
}
