#include "args.hpp"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <limits>

namespace ascii {

static bool parse_color_mode(const std::string& s, ColorMode& mode) {
    if (s == "none") mode = ColorMode::None;
    else if (s == "16") mode = ColorMode::Ansi16;
    else if (s == "256") mode = ColorMode::Ansi256;
    else if (s == "truecolor") mode = ColorMode::Truecolor;
    else if (s == "blockart") mode = ColorMode::BlockArt;
    else return false;
    return true;
}

static bool validate_path(const std::string& path) {
    if (path.empty()) return false;
    if (path.find("..") != std::string::npos) return false;
    if (path.find('\0') != std::string::npos) return false;
    return true;
}

static bool parse_int(const char* text, int min_value, int max_value, int& value) {
    if (!text || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        parsed < min_value || parsed > max_value) {
        return false;
    }
    value = static_cast<int>(parsed);
    return true;
}

static bool parse_float(const char* text, float min_value, float max_value, float& value) {
    if (!text || *text == '\0') return false;
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(text, &end);
    if (errno == ERANGE || end == text || *end != '\0' || !std::isfinite(parsed) ||
        parsed < min_value || parsed > max_value) {
        return false;
    }
    value = parsed;
    return true;
}

static bool take_value(int argc, char* argv[], int& i, Args& args, const char* option, const char*& value) {
    if (i + 1 >= argc) {
        args.valid = false;
        args.error = std::string("Missing value for ") + option;
        return false;
    }
    value = argv[++i];
    return true;
}

static bool set_path(const char* value, std::string& destination, Args& args, const char* option) {
    if (!validate_path(value)) {
        args.valid = false;
        args.error = std::string("Invalid path for ") + option + ": " + value;
        return false;
    }
    destination = value;
    return true;
}

Args parse_args(int argc, char* argv[]) {
    Args args;
    
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            args.show_help = true;
            return args;
        }
        
        const char* value = nullptr;
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !set_path(value, args.output, args, arg)) break;
        }
        else if (strcmp(arg, "--config") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !set_path(value, args.config_path, args, arg)) break;
        }
        else if (strcmp(arg, "--replay") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !set_path(value, args.replay_path, args, arg)) break;
        }
        else if (strcmp(arg, "--inspect-replay") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !set_path(value, args.inspect_replay_path, args, arg)) break;
        }
        else if (strcmp(arg, "--play-replay") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !set_path(value, args.play_replay_path, args, arg)) break;
        }
        else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--fps") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            if (!parse_int(value, 1, 120, args.fps)) { args.valid = false; args.error = std::string("Invalid value for ") + arg + ": " + value; break; }
            args.fps_set = true;
        }
        else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--cols") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            if (!parse_int(value, 1, 500, args.cols)) { args.valid = false; args.error = std::string("Invalid value for ") + arg + ": " + value; break; }
            args.cols_set = true;
        }
        else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--rows") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            if (!parse_int(value, 1, 200, args.rows)) { args.valid = false; args.error = std::string("Invalid value for ") + arg + ": " + value; break; }
            args.rows_set = true;
        }
        else if (strcmp(arg, "--char-set") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            std::string cs = value;
            if (cs != "basic" && cs != "blocks" && cs != "line-art") { args.valid = false; args.error = "Invalid value for --char-set: " + cs; break; }
            args.char_set = cs;
            args.char_set_set = true;
        }
        else if (strcmp(arg, "--profile") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            std::string p = value;
            if (p != "natural" && p != "anime" && p != "ui") { args.valid = false; args.error = "Invalid value for --profile: " + p; break; }
            args.profile = p;
        }
        else if (strcmp(arg, "--color") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            if (!parse_color_mode(value, args.color_mode)) { args.valid = false; args.error = std::string("Invalid value for --color: ") + value; break; }
            args.color_mode_set = true;
        }
        else if (strcmp(arg, "--edge-thresh") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            if (!parse_float(value, 0.0f, 1.0f, args.edge_threshold)) { args.valid = false; args.error = std::string("Invalid value for --edge-thresh: ") + value; break; }
            args.edge_threshold_set = true;
        }
        else if (strcmp(arg, "--contour-thresh") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_float(value, 0.0f, 1.0f, args.contour_threshold)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --contour-thresh: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--blur") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_float(value, 0.1f, 10.0f, args.blur_sigma)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --blur: ") + (value ? value : ""); } break; }
            args.blur_sigma_set = true;
        }
        else if (strcmp(arg, "--temporal") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_float(value, 0.0f, 1.0f, args.temporal_alpha)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --temporal: ") + (value ? value : ""); } break; }
            args.temporal_alpha_set = true;
        }
        else if (strcmp(arg, "--motion-solve-div") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_int(value, 1, 8, args.motion_solve_divisor)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --motion-solve-div: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--motion-reuse") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_int(value, 0, 32, args.motion_max_reuse_frames)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --motion-reuse: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--motion-reuse-thresh") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_float(value, 0.0f, 1.0f, args.motion_reuse_scene_threshold)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --motion-reuse-thresh: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--motion-reuse-decay") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_float(value, 0.0f, 1.0f, args.motion_reuse_confidence_decay)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --motion-reuse-decay: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--phase-interval") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_int(value, 1, 64, args.motion_phase_interval)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --phase-interval: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--phase-scene-trigger") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_float(value, 0.0f, 1.0f, args.motion_phase_scene_trigger)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --phase-scene-trigger: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--motion-still-thresh") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !parse_float(value, 0.0f, 1.0f, args.motion_still_scene_threshold)) { if (args.valid) { args.valid = false; args.error = std::string("Invalid value for --motion-still-thresh: ") + (value ? value : ""); } break; }
        }
        else if (strcmp(arg, "--scale") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            std::string sm = value;
            if (sm != "fit" && sm != "fill" && sm != "stretch") { args.valid = false; args.error = "Invalid value for --scale: " + sm; break; }
            args.scale_mode = sm;
            args.scale_mode_set = true;
        }
        else if (strcmp(arg, "--font") == 0) {
            if (!take_value(argc, argv, i, args, arg, value) || !set_path(value, args.font_path, args, arg)) break;
            args.font_path_set = true;
        }
        else if (strcmp(arg, "--no-audio") == 0) {
            args.no_audio = true;
            args.no_audio_set = true;
        }
        else if (strcmp(arg, "--no-hysteresis") == 0) {
            args.use_hysteresis = false;
            args.use_hysteresis_set = true;
        }
        else if (strcmp(arg, "--no-contours") == 0) {
            args.contours_enabled = false;
            args.contours_enabled_set = true;
        }
        else if (strcmp(arg, "--no-orientation") == 0) {
            args.use_orientation_matching = false;
            args.use_simple_orientation = false;
            args.orientation_mode_set = true;
        }
        else if (strcmp(arg, "--simple-orientation") == 0) {
            args.use_simple_orientation = true;
            args.use_orientation_matching = false;
            args.orientation_mode_set = true;
        }
        else if (strcmp(arg, "--debug") == 0) {
            if (!take_value(argc, argv, i, args, arg, value)) break;
            const std::string mode = value;
            if (mode != "grayscale" && mode != "edges" && mode != "orientation") {
                args.valid = false;
                args.error = "Invalid value for --debug: " + mode;
                break;
            }
            args.debug_mode = mode;
        }
        else if (strcmp(arg, "--profile-live") == 0) {
            args.profile_live = true;
            args.profile_live_set = true;
        }
        else if (strcmp(arg, "--strict-memory") == 0) {
            args.strict_memory = true;
            args.strict_memory_set = true;
        }
        else if (strcmp(arg, "--fast") == 0) {
            args.fast_mode = true;
        }
        else if (arg[0] != '-') {
            if (!args.input.empty()) {
                args.valid = false;
                args.error = std::string("Unexpected positional argument: ") + arg;
                break;
            }
            args.input = arg;
            if (!validate_path(args.input)) {
                args.valid = false;
                args.error = std::string("Invalid input path: ") + arg;
                break;
            }
        }
        else {
            args.valid = false;
            args.error = std::string("Unknown option: ") + arg;
            break;
        }
    }
    
    return args;
}

void print_help(const char* prog) {
    printf("Usage: %s [OPTIONS] <INPUT>\n\n", prog);
    printf("INPUT:\n");
    printf("  Path to video/image; \"webcam\" requires an optional OpenCV build\n\n");
    printf("OPTIONS:\n");
    printf("  -o, --output <FILE>     Output file (.txt frames, video, or still image)\n");
    printf("      --config <FILE>     Config file path (default: platform-specific)\n");
    printf("      --replay <FILE>     Write deterministic replay to .areplay file\n");
    printf("      --inspect-replay <FILE>  Print .areplay metadata\n");
    printf("      --play-replay <FILE>     Play .areplay in terminal, or export text with -o\n");
    printf("  -f, --fps <N>           Target FPS (default: 30, range: 1-120)\n");
    printf("  -c, --cols <N>          Max columns (default: auto-detect, range: 1-500)\n");
    printf("  -r, --rows <N>          Max rows (default: auto-detect, range: 1-200)\n");
    printf("      --char-set <NAME>   Character set: basic, blocks, line-art\n");
    printf("      --profile <NAME>    Content preset: natural, anime, ui\n");
    printf("      --color <MODE>      Color mode: none, 16, 256, truecolor, blockart\n");
    printf("      --edge-thresh <N>   Edge detection threshold (0.0-1.0)\n");
    printf("      --contour-thresh <N> Minimum contour occupancy per cell (0.0-1.0)\n");
    printf("      --blur <N>          Blur sigma (default: 1.0, range: 0.1-10.0)\n");
    printf("      --temporal <N>      Temporal smoothing alpha (0.0-1.0)\n");
    printf("      --motion-solve-div <N>  Motion solve downscale divisor (1-8)\n");
    printf("      --motion-reuse <N>  Reuse motion field for up to N stable frames\n");
    printf("      --motion-reuse-thresh <N>  Scene-change threshold for motion reuse (0-1)\n");
    printf("      --motion-reuse-decay <N>   Confidence decay applied on reused motion (0-1)\n");
    printf("      --phase-interval <N>       Recompute phase-correlation every N frames\n");
    printf("      --phase-scene-trigger <N>  Force phase refresh above scene-change threshold (0-1)\n");
    printf("      --motion-still-thresh <N>  Zero motion when scene-change is below threshold (0-1)\n");
    printf("      --scale <MODE>      Scaling: fit, fill, stretch\n");
    printf("      --font <PATH>       Font file to use (auto-detects system font if not set)\n");
    printf("      --no-audio          Disable audio playback\n");
    printf("      --no-hysteresis     Disable edge hysteresis\n");
    printf("      --no-contours       Disable default ASCII contour overlay\n");
    printf("      --no-orientation    Disable orientation-based glyph selection\n");
    printf("      --simple-orientation Use simple 8-direction orientation mapping\n");
    printf("      --debug <MODE>      Debug view: grayscale, edges, orientation\n");
    printf("      --profile-live      Output per-frame profiling as JSONL to stderr\n");
    printf("      --strict-memory     Fail if memory budget exceeded\n");
    printf("      --fast              Fast preview mode (disables expensive analysis modules)\n");
    printf("  -h, --help              Show this help\n");
    printf("\nINTERACTIVE CONTROLS (during playback):\n");
    printf("  SPACE                   Pause/resume\n");
    printf("  q/Esc                   Quit\n");
    printf("  c                       Cycle color mode (none -> 16 -> 256 -> truecolor -> blockart)\n");
    printf("  +/=                     Increase edge threshold\n");
    printf("  -                       Decrease edge threshold\n");
    printf("\nCONFIG FILE:\n");
    printf("  Default locations:\n");
    printf("    Linux:   ~/.config/ascii-engine/config.toml\n");
    printf("    macOS:   ~/Library/Application Support/ascii-engine/config.toml\n");
    printf("    Windows: %%APPDATA%%\\ascii-engine\\config.toml\n");
}

}
