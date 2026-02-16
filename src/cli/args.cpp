#include "args.hpp"
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <algorithm>

namespace ascii {

ColorMode parse_color_mode(const std::string& s) {
    if (s == "none") return ColorMode::None;
    if (s == "16") return ColorMode::Ansi16;
    if (s == "256") return ColorMode::Ansi256;
    if (s == "blockart") return ColorMode::BlockArt;
    return ColorMode::Truecolor;
}

static int clamp_int(int val, int min_val, int max_val, int default_val) {
    if (val < min_val || val > max_val) return default_val;
    return val;
}

static float clamp_float(float val, float min_val, float max_val, float default_val) {
    if (val < min_val || val > max_val) return default_val;
    return val;
}

static bool validate_path(const std::string& path) {
    if (path.empty()) return false;
    if (path.find("..") != std::string::npos) return false;
    if (path.find('\0') != std::string::npos) return false;
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
        
        if (strcmp(arg, "-o") == 0 || strcmp(arg, "--output") == 0) {
            if (i + 1 < argc) {
                args.output = argv[++i];
                if (!validate_path(args.output)) {
                    args.output.clear();
                }
            }
        }
        else if (strcmp(arg, "--config") == 0) {
            if (i + 1 < argc) {
                args.config_path = argv[++i];
                if (!validate_path(args.config_path)) {
                    args.config_path.clear();
                }
            }
        }
        else if (strcmp(arg, "--replay") == 0) {
            if (i + 1 < argc) {
                args.replay_path = argv[++i];
                if (!validate_path(args.replay_path)) {
                    args.replay_path.clear();
                }
            }
        }
        else if (strcmp(arg, "-f") == 0 || strcmp(arg, "--fps") == 0) {
            if (i + 1 < argc) args.fps = clamp_int(std::atoi(argv[++i]), 1, 120, 30);
        }
        else if (strcmp(arg, "-c") == 0 || strcmp(arg, "--cols") == 0) {
            if (i + 1 < argc) args.cols = clamp_int(std::atoi(argv[++i]), 1, 500, 0);
        }
        else if (strcmp(arg, "-r") == 0 || strcmp(arg, "--rows") == 0) {
            if (i + 1 < argc) args.rows = clamp_int(std::atoi(argv[++i]), 1, 200, 0);
        }
        else if (strcmp(arg, "--char-set") == 0) {
            if (i + 1 < argc) {
                std::string cs = argv[++i];
                if (cs == "basic" || cs == "blocks" || cs == "line-art") {
                    args.char_set = cs;
                }
            }
        }
        else if (strcmp(arg, "--profile") == 0) {
            if (i + 1 < argc) {
                std::string p = argv[++i];
                if (p == "natural" || p == "anime" || p == "ui") {
                    args.profile = p;
                }
            }
        }
        else if (strcmp(arg, "--color") == 0) {
            if (i + 1 < argc) args.color_mode = parse_color_mode(argv[++i]);
        }
        else if (strcmp(arg, "--edge-thresh") == 0) {
            if (i + 1 < argc) args.edge_threshold = clamp_float(static_cast<float>(std::atof(argv[++i])), 0.0f, 1.0f, 0.1f);
        }
        else if (strcmp(arg, "--blur") == 0) {
            if (i + 1 < argc) args.blur_sigma = clamp_float(static_cast<float>(std::atof(argv[++i])), 0.1f, 10.0f, 1.0f);
        }
        else if (strcmp(arg, "--temporal") == 0) {
            if (i + 1 < argc) args.temporal_alpha = clamp_float(static_cast<float>(std::atof(argv[++i])), 0.0f, 1.0f, 0.3f);
        }
        else if (strcmp(arg, "--scale") == 0) {
            if (i + 1 < argc) {
                std::string sm = argv[++i];
                if (sm == "fit" || sm == "fill" || sm == "stretch") {
                    args.scale_mode = sm;
                }
            }
        }
        else if (strcmp(arg, "--font") == 0) {
            if (i + 1 < argc) {
                args.font_path = argv[++i];
                if (!validate_path(args.font_path)) {
                    args.font_path.clear();
                }
            }
        }
        else if (strcmp(arg, "--no-audio") == 0) {
            args.no_audio = true;
        }
        else if (strcmp(arg, "--no-hysteresis") == 0) {
            args.use_hysteresis = false;
        }
        else if (strcmp(arg, "--no-orientation") == 0) {
            args.use_orientation_matching = false;
        }
        else if (strcmp(arg, "--simple-orientation") == 0) {
            args.use_simple_orientation = true;
            args.use_orientation_matching = false;
        }
        else if (strcmp(arg, "--debug") == 0) {
            if (i + 1 < argc) {
                args.debug_mode = argv[++i];
            }
        }
        else if (strcmp(arg, "--profile-live") == 0) {
            args.profile_live = true;
        }
        else if (strcmp(arg, "--strict-memory") == 0) {
            args.strict_memory = true;
        }
        else if (arg[0] != '-') {
            args.input = arg;
            if (!validate_path(args.input)) {
                args.input.clear();
            }
        }
    }
    
    return args;
}

void print_help(const char* prog) {
    printf("Usage: %s [OPTIONS] <INPUT>\n\n", prog);
    printf("INPUT:\n");
    printf("  Path to video file, image, or \"webcam\" for live capture\n\n");
    printf("OPTIONS:\n");
    printf("  -o, --output <FILE>     Output file (.mp4 or .gif video, or .txt frames)\n");
    printf("      --config <FILE>     Config file path (default: platform-specific)\n");
    printf("      --replay <FILE>     Write deterministic replay to .areplay file\n");
    printf("  -f, --fps <N>           Target FPS (default: 30, range: 1-120)\n");
    printf("  -c, --cols <N>          Max columns (default: auto-detect, range: 1-500)\n");
    printf("  -r, --rows <N>          Max rows (default: auto-detect, range: 1-200)\n");
    printf("      --char-set <NAME>   Character set: basic, blocks, line-art\n");
    printf("      --profile <NAME>    Content preset: natural, anime, ui\n");
    printf("      --color <MODE>      Color mode: none, 16, 256, truecolor, blockart\n");
    printf("      --edge-thresh <N>   Edge detection threshold (0.0-1.0)\n");
    printf("      --blur <N>          Blur sigma (default: 1.0, range: 0.1-10.0)\n");
    printf("      --temporal <N>      Temporal smoothing alpha (0.0-1.0)\n");
    printf("      --scale <MODE>      Scaling: fit, fill, stretch\n");
    printf("      --font <PATH>       Font file to use (auto-detects system font if not set)\n");
    printf("      --no-audio          Disable audio playback\n");
    printf("      --no-hysteresis     Disable edge hysteresis\n");
    printf("      --no-orientation    Disable orientation-based glyph selection\n");
    printf("      --simple-orientation Use simple 8-direction orientation mapping\n");
    printf("      --debug <MODE>      Debug view: grayscale, edges, orientation\n");
    printf("      --profile-live      Output per-frame profiling as JSONL to stderr\n");
    printf("      --strict-memory     Fail if memory budget exceeded\n");
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
