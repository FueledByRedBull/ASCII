#include <iostream>
#include <cassert>
#include <filesystem>
#include <cmath>

#include <opencv2/opencv.hpp>

#include "../src/core/types.hpp"
#include "../src/core/pipeline.hpp"
#include "../src/core/frame_source.hpp"

using namespace ascii;

#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running " #name "... "; \
    try { \
        test_##name(); \
        std::cout << "PASSED\n"; \
    } catch (const std::exception& e) { \
        std::cout << "FAILED: " << e.what() << "\n"; \
        failures++; \
    } catch (...) { \
        std::cout << "FAILED: unknown exception\n"; \
        failures++; \
    } \
} while(0)

int failures = 0;

TEST(pipeline_target_dimensions_fit) {
    FrameBuffer frame(320, 180, Color(32, 64, 128, 255));
    
    Pipeline::Config cfg;
    cfg.target_cols = 40;
    cfg.target_rows = 20;
    cfg.cell_width = 8;
    cfg.cell_height = 16;
    cfg.scale_mode = "fit";
    
    Pipeline pipeline(cfg);
    auto result = pipeline.process(frame);
    
    assert(result.luminance.width() == 40 * 8);
    assert(result.luminance.height() == 20 * 16);
    assert(result.color_buffer.width() == result.luminance.width());
    assert(result.color_buffer.height() == result.luminance.height());
    assert(result.grid_cols == 40);
    assert(result.grid_rows == 20);
}

TEST(pipeline_target_dimensions_fill) {
    FrameBuffer frame(1920, 1080, Color(255, 200, 100, 255));
    
    Pipeline::Config cfg;
    cfg.target_cols = 80;
    cfg.target_rows = 24;
    cfg.cell_width = 8;
    cfg.cell_height = 16;
    cfg.scale_mode = "fill";
    
    Pipeline pipeline(cfg);
    auto result = pipeline.process(frame);
    
    assert(result.luminance.width() == 80 * 8);
    assert(result.luminance.height() == 24 * 16);
    assert(result.color_buffer.width() == result.luminance.width());
    assert(result.color_buffer.height() == result.luminance.height());
    assert(result.grid_cols == 80);
    assert(result.grid_rows == 24);
}

TEST(frame_source_factory_extended_types) {
    auto pipe = create_source("pipe:16x8:rgb");
    assert(dynamic_cast<PipeSource*>(pipe.get()) != nullptr);
    
    auto sequence = create_source("frames/*.png");
    assert(dynamic_cast<ImageSequenceSource*>(sequence.get()) != nullptr);
}

TEST(image_sequence_source_reading) {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "ascii_engine_imgseq_test";
    fs::create_directories(dir);
    
    cv::Mat img1(8, 8, CV_8UC3, cv::Scalar(0, 0, 255));
    cv::Mat img2(8, 8, CV_8UC3, cv::Scalar(0, 255, 0));
    cv::imwrite((dir / "0001.png").string(), img1);
    cv::imwrite((dir / "0002.png").string(), img2);
    
    ImageSequenceSource source;
    bool opened = source.open((dir / "*.png").string());
    assert(opened);
    assert(source.is_open());
    assert(source.frame_size().width == 8);
    assert(source.frame_size().height == 8);
    
    FrameBuffer out;
    assert(source.read(out));
    assert(out.width() == 8 && out.height() == 8);
    
    assert(source.read(out));
    assert(!source.read(out));
    
    fs::remove_all(dir);
}

TEST(pipe_source_uri_parsing) {
    PipeSource source;
    assert(source.open("pipe:4x2:rgba:25"));
    assert(source.is_open());
    Size s = source.frame_size();
    assert(s.width == 4);
    assert(s.height == 2);
    assert(std::abs(source.fps() - 25.0) < 0.001);
    
    PipeSource invalid;
    assert(!invalid.open("pipe:bad"));
}

int main() {
    std::cout << "=== ASCII Engine Integration Test Suite ===\n\n";
    
    RUN_TEST(pipeline_target_dimensions_fit);
    RUN_TEST(pipeline_target_dimensions_fill);
    RUN_TEST(frame_source_factory_extended_types);
    RUN_TEST(image_sequence_source_reading);
    RUN_TEST(pipe_source_uri_parsing);
    
    std::cout << "\n=== Test Summary ===\n";
    std::cout << "Failures: " << failures << "\n";
    
    if (failures == 0) {
        std::cout << "\nAll integration tests passed.\n";
        return 0;
    }
    
    std::cout << "\nSome integration tests failed.\n";
    return 1;
}
