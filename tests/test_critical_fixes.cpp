#include <iostream>
#include <cassert>
#include <fstream>
#include "../src/core/temporal.hpp"
#include "../src/core/edge_detector.hpp"
#include "../src/core/types.hpp"
#include "../src/glyph/font_loader.hpp"

using namespace ascii;

void test_temporal_initialization() {
    std::cout << "Testing temporal smoothing initialization...\n";
    
    TemporalSmoother smoother;
    smoother.initialize(2, 2);
    
    // Test that both smoothing methods set initialized flag properly
    float lum1 = smoother.smooth_luminance(0, 0.5f);
    assert(smoother.frame_state()[0].initialized);
    assert(lum1 == 0.5f);
    
    // Reset for edge test
    smoother.reset();
    float edge1 = smoother.smooth_edge_strength(0, 0.3f);
    assert(smoother.frame_state()[0].initialized);
    assert(edge1 == 0.3f);
    
    std::cout << "✓ Temporal initialization test passed\n";
}

void test_edge_bounds_checking() {
    std::cout << "Testing edge detection bounds checking...\n";
    
    // Create a small test image
    FloatImage test_img(3, 3);
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            test_img.set(x, y, 0.5f);
        }
    }
    
    // Set center to higher value
    test_img.set(1, 1, 1.0f);
    
    EdgeDetector detector;
    EdgeDetector::Config config;
    config.blur_sigma = 1.0f;
    config.low_threshold = 0.1f;
    config.high_threshold = 0.3f;
    detector.set_config(config);
    
    auto edges = detector.detect(test_img);
    
    // Should not crash and should produce valid results
    assert(!edges.magnitude.empty());
    assert(edges.magnitude.width() == 3);
    assert(edges.magnitude.height() == 3);
    
    std::cout << "✓ Edge bounds checking test passed\n";
}

void test_font_validation() {
    std::cout << "Testing font security validation...\n";
    
    // Test font loader with invalid file
    FontLoader loader;
    auto result = loader.load("nonexistent_font.ttf");
    assert(!result.success());
    
    // Test font loader with invalid data in memory (too small)
    uint8_t small_data[] = {0x00, 0x01, 0x00, 0x00}; // TrueType signature but tiny
    auto mem_result = loader.load_from_memory(small_data, 4);
    assert(!mem_result.success());
    
    std::cout << "✓ Font validation test passed\n";
}

int main() {
    std::cout << "Running critical fixes validation tests...\n\n";
    
    try {
        test_temporal_initialization();
        test_edge_bounds_checking();
        test_font_validation();
        
        std::cout << "\n✓ All critical fixes tests passed!\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "✗ Test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "✗ Test failed with unknown exception\n";
        return 1;
    }
}