#include "core/motion.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>

#ifdef HAS_OPENMP
#include <omp.h>
#endif

using namespace ascii;

namespace {

FloatImage pattern(int width, int height, int offset) {
    FloatImage image(width, height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const uint32_t value = static_cast<uint32_t>(x + offset) * 1664525u ^
                                   static_cast<uint32_t>(y + offset / 2) * 1013904223u;
            image.set(x, y, static_cast<float>((value >> 8) & 255u) / 255.0f);
        }
    }
    return image;
}

void set_threads(int count) {
#ifdef HAS_OPENMP
    omp_set_num_threads(count);
#else
    (void)count;
#endif
}

void test_parallel_motion_is_bit_exact() {
    MotionEstimator::Config config;
    config.max_reuse_frames = 0;
    config.still_scene_threshold = 0.0f;
    config.phase_interval = 1;
    MotionEstimator serial(config), parallel(config);
    uint64_t compared = 0;
    bool has_nonzero_motion = false;
    bool has_confidence = false;
    for (const auto [width, height] : {std::pair{960,640}, {961,641}, {127,89}, {100,70}}) {
        auto previous = pattern(width, height, 0);
        for (int offset : {1,2,3}) {
            auto current = pattern(width, height, offset);
            set_threads(1);
            serial.compute_flow(previous, current);
            set_threads(8);
            parallel.compute_flow(previous, current);
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const auto& expected = serial.get_motion(x, y);
                    const auto& actual = parallel.get_motion(x, y);
                    for (const auto [a, b] : {std::pair{expected.dx, actual.dx},
                                             {expected.dy, actual.dy},
                                             {expected.confidence, actual.confidence}}) {
                        if (!std::isfinite(b) || std::bit_cast<uint32_t>(a) != std::bit_cast<uint32_t>(b)) {
                            throw std::runtime_error("parallel motion changed a flow component");
                        }
                        ++compared;
                    }
                    has_nonzero_motion |= actual.dx != 0.0f || actual.dy != 0.0f;
                    has_confidence |= actual.confidence > 0.0f;
                }
            }
            previous = current;
        }
    }
    if (!has_nonzero_motion || !has_confidence) {
        throw std::runtime_error("motion comparison must exercise nonzero confident flow");
    }
    std::cout << "Bit-exact serial/parallel motion components: " << compared << '\n';
}

}  // namespace

int main() {
#ifdef HAS_OPENMP
    const int previous_threads = omp_get_max_threads();
#endif
    int result = 0;
    try {
        test_parallel_motion_is_bit_exact();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        result = 1;
    }
#ifdef HAS_OPENMP
    omp_set_num_threads(previous_threads);
#endif
    return result;
}
