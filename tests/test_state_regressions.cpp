#include "core/color_space.hpp"
#include "core/pipeline_runtime_cache.hpp"
#include "glyph/font_loader.hpp"

#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace ascii;

namespace {

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_color_initialization() {
    std::array<uint8_t, 8193> before;
    for (size_t i = 0; i < before.size(); ++i) {
        before[i] = ColorSpace::linear_to_srgb(static_cast<float>(i) / 8192.0f);
    }
    ColorSpace::init();
    for (size_t i = 0; i < before.size(); ++i) {
        expect(before[i] == ColorSpace::linear_to_srgb(static_cast<float>(i) / 8192.0f),
               "color conversion must not depend on an explicit initialization call");
    }
}

void test_concurrent_color_initialization() {
    constexpr int thread_count = 8;
    std::barrier start(thread_count);
    std::atomic<int> failures = 0;
    std::vector<std::thread> threads;
    for (int thread = 0; thread < thread_count; ++thread) {
        threads.emplace_back([&] {
            start.arrive_and_wait();
            for (int repeat = 0; repeat < 32; ++repeat) {
                ColorSpace::init();
                for (int value = 0; value < 256; ++value) {
                    const float linear = ColorSpace::srgb_to_linear(static_cast<uint8_t>(value));
                    const int encoded = ColorSpace::linear_to_srgb(linear);
                    if (!std::isfinite(linear) || std::abs(encoded - value) > 1) ++failures;
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();
    expect(failures == 0, "concurrent color conversions must remain deterministic");
}

void test_cache_color_stats_upgrade(bool need_color_buffer) {
    PipelineRuntimeCache cache;
    FrameBuffer frame(8, 16, Color(255, 0, 0));
    PipelineRuntimeCache::Query query;
    query.reuse_limit = 4;
    query.need_color_buffer = false;
    query.need_color_stats = false;
    expect(!cache.begin_frame(frame, query).reuse_pipeline_result, "empty cache cannot reuse a result");
    Pipeline::Result result{};
    result.grid_cols = result.grid_rows = 1;
    result.cell_stats.resize(1);
    cache.commit_processed_result(result, false, false, false);

    query.need_color_stats = true;
    query.need_color_buffer = need_color_buffer;
    const auto decision = cache.begin_frame(frame, query);
    expect(!decision.reuse_pipeline_result, "color-statistics upgrade must recompute a monochrome result");
    expect(!decision.reuse_cell_stats, "color-statistics upgrade must recompute monochrome cell statistics");
    expect(decision.process_options.need_color_stats, "upgrade must request color means from the pipeline");
    expect(!decision.process_options.reuse_cell_stats, "upgrade must not supply incomplete cached means");

    result.cell_stats[0].mean_r = 1.0f;
    if (need_color_buffer) result.color_buffer = frame;
    cache.commit_processed_result(result, need_color_buffer, true, false);
    const auto reused = cache.begin_frame(frame, query);
    expect(reused.reuse_pipeline_result && reused.reused_result->cell_stats[0].mean_r == 1.0f,
           "complete color statistics should remain reusable");
    if (!need_color_buffer) {
        query.need_color_buffer = true;
        query.need_color_stats = false;
        const auto cells = cache.begin_frame(frame, query);
        expect(!cells.reuse_pipeline_result && cells.reuse_cell_stats,
               "a new color buffer may reuse previously complete cell statistics");
        result.color_buffer = frame;
        cache.commit_processed_result(result, true, false, true);
        query.need_color_stats = true;
        expect(cache.begin_frame(frame, query).reuse_pipeline_result,
               "reusing complete cells must retain their color-statistics capability");
    }
    cache.invalidate();
    expect(!cache.begin_frame(frame, query).reuse_pipeline_result, "invalidation must drop cached capabilities");
}

bool same_bitmap(const GlyphBitmap& a, const GlyphBitmap& b) {
    return a.width == b.width && a.height == b.height && a.advance == b.advance &&
           a.bearing_x == b.bearing_x && a.bearing_y == b.bearing_y && a.pixels == b.pixels;
}

std::string installed_font() {
    const auto path = FontLoader::find_system_monospace_font();
    expect(!path.empty(), "font regressions require an installed monospace font");
    return path;
}

void test_font_reload() {
    const auto path = installed_font();
    FontLoader loader;
    expect(loader.load(path, 16.0f).success(), "installed font must load at 16 pixels");
    const auto small = loader.render_glyph('M');
    expect(!small.empty(), "installed font must render M");
    expect(loader.load(path, 32.0f).success(), "installed font must reload at 32 pixels");
    FontLoader fresh;
    expect(fresh.load(path, 32.0f).success(), "comparison font must load");
    const auto large = loader.render_glyph('M');
    expect(same_bitmap(large, fresh.render_glyph('M')), "font reload must discard stale glyph bitmaps");
    expect(!same_bitmap(small, large), "font size change must alter rendered glyphs");
}

void test_font_height_and_failed_reload() {
    const auto path = installed_font();
    FontLoader loader;
    expect(loader.load(path, 16.0f).success(), "installed font must load");
    const auto before = loader.render_glyph('M');
    const int line_height = loader.line_height();
    for (float height : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(), 1025.0f, 1.0e30f}) {
        expect(!loader.load(path, height).success(), "invalid font height must be rejected");
        expect(loader.is_loaded() && loader.pixel_height() == 16.0f && loader.line_height() == line_height,
               "failed reload must preserve the previous valid font metrics");
        expect(same_bitmap(before, loader.render_glyph('M')), "failed reload must preserve valid glyph data");
    }
    expect(!loader.load("missing-state-regression-font.ttf").success(), "missing font must fail");
    expect(same_bitmap(before, loader.render_glyph('M')), "missing replacement must preserve the previous font");
    FontLoader empty;
    expect(!empty.load("missing-state-regression-font.ttf").success(), "missing initial font must fail");
    expect(!empty.is_loaded() && empty.render_glyph('M').empty(), "failed initial load cannot claim a usable font");
}

void test_memory_font_ownership() {
    const auto path = installed_font();
    FontLoader loader;
    {
        std::ifstream file(path, std::ios::binary);
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(file)), {});
        expect(!bytes.empty(), "installed font bytes must be readable");
        expect(loader.load_from_memory(bytes.data(), bytes.size(), 24.0f).success(), "memory font must load");
    }
    FontLoader fresh;
    expect(fresh.load(path, 24.0f).success(), "comparison font must load");
    expect(same_bitmap(loader.render_glyph('W'), fresh.render_glyph('W')),
           "memory font must remain usable after the caller releases its input buffer");
}

} // namespace

int main() {
    try {
        test_color_initialization();
        test_concurrent_color_initialization();
        test_cache_color_stats_upgrade(false);
        test_cache_color_stats_upgrade(true);
        test_font_reload();
        test_font_height_and_failed_reload();
        test_memory_font_ownership();
        std::cout << "[OK] State regression tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
