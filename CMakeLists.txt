cmake_minimum_required(VERSION 3.16)
project(ascii_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2")
    set(CMAKE_CXX_FLAGS_DEBUG "/Zi /Od")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
endif()

option(ASCII_USE_OPENCV "Enable OpenCV utility paths" ON)
option(ASCII_ENABLE_AVX2 "Enable AVX2 optimized kernels where supported" OFF)

if(ASCII_USE_OPENCV)
    find_package(OpenCV 4.0 REQUIRED)
    add_compile_definitions(ASCII_USE_OPENCV)
endif()

if(WIN32)
    find_package(SDL2 QUIET CONFIG)
    if(SDL2_FOUND)
        set(SDL2_LIBRARIES SDL2::SDL2)
    else()
        find_path(SDL2_INCLUDE_DIR SDL.h PATH_SUFFIXES SDL2 include)
        find_library(SDL2_LIBRARY NAMES SDL2)
        if(NOT SDL2_INCLUDE_DIR OR NOT SDL2_LIBRARY)
            message(FATAL_ERROR "SDL2 not found. Install SDL2 dev files or provide SDL2 via package manager.")
        endif()
        set(SDL2_INCLUDE_DIRS ${SDL2_INCLUDE_DIR})
        set(SDL2_LIBRARIES ${SDL2_LIBRARY})
    endif()

    find_path(LIBAV_INCLUDE_DIR libavcodec/avcodec.h PATH_SUFFIXES include)
    find_library(AVFORMAT_LIB NAMES avformat)
    find_library(AVCODEC_LIB NAMES avcodec)
    find_library(AVUTIL_LIB NAMES avutil)
    find_library(SWSCALE_LIB NAMES swscale)
    find_library(SWRESAMPLE_LIB NAMES swresample)
    if(NOT LIBAV_INCLUDE_DIR OR NOT AVFORMAT_LIB OR NOT AVCODEC_LIB OR NOT AVUTIL_LIB OR NOT SWSCALE_LIB OR NOT SWRESAMPLE_LIB)
        message(FATAL_ERROR "FFmpeg libraries not found. Ensure avformat, avcodec, avutil, swscale, and swresample are installed.")
    endif()
    set(LIBAV_INCLUDE_DIRS ${LIBAV_INCLUDE_DIR})

    find_path(ZSTD_INCLUDE_DIR zstd.h PATH_SUFFIXES include)
    find_library(ZSTD_LIB NAMES zstd)
    if(NOT ZSTD_INCLUDE_DIR OR NOT ZSTD_LIB)
        message(FATAL_ERROR "zstd not found. Install zstd dev files.")
    endif()
    set(ZSTD_INCLUDE_DIRS ${ZSTD_INCLUDE_DIR})
    set(ZSTD_LIBRARIES ${ZSTD_LIB})
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2 REQUIRED sdl2)
    pkg_check_modules(LIBAV REQUIRED libavcodec libavformat libavutil libswscale libswresample)
    pkg_check_modules(ZSTD REQUIRED libzstd)

    find_library(AVCODEC_LIB avcodec)
    find_library(AVFORMAT_LIB avformat)
    find_library(AVUTIL_LIB avutil)
    find_library(SWSCALE_LIB swscale)
    find_library(SWRESAMPLE_LIB swresample)
    find_library(ZSTD_LIB zstd)
endif()

find_package(OpenMP)

add_executable(ascii-engine
    src/main.cpp
    src/core/config.cpp
    src/core/color_space.cpp
    src/core/replay.cpp
    src/core/motion.cpp
    src/core/frame_source.cpp
    src/core/pipeline.cpp
    src/core/cell_stats.cpp
    src/core/edge_detector.cpp
    src/core/temporal.cpp
    src/glyph/font_loader.cpp
    src/glyph/glyph_cache.cpp
    src/glyph/char_sets.cpp
    src/mapping/char_selector.cpp
    src/mapping/color_mapper.cpp
    src/mapping/bilateral_grid.cpp
    src/render/dither.cpp
    src/render/terminal_renderer.cpp
    src/render/block_renderer.cpp
    src/render/bitmap_renderer.cpp
    src/render/video_encoder.cpp
    src/audio/audio_player.cpp
    src/terminal/terminal.cpp
    src/cli/args.cpp
)

target_include_directories(ascii-engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor
    ${SDL2_INCLUDE_DIRS}
    ${LIBAV_INCLUDE_DIRS}
    ${ZSTD_INCLUDE_DIRS}
)

if(ASCII_USE_OPENCV)
    target_include_directories(ascii-engine PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

if(SDL2_LIBRARY_DIRS OR LIBAV_LIBRARY_DIRS OR ZSTD_LIBRARY_DIRS)
    target_link_directories(ascii-engine PRIVATE
        ${SDL2_LIBRARY_DIRS}
        ${LIBAV_LIBRARY_DIRS}
        ${ZSTD_LIBRARY_DIRS}
    )
endif()

target_link_libraries(ascii-engine PRIVATE
    ${SDL2_LIBRARIES}
    ${AVFORMAT_LIB}
    ${AVCODEC_LIB}
    ${AVUTIL_LIB}
    ${SWSCALE_LIB}
    ${SWRESAMPLE_LIB}
    ${ZSTD_LIBRARIES}
)

if(ASCII_USE_OPENCV)
    target_link_libraries(ascii-engine PRIVATE ${OpenCV_LIBS})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(ascii-engine PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(ascii-engine PRIVATE HAS_OPENMP)
endif()

if(MSVC)
    target_compile_options(ascii-engine PRIVATE /W4)
    target_compile_definitions(ascii-engine PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(ascii-engine PRIVATE -Wall -Wextra)
endif()

if(ASCII_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(ascii-engine PRIVATE /arch:AVX2)
    else()
        target_compile_options(ascii-engine PRIVATE -mavx2)
    endif()
endif()

if(WIN32)
    target_compile_definitions(ascii-engine PRIVATE _WIN32 WIN32_LEAN_AND_MEAN NOMINMAX)
    target_link_libraries(ascii-engine PRIVATE ws2_32)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT MSVC)
    target_link_options(ascii-engine PRIVATE -fsanitize=address)
endif()

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

install(TARGETS ascii-engine RUNTIME DESTINATION bin)
