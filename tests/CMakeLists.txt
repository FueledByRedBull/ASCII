cmake_minimum_required(VERSION 3.16)

add_executable(test_comprehensive test_comprehensive.cpp
    ../src/core/temporal.cpp
    ../src/core/edge_detector.cpp
    ../src/core/cell_stats.cpp
    ../src/glyph/font_loader.cpp
    ../src/glyph/glyph_cache.cpp
    ../src/mapping/color_mapper.cpp
    ../src/mapping/char_selector.cpp
    ../src/terminal/terminal.cpp
    ../src/cli/args.cpp
)

target_include_directories(test_comprehensive PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_comprehensive 
    ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_comprehensive OpenMP::OpenMP_CXX)
    target_compile_definitions(test_comprehensive PRIVATE HAS_OPENMP)
endif()

add_test(NAME ComprehensiveTest COMMAND test_comprehensive)

add_executable(test_critical_fixes test_critical_fixes.cpp
    ../src/core/temporal.cpp
    ../src/core/edge_detector.cpp
    ../src/glyph/font_loader.cpp
)

target_include_directories(test_critical_fixes PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_critical_fixes 
    ${OpenCV_LIBS}
)

add_test(NAME CriticalFixesTest COMMAND test_critical_fixes)

add_executable(test_integration test_integration.cpp
    ../src/core/frame_source.cpp
    ../src/core/pipeline.cpp
    ../src/core/cell_stats.cpp
    ../src/core/edge_detector.cpp
)

target_include_directories(test_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../src
    ${CMAKE_CURRENT_SOURCE_DIR}/../vendor
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_integration
    ${OpenCV_LIBS}
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(test_integration OpenMP::OpenMP_CXX)
    target_compile_definitions(test_integration PRIVATE HAS_OPENMP)
endif()

add_test(NAME IntegrationTest COMMAND test_integration)

if(TARGET ascii-engine)
    add_test(NAME MainHelpTest COMMAND $<TARGET_FILE:ascii-engine> --help)
endif()
