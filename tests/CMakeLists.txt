cmake_minimum_required(VERSION 3.16)

add_executable(test_comprehensive test_comprehensive.cpp)
target_link_libraries(test_comprehensive PRIVATE ascii-engine-core)

add_test(NAME ComprehensiveTest COMMAND test_comprehensive)

add_executable(test_critical_fixes test_critical_fixes.cpp)
target_link_libraries(test_critical_fixes PRIVATE ascii-engine-core)

add_test(NAME CriticalFixesTest COMMAND test_critical_fixes)

add_executable(test_replay test_replay.cpp)
target_link_libraries(test_replay PRIVATE ascii-engine-core)

add_test(NAME ReplayTest COMMAND test_replay)

add_executable(test_terminal_renderer test_terminal_renderer.cpp)
target_link_libraries(test_terminal_renderer PRIVATE ascii-engine-core)

add_test(NAME TerminalRendererTest COMMAND test_terminal_renderer)

add_executable(test_contours test_contours.cpp)
target_link_libraries(test_contours PRIVATE ascii-engine-core)

add_test(NAME ContourTest COMMAND test_contours)

if(ASCII_USE_OPENCV)
    add_executable(test_integration test_integration.cpp)
    target_link_libraries(test_integration PRIVATE ascii-engine-core)
    add_test(NAME IntegrationTest COMMAND test_integration)
endif()

if(TARGET ascii-engine)
    add_test(NAME MainHelpTest COMMAND $<TARGET_FILE:ascii-engine> --help)
    add_test(NAME HelpNoContoursTest COMMAND $<TARGET_FILE:ascii-engine> --help)
    set_tests_properties(HelpNoContoursTest PROPERTIES
        PASS_REGULAR_EXPRESSION "--no-contours")
    add_test(NAME HelpContourThresholdTest COMMAND $<TARGET_FILE:ascii-engine> --help)
    set_tests_properties(HelpContourThresholdTest PROPERTIES
        PASS_REGULAR_EXPRESSION "--contour-thresh")
    add_test(NAME UnsupportedOutputTest
        COMMAND $<TARGET_FILE:ascii-engine> dummy-input --no-audio -o out.unsupported)
    set_tests_properties(UnsupportedOutputTest PROPERTIES
        WILL_FAIL TRUE
        PASS_REGULAR_EXPRESSION "Unsupported output extension")
    add_test(NAME ImageTextSmoke
        COMMAND $<TARGET_FILE:ascii-engine>
            ${CMAKE_SOURCE_DIR}/images/potsepis.png
            --cols 16 --rows 8 --fast --no-audio
            -o ${CMAKE_CURRENT_BINARY_DIR}/smoke.txt)
    add_test(NAME ImageStillSmoke
        COMMAND $<TARGET_FILE:ascii-engine>
            ${CMAKE_SOURCE_DIR}/images/potsepis.png
            --cols 16 --rows 8 --fast --no-audio
            -o ${CMAKE_CURRENT_BINARY_DIR}/smoke.jpg)
    add_test(NAME ImageReplaySmoke
        COMMAND $<TARGET_FILE:ascii-engine>
            ${CMAKE_SOURCE_DIR}/images/potsepis.png
            --cols 16 --rows 8 --fast --no-audio
            --replay ${CMAKE_CURRENT_BINARY_DIR}/smoke.areplay
            -o ${CMAKE_CURRENT_BINARY_DIR}/smoke_replay.txt)
    add_test(NAME ReplayInspectSmoke
        COMMAND $<TARGET_FILE:ascii-engine>
            --inspect-replay ${CMAKE_CURRENT_BINARY_DIR}/smoke.areplay)
    set_tests_properties(ReplayInspectSmoke PROPERTIES
        DEPENDS ImageReplaySmoke
        PASS_REGULAR_EXPRESSION "frames: 1")
endif()
