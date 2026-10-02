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

add_executable(test_analysis_regressions test_analysis_regressions.cpp)
target_link_libraries(test_analysis_regressions PRIVATE ascii-engine-core)
add_test(NAME AnalysisRegressionsTest COMMAND test_analysis_regressions)

add_executable(test_mapping_regressions test_mapping_regressions.cpp)
target_link_libraries(test_mapping_regressions PRIVATE ascii-engine-core)
add_test(NAME MappingRegressionsTest COMMAND test_mapping_regressions)

add_executable(test_color_regressions test_color_regressions.cpp)
target_link_libraries(test_color_regressions PRIVATE ascii-engine-core)
add_test(NAME ColorRegressionsTest COMMAND test_color_regressions)

add_executable(test_bitmap_regressions test_bitmap_regressions.cpp)
target_link_libraries(test_bitmap_regressions PRIVATE ascii-engine-core)
add_test(NAME BitmapRegressionsTest COMMAND test_bitmap_regressions)

add_executable(test_boundary_regressions test_boundary_regressions.cpp)
target_link_libraries(test_boundary_regressions PRIVATE ascii-engine-core)
add_test(NAME BoundaryRegressionsTest COMMAND test_boundary_regressions)

add_executable(test_descriptor_regressions test_descriptor_regressions.cpp)
target_link_libraries(test_descriptor_regressions PRIVATE ascii-engine-core)
add_test(NAME DescriptorRegressionsTest COMMAND test_descriptor_regressions)

add_executable(test_aggregation_regressions test_aggregation_regressions.cpp)
target_link_libraries(test_aggregation_regressions PRIVATE ascii-engine-core)
add_test(NAME AggregationRegressionsTest COMMAND test_aggregation_regressions)

add_executable(test_edge_regressions test_edge_regressions.cpp)
target_link_libraries(test_edge_regressions PRIVATE ascii-engine-core)
add_test(NAME EdgeRegressionsTest COMMAND test_edge_regressions)

add_executable(test_motion_regressions test_motion_regressions.cpp)
target_link_libraries(test_motion_regressions PRIVATE ascii-engine-core)
add_test(NAME MotionRegressionsTest COMMAND test_motion_regressions)

add_executable(test_state_regressions test_state_regressions.cpp)
target_link_libraries(test_state_regressions PRIVATE ascii-engine-core)
add_test(NAME StateRegressionsTest COMMAND test_state_regressions)

add_executable(test_selector_regressions test_selector_regressions.cpp)
target_link_libraries(test_selector_regressions PRIVATE ascii-engine-core)
add_test(NAME SelectorRegressionsTest COMMAND test_selector_regressions)

add_executable(test_source_regressions test_source_regressions.cpp)
target_include_directories(test_source_regressions PRIVATE ${LIBAV_INCLUDE_DIRS})
target_link_libraries(test_source_regressions PRIVATE ascii-engine-core)
add_test(NAME SourceRegressionsTest COMMAND test_source_regressions)

add_executable(test_audio_regressions test_audio_regressions.cpp)
target_include_directories(test_audio_regressions PRIVATE ${SDL2_INCLUDE_DIRS} ${LIBAV_INCLUDE_DIRS})
target_link_libraries(test_audio_regressions PRIVATE ascii-engine-core)
add_test(NAME AudioRegressionsTest COMMAND test_audio_regressions)

add_executable(test_contours test_contours.cpp)
target_link_libraries(test_contours PRIVATE ascii-engine-core)

add_test(NAME ContourTest COMMAND test_contours)

add_executable(test_media_fixture test_media_fixture.cpp)
target_link_libraries(test_media_fixture PRIVATE ascii-engine-core)

if(ASCII_USE_OPENCV)
    add_executable(test_integration test_integration.cpp)
    target_link_libraries(test_integration PRIVATE ascii-engine-core)
    add_test(NAME IntegrationTest COMMAND test_integration)
endif()

if(TARGET ascii-engine)
    function(add_expected_failure_test name expected)
        add_test(NAME ${name}
            COMMAND ${CMAKE_COMMAND}
                "-DPROGRAM=$<TARGET_FILE:ascii-engine>"
                "-DEXPECT=${expected}"
                "-DPROGRAM_ARGS=${ARGN}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/expect_failure.cmake)
    endfunction()

    add_test(NAME MainHelpTest COMMAND $<TARGET_FILE:ascii-engine> --help)
    add_test(NAME HelpNoContoursTest COMMAND $<TARGET_FILE:ascii-engine> --help)
    set_tests_properties(HelpNoContoursTest PROPERTIES
        PASS_REGULAR_EXPRESSION "--no-contours")
    add_test(NAME HelpContourThresholdTest COMMAND $<TARGET_FILE:ascii-engine> --help)
    set_tests_properties(HelpContourThresholdTest PROPERTIES
        PASS_REGULAR_EXPRESSION "--contour-thresh")
    add_expected_failure_test(UnsupportedOutputTest "Unsupported output extension"
        dummy-input --no-audio -o out.unsupported)
    add_expected_failure_test(MalformedCliTest "Invalid value for --fps"
        dummy-input --fps not-a-number)
    add_expected_failure_test(TextOutputFailureTest "Failed to write text output"
            ${CMAKE_SOURCE_DIR}/images/potsepis.png
            --cols 16 --rows 8 --fast --no-audio
            -o ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt/out.txt)
    add_expected_failure_test(ReplayOutputFailureTest "Failed to open replay output"
            ${CMAKE_SOURCE_DIR}/images/potsepis.png
            --cols 16 --rows 8 --fast --no-audio
            --replay ${CMAKE_CURRENT_BINARY_DIR}/CMakeCache.txt/out.areplay)
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

    set(media_fixture ${CMAKE_CURRENT_BINARY_DIR}/media_fixture.mkv)
    set(media_text ${CMAKE_CURRENT_BINARY_DIR}/media_frames.txt)
    set(media_encoded ${CMAKE_CURRENT_BINARY_DIR}/media_encoded.mkv)
    set(media_still ${CMAKE_CURRENT_BINARY_DIR}/media_first.jpg)
    set(media_truncated ${CMAKE_CURRENT_BINARY_DIR}/media_truncated.mkv)
    set(blockart_still ${CMAKE_CURRENT_BINARY_DIR}/blockart_first.jpg)
    set(blockart_video ${CMAKE_CURRENT_BINARY_DIR}/blockart_video.mkv)
    set(golden_replay ${CMAKE_CURRENT_BINARY_DIR}/golden.areplay)
    set(golden_replay_text ${CMAKE_CURRENT_BINARY_DIR}/golden.txt)

    add_test(NAME GenerateMediaFixture
        COMMAND $<TARGET_FILE:test_media_fixture> generate ${media_fixture} 8)
    add_test(NAME VerifyMediaFixture
        COMMAND $<TARGET_FILE:test_media_fixture> verify-video ${media_fixture} 8 12)
    set_tests_properties(VerifyMediaFixture PROPERTIES DEPENDS GenerateMediaFixture)

    add_test(NAME VideoTextExport
        COMMAND $<TARGET_FILE:ascii-engine> ${media_fixture}
            --cols 16 --rows 8 --fast --no-audio -o ${media_text})
    set_tests_properties(VideoTextExport PROPERTIES DEPENDS GenerateMediaFixture)
    add_test(NAME VerifyVideoTextExport
        COMMAND $<TARGET_FILE:test_media_fixture> verify-text ${media_text} 8)
    set_tests_properties(VerifyVideoTextExport PROPERTIES DEPENDS VideoTextExport)

    add_test(NAME VideoEncodedExport
        COMMAND $<TARGET_FILE:ascii-engine> ${media_fixture}
            --cols 16 --rows 8 --fast --no-audio -o ${media_encoded})
    set_tests_properties(VideoEncodedExport PROPERTIES DEPENDS GenerateMediaFixture)
    add_test(NAME VerifyVideoEncodedExport
        COMMAND $<TARGET_FILE:test_media_fixture> verify-video ${media_encoded} 8 30)
    set_tests_properties(VerifyVideoEncodedExport PROPERTIES DEPENDS VideoEncodedExport)

    add_test(NAME VideoStillExport
        COMMAND $<TARGET_FILE:ascii-engine> ${media_fixture}
            --cols 16 --rows 8 --fast --no-audio -o ${media_still})
    set_tests_properties(VideoStillExport PROPERTIES DEPENDS GenerateMediaFixture)
    add_test(NAME VerifyVideoStillExport
        COMMAND $<TARGET_FILE:test_media_fixture> verify-image ${media_still})
    set_tests_properties(VerifyVideoStillExport PROPERTIES DEPENDS VideoStillExport)

    add_test(NAME BlockArtStillExport
        COMMAND $<TARGET_FILE:ascii-engine> ${media_fixture}
            --cols 16 --rows 8 --fast --no-audio --color blockart -o ${blockart_still})
    set_tests_properties(BlockArtStillExport PROPERTIES DEPENDS GenerateMediaFixture)
    add_test(NAME VerifyBlockArtStillExport
        COMMAND $<TARGET_FILE:test_media_fixture> verify-image ${blockart_still})
    set_tests_properties(VerifyBlockArtStillExport PROPERTIES DEPENDS BlockArtStillExport)

    add_test(NAME BlockArtVideoExport
        COMMAND $<TARGET_FILE:ascii-engine> ${media_fixture}
            --cols 16 --rows 8 --fast --no-audio --color blockart -o ${blockart_video})
    set_tests_properties(BlockArtVideoExport PROPERTIES DEPENDS GenerateMediaFixture)
    add_test(NAME VerifyBlockArtVideoExport
        COMMAND $<TARGET_FILE:test_media_fixture> verify-video ${blockart_video} 8 30)
    set_tests_properties(VerifyBlockArtVideoExport PROPERTIES DEPENDS BlockArtVideoExport)

    add_test(NAME CreateTruncatedMedia
        COMMAND $<TARGET_FILE:test_media_fixture> truncate ${media_fixture} ${media_truncated})
    set_tests_properties(CreateTruncatedMedia PROPERTIES DEPENDS GenerateMediaFixture)
    add_expected_failure_test(TruncatedMediaFailure "Error:"
        ${media_truncated} --cols 16 --rows 8 --fast --no-audio -o ${CMAKE_CURRENT_BINARY_DIR}/truncated.txt)
    set_tests_properties(TruncatedMediaFailure PROPERTIES DEPENDS CreateTruncatedMedia)

    function(add_runtime_contract name case input)
        add_test(NAME ${name} COMMAND ${CMAKE_COMMAND}
            "-DPROGRAM=$<TARGET_FILE:ascii-engine>"
            "-DCASE=${case}" "-DINPUT=${input}"
            "-DMEDIA_HELPER=$<TARGET_FILE:test_media_fixture>"
            "-DWORK=${CMAKE_CURRENT_BINARY_DIR}/${name}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/runtime_contracts.cmake)
    endfunction()
    add_runtime_contract(InputOutputCollision collision ${media_fixture})
    set_tests_properties(InputOutputCollision PROPERTIES DEPENDS GenerateMediaFixture)
    add_runtime_contract(NumberedTextCollision numbered-collision ${media_fixture})
    set_tests_properties(NumberedTextCollision PROPERTIES DEPENDS GenerateMediaFixture)
    add_runtime_contract(FailedDecodePreservesOutput preserve ${media_truncated})
    set_tests_properties(FailedDecodePreservesOutput PROPERTIES DEPENDS CreateTruncatedMedia)
    add_runtime_contract(ConfigColorOutput config-color ${CMAKE_SOURCE_DIR}/images/potsepis.png)
    add_runtime_contract(DebugReplayOutput debug-replay ${CMAKE_SOURCE_DIR}/images/potsepis.png)
    add_runtime_contract(DirectoryOutputRejected destination-directory ${CMAKE_SOURCE_DIR}/images/potsepis.png)
    add_runtime_contract(SequenceOutputCollision sequence-collision ${CMAKE_SOURCE_DIR}/images/potsepis.png)
    add_runtime_contract(UnicodePathContracts unicode-paths ${media_fixture})
    set_tests_properties(UnicodePathContracts PROPERTIES DEPENDS GenerateMediaFixture)

    add_test(NAME GenerateGoldenReplay
        COMMAND $<TARGET_FILE:test_media_fixture> generate-replay ${golden_replay})
    add_test(NAME GoldenReplayTextExport
        COMMAND $<TARGET_FILE:ascii-engine> --play-replay ${golden_replay} -o ${golden_replay_text})
    set_tests_properties(GoldenReplayTextExport PROPERTIES DEPENDS GenerateGoldenReplay)
    add_runtime_contract(ReplayNumberedTextCollision replay-numbered-collision ${golden_replay})
    set_tests_properties(ReplayNumberedTextCollision PROPERTIES DEPENDS GenerateGoldenReplay)
    add_test(NAME VerifyGoldenReplayText
        COMMAND $<TARGET_FILE:test_media_fixture> verify-replay-text ${golden_replay_text})
    set_tests_properties(VerifyGoldenReplayText PROPERTIES DEPENDS GoldenReplayTextExport)
    add_test(NAME EncoderPartialFailure
        COMMAND $<TARGET_FILE:test_media_fixture> verify-encoder-partial-failure
            ${CMAKE_CURRENT_BINARY_DIR}/partial_failure.mkv)
    add_test(NAME EncoderUnfinishedOutput
        COMMAND $<TARGET_FILE:test_media_fixture> verify-encoder-unfinished
            ${CMAKE_CURRENT_BINARY_DIR}/unfinished.mkv)
    add_test(NAME EncoderGeometry
        COMMAND $<TARGET_FILE:test_media_fixture> verify-encoder-geometry
            ${CMAKE_CURRENT_BINARY_DIR}/encoder_geometry.mkv)
endif()
