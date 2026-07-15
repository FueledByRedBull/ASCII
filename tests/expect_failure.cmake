if(NOT DEFINED PROGRAM OR NOT DEFINED EXPECT)
    message(FATAL_ERROR "PROGRAM and EXPECT are required")
endif()

execute_process(
    COMMAND "${PROGRAM}" ${PROGRAM_ARGS}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
)

if(result EQUAL 0)
    message(FATAL_ERROR "Command unexpectedly succeeded\n${stdout}${stderr}")
endif()

set(combined "${stdout}${stderr}")
string(FIND "${combined}" "${EXPECT}" match_index)
if(match_index EQUAL -1)
    message(FATAL_ERROR "Expected diagnostic not found: ${EXPECT}\n${combined}")
endif()
