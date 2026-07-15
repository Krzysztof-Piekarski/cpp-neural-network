include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_executable(cnn_tests
    MatrixTests.cc
    ActivationTests.cc
)

target_link_libraries(cnn_tests
    PRIVATE
        cnn
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(cnn_tests)