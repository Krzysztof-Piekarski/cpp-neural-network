cmake_minimum_required(VERSION 3.24)

project(cpp-neural-network
    VERSION 0.1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cnn
    src/Matrix.cc
    src/Activation.cc
    src/Initializer.cc
    src/DenseLayer.cc
    src/Loss.cc
)

target_include_directories(cnn
    PUBLIC
        ${CMAKE_SOURCE_DIR}/inc
)

target_compile_features(cnn PUBLIC cxx_std_20)

option(BUILD_TESTING "Build unit tests" ON)

if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

if (MSVC)
    target_compile_options(cnn PRIVATE /W4)
else()
    target_compile_options(cnn PRIVATE -Wall -Wextra -Wpedantic)
endif()