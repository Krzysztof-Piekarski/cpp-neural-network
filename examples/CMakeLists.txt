add_executable(xor
    xor.cc
)

target_link_libraries(xor
    PRIVATE
        cnn
)