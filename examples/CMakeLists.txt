file(GLOB EXAMPLE_SOURCES "*.cc")

foreach(example ${EXAMPLE_SOURCES})
    get_filename_component(name ${example} NAME_WE)

    add_executable(${name} ${example})
    target_link_libraries(${name} PRIVATE cnn)
endforeach()