cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(example)

find_package(Torch REQUIRED)

add_executable(main.x main.cc)
target_link_libraries(main.x "${TORCH_LIBRARIES}")
set_property(TARGET main.x PROPERTY CXX_STANDARD 11)