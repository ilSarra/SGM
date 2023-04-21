cmake_minimum_required(VERSION 3.0)
project(3DP_Lab_1)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRECTORIES})

add_library(sgm STATIC sgm.h sgm.cpp)
add_executable(main main.cpp)

target_link_libraries(main sgm ${OpenCV_LIBS})
target_link_libraries(sgm ${OpenCV_LIBS})