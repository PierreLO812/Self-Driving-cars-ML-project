cmake_minimum_required(VERSION 3.10)
project(VisionController)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(vision_controller vision_controller.cpp)
target_link_libraries(vision_controller ${OpenCV_LIBS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
