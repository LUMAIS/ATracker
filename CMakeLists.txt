cmake_policy(SET CMP0048 NEW)

project(detector VERSION 0.4.4
                LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)# core highgui imgproc imgcodecs)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(src)

message("PROJECT_BINARY_DIR -> " ${PROJECT_BINARY_DIR})
message("PROJECT_SOURCE_DIR -> " ${PROJECT_SOURCE_DIR})

file(COPY data DESTINATION ${PROJECT_BINARY_DIR}/)