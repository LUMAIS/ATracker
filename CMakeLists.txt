if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(atracker VERSION 1.0.0
            DESCRIPTION "Basic ant tracker (former AntDetect), whose stable functionality is integrated into LAFFTrack/artemis"
            HOMEPAGE_URL https://github.com/LUMAIS/atracker
            LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)

message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

add_subdirectory(src)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
# add_executable(test src/test.cpp)
# target_link_libraries(test ${OpenCV_LIBS})

IF(EXISTS data)
    file(COPY data DESTINATION ${PROJECT_BINARY_DIR}/)
endif()
