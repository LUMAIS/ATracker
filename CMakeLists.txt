if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(atracker VERSION 1.0.0
            DESCRIPTION "Basic ant tracker (former AntDetect), whose stable functionality is integrated into LAFFTrack/artemis"
            HOMEPAGE_URL https://github.com/LUMAIS/atracker
            LANGUAGES C CXX)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
            
if(NOT DEFINED ${CMAKE_CUDA_COMPILER})
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()
message(DEBUG "CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
if(NOT DEFINED ${CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES all)
endif()
message(DEBUG "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
# message(STATUS "CUDAToolkit_ROOT: ${CUDAToolkit_ROOT}")
# message(STATUS "Torch_DIR: ${Torch_DIR}")

set(CMAKE_CXX_STANDARD 17)

message(VERBOSE "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

add_subdirectory(src)

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
# add_executable(test src/test.cpp)
# target_link_libraries(test ${OpenCV_LIBS})

set(DATA data)
IF(EXISTS ${DATA})
    file(CREATE_LINK ${PROJECT_SOURCE_DIR}/${DATA}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${DATA}
        RESULT errcode COPY_ON_ERROR SYMBOLIC)
endif()
