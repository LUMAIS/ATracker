cmake_minimum_required(VERSION 3.14)

if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)

project(atracker VERSION 1.0.1
            DESCRIPTION "Basic ant tracker (former AntDetect), whose stable functionality is integrated into LAFFTrack/artemis"
            HOMEPAGE_URL https://github.com/LUMAIS/atracker
            LANGUAGES C CXX)

find_package(Git)
if(GIT_FOUND)
    execute_process(
        COMMAND git describe --always --dirty=+ --broken=-
        # PROJECT_SOURCE_DIR
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_SRC_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        )
    set(CMDLINE_PARSER_VERSION "${CMAKE_PROJECT_VERSION} (${GIT_SRC_VERSION})")
    message(STATUS CMDLINE_PARSER_VERSION="${CMDLINE_PARSER_VERSION}")
    # Transfer some definitions to the source code
    add_definitions(-DGIT_SRC_VERSION="${GIT_SRC_VERSION}")
    add_definitions(-DCMDLINE_PARSER_VERSION="${CMDLINE_PARSER_VERSION}")
    # # generate version.cc
    # configure_file("${CMAKE_CURRENT_SOURCE_DIR}/version.cc.in" "${CMAKE_CURRENT_BINARY_DIR}/version.cc" @ONLY)
    # list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/version.cc" version.hh)
endif(GIT_FOUND)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/lib)
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

# file(CREATE_LINK <original> <linkname> [RESULT <result>] [COPY_ON_ERROR] [SYMBOLIC])
# set(DATA data)
foreach(DATA data)
    # IF(EXISTS ${DATA})
    message(STATUS "Transferring to the binaries: DATA=${DATA} to ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/")
    file(CREATE_LINK ${PROJECT_SOURCE_DIR}/${DATA}
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${DATA}
        RESULT errcode COPY_ON_ERROR SYMBOLIC)
    if(${errcode})
        message(SEND_ERROR "Resource (${DATA}) transfer to the target directory (${PROJECT_BINARY_DIR}) failed: ${errcode}")
    endif()
    # endif()
endforeach()
