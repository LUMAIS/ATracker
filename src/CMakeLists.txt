find_package(Gengetopt QUIET)
include_directories(autogen)

if(GENGETOPT_FOUND AND EXISTS "GenerateArgparser.sh")
    execute_process(sh "GenerateArgparser.sh"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        RESULT_VARIABLE GENOPT_ERRCODE)
    if(NOT GENOPT_ERRCODE EQUAL "0")
        message(WARNING "Application options generation failed. Using original precompiled options.")
    endif()
endif()

find_package(OpenCV REQUIRED)  # core highgui imgproc imgcodecs
include_directories(${OpenCV_INCLUDE_DIRS})

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")  # /usr/lib/x86_64-linux-gnu/cmake/opencv4;  /usr/lib/x86_64-linux-gnu/
#----------------------------libtorch-----------------------------------
# ATTENTION: the project requires cxx11 ABI (wotks with CPU version, but might fail on GPU) of Libtorch/PyTorch, which differs from the one in Python libs
if(NOT ${TORCH_INSTALL_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "libtorch")  # if libtorch is included into the src of this project
endif()
find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(${TORCH_INCLUDE_DIRS})

message(STATUS "TORCH_INCLUDE_DIRS: ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")
#----------------------------libtorch-----------------------------------

set(SOURCES main.cpp autogen/cmdline.c)
#  lib/utils.cpp
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
#set(HDR_FILES main.hpp)

# AntDetect with shared lib ----
set(PROJ_LIBNAME tracker)
include_directories(lib)
# link_directories(${CMAKE_SOURCE_DIR}/lib)
# list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/version.cc" version.hh)
# add_subdirectory(${CMAKE_SOURCE_DIR}/lib ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
link_directories(${CMAKE_SOURCE_DIR} .)
file(GLOB LIB_SRC CONFIGURE_DEPENDS
     "lib/*.cpp"
)
# list(REMOVE_ITEM LIB_SRC "lib/outdated.cpp")
message(STATUS LIB_SRC="${LIB_SRC}")
# ${CMAKE_CURRENT_SOURCE_DIR}/lib/
add_library(${PROJ_LIBNAME} SHARED ${LIB_SRC})
add_executable(${PROJECT_NAME} ${SOURCES})  # detector
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib)
# ATTENTION: the project requires cxx11 ABI (wotks with CPU version, but might fail on GPU) of Libtorch/PyTorch, which differs from the one in Python libs
target_link_libraries(${PROJECT_NAME} ${PROJ_LIBNAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# AntDetect without shared lib ----
# add_executable(${PROJECT_NAME} ${SOURCES} lib/tracker.cpp)  # detector
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

# Test PROJECT_NAME ----
# add_executable(${PROJECT_NAME} test.cpp)  # detector
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# # target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} ${OpenCV_LIBS})

# Test ----
# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
# add_executable(test test.cpp)
# target_link_libraries(test ${OpenCV_LIBS})
