set(SRC_FILES main.cpp)
#set(HDR_FILES main.hpp)

find_package(OpenCV REQUIRED)

#----------------------------libtorch-----------------------------------
list(APPEND CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
#----------------------------libtorch-----------------------------------



ADD_LIBRARY(antdetect SHARED trophallaxis/antdetect.cpp)

add_executable(detector main.cpp)

target_link_libraries(detector antdetect ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES})
