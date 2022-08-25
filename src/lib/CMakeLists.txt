message("Built tracker lib")

set(CMAKE_BUILD_TYPE RelWithDebInfo)  # Debug , Release , RelWithDebInfo

# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h onnxruntime-linux-x64-1.8.1/include)
# message("ONNX_RUNTIME_SESSION_INCLUDE_DIRS -> " ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS})
# find_path(ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS cuda_provider_factory.h HINTS onnxruntime-linux-x64-gpu-1.8.1/include/)

# include_directories(${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} 
# 					${ONNX_RUNTIME_PROVIDERS_INCLUDE_DIRS} 
# 					${ONNX_VERSION_CONVERTER_INCLUDE_DIRS}
# 	                )

add_library(${PROJECT_NAME} SHARED tracker.cpp)