# ATracker
Ant event detector and tracker, involving YOLO-based detection and several tracking techniques.  
The application should be cross-platform, however, it has be validated only on Linux Ubuntu 20.04 LTS / Debian.

# Installation and Prerequisites
ATracker depends on Torch (libtorch with C++11 ABI) and OpenCV.

Torch installation (latest stable version of LibTorch C++) is provided on https://pytorch.org/get-started/locally/, see https://pytorch.org/TensorRT/tutorials/installation.html for details.

OpenCV can be installed on Linux Ubuntu/Debian via apt:
```sh
$ sudo apt install libopencv-dev
```
CMake requires specification of the `OpenCV_DIR` environment variable, which is `/usr/include/opencv4/opencv2` by default on Ubuntu 20.04 LTS.  
In case of issues or dependency resolution errors, OpenCV v4.2.0 can be installed manually: https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html, where the default installation path is `/usr/local` and the cmake configuration is present in the created `./build/`.

A custom configuration of environment variables for CMake can be the following:
```sh
TORCH_INSTALL_PREFIX=/opt/xdk/libtorch-cxx11-gpu
Torch_DIR=/opt/xdk/libtorch-cxx11-gpu/share/cmake/Torch
OpenCV_DIR=/opt/xdk/opencv/build
CUDAToolkit_ROOT=/usr/local/cuda
CMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
CMAKE_CUDA_ARCHITECTURES=all
```
Those variables can be specified for CMake via a terminal or using VSCode IDE setting (`Ctrl + ,`) with the filter `@id:cmake.environment @ext:ms-vscode.cmake-tools` (environment settings of the `CMake Tools` extension).

A manual build with cmake:
```sh
$ mkdir build && cd build && \
  cmake -DCMAKE_CUDA_ARCHITECTURES=all -DCUDAToolkit_ROOT=/usr/local/cuda -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DOpenCV_DIR=/opt/xdk/opencv/build -DTORCH_INSTALL_PREFIX=/opt/xdk/libtorch-cxx11-gpu -DTorch_DIR=/opt/xdk/libtorch-cxx11-gpu/share/cmake/Torch .. && \
  cmake --build . --config Release -j 4
```
