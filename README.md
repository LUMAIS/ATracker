# ATracker
Ant event detector and tracker, involving YOLO-based detection and several tracking techniques.  
The application should be cross-platform, however, it has be validated only on Linux Ubuntu 20.04 LTS / Debian.

Authors:  (c) Artem Lutov &lt;&#108;&#97;v&commat;lumais&#46;&#99;om&gt;, Serhii Oleksenko &lt;serhii&commat;lumais&#46;&#99;om&gt;  
License: [Apache License, Version 2](www.apache.org/licenses/LICENSE-2.0.html)  
Organizations: [UNIFR](https://www.unifr.ch), [Lutov Analytics](https://lutan.ch), [LUMAIS](http://lumais.com)

__Table of Contents__
- [Installation and Prerequisites](#installation-and-prerequisites)
- [Usage](#usage)

## Installation and Prerequisites
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

## Usage

```sh
build/bin$ ./atracker -h

Usage: atracker [OPTIONS]

Examples:
  $ ./atracker -m data/AntED_yolo5_traced_992.pt -v data/3.mp4 -s 1 -n 8
  $ ./atracker data/Cflo_troph_count_3-38_3-52.mp4 -n 32

Basic ant tracker (former AntDetect), whose stable functionality is integrated
into LAFFTrack/artemis

Basic ant tracker, which uses YOLO-based ant events detector and several
tracking techniques to track ant-related objects (e.g., ant, larva, pupa) and
their interaction events (e.g., trophallaxis).
NOTE: this application is used mainly for internal evaluation and valudation
purposes before integrating selected functionality into LAFFTrack/artemis.


  -h, --help                Print help and exit
  -V, --version             Print version and exit
  -o, --output=filename     output directory  (default=`.')
  -f, --fout_suffix=STRING  additional suffix for the resulting output files

 Group: detection
  Object detection parameters
  -m, --model=filename      path to the object detector (PyTorch ML model)
  -a, --ant-length=INT      expected ant length  (default=`80')
  -c, --confidence=FLOAT    confidence threshold for the calling object
                              detector model, typically [0.25, 0.85] for a
                              YOLOv5-based model  (default=`0.32')
  -r, --rescale=FLOAT       extend and rescale canvas of the input frames to
                              ensure the expected size of ants E (0, 1). NOTE:
                              causes a computational overhead without affecting
                              original coordinates  (default=`1')
  -g, --cuda                computational device for the object detector (CUDA
                              GPU or CPU}  (default=off)

 Group: input
  Input data
  -i, --img=filename        path to the input image
  -v, --video=filename      path to the input video
  -s, --frame_start=INT     start frame index  (default=`0')
  -n, --frame_num=INT       the number of frames  (default=`-1')
```

Execution example:
```sh
build/bin$ ./atracker -m data/models/AntED_yolo5_traced_992.pt -v data/video/NontaggedAnts/6.mp4 -o runs -n 5
```
Executes the specified _YOLO5 ant detector_ model on _5 first frames_ of the __6.mp4__ input file, tracking those ants (recovering their ids between frames) and outputs results to the __runs__ directory, automatically adding a _suffix_ to the resulting files. The suffix includes execution parameters and the git version hash of the sources.
```sh
build/bin$ du -sh runs/*
208K	runs/6_i0-5_c0.32_f1f86fb+
84K	runs/6_i0-5_c0.32_f1f86fb+_0.jpg
120K	runs/6_i0-5_c0.32_f1f86fb+_demo.mp4

build/bin$ git rev-parse --short HEAD
f1f86fb

$ ll -sh
total 6.4M
4.0K drwxrwxr-x 4 lav lav 4.0K Jul 24 15:09 ./
4.0K drwxrwxr-x 6 lav lav 4.0K Jul 24 14:50 ../
6.4M -rwxrwxr-x 1 lav lav 6.4M Jul 24 15:09 atracker*
   0 lrwxrwxrwx 1 lav lav   58 Jul 24 14:50 data -> ../../data/
4.0K drwxrwxr-x 2 lav lav 4.0K Jul 24 15:09 lib/
4.0K drwxrwxr-x 3 lav lav 4.0K Jul 24 15:15 runs/
```
