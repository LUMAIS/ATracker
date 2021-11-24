#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>

#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

enum Det {
    tl_x = 0,
    tl_y = 1,
    br_x = 2,
    br_y = 3,
    score = 4,
    class_idx = 5
};

struct Detection {
    cv::Rect bbox;
    float score;
    int class_idx;
};

void testtorch();
int testmodule (std::string strpath);

std::vector<cv::Point2f> detectorT (torch::jit::script::Module module, cv::Mat imageBGR, torch::DeviceType device_type);
std::vector<cv::Mat> LoadVideo (const std::string & paths);
std::vector<cv::Point2f> detectorT2 (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type);
std::vector<cv::Point2f> detectorT3 (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type);//latest version with CUDA support 

/*
template <class T>
void tuplesize(T value)
{
     int a[std::tuple_size_v<T>]; // can be used at compile time
 
    std::cout << std::tuple_size<T>{} << ' ' // or at run time
              << sizeof a << ' ' 
              << sizeof value << '\n';
}*/