#pragma once

#include <string>
#include <vector>
#include <utility> 
#include <limits>
#include <cmath>
#include <numeric>

#include <torch/torch.h>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

// #include "utils.h"

using std::string;
using std::vector;
using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Size2f;

constexpr float  dftConf = 0.32;  // Default confidence of the detecting objects. For YOLOv5:  0.25, 0.32 .. 0.5; 0.5
extern const uint16_t model_resolution; // frame resizing for model (992)
extern uint16_t objhsz; // area half size for a moving object
// Frame resolution for motion detection using non-adaptive thresholding:  (good value 248), 400
// Ant size should be >= 8 px (=> objhsz >= 4)
constexpr uint16_t antLenMin = 8;  // Minimal length of an ant


enum class ObjClass: uint8_t;

// Original Serhii's classes
// enum class ObjClass: uint8_t {
// 	TROPH_ANT,  // Ant-trophallaxis
// 	ANT,
// 	ANT_HEAD,
// 	TROPH_LARVA,
// 	LARVA,
// 	FOOD_NOISE,
// 	PUPA,
// 	BARCODE,
// 	UNCATECORIZED
// };

const char* objClassTitle(ObjClass objClass) noexcept;

class ClsObjR;

class ALObject; // AntLab Object

struct OBJdetect;

vector<Point2f> detectorT(torch::jit::script::Module module, Mat imageBGR, torch::DeviceType device_type);
vector<OBJdetect> detectorV4(const string& pathmodel, Mat frame, torch::DeviceType device_type, float confidence=dftConf, const string& outfile="");  // 0.5; latest version with CUDA support
// Mat trackingMotV2(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);
// void trackingMotV2b(Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId);

struct IdFix {
	uint8_t  type;
	uint16_t  idOld;
	uint16_t  id;
};
struct Obj;

// TODO: reimplement fixIds() to probalistically fix IDs without using an object detector.

//! @brief Fix object IDs for a set of frames
//! @pre object detection used in fixIds is expected to yield exactly the same number of objects on each frame as given by the user in objs
//!
//! @param[in, out] objs  - input and resulting objects fo all input frames
//! @param[out] fixedIds  - fixed ids for each object
//! @param d_images  - processing frames
//! @param confidence  - target confidence
//! @param framesize  - target frame size, which should be equal to the ML model resolution
//! @param pathmodel  - path to the object detector (ML model)
//! @param device  - target device 
void fixIds(const vector<vector<Obj>>&objs, vector<std::pair<uint,IdFix>>&fixedIds, vector<Mat> &d_images, float confidence=dftConf, uint16_t framesize=model_resolution, const string& pathmodel="", torch::DeviceType device=torch::kCPU);

//! @brief Fix object IDs for a pair of frames
//! @pre object detection used in fixIds is expected to yield exactly the same number of objects on each frame as given by the user in objs
//!
//! @param[in, out] objs  - input and resulting objects for the specified pair of frames
//! @param[out] fixedIds  - fixed ids for each object
//! @param frame0  - first frame
//! @param frame  - second frame or the first frame when the initialization is performed
//! @param ifr  - frame index for the first frame starting from 0: frame0, frame0, 0 to perform the initialization
//! @param confidence  - target confidence
//! @param framesize  - target frame size, which should be equal to the ML model resolution
//! @param pathmodel  - path to the object detector (ML model)
//! @param device  - target device 
void fixIds(const vector<vector<Obj>>&objs, vector<std::pair<uint,IdFix>>&fixedIds, Mat frame0, Mat frame, size_t ifr, float confidence=dftConf, uint16_t framesize=model_resolution, const string& pathmodel="", torch::DeviceType device=torch::kCPU);

vector<std::pair<Point2f,uint16_t>> trackingMotV2_1_artemis(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);
Mat trackingMotV2_1(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf, const string& outfileBase="");

// ORB-related routines --------------------------------------------------------------------------------------
std::tuple<vector<Point2f>,vector<Point2f>,Mat> detectORB(Mat &im1, Mat &im2, float reskoef);
Mat trackingMotV2_2(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);
Mat trackingMotV2_3(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);

// Accessory utils -------------------------------------------------------------------------------------------
void OBJdetectsToObjs(vector<OBJdetect> objdetects, vector<Obj> &objs);

/*!
 * @brief Trace objects into CSV files
 * 
 * @param objects  objects to be traced
 * @param odir  output directory
 */
// @param fsize  frame size
void traceObjects(const vector<ALObject> &objects, const string& odir);  // , const cv::Size& frame

vector<Mat> LoadVideo(const string &paths, uint16_t startframe, uint16_t getframes);
