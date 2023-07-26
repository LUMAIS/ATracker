#pragma once

#include <string>
#include <vector>
#include <utility> 
#include <iostream>
#include <fstream>
#include <exception>
#include <limits>
#include <cassert>
#include <chrono>
#include <cmath>
#include <numeric>
#include <memory>
#include <filesystem>

#include <torch/torch.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>

using std::string;
using std::vector;
using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Size2f;
using cv::Scalar;

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::to_string;
namespace fs = std::filesystem;

extern const uint16_t model_resolution; // frame resizing for model (992)

extern uint16_t frame_resolution;  // frame frame_resolution
extern int resolution;  // 1200;  //actual frame resolution

extern uint16_t extr;  // sidebar size
extern uint16_t objhsz; // area half size for a moving object
extern float reduseres;   // (good value 248)


// Declarations -------------------------------------------------------------------- 

// string objClassTitle[] = {"ant": 0,
//                         "ant-head": 1,
//                         "trophallaxis-ant": 2,
//                         "larva": 3,
//                         "trophallaxis-larva": 4,
//                         "food-noise": 5,  // fn
//                         "pupa": 6,
//                         "barcode": 7} #,"uncategorized": 8}
enum class ObjClass: uint8_t {
	ANT,
	ANT_HEAD,
	TROPH_ANT,  // Ant-ant trophallaxis
	LARVA,
	TROPH_LARVA,
	FOOD_NOISE,
	PUPA,
	BARCODE,
	UNCATECORIZED

	//operator uint8_t(enum class ObjClass oc) noexcept { return static_cast<uint8_t>(oc); }
};

class ClsObjR
{
	public:
	size_t cls_id;
	size_t obj_id;
	float r;
};

struct Trace {
	size_t  frame;  /// Frame id
	Point  center;  /// Object center
	cv::Size  size;  /// Object size
	float  prob;  /// Object identification probability

	Trace(size_t frame, const Point& center, const cv::Size& size, float prob) noexcept
		:frame(frame), center(center), size(size), prob(prob)  {}
	Trace(size_t frame, float x, float y, float w, float h, float prob) noexcept
		:frame(frame), center(Point(roundf(x), roundf(y))), size(cv::Size(roundf(w), roundf(h))), prob(prob)  {}
};

class ALObject // AntLab Object
{
public:
	static constexpr uint16_t  trackptsMax = 32;  // Max number of track points (historical cluster centers)

	uint16_t id;
	ObjClass type;
	Size2f size;  /// w, h
	Point2f model_center;
	Point2f cluster_center;
	vector<Point2f> cluster_points;
	//size_t frameId;  /// Latest frame index
	Mat img;
	float  prob;  /// Object identification probability on the last frame
	bool det_mc = false;  // Cache attribute, whether the model center is identified on the current frame  <=> prob > 0
	bool det_pos = false;  // Cluster center by motion has not been detected; set by center_determine()

	vector<Mat> samples;
	vector<Point2f> coords;
	//vector<IMGsamples> moution_samples;

	vector<Point2f> track_points;  // For the latest frames contains model_center if available, otherwise cluster_center
	// TODO: maintain traces only for the last few frames, performing incremental tracing
	vector<Trace>  traces;  // Traces of the object, where it was tracked

	vector<uint> ORB_ids;

	ALObject(uint16_t id, ObjClass type, const Size2f& size, const Point2f& mcenter
		, vector<Point2f> cluster_points, const Mat& img, float prob=0)
	: id(id), type(type), size(size), model_center(mcenter), cluster_center(mcenter)
		, cluster_points(cluster_points), img(img), prob(prob), det_mc(prob >= 0), det_pos(false)
	{
		if(!this->cluster_points.empty())
			center_determine(-1, true);
		// if (samplescreation)
		// 	samples_creation();
	}

	ALObject(uint16_t id, ObjClass type, vector<Point2f> cluster_points, const Mat& img)
	: ALObject(id, type, Size2f(0, 0), Point2f(-1, -1), cluster_points, img)  {}

	// Determine cluster center by cluster points
	void center_determine(size_t frame_id, bool samplescreation);

	void push_track_point(Point2f track_point);

	Point2f proposed_center();
};


struct OBJdetect
{
	Point2f center;  // x, y
	Size2f size;  // w, h
	ObjClass type;
	float prob;  // probability
};

// enum class Det
// {
// 	tl_x = 0,
// 	tl_y = 1,
// 	br_x = 2,
// 	br_y = 3,
// 	confidence = 4,
// 	class_idx = 5
// };

struct Detection
{
	cv::Rect bbox;
	float score;
	int class_idx;
};

struct intpoint
{
	int ipoint;
	Point2f mpoint;
};

struct Obj {
	uint8_t  type;// Object type
	uint16_t  id; // Object id
	uint16_t  x;  // Center x of the bounding box
	uint16_t  y;  // Center y of the bounding box
	uint16_t  w;  // Width of the bounding box
	uint16_t  h;  // Height of the bounding box
};

// General operations --------------------------------------------------------------
inline uint16_t max_u16(float a, float b=0) noexcept  { return std::max<int16_t>(roundf(a), roundf(b)); }
inline uint16_t min_u16(float a, float b) noexcept  { return std::min<int16_t>(roundf(a), roundf(b)); }

float distance(const Point2f& a, const Point2f& b) noexcept;

//! Harmonic mean
constexpr float hmean(float a, float b) noexcept  { return 2.f * a*b / (a+b); }
//! Geometric mean >= Harmonic mean
inline float gmean(float a, float b) noexcept  { return sqrtf(a*b); }

// Torch-related operations --------------------------------------------------------
void testtorch();

int testmodule(const string& strpath);


// IO-related operations -----------------------------------------------------------

// Get current datetime in the format YYYY-MM-DD_HH-mm-ss
string dateTime();

Mat frame_resizing(Mat frame, uint16_t framesize=model_resolution);

void drawrec(Mat &image, Point2f p1, int size, bool detect, float koef);

vector<Point2f> draw_map_prob(vector<int> npsamples, vector<Point2f> mpoints);

//! Evaluate resolution ratio for the proper object detection based on (ML) model resolution and expected object size
inline float evalReduseres(uint16_t objhsz) noexcept { return roundf(model_resolution * 12.f / objhsz / 32.f) * 32; }  // 480; 224;  12 is a reference length on an ant for motion processing


// Accessory operations ------------------------------------------------------------
Mat color_correction(Mat imag);


// Object-related operations -------------------------------------------------------
const char* objClassTitle(ObjClass objClass) noexcept;

Scalar objClassColor(ObjClass objClass) noexcept;

Scalar objColor(uint32_t id, uint8_t clrLow=32, uint8_t clrHigh=223) noexcept;

bool cmpIPt(intpoint a, intpoint b) noexcept;

Point2f cluster_center(vector<Point2f> cluster_points);

size_t samples_compV2(Mat sample1, Mat sample2);

bool compare_clsobj(const ClsObjR& a, const ClsObjR& b) noexcept;

vector<Point2f> map_prob(vector<int> npsamples, vector<Point2f> mpoints);

void OBJdetectsToObjs(vector<OBJdetect> objdetects, vector<Obj> &objs);

void ALObjectsToObjs(vector<ALObject> objects, vector<Obj> &objs);
