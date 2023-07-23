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

#include <torch/torch.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>

using std::string;
using std::vector;
using cv::Mat;
using cv::Point;
using cv::Point2f;
using cv::Size2f;

constexpr float  dftConf = 0.32;  // Default confidence of the detecting objects. For YOLOv5:  0.25, 0.32 .. 0.5; 0.5

extern const uint16_t model_resolution; // frame resizing for model (992)
extern uint16_t frame_resolution;  // frame frame_resolution
extern int resolution;  // 1200;  //actual frame resolution

extern uint16_t extr;  // sidebar size
extern uint16_t objhsz; // area half size for a moving object
extern float reduseres;   // (good value 248)

uint16_t max_u16(float a, float b=0) noexcept;
uint16_t min_u16(float a, float b) noexcept;

float distance(const Point2f& a, const Point2f& b) noexcept;

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

class IMGsamples
{
public:
	uint16_t num;
	vector<Mat> samples;
	vector<Point2f> coords; // coordinates relative to the center of the cluster

	IMGsamples(vector<Mat> samples, vector<Point2f> coords)
	{
		this->samples = samples;
		this->coords = coords;
		num = samples.size();
	}
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
	void center_determine(size_t frame_id, bool samplescreation)
	{
		vector<Point2f> cp;  // pairwise cluster points
		vector<float> l;  // distances

		for(int i=0; i < cluster_points.size(); i++)
		{
			const auto& pi = cluster_points[i];
			for(int j=i; j< cluster_points.size(); j++)
			{
				const auto& pj = cluster_points[j];
				cp.push_back(Point2f((pi.x + pj.x) / 2, (pi.y + pj.y) / 2));
				l.push_back(distance(pi, pj));
			}
		}

		Point2f sumcp;
		float suml = 0;

		sumcp.x = 0;
		sumcp.y = 0;

		for(int i =0; i< cp.size(); i++)
		{
			sumcp.x += cp.at(i).x*l.at(i);
			sumcp.y += cp.at(i).y*l.at(i);
			suml += l.at(i);
		}

		cluster_center.x = sumcp.x/suml;
		cluster_center.y = sumcp.y/suml;

		if(frame_id != static_cast<decltype(frame_id)>(-1)) {
			if(!traces.empty() && traces.back().frame == frame_id) {
				traces.back().center.x = roundf(cluster_center.x);
				traces.back().center.y = roundf(cluster_center.y);
			} else {
				assert((traces.empty() || traces.back().frame < frame_id) && "Unexpected frame number in the traces");
				traces.push_back(Trace{frame_id, cluster_center.x, cluster_center.y, size.width, size.height, prob});
			}
		}

		det_pos = true;
	}

	// void samples_creation()
	// {
	// 	Mat sample;
	// 	Point2f coord;

	// 	samples.clear();
	// 	coords.clear();

	// 	for (int i = 0; i < cluster_points.size(); i++)
	// 	{
	// 		// coord.x = cluster_points.at(i).x - cluster_center.x + objhsz;
	// 		// coord.y = cluster_points.at(i).y - cluster_center.y + objhsz;
	// 		coord.x = min_u16(cluster_points.at(i).x - roundf(frame_resolution / reduseres / 2.f));
	// 		coord.y = min_u16(cluster_points.at(i).y - roundf(frame_resolution / reduseres / 2.f));

	// 		sample = img(cv::Range(coord.y, min_u16(img.rows, coord.y + frame_resolution / reduseres))
	// 			, cv::Range(coord.x, min_u16(img.cols, coord.x + frame_resolution / reduseres)));
	// 		samples.push_back(sample);
	// 		coords.push_back(coord);
	// 	}

	// 	IMGsamples buf(samples, coords);
	// 	moution_samples.push_back(buf);
	// }

	void push_track_point(Point2f track_point)
	{
		track_points.push_back(track_point);

		if (track_points.size() > trackptsMax)
			track_points.erase(track_points.begin());
	}

	//! Expected center of the object based on it's movement in the opposite direction to the previous position
	Point2f proposed_center()
	{
		Point2f proposed(cluster_center);

		if (track_points.size() > 1)
		{
			proposed.x += 0.5f * (cluster_center.x - track_points.at(track_points.size() - 2).x);
			proposed.y += 0.5f * (cluster_center.y - track_points.at(track_points.size() - 2).y);
		}

		return proposed;
	}
};

struct OBJdetect
{
	Point2f center;  // x, y
	Size2f size;  // w, h
	ObjClass type;
	float prob;  // probability
};

enum class Det
{
	tl_x = 0,
	tl_y = 1,
	br_x = 2,
	br_y = 3,
	confidence = 4,
	class_idx = 5
};

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

struct IdFix {
	uint8_t  type;
	uint16_t  idOld;
	uint16_t  id;
}
;
struct Obj {
	uint8_t  type;// Object type
	uint16_t  id; // Object id
	uint16_t  x;  // Center x of the bounding box
	uint16_t  y;  // Center y of the bounding box
	uint16_t  w;  // Width of the bounding box
	uint16_t  h;  // Height of the bounding box
};

void testtorch();
int testmodule(string strpath);

vector<Point2f> detectorT(torch::jit::script::Module module, Mat imageBGR, torch::DeviceType device_type);
vector<Mat> LoadVideo(const string &paths, uint16_t startframe, uint16_t getframes);
vector<OBJdetect> detectorV4(const string& pathmodel, Mat frame, torch::DeviceType device_type, float confidence=dftConf, const string& outfile="");  // 0.5; latest version with CUDA support
Point2f cluster_center(vector<Point2f> cluster_points);
// Mat trackingMotV2(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);
// void trackingMotV2b(Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId);
void fixIDs(const vector<vector<Obj>>&objs, vector<std::pair<uint,IdFix>>&fixedIds, vector<Mat> &d_images, float confidence=dftConf, uint16_t framesize=model_resolution, const string& pathmodel="", torch::DeviceType device=torch::kCPU);
void OBJdetectsToObjs(vector<OBJdetect> objdetects,vector<Obj> &objs);
Mat trackingMotV2_1(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf, const string& outfileBase="");

vector<std::pair<Point2f,uint16_t>> trackingMotV2_1_artemis(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);

std::tuple<vector<Point2f>,vector<Point2f>,Mat> detectORB(Mat &im1, Mat &im2, float reskoef);
Mat trackingMotV2_2(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);
Mat trackingMotV2_3(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence=dftConf);

/// Get current datetime in the format YYYY-MM-DD_HH-mm-ss
string dateTime();

/**
 * @brief Trace objects into CSV files
 * 
 * @param objects  objects to be traced
 * @param odir  output directory
 */
// @param fsize  frame size
void traceObjects(const vector<ALObject> &objects, const string& odir);  // , const cv::Size& frame
