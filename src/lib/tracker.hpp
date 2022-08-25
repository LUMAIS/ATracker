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

constexpr float  dftConf = 0.32;  // Default confidence of the detecting objects. For YOLOv5:  0.25, 0.32 .. 0.5; 0.5

extern int resolution;  // 1200;  // frame size for model 992
// extern const int resolution;  // frame size for model 992

extern uint16_t extr;        // sidebar size
extern const uint16_t half_imgsize; // area half size for a moving object
extern const uint16_t model_resolution;  // frame resizing for model (992)
extern uint16_t frame_resolution;//frame frame_resolution
extern float reduseres;   // (good value 248)
extern uint16_t color_threshold; // 65-70

extern string class_name[];

uint16_t max_u16(float a, float b=0) noexcept;
uint16_t min_u16(float a, float b) noexcept;

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
  uint32_t  frame;  /// Frame id
  Point center;  /// Object center
  cv::Size  size;  /// Object size

  Trace(uint32_t frame, const Point& center, const cv::Size& size) noexcept
    :frame(frame), center(center), size(size)  {}
  Trace(size_t frame, float x, float y, float w, float h) noexcept
    :frame(frame), center(Point(roundf(x), roundf(y))), size(cv::Size(roundf(w), roundf(h)))  {}
};

class ALObject // AntLab Object
{
public:
  uint16_t id;
  string obj_type;
  vector<Point2f> cluster_points;
  Point2f cluster_center;
  Point2f model_center;
  bool det_mc = false;

  vector<Point2f> track_points;
  Point2f rectangle;  // w, h
  Mat img;

  vector<Mat> samples;
  vector<Point2f> coords;
  vector<IMGsamples> moution_samples;

  vector<Trace>  traces;  // Traces of the object, where it was tracked


  bool det_pos = false;

  vector<uint> ORB_ids;

  ALObject(uint16_t id, string obj_type, vector<Point2f> cluster_points, Mat img)
  {
    this->obj_type = obj_type;
    this->id = id;
    this->img = img;
    //this->model_center = cluster_center;
    if(!cluster_points.empty()) {
      this->cluster_points = cluster_points;
      center_determine(-1, true);
    }
  }

  void center_determine(uint32_t frame_id, bool samplescreation)
  {

    vector<Point2f> cp;
    vector<float> l;

    for(int i=0; i < cluster_points.size(); i++)
    {
      for(int j=i; j< cluster_points.size(); j++)
      {
        Point2f p;
        float r;

        p.x = (cluster_points.at(i).x + cluster_points.at(j).x)/2;
        p.y = (cluster_points.at(i).y + cluster_points.at(j).y)/2;

        cp.push_back(p);

        r = sqrt(pow((cluster_points.at(i).x - cluster_points.at(j).x),2) + pow((cluster_points.at(i).y - cluster_points.at(j).y),2));

        l.push_back(r);
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

    if(frame_id != -1) {
      if(!traces.empty() && traces.back().frame == frame_id) {
        traces.back().center.x = roundf(cluster_center.x);
        traces.back().center.y = roundf(cluster_center.y);
      } else {
        assert((traces.empty() || traces.back().frame < frame_id) && "Unexpected frame number in the traces");
        traces.push_back(Trace{frame_id, cluster_center.x, cluster_center.y, rectangle.x, rectangle.y});
      }
    }

    det_pos = true;

    if (samplescreation)
      samples_creation();
  }

  void samples_creation()
  {
    Mat sample;
    Point2f coord;

    samples.clear();
    coords.clear();

    for (int i = 0; i < cluster_points.size(); i++)
    {
      coord.y = cluster_points.at(i).y - cluster_center.y + half_imgsize;
      coord.x = cluster_points.at(i).x - cluster_center.x + half_imgsize;

      if (coord.y > 0 && coord.y < img.rows && (coord.y + frame_resolution / reduseres) < 2 * half_imgsize
      && coord.x > 0 && coord.y < img.cols && (coord.x + frame_resolution / reduseres) < 2 * half_imgsize)
      {
        sample = img(cv::Range(coord.y, min_u16(img.rows, coord.y + frame_resolution / reduseres))
          , cv::Range(coord.x, min_u16(img.cols, coord.x + frame_resolution / reduseres)));
        samples.push_back(sample);
        coords.push_back(coord);
      }
    }

    IMGsamples buf(samples, coords);
    moution_samples.push_back(buf);
  }

  void push_track_point(Point2f track_point)
  {
    track_points.push_back(track_point);

    if (track_points.size() > 33)
      track_points.erase(track_points.begin());
  }

  Point2f proposed_center()
  {
    Point2f proposed;

    if (track_points.size() > 1)
    {
      proposed.x = cluster_center.x + 0.5*(cluster_center.x - track_points.at(track_points.size() - 2).x);
      proposed.y = cluster_center.y + 0.5*(cluster_center.y - track_points.at(track_points.size() - 2).y);
    }
    else
      proposed = cluster_center;

    return proposed;
  }
};

class OBJdetect
{
public:
  Point2f detect;  // x, y
  Point2f rectangle;  // w, h
  string type;
};

enum Det
{
  tl_x = 0,
  tl_y = 1,
  br_x = 2,
  br_y = 3,
  score = 4,
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

struct idFix {
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
vector<OBJdetect> detectorV4(string pathmodel, Mat frame, torch::DeviceType device_type, const float confidence=dftConf);  // 0.5; latest version with CUDA support
Point2f cluster_center(vector<Point2f> cluster_points);
Mat DetectorMotionV2(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, bool usedetector);
void DetectorMotionV2b(Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame);
Mat DetectorMotionV3(Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame);
void fixIDs(const vector<vector<Obj>>&objs, vector<std::pair<uint,idFix>>&fixedIds, vector<Mat> &d_images, uint framesize=0);
void OBJdetectsToObjs(vector<OBJdetect> objdetects,vector<Obj> &objs);
Mat DetectorMotionV2_1(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, /*vector<cv::Scalar> class_name_color,*/ bool usedetector);

vector<std::pair<Point2f,uint16_t>> DetectorMotionV2_1_artemis(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, bool usedetector);

std::tuple<vector<Point2f>,vector<Point2f>,Mat> detectORB(Mat &im1, Mat &im2, float reskoef);
Mat DetectorMotionV2_2(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, /*vector<cv::Scalar> class_name_color,*/ bool usedetector);
Mat DetectorMotionV2_3(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, /*vector<cv::Scalar> class_name_color,*/ bool usedetector);

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
