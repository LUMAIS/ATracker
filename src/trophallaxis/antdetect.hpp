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
#include <utility> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int extr = 205;        // sidebar size
int half_imgsize = 80; // area half size for a moving object
int resolution = 992;  // frame size for model
int reduseres = 192;   // (good value 248)

std::string class_name[9] = {"ta", "a", "ah", "tl", "l", "fn", "u", "p", "b"};

class IMGsamples
{
public:
  uint16_t num;
  std::vector<cv::Mat> samples;
  std::vector<cv::Point2f> coords; // coordinates relative to the center of the cluster

  IMGsamples(std::vector<cv::Mat> samples, std::vector<cv::Point2f> coords)
  {
    this->samples = samples;
    this->coords = coords;
    num = samples.size();
  }
};

class ALObject // AntLab Object
{
public:
  uint16_t id;
  std::string obj_type;
  std::vector<cv::Point2f> claster_points;
  cv::Point2f claster_center;
  cv::Point2f model_center;
  std::vector<cv::Point2f> track_points;
  cv::Point2f rectangle;
  cv::Mat img;

  std::vector<cv::Mat> samples;
  std::vector<cv::Point2f> coords;

  std::vector<IMGsamples> moution_samples;

  ALObject(uint16_t id, std::string obj_type, std::vector<cv::Point2f> claster_points, cv::Mat img)
  {
    this->obj_type = obj_type;
    this->id = id;
    this->claster_points = claster_points;
    this->img = img;
    center_determine(true);
    model_center = claster_center;
  }

  void center_determine(bool samplescreation)
  {

    if (claster_points.size() > 0)
    {
      int powx = 0;
      int powy = 0;

      for (int i = 0; i < claster_points.size(); i++)
      {
        powx = powx + pow(claster_points[i].x, 2);
        powy = powy + pow(claster_points[i].y, 2);
      }

      claster_center.x = sqrt(powx / claster_points.size());
      claster_center.y = sqrt(powy / claster_points.size());
    }

    if (samplescreation == true)
    {
      samples_creation();
    }
  }

  void samples_creation()
  {
    cv::Mat sample;
    cv::Point2f coord;

    samples.clear();
    coords.clear();

    for (int i = 0; i < claster_points.size(); i++)
    {
      coord.y = claster_points.at(i).y - claster_center.y + half_imgsize;
      coord.x = claster_points.at(i).x - claster_center.x + half_imgsize;

      if (coord.y > 0 && (coord.y + resolution / reduseres) < 2 * half_imgsize && coord.x > 0 && (coord.x + resolution / reduseres) < 2 * half_imgsize)
      {
        sample = img(cv::Range(coord.y, coord.y + resolution / reduseres), cv::Range(coord.x, coord.x + resolution / reduseres));
        samples.push_back(sample);
        coords.push_back(coord);
      }
    }

    IMGsamples buf(samples, coords);
    moution_samples.push_back(buf);
  }

  void push_track_point(cv::Point2f track_point)
  {
    track_points.push_back(track_point);

    if (track_points.size() > 33)
      track_points.erase(track_points.begin());
  }

  cv::Point2f proposed_center()
  {
    cv::Point2f proposed;

    if (track_points.size() > 0)
    {
      proposed.x = claster_center.x + (claster_center.x - track_points.at(track_points.size() - 1).x) / 10;
      proposed.y = claster_center.y + (claster_center.y - track_points.at(track_points.size() - 1).y) / 10;
    }
    else
      proposed = claster_center;

    return proposed;
  }
};

class OBJdetect
{
public:
  cv::Point2f detect;
  cv::Point2f rectangle;
  std::string type;
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
  cv::Point2f mpoint;
};

struct idFix {
  uint8_t  type;
  uint16_t  idOld;
  uint16_t  id;
}
;
struct Obj {
  uint8_t  type;  // Object type
  uint16_t  id;  // Object id
  uint16_t  x;  // Center x of the bounding box
  uint16_t  y;  // Center y of the bounding box
  uint16_t  w;  // Width of the bounding box
  uint16_t  h;  // Height of the bounding box
};

void testtorch();
int testmodule(std::string strpath);

std::vector<cv::Point2f> detectorT(torch::jit::script::Module module, cv::Mat imageBGR, torch::DeviceType device_type);
std::vector<cv::Mat> LoadVideo(const std::string &paths, uint8_t startframe, uint8_t getframes);
std::vector<OBJdetect> detectorV4(std::string pathmodel, cv::Mat frame, torch::DeviceType device_type); // latest version with CUDA support

cv::Point2f claster_center(std::vector<cv::Point2f> claster_points);
cv::Mat DetectorMotionV2(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame, bool usedetector);
void DetectorMotionV2b(cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame);

cv::Mat DetectorMotionV3(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame, bool usedetector);

void fixIDs(const std::vector<std::vector<Obj>>&objs, std::vector<std::pair<uint,idFix>>&fixedIds, std::vector<cv::Mat> d_images);

void OBJdetectsToObjs(std::vector<OBJdetect> objdetects,std::vector<Obj> &objs);