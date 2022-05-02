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
int resolution = 992;  // frame size for model 992
int reduseres = 290;   // (good value 248)

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

class ClsObjR
{
  public:
  size_t cls_id;
  size_t obj_id;
  float r;
};

class ALObject // AntLab Object
{
public:
  uint16_t id;
  std::string obj_type;
  std::vector<cv::Point2f> claster_points;
  cv::Point2f claster_center;
  cv::Point2f model_center;
  bool det_mc = false;

  std::vector<cv::Point2f> track_points;
  cv::Point2f rectangle;
  cv::Mat img;

  std::vector<cv::Mat> samples;
  std::vector<cv::Point2f> coords;
  std::vector<IMGsamples> moution_samples;
  bool det_pos = false;

  ALObject(uint16_t id, std::string obj_type, std::vector<cv::Point2f> claster_points, cv::Mat img)
  {
    this->obj_type = obj_type;
    this->id = id;
    this->claster_points = claster_points;
    this->img = img;
    center_determine(true);
    //model_center = claster_center;
  }

  void center_determine(bool samplescreation)
  {

    std::vector<cv::Point2f> cp;
    std::vector<float> l;

    for(int i=0; i < claster_points.size(); i++)
    {
      for(int j=i; j< claster_points.size(); j++)
      {
        cv::Point2f p;
        float r;

        p.x = (claster_points.at(i).x + claster_points.at(j).x)/2;
        p.y = (claster_points.at(i).y + claster_points.at(j).y)/2;

        cp.push_back(p);

        r = sqrt(pow((claster_points.at(i).x - claster_points.at(j).x),2) + pow((claster_points.at(i).y - claster_points.at(j).y),2));

        l.push_back(r);
      }
    }

    cv::Point2f sumcp;
    float suml = 0;

    sumcp.x = 0;
    sumcp.y = 0;

    for(int i =0; i< cp.size(); i++)
    {
      sumcp.x += cp.at(i).x*l.at(i);
      sumcp.y += cp.at(i).y*l.at(i);
      suml += l.at(i);
    }

    claster_center.x = sumcp.x/suml;
    claster_center.y = sumcp.y/suml;

    det_pos = true;


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

    if (track_points.size() > 1)
    {
      proposed.x = claster_center.x + 0.5*(claster_center.x - track_points.at(track_points.size() - 2).x);
      proposed.y = claster_center.y + 0.5*(claster_center.y - track_points.at(track_points.size() - 2).y);
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
  uint8_t  type;// Object type
  uint16_t  id; // Object id
  uint16_t  x;  // Center x of the bounding box
  uint16_t  y;  // Center y of the bounding box
  uint16_t  w;  // Width of the bounding box
  uint16_t  h;  // Height of the bounding box
};

void testtorch();
int testmodule(std::string strpath);

std::vector<cv::Point2f> detectorT(torch::jit::script::Module module, cv::Mat imageBGR, torch::DeviceType device_type);
std::vector<cv::Mat> LoadVideo(const std::string &paths, uint16_t startframe, uint16_t getframes);
std::vector<OBJdetect> detectorV4(std::string pathmodel, cv::Mat frame, torch::DeviceType device_type); // latest version with CUDA support
cv::Point2f claster_center(std::vector<cv::Point2f> claster_points);
cv::Mat DetectorMotionV2(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame, bool usedetector);
void DetectorMotionV2b(cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame);
cv::Mat DetectorMotionV3(cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame);
void fixIDs(const std::vector<std::vector<Obj>>&objs, std::vector<std::pair<uint,idFix>>&fixedIds, std::vector<cv::Mat> &d_images, uint framesize=0);
void OBJdetectsToObjs(std::vector<OBJdetect> objdetects,std::vector<Obj> &objs);
cv::Mat DetectorMotionV2_1(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, size_t id_frame, bool usedetector);