#include "antdetect.hpp"

#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <tuple>

#include <chrono>
#include <sys/time.h>
#include <ctime>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

void testtorch()
{
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;
  std::cout << "testtorch() - OK!!" << std::endl;
}

int testmodule(std::string strpath)
{
  torch::jit::script::Module module;
  try
  {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(strpath);
  }
  catch (const c10::Error &e)
  {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  return 0;
}

void drawrec(cv::Mat &image, cv::Point2f p1, int size, bool detect, float koef)
{
  cv::Point2f p2;
  cv::Point2f p3;
  cv::Point2f p4;

  p2.x = p1.x + size;
  p2.y = p1.y;

  p3.x = p2.x;
  p3.y = p2.y + size;

  p4.x = p3.x - size;
  p4.y = p3.y;

  p1.x = p1.x / koef;
  p1.y = p1.y / koef;

  p2.x = p2.x / koef;
  p2.y = p2.y / koef;

  p3.x = p3.x / koef;
  p3.y = p3.y / koef;

  p4.x = p4.x / koef;
  p4.y = p4.y / koef;

  u_char R, G, B;

  if (detect == false)
  {
    R = 40;
    G = 160;
    B = 0;
  }
  else
  {
    R = 255;
    G = 0;
    B = 0;
  }

  cv::line(image, p1, p2, cv::Scalar(B, G, R), 2, 8, 0);
  cv::line(image, p2, p3, cv::Scalar(B, G, R), 2, 8, 0);
  cv::line(image, p3, p4, cv::Scalar(B, G, R), 2, 8, 0);
  cv::line(image, p4, p1, cv::Scalar(B, G, R), 2, 8, 0);
}

std::vector<cv::Mat> LoadVideo(const std::string &paths, uint8_t startframe, uint8_t getframes)
{
  std::vector<cv::Mat> d_images;
  cv::VideoCapture cap(paths);

  if (!cap.isOpened())
  {
    std::cout << "[StubVideoGrabber]: Cannot open the video file" << std::endl;
  }
  else
  {
    cv::Mat framebuf;
    for (int frame_count = startframe; frame_count < cap.get(cv::CAP_PROP_FRAME_COUNT); frame_count++)
    {
      cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);

      if (getframes != 0 && frame_count > startframe + getframes)
        break;

      if (!cap.read(framebuf))
      {
        std::cout << "[LoadVideo]: Failed to extract the frame " << frame_count << std::endl;
      }
      else
      {
        cv::Mat frame;
        cv::cvtColor(framebuf, frame, cv::COLOR_RGB2GRAY);
        d_images.push_back(frame);
        std::cout << "[LoadVideo]: Success to extracted the frame " << frame_count << std::endl;
      }
    }
  }
  return d_images;
}

cv::Mat frame_resizing(cv::Mat frame)
{
  int rows = frame.rows;
  int cols = frame.cols;

  float rwsize;
  float clsize;

  if (rows > cols)
  {
    rwsize = 992 * rows * 1.0 / cols;
    clsize = 992;
  }
  else
  {
    rwsize = 992;
    clsize = 992 * cols * 1.0 / rows;
  }

  cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rect(0, 0, 992, 992);

  return frame(rect);
}

std::vector<OBJdetect> detectorV4(std::string pathmodel, cv::Mat frame, torch::DeviceType device_type)
{
  std::vector<OBJdetect> obj_detects;
  auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  torch::jit::script::Module module = torch::jit::load(pathmodel);
  std::cout << "Load module +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << std::endl;
  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  int resolution = 992;
  int pointsdelta = 5;
  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsCent;
  std::vector<cv::Point2f> detectsRect;
  std::vector<uint8_t> Objtype;

  cv::Scalar class_name_color[9] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 0), cv::Scalar(255, 255, 255), cv::Scalar(200, 0, 200), cv::Scalar(100, 0, 255)};
  cv::Mat imageBGR;

  imageBGR = frame_resizing(frame);
  // cv::resize(frame, imageBGR,cv::Size(992, 992),cv::InterpolationFlags::INTER_CUBIC);

  cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
  imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
  auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();
  input_tensor = input_tensor.to(device_type);
  //----------------------------------
  // module.to(device_type);

  if (device_type != torch::kCPU)
  {
    input_tensor = input_tensor.to(torch::kHalf);
  }
  //----------------------------------

  // std::cout<<"input_tensor.to(device_type) - OK"<<std::endl;
  std::vector<torch::jit::IValue> input;
  input.emplace_back(input_tensor);
  // std::cout<<"input.emplace_back(input_tensor) - OK"<<std::endl;

  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  auto outputs = module.forward(input).toTuple();
  // std::cout << "Processing +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << std::endl;
  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

  // std::cout<<"module.forward(input).toTuple() - OK"<<std::endl;
  torch::Tensor detections = outputs->elements()[0].toTensor();

  int item_attr_size = 13;
  int batch_size = detections.size(0);
  auto num_classes = detections.size(2); // - item_attr_size;

  auto conf_thres = 0.50;
  auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

  std::vector<std::vector<Detection>> output;
  output.reserve(batch_size);

  for (int batch_i = 0; batch_i < batch_size; batch_i++)
  {
    // apply constrains to get filtered detections for current image
    auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes});
    // if none detections remain then skip and start to process next image

    if (0 == det.size(0))
    {
      continue;
    }

    for (size_t i = 0; i < det.size(0); ++i)
    {
      float x = det[i][0].item().toFloat() * imageBGR.cols / resolution;
      float y = det[i][1].item().toFloat() * imageBGR.rows / resolution;

      float h = det[i][2].item().toFloat() * imageBGR.cols / resolution;
      float w = det[i][3].item().toFloat() * imageBGR.rows / resolution;

      float wheit = 0;
      Objtype.push_back(8);

      for (int j = 4; j < det.size(1); j++)
      {
        if (det[i][j].item().toFloat() > wheit)
        {
          wheit = det[i][j].item().toFloat();
          Objtype.at(i) = j - 4;
        }
      }

      detectsCent.push_back(cv::Point(x, y));
      detectsRect.push_back(cv::Point(h, w));
    }
  }

  for (size_t i = 0; i < detectsCent.size(); i++)
  {
    if (detectsCent.at(i).x > 0)
    {
      for (size_t j = 0; j < detectsCent.size(); j++)
      {
        if (detectsCent.at(j).x > 0 && i != j)
        {
          if (sqrt(pow(detectsCent.at(i).x - detectsCent.at(j).x, 2) + pow(detectsCent.at(i).y - detectsCent.at(j).y, 2)) < pointsdelta)
          {
            detectsCent.at(i).x = (detectsCent.at(i).x + detectsCent.at(j).x) * 1.0 / 2;
            detectsCent.at(i).y = (detectsCent.at(i).y + detectsCent.at(j).y) * 1.0 / 2;

            detectsRect.at(i).x = (detectsRect.at(i).x + detectsRect.at(j).x) * 1.0 / 2;
            detectsRect.at(i).y = (detectsRect.at(i).y + detectsRect.at(j).y) * 1.0 / 2;

            detectsCent.at(j).x = -1;
          }
        }
      }
    }
  }

  for (size_t i = 0; i < detectsCent.size(); i++)
  {

    cv::Point2f pt1;
    cv::Point2f pt2;
    cv::Point2f ptext;

    if (detectsCent.at(i).x >= 0)
    {

      OBJdetect obj_buf;

      obj_buf.detect = detectsCent.at(i);
      obj_buf.rectangle = detectsRect.at(i);
      obj_buf.type = class_name[Objtype.at(i)];
      obj_detects.push_back(obj_buf);

      detects.push_back(detectsCent.at(i));
      pt1.x = detectsCent.at(i).x - detectsRect.at(i).x / 2;
      pt1.y = detectsCent.at(i).y - detectsRect.at(i).y / 2;

      pt2.x = detectsCent.at(i).x + detectsRect.at(i).x / 2;
      pt2.y = detectsCent.at(i).y + detectsRect.at(i).y / 2;

      ptext.x = detectsCent.at(i).x - 5;
      ptext.y = detectsCent.at(i).y + 5;

      rectangle(imageBGR, pt1, pt2, class_name_color[Objtype.at(i)], 1);

      cv::putText(imageBGR,                  // target image
                  class_name[Objtype.at(i)], // text
                  ptext,                     // top-left position
                  1,
                  0.8,
                  class_name_color[Objtype.at(i)], // font color
                  1);
    }
  }

  millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
  return obj_detects;
}

cv::Point2f claster_center(std::vector<cv::Point2f> claster_points)
{
  cv::Point2f claster_center;

  int powx = 0;
  int powy = 0;

  for (int i = 0; i < claster_points.size(); i++)
  {
    powx = powx + pow(claster_points[i].x, 2);
    powy = powy + pow(claster_points[i].y, 2);
  }

  claster_center.x = sqrt(powx / claster_points.size());
  claster_center.y = sqrt(powy / claster_points.size());

  return claster_center;
}

size_t samples_compV2(cv::Mat sample1, cv::Mat sample2)
{
  size_t npf = 0;

  sample1.convertTo(sample1, CV_8UC1);
  sample2.convertTo(sample2, CV_8UC1);

  cv::Point2f pm;

  for (int y = 0; y < sample1.rows; y++)
  {
    for (int x = 0; x < sample1.cols; x++)
    {
      uchar color1 = sample1.at<uchar>(cv::Point(x, y));
      uchar color2 = sample2.at<uchar>(cv::Point(x, y));

      npf += abs((int)color2 - (int)color1);
    }
  }

  return npf;
}

cv::Mat draw_object(ALObject obj, ALObject obj2, cv::Scalar color)
{
  int wh = 800;
  float hp = 1.0 * (resolution / reduseres) * wh / (2 * half_imgsize);

  cv::Mat imag(wh, wh, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat imag2(wh, wh, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat imgres(wh, wh * 3, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat imgbuf;

  cv::Point2f bufp;
  cv::Point2f pt1;
  cv::Point2f pt2;

  cv::Mat imgsm;

  cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);

  imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

  cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);

  imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

  /*
    for (int i = 0; i < obj.samples.size(); i++)
    {
      imgsm = obj.samples.at(i);
      cv::resize(imgsm, imgsm, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

      //--test color---
      size_t rows = imgsm.rows;
      size_t cols = imgsm.cols;
      uint8_t *pixelPtr1 = (uint8_t *)imgsm.data;
      int cn = imgsm.channels();
      cv::Scalar_<uint8_t> bgrPixel1;

      for (size_t x = 0; x < rows; x++)
      {
        for (size_t y = 0; y < cols; y++)
        {
          bgrPixel1.val[0] = pixelPtr1[x * imgsm.cols * cn + y * cn + 0]; // B
          bgrPixel1.val[1] = pixelPtr1[x * imgsm.cols * cn + y * cn + 1]; // G
          bgrPixel1.val[2] = pixelPtr1[x * imgsm.cols * cn + y * cn + 2]; // R

          if (bgrPixel1.val[0] > 30)
          {
            if(bgrPixel1.val[0] > 100)
              pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)255;
            else
              pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)100;
          }
          else
            pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)0;

          if (bgrPixel1.val[1] > 30)
          {
            if(bgrPixel1.val[0] > 100)
              pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)255;
            else
              pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)100;
          }
          else
            pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)0;

          if (bgrPixel1.val[2] > 30)
          {
            if(bgrPixel1.val[0] > 100)
              pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)255;
            else
              pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)100;
          }
          else
            pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)0;
        }
      }
      //--test color---/

      bufp.x = 1.0 * obj.coords.at(i).x * wh / (2 * half_imgsize);
      bufp.y = 1.0 * obj.coords.at(i).y * wh / (2 * half_imgsize);
      imgsm.copyTo(imag(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));
    }

    for (int i = 0; i < obj.claster_points.size(); i++)
    {
      bufp.x = ((obj.claster_points.at(i).x - obj.claster_center.x) * wh / (2 * half_imgsize) + wh / 2);
      bufp.y = ((obj.claster_points.at(i).y - obj.claster_center.y) * wh / (2 * half_imgsize) + wh / 2);

      pt1.x = bufp.x;
      pt1.y = bufp.y;

      pt2.x = bufp.x + hp;
      pt2.y = bufp.y + hp;

      rectangle(imag, pt1, pt2, color, 1);
    }
  */
  cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);

  //---------------------------------------------

  cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);

  for (int i = 0; i < obj2.samples.size(); i++)
  {
    imgsm = obj2.samples.at(i);
    cv::resize(imgsm, imgsm, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

    /*/--test color---
    size_t rows = imgsm.rows;
    size_t cols = imgsm.cols;
    uint8_t *pixelPtr1 = (uint8_t *)imgsm.data;
    int cn = imgsm.channels();
    cv::Scalar_<uint8_t> bgrPixel1;

    for (size_t x = 0; x < rows; x++)
    {
      for (size_t y = 0; y < cols; y++)
      {
        bgrPixel1.val[0] = pixelPtr1[x * imgsm.cols * cn + y * cn + 0]; // B
        bgrPixel1.val[1] = pixelPtr1[x * imgsm.cols * cn + y * cn + 1]; // G
        bgrPixel1.val[2] = pixelPtr1[x * imgsm.cols * cn + y * cn + 2]; // R

        if (bgrPixel1.val[0] > 30)
        {
          if(bgrPixel1.val[0] > 100)
            pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)255;
          else
            pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)100;
        }
        else
          pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)0;

        if (bgrPixel1.val[1] > 30)
        {
          if(bgrPixel1.val[0] > 100)
            pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)255;
          else
            pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)100;
        }
        else
          pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)0;

        if (bgrPixel1.val[2] > 30)
        {
          if(bgrPixel1.val[0] > 100)
            pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)255;
          else
            pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)100;
        }
        else
          pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)0;
      }
    }
    //--test color---*/

    bufp.x = 1.0 * obj2.coords.at(i).x * wh / (2 * half_imgsize);
    bufp.y = 1.0 * obj2.coords.at(i).y * wh / (2 * half_imgsize);
    imgsm.copyTo(imag2(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));
  }

  for (int i = 0; i < obj2.claster_points.size(); i++)
  {
    bufp.x = ((obj2.claster_points.at(i).x - obj2.claster_center.x) * wh / (2 * half_imgsize) + wh / 2);
    bufp.y = ((obj2.claster_points.at(i).y - obj2.claster_center.y) * wh / (2 * half_imgsize) + wh / 2);

    pt1.x = bufp.x;
    pt1.y = bufp.y;

    pt2.x = bufp.x + hp;
    pt2.y = bufp.y + hp;

    rectangle(imag2, pt1, pt2, color, 1);
  }

  cv::cvtColor(imag2, imag2, cv::COLOR_BGR2RGB);

  //-------using matchTemplate--bad idea---------------------
  /*
    cv::Mat result;
    result.create(imag.rows, imag.cols, CV_32FC1);

    for (size_t s2 = 0; s2 < obj2.samples.size(); s2++)
    {
      matchTemplate(imag, obj2.samples.at(s2), result, 2);

      //imshow("result", result);

      size_t R = rand() % 255;
      size_t G = rand() % 255;
      size_t B = rand() % 255;

      bufp.x = 1.0 * obj2.coords.at(s2).x * wh / (2 * half_imgsize);
      bufp.y = 1.0 * obj2.coords.at(s2).y * wh / (2 * half_imgsize);
      // imgsm.copyTo(imag2(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

      pt1.x = bufp.x;
      pt1.y = bufp.y;

      pt2.x = bufp.x + hp;
      pt2.y = bufp.y + hp;

      rectangle(imag2, pt1, pt2, cv::Scalar(R, G, B), 1);
      imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
      imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
      imshow("imgres", imgres);

      cv::waitKey(0);
    }*/
  //------------------------------

  /*imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
  imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
  imshow("imgres", imgres);*/

  cv::waitKey(0);

  if (3 == 3)
    do
    {
      size_t ns2 = 0;
      size_t ns1 = 0;
      size_t minnp = 0;

      for (size_t s1 = 0; s1 < obj.samples.size(); s1++)
      {
        for (size_t s2 = 0; s2 < obj2.samples.size(); s2++)
        {
          size_t np = samples_compV2(obj.samples.at(s1), obj2.samples.at(s2));
          if ((minnp > np || minnp == 0) && np > 0)
          {
            minnp = np;
            ns2 = s2;
            ns1 = s1;
          }
        }
      }

      size_t R = rand() % 255;
      size_t G = rand() % 255;
      size_t B = rand() % 255;

      std::cout << "minnp - " << minnp << std::endl;

      cv::Mat imgsm = obj2.samples.at(ns2);
      cv::resize(imgsm, imgsm, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

      bufp.x = 1.0 * obj.coords.at(ns1).x * wh / (2 * half_imgsize);
      bufp.y = 1.0 * obj.coords.at(ns1).y * wh / (2 * half_imgsize);

      cv::Point2f c_cir;

      c_cir.x = (half_imgsize - (obj2.coords.at(ns2).x - obj.coords.at(ns1).x)) * wh / (2 * half_imgsize);
      c_cir.y = (half_imgsize - (obj2.coords.at(ns2).y - obj.coords.at(ns1).y)) * wh / (2 * half_imgsize);

      std::cout << "c_cir.y - " << c_cir.y << std::endl;
      std::cout << "c_cir.x - " << c_cir.x << std::endl;
      std::cout << "half_imgsize - " << half_imgsize << std::endl;

      if (c_cir.y < 0 || c_cir.x < 0 || c_cir.y > 2 * half_imgsize * wh / (2 * half_imgsize) || c_cir.x > 2 * half_imgsize * wh / (2 * half_imgsize))
        goto dell;

      cv::circle(imag, c_cir, 3, cv::Scalar(R, G, B), 1);

      // imgsm.copyTo(imag(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

      pt1.x = bufp.x;
      pt1.y = bufp.y;

      pt2.x = bufp.x + hp;
      pt2.y = bufp.y + hp;

      rectangle(imag, pt1, pt2, cv::Scalar(R, G, B), 1);

      //---

      bufp.x = 1.0 * obj2.coords.at(ns2).x * wh / (2 * half_imgsize);
      bufp.y = 1.0 * obj2.coords.at(ns2).y * wh / (2 * half_imgsize);
      // imgsm.copyTo(imag2(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

      pt1.x = bufp.x;
      pt1.y = bufp.y;

      pt2.x = bufp.x + hp;
      pt2.y = bufp.y + hp;

      rectangle(imag2, pt1, pt2, cv::Scalar(R, G, B), 1);

      imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
      imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));

      imshow("imgres", imgres);
      cv::waitKey(0);
    dell:
      obj.samples.erase(obj.samples.begin() + ns1);
      obj2.samples.erase(obj2.samples.begin() + ns2);

      obj.coords.erase(obj.coords.begin() + ns1);
      obj2.coords.erase(obj2.coords.begin() + ns2);

    } while (obj.samples.size() > 0 && obj2.samples.size() > 0);
  //-----------------------------------

  imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
  imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
  imshow("imgres", imgres);
  cv::waitKey(0);

  return imgres;
}

cv::Mat draw_compare(ALObject obj, ALObject obj2, cv::Scalar color)
{
  int wh = 800;
  float hp = 1.0 * (resolution / reduseres) * wh / (2 * half_imgsize);

  cv::Mat imag(wh, wh, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat imag2(wh, wh, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat imgres(wh, wh * 2, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat imgbuf;

  cv::Mat sample;
  cv::Mat sample2;

  cv::Point2f bufp;
  cv::Point2f pt1;
  cv::Point2f pt2;

  std::vector<cv::Point2f> center;
  std::vector<int> npsamples;

  cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
  imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

  cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
  imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

  int st = 5; // step for samles compare

  cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
  imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

  cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
  imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

  for (int step_x = -(int)(imag.cols / st) / 2; step_x < (int)(imag.cols / st) / 2; step_x++)
  {
    for (int step_y = -(int)(imag.rows / st) / 2; step_y < (int)(imag.rows / st) / 2; step_y++)
    {
      // cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
      // imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

      // cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
      // imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

      int np = 0;
      int ns = 0;
      for (int i = 0; i < obj2.samples.size(); i++)
      {
        sample2 = obj2.samples.at(i);
        // cv::resize(sample2, sample2, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

        bufp.x = 1.0 * obj2.coords.at(i).x * wh / (2 * half_imgsize);
        bufp.y = 1.0 * obj2.coords.at(i).y * wh / (2 * half_imgsize);

        if ((bufp.y + step_y * st + hp) > imag.rows || (bufp.x + step_x * st + hp) > imag.cols || (bufp.y + step_y * st) < 0 || (bufp.x + step_x * st) < 0)
          continue;

        sample = obj.img(cv::Range(obj2.coords.at(i).y + step_y * st * obj2.img.rows / imag.rows, obj2.coords.at(i).y + step_y * st * obj2.img.rows / imag.rows + resolution / reduseres), cv::Range(obj2.coords.at(i).x + step_x * st * obj2.img.cols / imag.cols, obj2.coords.at(i).x + step_x * st * obj2.img.cols / imag.cols + resolution / reduseres));

        np += samples_compV2(obj2.samples.at(i), sample); // sample compare
        ns++;

        /*
        cv::resize(sample, sample, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

        sample.copyTo(imag2(cv::Rect(bufp.x + step_x*st, bufp.y + step_y*st, sample.cols, sample.rows)));
        sample2.copyTo(imag(cv::Rect(bufp.x + step_x*st, bufp.y + step_y*st, sample2.cols, sample2.rows)));

        pt1.x = bufp.x + step_x*st-1;
        pt1.y = bufp.y + step_y*st-1;

        pt2.x = bufp.x + hp + step_x*st;
        pt2.y = bufp.y + hp + step_y*st;

        rectangle(imag, pt1, pt2, color, 1);
        rectangle(imag2, pt1, pt2, color, 1);
        */
      }

      bufp.y = half_imgsize + step_y * st * obj2.img.rows / imag.rows;
      bufp.x = half_imgsize + step_x * st * obj2.img.cols / imag.cols;

      if (bufp.y > 0 && bufp.y < 2 * half_imgsize && bufp.x > 0 && bufp.x < 2 * half_imgsize)
      {
        npsamples.push_back(np);
        center.push_back(bufp);
      }

      // imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
      // imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
      // imshow("imgres", imgres);
      // cv::waitKey(0);
    }
  }

  int max = npsamples.at(0);
  int min = npsamples.at(1);
  for (int i = 0; i < npsamples.size(); i++)
  {
    if (max < npsamples.at(i))
      max = npsamples.at(i);

    if (min > npsamples.at(i))
      min = npsamples.at(i);
  }

  for (int i = 0; i < npsamples.size(); i++)
    npsamples.at(i) -= min;

  int color_step = (max - min) / 255;

  cv::Mat imgcenter(2 * half_imgsize, 2 * half_imgsize, CV_8UC1, cv::Scalar(0, 0, 0));

  for (int i = 0; i < npsamples.size(); i++)
  {
    imgcenter.at<uchar>(center.at(i).y, center.at(i).x) = 255 - npsamples.at(i) / color_step;
  }

  cv::resize(imgcenter, imgcenter, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
  imgcenter.convertTo(imgcenter, CV_8UC3);

  imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
  imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));

  imshow("imgres", imgres);
  imshow("imgcenter", imgcenter);
  cv::waitKey(0);

  return imgres;
}

cv::Mat DetectorMotionV2(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame, bool usedetector)
{
  cv::Scalar class_name_color[20] = {
      cv::Scalar(255, 0, 0),
      cv::Scalar(0, 20, 200),
      cv::Scalar(0, 255, 0),
      cv::Scalar(255, 0, 255),
      cv::Scalar(0, 255, 255),
      cv::Scalar(255, 255, 0),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 0, 200),
      cv::Scalar(100, 0, 255),
      cv::Scalar(255, 0, 100),
      cv::Scalar(30, 20, 200),
      cv::Scalar(25, 255, 0),
      cv::Scalar(255, 44, 255),
      cv::Scalar(88, 255, 255),
      cv::Scalar(255, 255, 39),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 46, 200),
      cv::Scalar(100, 79, 255),
      cv::Scalar(200, 46, 150),
      cv::Scalar(140, 70, 205),
  };

  std::vector<std::vector<cv::Point2f>> clasters;
  std::vector<cv::Point2f> motion;
  std::vector<cv::Mat> imgs;

  cv::Mat imageBGR0;
  cv::Mat imageBGR;

  cv::Mat imag;
  cv::Mat imagbuf;
  cv::Mat framebuf = frame;

  int mpc = 15;   // minimum number of points for a cluster (good value 15)
  int nd = 9;     //(good value 6-15)
  int rcobj = 17; //(good value 15)
  int robj = 17;  //(good value 17)
  int mdist = 10; // maximum distance from cluster center (good value 10)
  int pft = 9;    // points fixation threshold (good value 9)

  cv::Mat img;

  std::vector<OBJdetect> detects;

  //--------------------<detection using a classifier>----------
  if (usedetector)
  {
    detects = detectorV4(pathmodel, frame, device_type);

    for (int i = 0; i < objects.size(); i++)
    {
      objects[i].model_center.x = -1;
      objects[i].model_center.y = -1;
    }

    for (int i = 0; i < detects.size(); i++)
    {
      if (detects.at(i).type != "a")
      {
        detects.erase(detects.begin() + i);
        i--;
      }
    }

    for (int i = 0; i < detects.size(); i++)
    {
      std::vector<cv::Point2f> claster_points;
      claster_points.push_back(detects.at(i).detect);
      imagbuf = frame_resizing(framebuf);
      img = imagbuf(cv::Range(detects.at(i).detect.y - half_imgsize, detects.at(i).detect.y + half_imgsize), cv::Range(detects.at(i).detect.x - half_imgsize, detects.at(i).detect.x + half_imgsize));
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      img.convertTo(img, CV_8UC3);

      ALObject obj(objects.size(), detects.at(i).type, claster_points, img);
      obj.model_center = detects.at(i).detect;
      obj.claster_center = detects.at(i).detect;
      obj.rectangle = detects.at(i).rectangle;
      // obj.track_points.push_back(detects.at(i).detect);
      // obj.push_track_point(detects.at(i).detect);

      float rm = rcobj * 1.0 * resolution / reduseres;
      int n = -1;

      if (objects.size() > 0)
      {
        rm = sqrt(pow((objects[0].claster_center.x - obj.claster_center.x), 2) + pow((objects[0].claster_center.y - obj.claster_center.y), 2));
        // rm = sqrt(pow((objects[0].proposed_center().x - obj.claster_center.x), 2) + pow((objects[0].proposed_center().y - obj.claster_center.y), 2));
        if (rm < rcobj * 1.0 * resolution / reduseres && rm < rcobj)
          n = 0;
      }

      for (int j = 1; j < objects.size(); j++)
      {
        float r = sqrt(pow((objects[j].claster_center.x - obj.claster_center.x), 2) + pow((objects[j].claster_center.y - obj.claster_center.y), 2));
        // float r = sqrt(pow((objects[j].proposed_center().x - obj.claster_center.x), 2) + pow((objects[j].proposed_center().y - obj.claster_center.y), 2));
        if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
        {
          rm = r;
          n = j;
        }
      }

      if (n > -1)
      {
        objects[n].claster_center = obj.model_center;
        objects[n].model_center = obj.model_center;
        objects[n].rectangle = obj.rectangle;
        // objects[n].track_points.push_back(obj.claster_center);
        // objects[n].push_track_point(obj.claster_center);
        objects[n].img = obj.img;
      }
      else
      {
        objects.push_back(obj);
      }
    }
  }
  //--------------------</detection using a classifier>---------

  //--------------------<moution detections>--------------------
  int rows = frame.rows;
  int cols = frame.cols;

  float rwsize;
  float clsize;

  imagbuf = frame;
  if (rows > cols)
  {
    rwsize = resolution * rows * 1.0 / cols;
    clsize = resolution;
  }
  else
  {
    rwsize = resolution;
    clsize = resolution * cols * 1.0 / rows;
  }

  cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rectb(0, 0, resolution, resolution);
  imag = imagbuf(rectb);

  cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
  imag.convertTo(imag, CV_8UC3);

  if (rows > cols)
  {
    rwsize = reduseres * rows * 1.0 / cols;
    clsize = reduseres;
  }
  else
  {
    rwsize = reduseres;
    clsize = reduseres * cols * 1.0 / rows;
  }

  cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rect0(0, 0, reduseres, reduseres);
  imageBGR0 = frame0(rect0);
  imageBGR0.convertTo(imageBGR0, CV_8UC1);

  cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

  cv::Rect rect(0, 0, reduseres, reduseres);
  imageBGR = frame(rect);

  imageBGR.convertTo(imageBGR, CV_8UC1);

  cv::Point2f pm;

  for (int y = 0; y < imageBGR0.rows; y++)
  {
    for (int x = 0; x < imageBGR0.cols; x++)
    {
      uchar color1 = imageBGR0.at<uchar>(cv::Point(x, y));
      uchar color2 = imageBGR.at<uchar>(cv::Point(x, y));

      if (((int)color2 - (int)color1) > pft)
      {
        pm.x = x * resolution / reduseres;
        pm.y = y * resolution / reduseres;
        motion.push_back(pm);
      }
    }
  }

  cv::Point2f pt1;
  cv::Point2f pt2;

  for (int i = 0; i < motion.size(); i++) // visualization of the claster_points
  {
    pt1.x = motion.at(i).x;
    pt1.y = motion.at(i).y;

    pt2.x = motion.at(i).x + resolution / reduseres;
    pt2.y = motion.at(i).y + resolution / reduseres;

    rectangle(imag, pt1, pt2, cv::Scalar(255, 255, 255), 1);
  }

  uint16_t ncls = 0;
  uint16_t nobj;

  if (objects.size() > 0)
    nobj = 0;
  else
    nobj = -1;
  //--------------</moution detections>--------------------

  //--------------<layout of motion points by objects>-----

  if (objects.size() > 0)
  {
    for (int i = 0; i < objects.size(); i++)
    {
      objects[i].claster_points.clear();
    }

    for (int i = 0; i < motion.size(); i++)
    {
      for (int j = 0; j < objects.size(); j++)
      {
        if (objects[j].model_center.x < 0)
          continue;

        if (i < 0)
          break;

        if ((motion.at(i).x < (objects[j].model_center.x + objects[j].rectangle.x / 2)) && (motion.at(i).x > (objects[j].model_center.x - objects[j].rectangle.x / 2)) && (motion.at(i).y < (objects[j].model_center.y + objects[j].rectangle.y / 2)) && (motion.at(i).y > (objects[j].model_center.y - objects[j].rectangle.y / 2)))
        {
          objects[j].claster_points.push_back(motion.at(i));
          motion.erase(motion.begin() + i);
          i--;
        }
      }
    }

    float rm = rcobj * 1.0 * resolution / reduseres;

    for (int i = 0; i < motion.size(); i++)
    {
      rm = sqrt(pow((objects[0].claster_center.x - motion.at(i).x), 2) + pow((objects[0].claster_center.y - motion.at(i).y), 2));

      int n = -1;
      if (rm < rcobj * 1.0 * resolution / reduseres)
        n = 0;

      for (int j = 1; j < objects.size(); j++)
      {
        float r = sqrt(pow((objects[j].claster_center.x - motion.at(i).x), 2) + pow((objects[j].claster_center.y - motion.at(i).y), 2));
        if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
        {
          rm = r;
          n = j;
        }
      }

      if (n > -1)
      {
        objects[n].claster_points.push_back(motion.at(i));
        motion.erase(motion.begin() + i);
        i--;
        objects[n].center_determine(false);
        if (i < 0)
          break;
      }
    }
  }

  for (int j = 0; j < objects.size(); j++)
  {

    objects[j].center_determine(false);

    cv::Point2f clastercenter = objects[j].claster_center;

    imagbuf = frame_resizing(framebuf);
    imagbuf.convertTo(imagbuf, CV_8UC3);

    if (clastercenter.y - half_imgsize < 0)
      clastercenter.y = half_imgsize + 1;

    if (clastercenter.x - half_imgsize < 0)
      clastercenter.x = half_imgsize + 1;

    if (clastercenter.y + half_imgsize > imagbuf.rows)
      clastercenter.y = imagbuf.rows - 1;

    if (clastercenter.x + half_imgsize > imagbuf.cols)
      clastercenter.x = imagbuf.cols - 1;

    img = imagbuf(cv::Range(clastercenter.y - half_imgsize, clastercenter.y + half_imgsize), cv::Range(clastercenter.x - half_imgsize, clastercenter.x + half_imgsize));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_8UC3);
    objects[j].img = img;
    objects[j].samples_creation();
  }
  //--------------</layout of motion points by objects>----

  //--------------<claster creation>-----------------------

  while (motion.size() > 0)
  {
    cv::Point2f pc;

    if (nobj > -1 && nobj < objects.size())
    {
      pc = objects[nobj].claster_center;
      // pc = objects[nobj].proposed_center();
      nobj++;
    }
    else
    {
      pc = motion.at(0);
      motion.erase(motion.begin());
    }

    clasters.push_back(std::vector<cv::Point2f>());
    clasters[ncls].push_back(pc);

    for (int i = 0; i < motion.size(); i++)
    {
      float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
      if (r < nd * 1.0 * resolution / reduseres)
      {
        cv::Point2f cl_c = claster_center(clasters.at(ncls));
        r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
        if (r < mdist * 1.0 * resolution / reduseres)
        {
          clasters.at(ncls).push_back(motion.at(i));
          motion.erase(motion.begin() + i);
          i--;
        }
      }
    }

    int newp;
    do
    {
      newp = 0;

      for (int c = 0; c < clasters[ncls].size(); c++)
      {
        pc = clasters[ncls].at(c);
        for (int i = 0; i < motion.size(); i++)
        {
          float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

          if (r < nd * 1.0 * resolution / reduseres)
          {
            cv::Point2f cl_c = claster_center(clasters.at(ncls));
            r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
            if (r < mdist * 1.0 * resolution / reduseres)
            {
              clasters.at(ncls).push_back(motion.at(i));
              motion.erase(motion.begin() + i);
              i--;
              newp++;
            }
          }
        }
      }
    } while (newp > 0 && motion.size() > 0);

    ncls++;
  }
  //--------------</claster creation>----------------------

  //--------------<clusters to objects>--------------------
  for (int cls = 0; cls < ncls; cls++)
  {
    if (clasters[cls].size() > mpc) // if there are enough moving points
    {

      cv::Point2f clastercenter = claster_center(clasters[cls]);
      imagbuf = frame_resizing(framebuf);
      imagbuf.convertTo(imagbuf, CV_8UC3);
      img = imagbuf(cv::Range(clastercenter.y - half_imgsize, clastercenter.y + half_imgsize), cv::Range(clastercenter.x - half_imgsize, clastercenter.x + half_imgsize));
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      img.convertTo(img, CV_8UC3);

      ALObject obj(objects.size(), "a", clasters[cls], img);
      bool newobj = true;

      for (int i = 0; i < objects.size(); i++)
      {
        float r = sqrt(pow((objects[i].claster_center.x - obj.claster_center.x), 2) + pow((objects[i].claster_center.y - obj.claster_center.y), 2));
        // float r = sqrt(pow((objects[i].proposed_center().x - obj.claster_center.x), 2) + pow((objects[i].proposed_center().y - obj.claster_center.y), 2));
        if (r < robj * 1.0 * resolution / reduseres)
        {
          newobj = false;

          objects[i].img = obj.img;
          objects[i].claster_points = obj.claster_points;
          objects[i].center_determine(true);
          break;
        }
      }

      if (newobj == true)
        objects.push_back(obj);
    }
  }
  //--------------</clusters to objects>-------------------

  for (int i = 0; i < objects.size(); i++)
    objects[i].push_track_point(objects[i].claster_center);

  //--------------<visualization>--------------------------
  for (int i = 0; i < objects.size(); i++)
  {
    for (int j = 0; j < objects.at(i).claster_points.size(); j++) // visualization of the claster_points
    {
      pt1.x = objects.at(i).claster_points.at(j).x;
      pt1.y = objects.at(i).claster_points.at(j).y;

      pt2.x = objects.at(i).claster_points.at(j).x + resolution / reduseres;
      pt2.y = objects.at(i).claster_points.at(j).y + resolution / reduseres;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    if (objects.at(i).model_center.x > -1) // visualization of the classifier
    {
      pt1.x = objects.at(i).model_center.x - objects.at(i).rectangle.x / 2;
      pt1.y = objects.at(i).model_center.y - objects.at(i).rectangle.y / 2;

      pt2.x = objects.at(i).model_center.x + objects.at(i).rectangle.x / 2;
      pt2.y = objects.at(i).model_center.y + objects.at(i).rectangle.y / 2;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    for (int j = 0; j < objects.at(i).track_points.size(); j++)
      cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color[objects.at(i).id], 2);
  }
  //--------------</visualization>-------------------------

  //--------------<baseimag>-------------------------------
  cv::Mat baseimag(resolution, resolution + extr, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int i = 0; i < objects.size(); i++)
  {
    std::string text = objects.at(i).obj_type + " ID" + std::to_string(objects.at(i).id);

    cv::Point2f ptext;
    ptext.x = 20;
    ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

    cv::putText(baseimag, // target image
                text,     // text
                ptext,    // top-left position
                1,
                1,
                class_name_color[objects.at(i).id], // font color
                1);

    pt1.x = ptext.x - 1;
    pt1.y = ptext.y - 1 + 10;

    pt2.x = ptext.x + objects.at(i).img.cols + 1;
    pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

    if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
    {
      rectangle(baseimag, pt1, pt2, class_name_color[objects.at(i).id], 1);
      objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
    }
  }

  imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

  cv::Point2f p_idframe;
  p_idframe.x = resolution + extr - 95;
  p_idframe.y = 50;
  cv::putText(baseimag, std::to_string(id_frame), p_idframe, 1, 3, cv::Scalar(255, 255, 255), 2);
  cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
  //--------------</baseimag>-------------------------------

  imshow("Motion", baseimag);
  cv::waitKey(10);

  /*
    al_objs.push_back(objects.at(1));
    if (al_objs.size() > 1)
    {
      draw_compare(al_objs.at(al_objs.size() - 2), al_objs.at(al_objs.size() - 1), class_name_color[al_objs.at(0).id]);
    }
  */

  return baseimag;
}

void DetectorMotionV2b(cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame)
{
  cv::Scalar class_name_color[20] = {
      cv::Scalar(255, 0, 0),
      cv::Scalar(0, 20, 200),
      cv::Scalar(0, 255, 0),
      cv::Scalar(255, 0, 255),
      cv::Scalar(0, 255, 255),
      cv::Scalar(255, 255, 0),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 0, 200),
      cv::Scalar(100, 0, 255),
      cv::Scalar(255, 0, 100),
      cv::Scalar(30, 20, 200),
      cv::Scalar(25, 255, 0),
      cv::Scalar(255, 44, 255),
      cv::Scalar(88, 255, 255),
      cv::Scalar(255, 255, 39),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 46, 200),
      cv::Scalar(100, 79, 255),
      cv::Scalar(200, 46, 150),
      cv::Scalar(140, 70, 205),
  };

  std::vector<std::vector<cv::Point2f>> clasters;
  std::vector<cv::Point2f> motion;
  std::vector<cv::Mat> imgs;

  cv::Mat imageBGR0;
  cv::Mat imageBGR;

  cv::Mat imag;
  cv::Mat imagbuf;
  cv::Mat framebuf = frame;

  int mpc = 15;   // minimum number of points for a cluster (good value 15)
  int nd = 9;     //(good value 6-15)
  int rcobj = 15; //(good value 15)
  int robj = 17;  //(good value 17)
  int mdist = 12; // maximum distance from cluster center (good value 10)
  int pft = 9;    // points fixation threshold (good value 9)

  cv::Mat img;

  //--------------------<moution detections>--------------------
  int rows = frame.rows;
  int cols = frame.cols;

  float rwsize;
  float clsize;

  imagbuf = frame;
  if (rows > cols)
  {
    rwsize = resolution * rows * 1.0 / cols;
    clsize = resolution;
  }
  else
  {
    rwsize = resolution;
    clsize = resolution * cols * 1.0 / rows;
  }

  cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rectb(0, 0, resolution, resolution);
  imag = imagbuf(rectb);

  cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
  imag.convertTo(imag, CV_8UC3);

  if (rows > cols)
  {
    rwsize = reduseres * rows * 1.0 / cols;
    clsize = reduseres;
  }
  else
  {
    rwsize = reduseres;
    clsize = reduseres * cols * 1.0 / rows;
  }

  cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rect0(0, 0, reduseres, reduseres);
  imageBGR0 = frame0(rect0);
  imageBGR0.convertTo(imageBGR0, CV_8UC1);

  cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

  cv::Rect rect(0, 0, reduseres, reduseres);
  imageBGR = frame(rect);

  imageBGR.convertTo(imageBGR, CV_8UC1);

  cv::Point2f pm;

  for (int y = 0; y < imageBGR0.rows; y++)
  {
    for (int x = 0; x < imageBGR0.cols; x++)
    {
      uchar color1 = imageBGR0.at<uchar>(cv::Point(x, y));
      uchar color2 = imageBGR.at<uchar>(cv::Point(x, y));

      if (((int)color2 - (int)color1) > pft)
      {
        pm.x = x * resolution / reduseres;
        pm.y = y * resolution / reduseres;
        motion.push_back(pm);
      }
    }
  }

  cv::Point2f pt1;
  cv::Point2f pt2;

  for (int i = 0; i < motion.size(); i++) // visualization of the claster_points
  {
    pt1.x = motion.at(i).x;
    pt1.y = motion.at(i).y;

    pt2.x = motion.at(i).x + resolution / reduseres;
    pt2.y = motion.at(i).y + resolution / reduseres;

    rectangle(imag, pt1, pt2, cv::Scalar(255, 255, 255), 1);
  }

  uint16_t ncls = 0;
  uint16_t nobj;

  if (objects.size() > 0)
    nobj = 0;
  else
    nobj = -1;
  //--------------</moution detections>--------------------

  //--------------<layout of motion points by objects>-----

  if (objects.size() > 0)
  {
    for (int i = 0; i < objects.size(); i++)
    {
      objects[i].claster_points.clear();
    }

    for (int i = 0; i < motion.size(); i++)
    {
      for (int j = 0; j < objects.size(); j++)
      {
        if (objects[j].model_center.x < 0)
          continue;

        if (i < 0)
          break;

        if ((motion.at(i).x < (objects[j].model_center.x + objects[j].rectangle.x / 2)) && (motion.at(i).x > (objects[j].model_center.x - objects[j].rectangle.x / 2)) && (motion.at(i).y < (objects[j].model_center.y + objects[j].rectangle.y / 2)) && (motion.at(i).y > (objects[j].model_center.y - objects[j].rectangle.y / 2)))
        {
          objects[j].claster_points.push_back(motion.at(i));
          motion.erase(motion.begin() + i);
          i--;
        }
      }
    }

    float rm = rcobj * 1.0 * resolution / reduseres;

    for (int i = 0; i < motion.size(); i++)
    {
      rm = sqrt(pow((objects[0].claster_center.x - motion.at(i).x), 2) + pow((objects[0].claster_center.y - motion.at(i).y), 2));

      int n = -1;
      if (rm < rcobj * 1.0 * resolution / reduseres)
        n = 0;

      for (int j = 1; j < objects.size(); j++)
      {
        float r = sqrt(pow((objects[j].claster_center.x - motion.at(i).x), 2) + pow((objects[j].claster_center.y - motion.at(i).y), 2));
        if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
        {
          rm = r;
          n = j;
        }
      }

      if (n > -1)
      {
        objects[n].claster_points.push_back(motion.at(i));
        motion.erase(motion.begin() + i);
        i--;
        objects[n].center_determine(false);
        if (i < 0)
          break;
      }
    }
  }

  for (int j = 0; j < objects.size(); j++)
  {

    objects[j].center_determine(false);

    cv::Point2f clastercenter = objects[j].claster_center;

    imagbuf = frame_resizing(framebuf);
    imagbuf.convertTo(imagbuf, CV_8UC3);

    if (clastercenter.y - half_imgsize < 0)
      clastercenter.y = half_imgsize + 1;

    if (clastercenter.x - half_imgsize < 0)
      clastercenter.x = half_imgsize + 1;

    if (clastercenter.y + half_imgsize > imagbuf.rows)
      clastercenter.y = imagbuf.rows - 1;

    if (clastercenter.x + half_imgsize > imagbuf.cols)
      clastercenter.x = imagbuf.cols - 1;

    img = imagbuf(cv::Range(clastercenter.y - half_imgsize, clastercenter.y + half_imgsize), cv::Range(clastercenter.x - half_imgsize, clastercenter.x + half_imgsize));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_8UC3);
    objects[j].img = img;
    objects[j].samples_creation();
  }
  //--------------</layout of motion points by objects>----

  //--------------<claster creation>-----------------------

  while (motion.size() > 0)
  {
    cv::Point2f pc;

    if (nobj > -1 && nobj < objects.size())
    {
      pc = objects[nobj].claster_center;
      // pc = objects[nobj].proposed_center();
      nobj++;
    }
    else
    {
      pc = motion.at(0);
      motion.erase(motion.begin());
    }

    clasters.push_back(std::vector<cv::Point2f>());
    clasters[ncls].push_back(pc);

    for (int i = 0; i < motion.size(); i++)
    {
      float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
      if (r < nd * 1.0 * resolution / reduseres)
      {
        cv::Point2f cl_c = claster_center(clasters.at(ncls));
        r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
        if (r < mdist * 1.0 * resolution / reduseres)
        {
          clasters.at(ncls).push_back(motion.at(i));
          motion.erase(motion.begin() + i);
          i--;
        }
      }
    }

    int newp;
    do
    {
      newp = 0;

      for (int c = 0; c < clasters[ncls].size(); c++)
      {
        pc = clasters[ncls].at(c);
        for (int i = 0; i < motion.size(); i++)
        {
          float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

          if (r < nd * 1.0 * resolution / reduseres)
          {
            cv::Point2f cl_c = claster_center(clasters.at(ncls));
            r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
            if (r < mdist * 1.0 * resolution / reduseres)
            {
              clasters.at(ncls).push_back(motion.at(i));
              motion.erase(motion.begin() + i);
              i--;
              newp++;
            }
          }
        }
      }
    } while (newp > 0 && motion.size() > 0);

    ncls++;
  }
  //--------------</claster creation>----------------------

  //--------------<clusters to objects>--------------------
  for (int cls = 0; cls < ncls; cls++)
  {
    if (clasters[cls].size() > mpc) // if there are enough moving points
    {

      cv::Point2f clastercenter = claster_center(clasters[cls]);
      imagbuf = frame_resizing(framebuf);
      imagbuf.convertTo(imagbuf, CV_8UC3);
      img = imagbuf(cv::Range(clastercenter.y - half_imgsize, clastercenter.y + half_imgsize), cv::Range(clastercenter.x - half_imgsize, clastercenter.x + half_imgsize));
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      img.convertTo(img, CV_8UC3);

      ALObject obj(objects.size(), "a", clasters[cls], img);
      bool newobj = true;

      for (int i = 0; i < objects.size(); i++)
      {
        float r = sqrt(pow((objects[i].claster_center.x - obj.claster_center.x), 2) + pow((objects[i].claster_center.y - obj.claster_center.y), 2));
        // float r = sqrt(pow((objects[i].proposed_center().x - obj.claster_center.x), 2) + pow((objects[i].proposed_center().y - obj.claster_center.y), 2));
        if (r < robj * 1.0 * resolution / reduseres)
        {
          newobj = false;

          objects[i].img = obj.img;
          objects[i].claster_points = obj.claster_points;
          objects[i].center_determine(true);
          break;
        }
      }

      if (newobj == true)
        objects.push_back(obj);
    }
  }
  //--------------</clusters to objects>-------------------

  for (int i = 0; i < objects.size(); i++)
    objects[i].push_track_point(objects[i].claster_center);

  //--------------<visualization>--------------------------
  for (int i = 0; i < objects.size(); i++)
  {
    for (int j = 0; j < objects.at(i).claster_points.size(); j++) // visualization of the claster_points
    {
      pt1.x = objects.at(i).claster_points.at(j).x;
      pt1.y = objects.at(i).claster_points.at(j).y;

      pt2.x = objects.at(i).claster_points.at(j).x + resolution / reduseres;
      pt2.y = objects.at(i).claster_points.at(j).y + resolution / reduseres;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    if (objects.at(i).model_center.x > -1) // visualization of the classifier
    {
      pt1.x = objects.at(i).model_center.x - objects.at(i).rectangle.x / 2;
      pt1.y = objects.at(i).model_center.y - objects.at(i).rectangle.y / 2;

      pt2.x = objects.at(i).model_center.x + objects.at(i).rectangle.x / 2;
      pt2.y = objects.at(i).model_center.y + objects.at(i).rectangle.y / 2;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    for (int j = 0; j < objects.at(i).track_points.size(); j++)
      cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color[objects.at(i).id], 2);
  }
  //--------------</visualization>-------------------------

  //--------------<baseimag>-------------------------------
  cv::Mat baseimag(resolution, resolution + extr, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int i = 0; i < objects.size(); i++)
  {
    std::string text = objects.at(i).obj_type + " ID" + std::to_string(objects.at(i).id);

    cv::Point2f ptext;
    ptext.x = 20;
    ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

    cv::putText(baseimag, // target image
                text,     // text
                ptext,    // top-left position
                1,
                1,
                class_name_color[objects.at(i).id], // font color
                1);

    pt1.x = ptext.x - 1;
    pt1.y = ptext.y - 1 + 10;

    pt2.x = ptext.x + objects.at(i).img.cols + 1;
    pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

    if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
    {
      rectangle(baseimag, pt1, pt2, class_name_color[objects.at(i).id], 1);
      objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
    }
  }

  imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

  cv::Point2f p_idframe;
  p_idframe.x = resolution + extr - 95;
  p_idframe.y = 50;
  cv::putText(baseimag, std::to_string(id_frame), p_idframe, 1, 3, cv::Scalar(255, 255, 255), 2);
  cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
  //--------------</baseimag>-------------------------------
  //imshow("Motion", baseimag);
  //cv::waitKey(0);
}

void OBJdetectsToObjs(std::vector<OBJdetect> objdetects, std::vector<Obj> &objs)
{
  objs.clear();
  Obj objbuf;
  for (int i = 0; i < objdetects.size(); i++)
  {
    int j=0;
    for(j=0; j < sizeof(class_name)/sizeof(*class_name); j++)
      if(objdetects.at(i).type == class_name[j])
        break;

    objbuf.type = j;                         // Object type
    objbuf.id = i;                           // Object id
    objbuf.x = objdetects.at(i).detect.x;     // Center x of the bounding box
    objbuf.y = objdetects.at(i).detect.y;     // Center y of the bounding box
    objbuf.w = objdetects.at(i).rectangle.x;  // Width of the bounding box
    objbuf.h = objdetects.at(i).rectangle.y;  // Height of the bounding box

    objs.push_back(objbuf);
  }
}

void ALObjectsToObjs (std::vector<ALObject> objects, std::vector<Obj> &objs)
{
  objs.clear();
  Obj objbuf;
  for (int i = 0; i < objects.size(); i++)
  {
    int j=0;
    for(j=0; j < sizeof(class_name)/sizeof(*class_name); j++)
      if(objects.at(i).obj_type == class_name[j])
        break;

    objbuf.type = j;                         // Object type
    objbuf.id = objects.at(i).id;                           // Object id
    objbuf.x = objects.at(i).claster_center.x;     // Center x of the bounding box
    objbuf.y = objects.at(i).claster_center.y;     // Center y of the bounding box
    objbuf.w = 0;  // Width of the bounding box
    objbuf.h = 0;  // Height of the bounding box

    objs.push_back(objbuf);
  }
}

void fixIDs(const std::vector<std::vector<Obj>> &objs, std::vector<std::pair<uint, idFix>> &fixedIds, std::vector<cv::Mat> d_images)
{
  std::vector<ALObject> objects;
  std::vector<Obj> objsbuf;
  std::vector<std::vector<Obj>> fixedobjs;

  idFix idfix;

  const float maxr = 17.0;

  fixedIds.clear();

  fixedobjs.push_back(objs.at(0));//first frame can't be fixed

  for (int i = 0; i < d_images.size() - 1; i++)
  {
    DetectorMotionV2b(d_images.at(i), d_images.at(i + 1), objects, i);
    ALObjectsToObjs(objects,objsbuf);
    fixedobjs.push_back(objsbuf);
  }

  //std::cout<<"objs.size() - "<<objs.size()<<std::endl;
  //std::cout<<"objects.size() - "<<objects.size()<<std::endl;
  //std::cout<<"fixedobjs.size() - "<<fixedobjs.size()<<std::endl;

  for(int i=0; i<objs.size(); i++)
  {
    for(int j=0; j<objs.at(i).size(); j++)
    {
      if(objs.at(i).at(j).type != 1)
        continue;

      float minr = sqrt(pow((float)objs.at(i).at(j).x - (float)fixedobjs.at(i).at(0).x,2) + pow((float)objs.at(i).at(j).y - (float)fixedobjs.at(i).at(0).y,2));
      int ind = 0;

      for(int n=1; n<fixedobjs.at(i).size(); n++)
      {
        float r = sqrt(pow((float)objs.at(i).at(j).x - (float)fixedobjs.at(i).at(n).x,2) + pow((float)objs.at(i).at(j).y - (float)fixedobjs.at(i).at(n).y,2));
        if(r < minr)
        {
          minr = r;
          ind = n;
        }
      }

      if(minr < maxr && objs.at(i).at(j).id != fixedobjs.at(i).at(ind).id)
      {
        idfix.id = fixedobjs.at(i).at(ind).id;
        idfix.idOld = objs.at(i).at(j).id;
        idfix.type = objs.at(i).at(j).type;//not fixed
        fixedIds.push_back(std::make_pair((uint)i,idfix));
        fixedobjs.at(i).erase(fixedobjs.at(i).begin() + ind);
      }
    }
  }

  /*
  std::cout<<"fixedIds.size() - "<<fixedIds.size()<<std::endl;

  for(int i=0; i<fixedIds.size(); i++)
  {
    std::cout<<"fixedIds.at("<<i<<").second.id - "<<fixedIds.at(i).second.id<<std::endl;
    std::cout<<"fixedIds.at("<<i<<").second.idOld- "<<fixedIds.at(i).second.idOld<<std::endl;
  }
  */
}

bool compare(intpoint a, intpoint b)
{
  if (a.ipoint < b.ipoint)
    return 1;
  else
    return 0;
}

std::vector<cv::Point2f> draw_map_prob(std::vector<int> npsamples, std::vector<cv::Point2f> mpoints)
{
  int maxprbp = 9;

  maxprbp = 0;

  std::vector<cv::Point2f> probapoints;

  int max = npsamples.at(0);
  int min = npsamples.at(1);

  cv::Point2f mpoint;

  std::vector<intpoint> top;
  intpoint bufnpmp;

  for (int i = 0; i < npsamples.size(); i++)
  {
    bufnpmp.ipoint = npsamples.at(i);
    bufnpmp.mpoint = mpoints.at(i);
    top.push_back(bufnpmp);
  }

  sort(top.begin(), top.end(), compare);

  for (int i = 0; i < npsamples.size(); i++)
  {
    if (max < npsamples.at(i))
      max = npsamples.at(i);

    if (min > npsamples.at(i))
    {
      min = npsamples.at(i);
      mpoint = mpoints.at(i);
    }
  }

  cv::Point2f bufmp;
  int bufnp;

  /*std::cout << "----------------------" << std::endl;
  for (int i = 0; i < 10; i++)
  {
    std::cout << i << ". " << top.at(i).ipoint << std::endl;
    std::cout << i << ". " << top.at(i).mpoint << std::endl;
  }

  std::cout << "min - " << min << std::endl;
  std::cout << "mpoint.x - " << mpoint.x << std::endl;
  std::cout << "mpoint.y - " << mpoint.y << std::endl;
  */

  for (int i = 0; i < npsamples.size(); i++)
    npsamples.at(i) -= min;

  int color_step = (int)((max - min) / (3 * 255));

  cv::Mat imgpmap(2 * half_imgsize, 2 * half_imgsize, CV_8UC3, cv::Scalar(0, 0, 0));

  uint8_t *pixelPtr1 = (uint8_t *)imgpmap.data;
  int cn = imgpmap.channels();
  cv::Scalar_<uint8_t> bgrPixel1;

  for (int i = 0; i < npsamples.size(); i++)
  {
    int Pcolor = 3 * 255 - (int)(npsamples.at(i) / color_step);

    uint8_t red;
    uint8_t blue;
    if (Pcolor < 255)
    {
      red = (uint8_t)0;
      blue = (uint8_t)Pcolor;
    }
    else if (Pcolor > 255 && Pcolor < 2 * 255)
    {
      red = (uint8_t)(Pcolor - 255);
      blue = (uint8_t)255;
    }
    else
    {
      red = (uint8_t)255;
      blue = (uint8_t)(3 * 255 - Pcolor);
    }

    pixelPtr1[(size_t)mpoints.at(i).y * imgpmap.cols * cn + (size_t)mpoints.at(i).x * cn + 0] = blue;       // B
    pixelPtr1[(size_t)mpoints.at(i).y * imgpmap.cols * cn + (size_t)mpoints.at(i).x * cn + 1] = (uint8_t)0; // G
    pixelPtr1[(size_t)mpoints.at(i).y * imgpmap.cols * cn + (size_t)mpoints.at(i).x * cn + 2] = red;        // R
  }

  cv::circle(imgpmap, mpoint, 1, cv::Scalar(0, 255, 0), 1);

  for (int i = 0; i < top.size(); i++)
  {

    if (i >= maxprbp)
      break;

    cv::circle(imgpmap, top.at(i).mpoint, 1, cv::Scalar(0, 255, 255), 1);

    probapoints.push_back(top.at(i).mpoint);
  }

  cv::resize(imgpmap, imgpmap, cv::Size(imgpmap.rows * 5, imgpmap.cols * 5), cv::InterpolationFlags::INTER_CUBIC);
  imshow("imgpmap", imgpmap);
  cv::waitKey(0);

  return probapoints;
}

void objdeterm(std::vector<cv::Point2f> claster_points, cv::Mat frame, ALObject &obj, int id_frame)
{
  int wh = 800;
  std::vector<cv::Mat> claster_samples;
  cv::Point2f minp;
  cv::Point2f maxp;
  cv::Point2f bufp;

  cv::Point2f p1;
  cv::Point2f p2;

  minp.y = frame.rows;
  minp.x = frame.cols;

  maxp.x = 0;
  maxp.y = 0;
  cv::Mat bufimg;

  std::vector<int> npforpoints;
  std::vector<int> npsamples;
  std::vector<cv::Point2f> mpoints;

  std::vector<std::vector<int>> all_npforpoints;

  for (int i = 0; i < claster_points.size(); i++)
  {
    if (claster_points.at(i).y < minp.y)
      minp.y = claster_points.at(i).y;

    if (claster_points.at(i).x < minp.x)
      minp.x = claster_points.at(i).x;

    if (claster_points.at(i).y > maxp.y)
      maxp.y = claster_points.at(i).y;

    if (claster_points.at(i).x > maxp.x)
      maxp.x = claster_points.at(i).x;

    bufimg = frame(cv::Range(claster_points.at(i).y, claster_points.at(i).y + resolution / reduseres), cv::Range(claster_points.at(i).x, claster_points.at(i).x + resolution / reduseres));
    claster_samples.push_back(bufimg);

    npforpoints.push_back(0);
  }

  int st = 1; // step for samles compare

  cv::Mat imag = obj.img;

  int start_y = 0;
  int start_x = 0;

  int and_y = (int)((imag.rows - resolution / reduseres) / st);
  int and_x = (int)((imag.cols - resolution / reduseres) / st);

  for (int step_x = start_x; step_x < and_x; step_x++)
  {
    for (int step_y = start_y; step_y < and_y; step_y++)
    {
      // int np = 0;
      // int ns = 0;

      // cv::Mat resimg(maxp.y - minp.y + resolution / reduseres, maxp.x - minp.x + resolution / reduseres, CV_8UC3, cv::Scalar(0, 0, 50));

      /*
      for (int i = 0; i < claster_samples.size(); i++)
      {
        bufp.x = claster_points.at(i).x - minp.x;
        bufp.y = claster_points.at(i).y - minp.y;

        if ((bufp.x + step_x * st) + resolution / reduseres > imag.cols || (bufp.x + step_x * st) < 0)
        {
          continue;
        }

        if ((bufp.y + step_y * st) + resolution / reduseres > imag.rows || (bufp.y + step_y * st) < 0)
        {
          continue;
        }

        cv::Mat sample = imag(cv::Range(bufp.y + step_y * st, bufp.y + step_y * st + resolution / reduseres), cv::Range(bufp.x + step_x * st, bufp.x + step_x * st + resolution / reduseres));
        //sample.copyTo(resimg(cv::Rect(bufp.x, bufp.y, resolution / reduseres, resolution / reduseres)));

        npforpoints.at(i) = samples_compV2(claster_samples.at(i), sample);
        np += npforpoints.at(i);// sample compare
        ns++;
        //std::cout << "np - " << np << std::endl;
      }
      */

      for (int i = 0; i < claster_samples.size(); i++)
      {
        cv::Mat sample;
        sample = imag(cv::Range(step_y * st, step_y * st + resolution / reduseres), cv::Range(step_x * st, step_x * st + resolution / reduseres));
        int npbuf = samples_compV2(claster_samples.at(i), sample);
        npforpoints.at(i) = npbuf;
        // np += npbuf; // sample compare
        // ns++;
      }

      // np = (int)(np*npforpoints.at(test));

      // np = (int)(np-npforpoints.at(test));

      // np = npforpoints.at(test);

      // np = (int)np/(ns*2+1);

      // std::cout << "---------------------------------" << std::endl;
      // std::cout << "np - " << np << std::endl;
      // std::cout << "---------------------------------" << std::endl;

      if (2 == 3)
      {
        // cv::resize(resimg, resimg, cv::Size((maxp.x - minp.x + resolution / reduseres) * 5, (maxp.y - minp.y + resolution / reduseres) * 5), cv::InterpolationFlags::INTER_CUBIC);
        // imshow("resimg", resimg);
        // cv::waitKey(0);
      }

      // bufp.x = imag.rows - step_y*st;
      // bufp.y = imag.cols - step_x*st;

      bufp.y = step_y * st;
      bufp.x = step_x * st;

      if (bufp.y > 0 && bufp.y < imag.rows && bufp.x > 0 && bufp.x < imag.cols)
      {
        // npsamples.push_back(np);
        mpoints.push_back(bufp);
        all_npforpoints.push_back(npforpoints);
      }
    }
  }

  std::vector<int> sample_np;
  std::vector<std::vector<cv::Point2f>> alt_claster_points;

  std::vector<intpoint> chain;
  std::vector<std::vector<intpoint>> chains;

  std::cout << "claster_points.size() - " << claster_points.size() << std::endl;
  for (int ci = 0; ci < claster_points.size(); ci++)
  {
    /*/------------------<TESTING>-----------------------------------
    cv::Mat resimg = imag;
    cv::Point2f correct;

    correct.x = 0;
    correct.y = 0;

    for (int i = 0; i < claster_points.size(); i++)
    {
      p1.x = claster_points.at(i).x - minp.x;
      p1.y = claster_points.at(i).y - minp.y;
      p2.x = claster_points.at(i).x - minp.x + resolution / reduseres;
      p2.y = claster_points.at(i).y - minp.y + resolution / reduseres;

      p1 += correct;
      p2 += correct;

      cv::Mat s_imag = claster_samples.at(i);
      cv::cvtColor(s_imag, s_imag, cv::COLOR_BGR2RGB);
      s_imag.convertTo(s_imag, CV_8UC3);
      s_imag.copyTo(resimg(cv::Rect(p1.x, p1.y, resolution / reduseres, resolution / reduseres)));
      rectangle(resimg, p1, p2, cv::Scalar(0, 255, 0), 1);


      p1.x = p1.x + (resolution / reduseres) / 2;
      p1.y = p1.y + (resolution / reduseres) / 2;

      if (i == ci)
        cv::circle(resimg, p1, 1, cv::Scalar(0, 0, 255), 1);
    }

    cv::resize(resimg, resimg, cv::Size(resimg.cols * 5, resimg.rows * 5), cv::InterpolationFlags::INTER_CUBIC);
    imshow("resimg", resimg);
    cv::waitKey(0);

    //------------------</TESTING>-----------------------------------*/

    sample_np.clear();
    for (int i = 0; i < all_npforpoints.size(); i++)
      sample_np.push_back(all_npforpoints.at(i).at(ci));

    alt_claster_points.push_back(draw_map_prob(sample_np, mpoints));
  }

  int half_range = 8;

  for (int i = 0; i < alt_claster_points.size(); i++)
  {
    for (int ci = 0; ci < alt_claster_points.at(i).size(); ci++)
    {
      chain.clear();
      intpoint bufch;
      bufch.ipoint = i;
      bufch.mpoint = alt_claster_points.at(i).at(ci);
      chain.push_back(bufch);

      for (int j = i + 1; j < alt_claster_points.size(); j++)
      {
        cv::Point2f dp;
        dp.x = claster_points.at(i).x - claster_points.at(j).x;
        dp.y = claster_points.at(i).y - claster_points.at(j).y;

        for (int cj = 0; cj < alt_claster_points.at(j).size(); cj++)
        {
          if (abs(alt_claster_points.at(i).at(ci).x - alt_claster_points.at(j).at(cj).x - dp.x) < half_range && abs(alt_claster_points.at(i).at(ci).y - alt_claster_points.at(j).at(cj).y - dp.y) == 1)
          {
            bufch.ipoint = j;
            bufch.mpoint = alt_claster_points.at(j).at(cj);
            chain.push_back(bufch);
            break;
          }
        }
      }
      chains.push_back(chain);
    }
  }

  int maxchains = 0;
  for (int i = 0; i < chains.size(); i++)
  {
    if (maxchains < chains.at(i).size())
      maxchains = i;
  }

  // maxchains = 0;
  cv::Mat resimg = imag;
  cv::Mat resimg2(resimg.rows, resimg.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  cv::Mat res2x(resimg.rows, resimg.cols * 2, CV_8UC3, cv::Scalar(0, 0, 0));

  p1.y = minp.y + (maxp.y - minp.y) / 2 - half_imgsize;
  p1.x = minp.x + (maxp.x - minp.x) / 2 - half_imgsize;

  p2.y = minp.y + (maxp.y - minp.y) / 2 + half_imgsize;
  p2.x = minp.x + (maxp.x - minp.x) / 2 + half_imgsize;

  bufimg = frame(cv::Range(p1.y, p2.y), cv::Range(p1.x, p2.x));

  cv::cvtColor(bufimg, bufimg, cv::COLOR_BGR2RGB);
  bufimg.convertTo(bufimg, CV_8UC3);
  bufimg.copyTo(resimg2(cv::Rect(0, 0, bufimg.cols, bufimg.rows)));

  for (int i = 0; i < claster_points.size(); i++)
  {
    std::cout << "p1.x - " << p1.x << std::endl;
    p1.x = claster_points.at(i).x - minp.x - (maxp.x - minp.x) / 2 + half_imgsize;
    p1.y = claster_points.at(i).y - minp.y - (maxp.y - minp.y) / 2 + half_imgsize;
    p2.x = p1.x + resolution / reduseres;
    p2.y = p1.y + resolution / reduseres;

    rectangle(resimg2, p1, p2, cv::Scalar(0, 0, 255), 1);
  }

  for (int i = 0; i < chains.at(maxchains).size(); i++)
  {

    size_t R = rand() % 255;
    size_t G = rand() % 255;
    size_t B = rand() % 255;

    // std::cout << "chains.at(" << maxchains << ").at(" << i << ").mpoint - " << chains.at(maxchains).at(i).mpoint << std::endl;
    // std::cout << "chains.at(" << maxchains << ").at(" << i << ").ipoint - " << chains.at(maxchains).at(i).ipoint << std::endl;
    // std::cout << "------------------------------" << std::endl;

    p1.x = chains.at(maxchains).at(i).mpoint.x;
    p1.y = chains.at(maxchains).at(i).mpoint.y;
    p2.x = p1.x + resolution / reduseres;
    p2.y = p1.y + resolution / reduseres;

    rectangle(resimg, p1, p2, cv::Scalar(R, G, B), 1);

    p1.x = claster_points.at(chains.at(maxchains).at(i).ipoint).x - minp.x - (maxp.x - minp.x) / 2 + half_imgsize;
    p1.y = claster_points.at(chains.at(maxchains).at(i).ipoint).y - minp.y - (maxp.y - minp.y) / 2 + half_imgsize;
    p2.x = p1.x + resolution / reduseres;
    p2.y = p1.y + resolution / reduseres;

    rectangle(resimg2, p1, p2, cv::Scalar(R, G, B), 1);
  }

  resimg2.copyTo(res2x(cv::Rect(0, 0, resimg2.cols, resimg2.rows)));

  resimg.copyTo(res2x(cv::Rect(resimg2.cols, 0, resimg2.cols, resimg2.rows)));

  cv::resize(res2x, res2x, cv::Size(res2x.cols * 5, res2x.rows * 5), cv::InterpolationFlags::INTER_CUBIC);

  imshow("res2x", res2x);
  cv::waitKey(0);
}

cv::Mat DetectorMotionV3(std::string pathmodel, torch::DeviceType device_type, cv::Mat frame0, cv::Mat frame, std::vector<ALObject> &objects, int id_frame, bool usedetector)
{
  cv::Scalar class_name_color[20] = {
      cv::Scalar(255, 0, 0),
      cv::Scalar(0, 20, 200),
      cv::Scalar(0, 255, 0),
      cv::Scalar(255, 0, 255),
      cv::Scalar(0, 255, 255),
      cv::Scalar(255, 255, 0),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 0, 200),
      cv::Scalar(100, 0, 255),
      cv::Scalar(255, 0, 100),
      cv::Scalar(30, 20, 200),
      cv::Scalar(25, 255, 0),
      cv::Scalar(255, 44, 255),
      cv::Scalar(88, 255, 255),
      cv::Scalar(255, 255, 39),
      cv::Scalar(255, 255, 255),
      cv::Scalar(200, 46, 200),
      cv::Scalar(100, 79, 255),
      cv::Scalar(200, 46, 150),
      cv::Scalar(140, 70, 205),
  };

  std::vector<std::vector<cv::Point2f>> clasters;
  std::vector<cv::Point2f> motion;
  std::vector<cv::Mat> imgs;

  cv::Mat imageBGR0;
  cv::Mat imageBGR;

  cv::Mat imag;
  cv::Mat imagbuf;
  cv::Mat framebuf = frame_resizing(frame);

  int mpc = 15;   // minimum number of points for a cluster (good value 15)
  int nd = 9;     //(good value 6-15)
  int rcobj = 15; //(good value 15)
  int robj = 17;  //(good value 17)
  int mdist = 15; // maximum distance from cluster center (good value 16)
  int pft = 15;   // points fixation threshold (good value 9)

  float rdet = 50;
  int minsc = 15;
  cv::Mat img;

  std::vector<OBJdetect> detects;

  //--------------------<detection using a classifier>----------
  if (usedetector)
  {
    detects = detectorV4(pathmodel, frame, device_type);

    for (int i = 0; i < objects.size(); i++)
    {
      objects[i].model_center.x = -1;
      objects[i].model_center.y = -1;
    }

    for (int i = 0; i < detects.size(); i++)
    {
      if (detects.at(i).type != "a")
      {
        detects.erase(detects.begin() + i);
        i--;
      }
    }

    for (int i = 0; i < detects.size(); i++)
    {
      std::vector<cv::Point2f> claster_points;
      claster_points.push_back(detects.at(i).detect);
      imagbuf = framebuf;
      img = imagbuf(cv::Range(detects.at(i).detect.y - half_imgsize, detects.at(i).detect.y + half_imgsize), cv::Range(detects.at(i).detect.x - half_imgsize, detects.at(i).detect.x + half_imgsize));
      cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
      img.convertTo(img, CV_8UC3);

      ALObject obj(objects.size(), detects.at(i).type, claster_points, img);
      obj.model_center = detects.at(i).detect;
      obj.claster_center = detects.at(i).detect;
      obj.rectangle = detects.at(i).rectangle;
      // obj.track_points.push_back(detects.at(i).detect);
      // obj.push_track_point(detects.at(i).detect);

      float rm = rcobj * 1.0 * resolution / reduseres;
      int n = -1;

      if (objects.size() > 0)
      {
        rm = sqrt(pow((objects[0].claster_center.x - obj.claster_center.x), 2) + pow((objects[0].claster_center.y - obj.claster_center.y), 2));
        // rm = sqrt(pow((objects[0].proposed_center().x - obj.claster_center.x), 2) + pow((objects[0].proposed_center().y - obj.claster_center.y), 2));
        if (rm < rcobj * 1.0 * resolution / reduseres && rm < rcobj)
          n = 0;
      }

      for (int j = 1; j < objects.size(); j++)
      {
        float r = sqrt(pow((objects[j].claster_center.x - obj.claster_center.x), 2) + pow((objects[j].claster_center.y - obj.claster_center.y), 2));
        // float r = sqrt(pow((objects[j].proposed_center().x - obj.claster_center.x), 2) + pow((objects[j].proposed_center().y - obj.claster_center.y), 2));
        if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
        {
          rm = r;
          n = j;
        }
      }

      if (n > -1)
      {
        objects[n].claster_center = obj.model_center;
        objects[n].model_center = obj.model_center;
        objects[n].rectangle = obj.rectangle;
        // objects[n].track_points.push_back(obj.claster_center);
        // objects[n].push_track_point(obj.claster_center);
        objects[n].img = obj.img;
      }
      else
      {
        objects.push_back(obj);
      }
    }
  }
  //--------------------</detection using a classifier>---------

  //--------------------<moution detections>--------------------
  int rows = frame.rows;
  int cols = frame.cols;

  float rwsize;
  float clsize;

  imagbuf = frame;
  if (rows > cols)
  {
    rwsize = resolution * rows * 1.0 / cols;
    clsize = resolution;
  }
  else
  {
    rwsize = resolution;
    clsize = resolution * cols * 1.0 / rows;
  }

  cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rectb(0, 0, resolution, resolution);
  imag = imagbuf(rectb);

  cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
  imag.convertTo(imag, CV_8UC3);

  if (rows > cols)
  {
    rwsize = reduseres * rows * 1.0 / cols;
    clsize = reduseres;
  }
  else
  {
    rwsize = reduseres;
    clsize = reduseres * cols * 1.0 / rows;
  }

  cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
  cv::Rect rect0(0, 0, reduseres, reduseres);
  imageBGR0 = frame0(rect0);
  imageBGR0.convertTo(imageBGR0, CV_8UC1);

  cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

  cv::Rect rect(0, 0, reduseres, reduseres);
  imageBGR = frame(rect);

  imageBGR.convertTo(imageBGR, CV_8UC1);

  cv::Point2f pm;

  for (int y = 0; y < imageBGR0.rows; y++)
  {
    for (int x = 0; x < imageBGR0.cols; x++)
    {
      uchar color1 = imageBGR0.at<uchar>(cv::Point(x, y));
      uchar color2 = imageBGR.at<uchar>(cv::Point(x, y));

      if (((int)color2 - (int)color1) > pft)
      {
        pm.x = x * resolution / reduseres;
        pm.y = y * resolution / reduseres;
        motion.push_back(pm);
      }
    }
  }

  cv::Point2f pt1;
  cv::Point2f pt2;

  for (int i = 0; i < motion.size(); i++) // visualization of the claster_points
  {
    pt1.x = motion.at(i).x;
    pt1.y = motion.at(i).y;

    pt2.x = motion.at(i).x + resolution / reduseres;
    pt2.y = motion.at(i).y + resolution / reduseres;

    rectangle(imag, pt1, pt2, cv::Scalar(255, 255, 255), 1);
  }

  uint16_t ncls = 0;
  uint16_t nobj;

  if (objects.size() > 0)
    nobj = 0;
  else
    nobj = -1;
  //--------------</moution detections>--------------------

  //--------------<layout of motion points by objects>-----
  // deleted
  //--------------</layout of motion points by objects>----

  //--------------<claster creation>-----------------------

  while (motion.size() > 0)
  {
    cv::Point2f pc;

    if (nobj > -1 && nobj < objects.size())
    {
      pc = objects[nobj].claster_center;
      // pc = objects[nobj].proposed_center();
      nobj++;
    }
    else
    {
      pc = motion.at(0);
      motion.erase(motion.begin());
    }

    clasters.push_back(std::vector<cv::Point2f>());
    clasters[ncls].push_back(pc);

    for (int i = 0; i < motion.size(); i++)
    {
      float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
      if (r < nd * 1.0 * resolution / reduseres)
      {
        cv::Point2f cl_c = claster_center(clasters.at(ncls));
        r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
        if (r < mdist * 1.0 * resolution / reduseres)
        {
          clasters.at(ncls).push_back(motion.at(i));
          motion.erase(motion.begin() + i);
          i--;
        }
      }
    }

    int newp;
    do
    {
      newp = 0;

      for (int c = 0; c < clasters[ncls].size(); c++)
      {
        pc = clasters[ncls].at(c);
        for (int i = 0; i < motion.size(); i++)
        {
          float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

          if (r < nd * 1.0 * resolution / reduseres)
          {
            cv::Point2f cl_c = claster_center(clasters.at(ncls));
            r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
            if (r < mdist * 1.0 * resolution / reduseres)
            {
              clasters.at(ncls).push_back(motion.at(i));
              motion.erase(motion.begin() + i);
              i--;
              newp++;
            }
          }
        }
      }
    } while (newp > 0 && motion.size() > 0);

    ncls++;
  }
  //--------------</claster creation>----------------------

  //--------------<clusters to objects>--------------------
  std::cout << "objects.size() - " << objects.size() << std::endl;
  if (objects.size() == 0)
  {
    for (int cls = 0; cls < ncls; cls++)
    {
      if (clasters[cls].size() > mpc) // if there are enough moving points
      {
        cv::Point2f clastercenter = claster_center(clasters[cls]);
        imagbuf = framebuf;
        imagbuf.convertTo(imagbuf, CV_8UC3);
        img = imagbuf(cv::Range(clastercenter.y - half_imgsize, clastercenter.y + half_imgsize), cv::Range(clastercenter.x - half_imgsize, clastercenter.x + half_imgsize));
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        img.convertTo(img, CV_8UC3);
        ALObject obj(objects.size(), "a", clasters[cls], img);
        objects.push_back(obj);
      }
    }
  }
  else
  {
    for (int cls = 0; cls < ncls; cls++)
    {
      if (clasters[cls].size() > minsc)
      {
        std::cout << "cls - " << cls << std::endl;
        cv::Point2f clastercenter = claster_center(clasters[cls]);
        for (int i = 0; i < objects.size(); i++)
        {
          float rm = sqrt(pow((objects[i].claster_center.x - clastercenter.x), 2) + pow((objects[i].claster_center.y - clastercenter.y), 2));

          std::cout << "rm - " << rm << std::endl;

          if (rm < rdet)
          {
            objdeterm(clasters[cls], framebuf, objects[i], id_frame);
          }
        }
      }
    }
  }

  //--------------</clusters to objects>-------------------
  goto jumpto1;
  for (int i = 0; i < objects.size(); i++)
    objects[i].push_track_point(objects[i].claster_center);

  //--------------<objects visualization>--------------------------
  for (int i = 0; i < objects.size(); i++)
  {
    for (int j = 0; j < objects.at(i).claster_points.size(); j++) // visualization of the claster_points
    {
      pt1.x = objects.at(i).claster_points.at(j).x;
      pt1.y = objects.at(i).claster_points.at(j).y;

      pt2.x = objects.at(i).claster_points.at(j).x + resolution / reduseres;
      pt2.y = objects.at(i).claster_points.at(j).y + resolution / reduseres;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    if (objects.at(i).model_center.x > -1) // visualization of the classifier
    {
      pt1.x = objects.at(i).model_center.x - objects.at(i).rectangle.x / 2;
      pt1.y = objects.at(i).model_center.y - objects.at(i).rectangle.y / 2;

      pt2.x = objects.at(i).model_center.x + objects.at(i).rectangle.x / 2;
      pt2.y = objects.at(i).model_center.y + objects.at(i).rectangle.y / 2;

      rectangle(imag, pt1, pt2, class_name_color[objects.at(i).id], 1);
    }

    for (int j = 0; j < objects.at(i).track_points.size(); j++)
      cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color[objects.at(i).id], 2);
  }
//--------------</objects visualization>-------------------------
jumpto1:

  //--------------<clasters visualization>--------------------------
  for (int i = 0; i < ncls; i++)
  {
    for (int j = 0; j < clasters[i].size(); j++) // visualization of the claster_points
    {
      pt1.x = clasters[i].at(j).x;
      pt1.y = clasters[i].at(j).y;

      pt2.x = clasters[i].at(j).x + resolution / reduseres;
      pt2.y = clasters[i].at(j).y + resolution / reduseres;

      rectangle(imag, pt1, pt2, class_name_color[i], 1);
    }
  }
  //--------------</clasters visualization>-------------------------

  //--------------<baseimag>-------------------------------
  cv::Mat baseimag(resolution, resolution + extr, CV_8UC3, cv::Scalar(0, 0, 0));

  for (int i = 0; i < objects.size(); i++)
  {
    std::string text = objects.at(i).obj_type + " ID" + std::to_string(objects.at(i).id);
    cv::Point2f ptext;
    ptext.x = 20;
    ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

    cv::putText(baseimag, // target image
                text,     // text
                ptext,    // top-left position
                1,
                1,
                class_name_color[objects.at(i).id], // font color
                1);

    pt1.x = ptext.x - 1;
    pt1.y = ptext.y - 1 + 10;

    pt2.x = ptext.x + objects.at(i).img.cols + 1;
    pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

    if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
    {
      rectangle(baseimag, pt1, pt2, class_name_color[objects.at(i).id], 1);
      objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
    }
  }

  imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

  cv::Point2f p_idframe;
  p_idframe.x = resolution + extr - 95;
  p_idframe.y = 50;
  cv::putText(baseimag, std::to_string(id_frame), p_idframe, 1, 3, cv::Scalar(255, 255, 255), 2);
  cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
  //--------------</baseimag>-------------------------------

  imshow("Motion", baseimag);
  cv::waitKey(10);

  /*
    al_objs.push_back(objects.at(1));
    if (al_objs.size() > 1)
    {
      draw_compare(al_objs.at(al_objs.size() - 2), al_objs.at(al_objs.size() - 1), class_name_color[al_objs.at(0).id]);
    }
  */

  return baseimag;
}