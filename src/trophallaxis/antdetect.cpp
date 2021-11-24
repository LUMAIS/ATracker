#include "antdetect.hpp"

#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include <tuple>

void testtorch()
{
    torch::Tensor tensor = torch::eye(3);
    std::cout << tensor << std::endl;
    std::cout<<"testtorch() - OK!!"<<std::endl;
}

int testmodule (std::string strpath)
{
    torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(strpath);
  }
  catch (const c10::Error& e) {
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

    p2.x = p1.x+size;
    p2.y = p1.y;

    p3.x = p2.x;
    p3.y = p2.y+size;

    p4.x = p3.x-size;
    p4.y = p3.y;

    p1.x = p1.x/koef;
    p1.y = p1.y/koef;

    p2.x = p2.x/koef;
    p2.y = p2.y/koef;

    p3.x = p3.x/koef;
    p3.y = p3.y/koef;

    p4.x = p4.x/koef;
    p4.y = p4.y/koef;

    u_char R, G, B;

    if(detect == false)
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

std::vector<cv::Point2f> detectorT (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type)
{ 
  std::vector<std::string> labels;
  labels.push_back("more than one ant");
  labels.push_back("no ants");
  labels.push_back("one ant");
  labels.push_back("trophallaxis");

  float koef = frame.rows*1.0/1000;

  //std::cout << koef << '\n';

  int pinput = 175;
  int step = 100;

  cv::Point2f pd;
  cv::Point2f p1;
  p1.x = 0;
  p1.y = 0;

  cv::Mat imageBGR;
  cv::Mat bufimageBGR;

  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsforcircle;
  
  while(p1.y + step < frame.rows)
  {
    bool detect = false;
    imageBGR = frame(cv::Rect(p1.x, p1.y, pinput, pinput));
    //cv::resize(imageBGR, imageBGR, cv::Size(pinput, pinput), cv::INTER_CUBIC);
    cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
    imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
    auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
    input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();

    //torch::DeviceType device_type = torch::kCPU; 
    //if (torch::cuda::is_available()) {
    //  device_type = torch::kCUDA;
    //}

    input_tensor = input_tensor.to(device_type);
    std::vector<torch::jit::IValue> input;
    input.emplace_back(input_tensor);
    at::Tensor output = module.forward(input).toTensor();
  
    //std::cout << labels[output.argmax(1).item().toInt()] << '\n';

    if(labels[output.argmax(1).item().toInt()] == "trophallaxis")
    {
      pd.x = p1.x+pinput/2;
      pd.y = p1.y+pinput/2;
      detects.push_back(pd);
      pd.x = pd.x/koef;
      pd.y = pd.y/koef;
      detectsforcircle.push_back(pd);
      detect = true;
    }

    cv::resize(frame, bufimageBGR,cv::Size(1000, 1000),cv::InterpolationFlags::INTER_CUBIC);
    drawrec(bufimageBGR,p1,pinput,detect,koef);

    for(int j=0; j<detects.size(); j++)
        cv::circle(bufimageBGR, detectsforcircle.at(j), 5, cv::Scalar(0, 0, 255), 10);
      
    imshow("Detector", bufimageBGR);
    cv::waitKey(10);

    p1.x += step;
    if(p1.x + pinput >= frame.cols)
    {
        p1.x = 0;
        p1.y += step;
    }
  }
  return detects;
}

std::vector<cv::Mat> LoadVideo (const std::string & paths){
  std::vector<cv::Mat> d_images;
  cv::VideoCapture cap(paths);

  if (!cap.isOpened()) {
			std::cout << "[StubVideoGrabber]: Cannot open the video file"<<std::endl;
  }
	else
	{
		cv::Mat framebuf;
		for (int frame_count = 0; frame_count < cap.get(cv::CAP_PROP_FRAME_COUNT); frame_count++) 
		{
      cap.set(cv::CAP_PROP_POS_FRAMES, frame_count);
          	
      if (!cap.read(framebuf)) {
				std::cout << "[LoadVideo]: Failed to extract the frame "<<frame_count<<std::endl;
      }
			else
			{
				//std::string ty =  type2str(framebuf.type());
				//printf("Matrix 1: %s(%d) %dx%d \n", ty.c_str(),framebuf.type(), framebuf.cols, framebuf.rows );
				
				cv::Mat frame;

				cv::cvtColor(framebuf, frame, cv::COLOR_RGB2GRAY);
				
				d_images.push_back(frame);

				std::cout<< "[LoadVideo]: Success to extracted the frame "<<frame_count<<std::endl;
			}
    	}
	}
  return d_images;
}

std::vector<cv::Point2f> detectorT2 (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type)
{
  int resolution = 992;
  int pointsdelta = 30;
  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsbuf;
  cv::Mat imageBGR;
  cv::resize(frame, imageBGR,cv::Size(992, 992),cv::InterpolationFlags::INTER_CUBIC);

  cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
  imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
  auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();
  input_tensor = input_tensor.to(device_type);

  std::vector<torch::jit::IValue> input;
  input.emplace_back(input_tensor);
  auto outputs = module.forward(input).toTuple();
  torch::Tensor detections = outputs->elements()[0].toTensor();
//------------------------------------------------------------------------

  int item_attr_size = 13;
  int batch_size = detections.size(0);
  auto num_classes = detections.size(2);// - item_attr_size;

  auto conf_thres = 0.60 ;
  auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

  std::vector<std::vector<Detection>> output;
  output.reserve(batch_size);

for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        // apply constrains to get filtered detections for current image
        auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes});

        // if none detections remain then skip and start to process next image

        std::cout << "det.size(0) - " << det.size(0) << '\n';
        if (0 == det.size(0)) {
            continue;
        }

        for (size_t i=0; i < det.size(0); ++ i)
        {

          float left = det[i][0].item().toFloat() * imageBGR.cols / resolution;
          float top = det[i][1].item().toFloat() * imageBGR.rows / resolution;
          //float right = det[i][2].item().toFloat() * imageBGR.cols / resolution;
          //float bottom = det[i][3].item().toFloat() * imageBGR.rows / resolution;

          detectsbuf.push_back(cv::Point(left,top));
        }
}
//-------------------------------------------------------------------------

  for(size_t i=0; i < detectsbuf.size(); i++)
  {
    if(detectsbuf.at(i).x > 0) 
    {
      for(size_t j=0; j < detectsbuf.size(); j++)
      {
        if(detectsbuf.at(j).x > 0 && i != j)
        {
          if(sqrt(pow(detectsbuf.at(i).x - detectsbuf.at(j).x,2) + pow(detectsbuf.at(i).y - detectsbuf.at(j).y,2)) < pointsdelta)
          {
            detectsbuf.at(i).x = (detectsbuf.at(i).x + detectsbuf.at(j).x)*1.0/2;
            detectsbuf.at(i).y = (detectsbuf.at(i).y + detectsbuf.at(j).y)*1.0/2;
            detectsbuf.at(j).x = -1;
          }
        }
      }
    }
  }

  for(size_t i=0; i < detectsbuf.size(); i++)
  {
    if(detectsbuf.at(i).x >=0)
    {
      detects.push_back(detectsbuf.at(i));
    }
  }

  for(size_t i=0; i < detects.size(); i++)
  {
    cv::circle(imageBGR, detects.at(i), 3, cv::Scalar(0, 0, 255), 3);
  }

  imshow("DetectorT2", imageBGR);
  cv::waitKey(10);
  
  return detects;
}

std::vector<cv::Point2f> detectorT3 (torch::jit::script::Module module, cv::Mat frame, torch::DeviceType device_type)
{
  int resolution = 992;
  int pointsdelta = 30;
  std::vector<cv::Point2f> detects;
  std::vector<cv::Point2f> detectsbuf;
  cv::Mat imageBGR;
  cv::resize(frame, imageBGR,cv::Size(992, 992),cv::InterpolationFlags::INTER_CUBIC);

  cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
  imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
  auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();
  input_tensor = input_tensor.to(device_type);
  
  //----------------------------------
  if (device_type != torch::kCPU) {
      //module.to(torch::kHalf);
      input_tensor = input_tensor.to(torch::kHalf);
      std::cout<<"....to(torch::kHalf)!!!"<<std::endl;
  }
  //----------------------------------

  std::vector<torch::jit::IValue> input;
  input.emplace_back(input_tensor);
  auto outputs = module.forward(input).toTuple();
  torch::Tensor detections = outputs->elements()[0].toTensor();

  int item_attr_size = 13;
  int batch_size = detections.size(0);
  auto num_classes = detections.size(2);// - item_attr_size;

  auto conf_thres = 0.60 ;
  auto conf_mask = detections.select(2, 4).ge(conf_thres).unsqueeze(2);

  std::vector<std::vector<Detection>> output;
  output.reserve(batch_size);

for (int batch_i = 0; batch_i < batch_size; batch_i++) {
        // apply constrains to get filtered detections for current image
        auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes});

        // if none detections remain then skip and start to process next image

        //std::cout << "det.size(0) - " << det.size(0) << '\n';
        if (0 == det.size(0)) {
            continue;
        }

        for (size_t i=0; i < det.size(0); ++ i)
        {
            float left = det[i][0].item().toFloat() * imageBGR.cols / resolution;
            float top = det[i][1].item().toFloat() * imageBGR.rows / resolution;
            //float right = det[i][2].item().toFloat() * imageBGR.cols / resolution;
            //float bottom = det[i][3].item().toFloat() * imageBGR.rows / resolution;
            detectsbuf.push_back(cv::Point(left,top));
        }
}
  for(size_t i=0; i < detectsbuf.size(); i++)
  {
    if(detectsbuf.at(i).x > 0) 
    {
      for(size_t j=0; j < detectsbuf.size(); j++)
      {
        if(detectsbuf.at(j).x > 0 && i != j)
        {
          if(sqrt(pow(detectsbuf.at(i).x - detectsbuf.at(j).x,2) + pow(detectsbuf.at(i).y - detectsbuf.at(j).y,2)) < pointsdelta)
          {
            detectsbuf.at(i).x = (detectsbuf.at(i).x + detectsbuf.at(j).x)*1.0/2;
            detectsbuf.at(i).y = (detectsbuf.at(i).y + detectsbuf.at(j).y)*1.0/2;
            detectsbuf.at(j).x = -1;
          }
        }
      }
    }
  }

  for(size_t i=0; i < detectsbuf.size(); i++)
  {
    if(detectsbuf.at(i).x >=0)
    {
      detects.push_back(detectsbuf.at(i));
    }
  }

   for(size_t i=0; i < detects.size(); i++)
  {
    cv::circle(imageBGR, detects.at(i), 3, cv::Scalar(0, 0, 255), 3);
  }

  imshow("DetectorT3", imageBGR);
  cv::waitKey(10);
  
  return detects;
}