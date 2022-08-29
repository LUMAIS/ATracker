#include <memory>
#include <tuple>
#include <filesystem>
//#include <sys/time.h>
#include <ctime>

#include <torch/script.h> // One-stop header.
#include "tracker.hpp"

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;
using std::to_string;
using cv::Scalar;
namespace fs = std::filesystem;


int resolution = 992;  // 1200;  // frame size for model 992
const float kres = resolution / 1200.f;

uint16_t extr = 204;  // sidebar size
uint16_t const half_imgsize = roundf(80 * kres); // area half size for a moving object
const uint16_t model_resolution = resolution;  // frame resizing for model (992)
uint16_t frame_resolution = resolution;//frame frame_resolution
float reduseres = roundf(400 * kres);   // (good value 248)
uint16_t color_threshold = 80; // 65-70

string class_name[] = {"ta", "a", "ah", "tl", "l", "fn", "u", "p", "b"};  // class_name[9]
// string class_name[] = {"a", "ah", "ta", "l", "tl", "fn", "p", "b", "u"};  // class_name[9]
// constexpr uint8_t  cname_undefined = (sizeof class_name / sizeof  class_name[0]);  // TODO: validate correctness of this assignment
// // string class_name[] = {"ant": 0,
// //                         "ant-head": 1,
// //                         "trophallaxis-ant": 2,
// //                         "larva": 3,
// //                         "trophallaxis-larva": 4,
// //                         "food-noise": 5,  // fn
// //                         "pupa": 6,
// //                         "barcode": 7} #,"uncategorized": 8}

Scalar class_name_color(uint32_t id, uint8_t clrLow=32, uint8_t clrHigh=223) noexcept
{
	// Scalar class_name_color[9] = {Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255), Scalar(255, 255, 0), Scalar(255, 255, 255), Scalar(200, 0, 200), Scalar(100, 0, 255)};
	static Scalar  baseclrs[20] = {
			Scalar(255, 0, 0),
			Scalar(0, 20, 200),
			Scalar(0, 255, 0),
			Scalar(255, 0, 255),
			Scalar(0, 255, 255),
			Scalar(255, 255, 0),
			Scalar(255, 255, 255),
			Scalar(200, 0, 200),
			Scalar(100, 0, 255),
			Scalar(255, 0, 100),
			Scalar(30, 20, 200),
			Scalar(25, 255, 0),
			Scalar(255, 44, 255),
			Scalar(88, 255, 255),
			Scalar(255, 255, 39),
			Scalar(255, 255, 255),
			Scalar(200, 46, 200),
			Scalar(100, 79, 255),
			Scalar(200, 46, 150),
			Scalar(140, 70, 205),
	};

	const uint8_t  clrRange = clrHigh - clrLow;
	srand(id);  // Display the same objects with the same colors on following frames
	Scalar  clr = id < 20 ? baseclrs[id] : Scalar(clrLow + rand() % clrRange, clrLow + rand() % clrRange, clrLow + rand() % clrRange);
	// cout << "#" << id << " color: " << clr << endl;
	return clr;
}

uint16_t max_u16(float a, float b) noexcept  { return std::max<int16_t>(roundf(a), roundf(b)); }
uint16_t min_u16(float a, float b) noexcept  { return std::min<int16_t>(roundf(a), roundf(b)); }

void testtorch()
{
	torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << endl;
	std::cout << "testtorch() - OK!!" << endl;
}

int testmodule(string strpath)
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

void drawrec(Mat &image, Point2f p1, int size, bool detect, float koef)
{
	Point2f p2;
	Point2f p3;
	Point2f p4;

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

	cv::line(image, p1, p2, Scalar(B, G, R), 2, 8, 0);
	cv::line(image, p2, p3, Scalar(B, G, R), 2, 8, 0);
	cv::line(image, p3, p4, Scalar(B, G, R), 2, 8, 0);
	cv::line(image, p4, p1, Scalar(B, G, R), 2, 8, 0);
}

vector<Mat> LoadVideo(const string &paths, uint16_t startframe, uint16_t getframes)
{
	vector<Mat> d_images;
	cv::VideoCapture cap(paths);

	if (!cap.isOpened())
	{
		std::cout << "[StubVideoGrabber]: Cannot open the video file" << endl;
	}
	else
	{
		Mat framebuf;

		cap.set(cv::CAP_PROP_POS_FRAMES, startframe);

		for (int frame_count = 0; frame_count < cap.get(cv::CAP_PROP_FRAME_COUNT); frame_count++)
		{
			if (frame_count > getframes)
			{
				std::cout << "[LoadVideo]: (int)getframes -  " << (int)getframes << endl;
				break;
			}

			if (!cap.read(framebuf))
			{
				std::cout << "[LoadVideo]: Failed to extract the frame " << frame_count << endl;
			}
			else
			{
				// Mat frame;
				// cv::cvtColor(framebuf, frame, cv::COLOR_RGB2GRAY);
				// cv::cvtColor(framebuf, framebuf, cv::COLOR_RGB2GRAY);

				framebuf.convertTo(framebuf, CV_8UC3);
				uint16_t res;

				if (framebuf.rows > framebuf.cols)
					res = framebuf.rows;
				else
					res = framebuf.cols;

				Mat frame(res, res, CV_8UC3, Scalar(0, 0, 0));

				if (framebuf.rows > framebuf.cols)
					framebuf.copyTo(frame(cv::Rect((framebuf.rows - framebuf.cols) / 2, 0, framebuf.cols, framebuf.rows)));
				else
					framebuf.copyTo(frame(cv::Rect(0, (framebuf.cols - framebuf.rows) / 2, framebuf.cols, framebuf.rows)));

				cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);

				d_images.push_back(frame);
				std::cout << "[LoadVideo]: Success to extracted the frame " << frame_count << endl;
			}
		}
	}
	return d_images;
}

// Cuts frame to rect area
Mat frame_resizing(Mat frame)
{
	uint16_t rows = frame.rows;
	uint16_t cols = frame.cols;

	float rwsize = model_resolution;
	float clsize = model_resolution;

	if (rows > cols)
	{
		rwsize = (float)model_resolution * rows / cols;
		clsize = (float)model_resolution;
	}
	else
	{
		rwsize = (float)model_resolution;
		clsize = (float)model_resolution * cols / rows;
	}

	if (clsize != frame.cols || rwsize != frame.rows)
		cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	if(clsize != rwsize) {
		cv::Rect rect(0, 0, model_resolution, model_resolution);
		return frame(rect);
	}
	return frame;
}

// Cuts frame to rect area
Mat frame_resizingV2(Mat frame, uint framesize)
{
	int rows = frame.rows;
	int cols = frame.cols;

	float rwsize = model_resolution;
	float clsize = model_resolution;

	if (rows > cols)
	{
		rwsize = framesize * rows * 1.0 / cols;
		clsize = framesize;
	}
	else
	{
		rwsize = framesize;
		clsize = framesize * cols * 1.0 / rows;
	}

	if (clsize != frame.cols || rwsize != frame.rows)
		cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	if(clsize != rwsize) {
		cv::Rect rect(0, 0, framesize, framesize);
		return frame(rect);
	}
	return frame;
}

vector<OBJdetect> detectorV4_old(string pathmodel, Mat frame, torch::DeviceType device_type, const float confidence=dftConf)  // 0.5
{
	vector<OBJdetect> obj_detects;
	auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	torch::jit::script::Module module = torch::jit::load(pathmodel);
	std::cout << "Load module +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << endl;
	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	int resolution = 992;
	int pointsdelta = 5;
	vector<Point2f> detects;
	vector<Point2f> detectsCent;
	vector<Point2f> detectsRect;
	vector<uint8_t> Objtype;
	Mat imageBGR;

	// imageBGR = frame_resizing(frame);
	frame.copyTo(imageBGR);
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

	// std::cout<<"input_tensor.to(device_type) - OK"<<endl;
	vector<torch::jit::IValue> input;
	input.emplace_back(input_tensor);
	// std::cout<<"input.emplace_back(input_tensor) - OK"<<endl;

	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	auto outputs = module.forward(input).toTuple();
	// std::cout << "Processing +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << endl;
	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	// std::cout<<"module.forward(input).toTuple() - OK"<<endl;
	torch::Tensor detections = outputs->elements()[0].toTensor();

	// int item_attr_size = 13;
	int batch_size = detections.size(0);
	auto num_classes = detections.size(2); // - item_attr_size;
	auto conf_mask = detections.select(2, 4).ge(confidence).unsqueeze(2);

	vector<vector<Detection>> output;
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
			const auto dcur = det[i];
			float x = dcur[0].item().toFloat() * imageBGR.cols / resolution;
			float y = dcur[1].item().toFloat() * imageBGR.rows / resolution;

			float h = dcur[2].item().toFloat() * imageBGR.cols / resolution;
			float w = dcur[3].item().toFloat() * imageBGR.rows / resolution;

			float wheit = 0;
			Objtype.push_back(8);

			for (int j = 4; j < det.size(1); j++)
			{
				if (dcur[j].item().toFloat() > wheit)
				{
					wheit = dcur[j].item().toFloat();
					Objtype.at(i) = j - 4;
				}
			}

			detectsCent.push_back(Point(x, y));
			detectsRect.push_back(Point(h, w));
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

		Point2f pt1;
		Point2f pt2;
		Point2f ptext;

		if (detectsCent.at(i).x >= 0)
		{

			OBJdetect obj_buf;

			obj_buf.detect = detectsCent.at(i);
			obj_buf.obj_size = detectsRect.at(i);
			obj_buf.type = class_name[Objtype.at(i)];
			obj_detects.push_back(obj_buf);

			detects.push_back(detectsCent.at(i));
			pt1.x = detectsCent.at(i).x - detectsRect.at(i).x / 2;
			pt1.y = detectsCent.at(i).y - detectsRect.at(i).y / 2;

			pt2.x = detectsCent.at(i).x + detectsRect.at(i).x / 2;
			pt2.y = detectsCent.at(i).y + detectsRect.at(i).y / 2;

			ptext.x = detectsCent.at(i).x - 5;
			ptext.y = detectsCent.at(i).y + 5;

			rectangle(imageBGR, pt1, pt2, class_name_color(Objtype.at(i)), 1);

			cv::putText(imageBGR,                  // target image
									class_name[Objtype.at(i)], // text
									ptext,                     // top-left position
									1,
									0.8,
									class_name_color(Objtype.at(i)), // font color
									1);
		}
	}

	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	return obj_detects;
}

vector<OBJdetect> detectorV4(string pathmodel, Mat frame, torch::DeviceType device_type, const float confidence)
{
	vector<OBJdetect> obj_detects;
	auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	static torch::jit::script::Module module = torch::jit::load(pathmodel);
	std::cout << "Model loading time: +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec
		<< "ms; " << "confidence threshold : " << confidence << endl;
	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	uint16_t pointsdelta = 5;
	vector<Point2f> detects;
	vector<Point2f> detectsCent;
	vector<Point2f> detectsRect;
	vector<uint8_t> Objtype;
	Mat imageBGR;

	// imageBGR = frame_resizing(frame);
	// std::cout << "Frame size: " << frame.cols  << "x" << frame.rows << endl;
	//cv::resize(frame, imageBGR, cv::Size(992, 992), cv::InterpolationFlags::INTER_CUBIC);
	frame.copyTo(imageBGR);

	// Ensure that the input image size corresponds to the model size

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

	// std::cout<<"input_tensor.to(device_type) - OK"<<endl;
	vector<torch::jit::IValue> input;
	input.emplace_back(input_tensor);
	// std::cout<<"input.emplace_back(input_tensor) - OK"<<endl;

	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	auto outputs = module.forward(input).toTuple();
	// cout << "module.forward(input).toTuple() - OK" << endl;
	std::cout << "Raw object detection +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << endl;
	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	torch::Tensor detections = outputs->elements()[0].toTensor();

	// int item_attr_size = 13;
	int batch_size = detections.size(0);
	auto num_classes = detections.size(2); // - item_attr_size;
	auto conf_mask = detections.select(2, 4).ge(confidence).unsqueeze(2);

	vector<vector<Detection>> output;
	output.reserve(batch_size);

	for (int batch_i = 0; batch_i < batch_size; batch_i++)
	{
		// apply constrains to get filtered detections for current image
		const auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes});
		// if none detections remain then skip and start to process next image

		if (0 == det.size(0))
			continue;

		for (size_t i = 0; i < det.size(0); ++i)
		{
			const auto dcur = det[i];
			float x = dcur[0].item().toFloat() * imageBGR.cols / model_resolution;
			float y = dcur[1].item().toFloat() * imageBGR.rows / model_resolution;

			float w = dcur[2].item().toFloat() * imageBGR.cols / model_resolution;
			float h = dcur[3].item().toFloat() * imageBGR.rows / model_resolution;

			float conf = dcur[4].item().toFloat();

			// // Fetch class scores and assign the object to the most probable class
			// const auto cscores = dcur + 5;
			// Objtype.push_back(cname_undefined);
			Objtype.push_back(8);  // Add max index
			float cscore = 0;

			// TODO: reimplement
			for (int j = 5; j < det.size(1); j++)
			{
				if (dcur[j].item().toFloat() > cscore)
				{
					cscore = dcur[j].item().toFloat();
					Objtype.at(i) = j - 4;
				}
			}

			detectsCent.push_back(Point2f(x, y));
			detectsRect.push_back(Point2f(w, h));
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

		Point2f pt1;
		Point2f pt2;
		Point2f ptext;

		if (detectsCent.at(i).x >= 0)
		{

			OBJdetect obj_buf;

			obj_buf.detect = detectsCent.at(i);
			obj_buf.obj_size = detectsRect.at(i);
			obj_buf.type = class_name[Objtype.at(i)];
			obj_detects.push_back(obj_buf);

			detects.push_back(detectsCent.at(i));
			pt1.x = detectsCent.at(i).x - detectsRect.at(i).x / 2;
			pt1.y = detectsCent.at(i).y - detectsRect.at(i).y / 2;

			pt2.x = detectsCent.at(i).x + detectsRect.at(i).x / 2;
			pt2.y = detectsCent.at(i).y + detectsRect.at(i).y / 2;

			ptext.x = detectsCent.at(i).x - 5;
			ptext.y = detectsCent.at(i).y + 5;

			rectangle(imageBGR, pt1, pt2, class_name_color(Objtype.at(i)), 1);

			cv::putText(imageBGR,                  // target image
									class_name[Objtype.at(i)], // text
									ptext,                     // top-left position
									1,
									0.8,
									class_name_color(Objtype.at(i)), // font color
									1);
		}
	}

	cout << "detectorV4(), detected: " << obj_detects.size() << endl;
	// millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	return obj_detects;
}

Point2f cluster_center(vector<Point2f> cluster_points)
{
	Point2f cluster_center;

	int powx = 0;
	int powy = 0;

	for (int i = 0; i < cluster_points.size(); i++)
	{
		powx = powx + pow(cluster_points[i].x, 2);
		powy = powy + pow(cluster_points[i].y, 2);
	}

	cluster_center.x = sqrt(powx / cluster_points.size());
	cluster_center.y = sqrt(powy / cluster_points.size());

	return cluster_center;
}

size_t samples_compV2(Mat sample1, Mat sample2)
{
	size_t npf = 0;

	sample1.convertTo(sample1, CV_8UC1);
	sample2.convertTo(sample2, CV_8UC1);

	Point2f pm;

	for (int y = 0; y < sample1.rows; y++)
	{
		for (int x = 0; x < sample1.cols; x++)
		{
			uchar color1 = sample1.at<uchar>(Point(x, y));
			uchar color2 = sample2.at<uchar>(Point(x, y));

			npf += abs((int)color2 - (int)color1);
		}
	}

	return npf;
}

Mat draw_object(ALObject obj, ALObject obj2, Scalar color)
{
	int wh = 800;
	float hp = 1.0 * (resolution / reduseres) * wh / (2 * half_imgsize);

	Mat imag(wh, wh, CV_8UC3, Scalar(0, 0, 0));
	Mat imag2(wh, wh, CV_8UC3, Scalar(0, 0, 0));
	Mat imgres(wh, wh * 3, CV_8UC3, Scalar(0, 0, 0));
	Mat imgbuf;

	Point2f bufp;
	Point2f pt1;
	Point2f pt2;

	Mat imgsm;

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

		for (int i = 0; i < obj.cluster_points.size(); i++)
		{
			bufp.x = ((obj.cluster_points.at(i).x - obj.cluster_center.x) * wh / (2 * half_imgsize) + wh / 2);
			bufp.y = ((obj.cluster_points.at(i).y - obj.cluster_center.y) * wh / (2 * half_imgsize) + wh / 2);

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

	for (int i = 0; i < obj2.cluster_points.size(); i++)
	{
		bufp.x = ((obj2.cluster_points.at(i).x - obj2.cluster_center.x) * wh / (2 * half_imgsize) + wh / 2);
		bufp.y = ((obj2.cluster_points.at(i).y - obj2.cluster_center.y) * wh / (2 * half_imgsize) + wh / 2);

		pt1.x = bufp.x;
		pt1.y = bufp.y;

		pt2.x = bufp.x + hp;
		pt2.y = bufp.y + hp;

		rectangle(imag2, pt1, pt2, color, 1);
	}

	cv::cvtColor(imag2, imag2, cv::COLOR_BGR2RGB);

	//-------using matchTemplate--bad idea---------------------
	/*
		Mat result;
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

			rectangle(imag2, pt1, pt2, Scalar(R, G, B), 1);
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

		std::cout << "minnp - " << minnp << endl;

		Mat imgsm = obj2.samples.at(ns2);
		cv::resize(imgsm, imgsm, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

		bufp.x = 1.0 * obj.coords.at(ns1).x * wh / (2 * half_imgsize);
		bufp.y = 1.0 * obj.coords.at(ns1).y * wh / (2 * half_imgsize);

		Point2f c_cir;

		c_cir.x = (half_imgsize - (obj2.coords.at(ns2).x - obj.coords.at(ns1).x)) * wh / (2 * half_imgsize);
		c_cir.y = (half_imgsize - (obj2.coords.at(ns2).y - obj.coords.at(ns1).y)) * wh / (2 * half_imgsize);

		std::cout << "c_cir.y - " << c_cir.y << endl;
		std::cout << "c_cir.x - " << c_cir.x << endl;
		std::cout << "half_imgsize - " << half_imgsize << endl;
		// Scalar  objClr = Scalar(rand() % 255, rand() % 255, rand() % 255);
		Scalar  objClr = class_name_color(obj.id);

		if (c_cir.y < 0 || c_cir.x < 0 || c_cir.y > 2 * half_imgsize * wh / (2 * half_imgsize) || c_cir.x > 2 * half_imgsize * wh / (2 * half_imgsize))
			goto dell;

		cv::circle(imag, c_cir, 3, objClr, 1);

		// imgsm.copyTo(imag(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

		pt1.x = bufp.x;
		pt1.y = bufp.y;

		pt2.x = bufp.x + hp;
		pt2.y = bufp.y + hp;

		rectangle(imag, pt1, pt2, objClr, 1);

		//---

		bufp.x = 1.0 * obj2.coords.at(ns2).x * wh / (2 * half_imgsize);
		bufp.y = 1.0 * obj2.coords.at(ns2).y * wh / (2 * half_imgsize);
		// imgsm.copyTo(imag2(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

		pt1.x = bufp.x;
		pt1.y = bufp.y;

		pt2.x = bufp.x + hp;
		pt2.y = bufp.y + hp;

		rectangle(imag2, pt1, pt2, objClr, 1);

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

Mat draw_compare(ALObject obj, ALObject obj2, Scalar color)
{
	int wh = 800;
	float hp = 1.0 * (resolution / reduseres) * wh / (2 * half_imgsize);

	Mat imag(wh, wh, CV_8UC3, Scalar(0, 0, 0));
	Mat imag2(wh, wh, CV_8UC3, Scalar(0, 0, 0));
	Mat imgres(wh, wh * 2, CV_8UC3, Scalar(0, 0, 0));
	Mat imgbuf;

	Mat sample;
	Mat sample2;

	Point2f bufp;
	Point2f pt1;
	Point2f pt2;

	vector<Point2f> center;
	vector<int> npsamples;

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

				sample = obj.img(cv::Range(obj2.coords.at(i).y + step_y * st * obj2.img.rows / imag.rows
					, min_u16(obj.img.rows, obj2.coords.at(i).y + step_y * st * obj2.img.rows / imag.rows + resolution / reduseres))
					, cv::Range(obj2.coords.at(i).x + step_x * st * obj2.img.cols / imag.cols
					,  min_u16(obj.img.cols, obj2.coords.at(i).x + step_x * st * obj2.img.cols / imag.cols + resolution / reduseres)));

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

	Mat imgcenter(2 * half_imgsize, 2 * half_imgsize, CV_8UC1, Scalar(0, 0, 0));

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

Mat trackingMotV2(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, bool usedetector, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;
	Mat framebuf = frame;

	int mpc = 15;   // minimum number of points for a cluster (good value 15)
	int nd = 9;     //(good value 6-15)
	int rcobj = 17; //(good value 15)
	int robj = 17;  //(good value 17)
	int mdist = 10; // maximum distance from cluster center (good value 10)
	int pft = 9;    // points fixation threshold (good value 9)

	Mat img;

	vector<OBJdetect> detects;

	//--------------------<detection using a classifier>----------
	if (usedetector)
	{

		detects = detectorV4(pathmodel, frame_resizing(frame), device_type, confidence);

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
			vector<Point2f> cluster_points;
			cluster_points.push_back(detects.at(i).detect);
			imagbuf = frame_resizing(framebuf);
			img = imagbuf(cv::Range(max_u16(detects.at(i).detect.y - half_imgsize), min_u16(imagbuf.rows, detects.at(i).detect.y + half_imgsize))
				, cv::Range(max_u16(detects.at(i).detect.x - half_imgsize), min_u16(imagbuf.cols, detects.at(i).detect.x + half_imgsize)));
			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), detects.at(i).type, cluster_points, img);
			obj.model_center = detects.at(i).detect;
			obj.cluster_center = detects.at(i).detect;
			obj.obj_size = detects.at(i).obj_size;
			// obj.track_points.push_back(detects.at(i).detect);
			// obj.push_track_point(detects.at(i).detect);

			float rm = rcobj * 1.0 * resolution / reduseres;
			int n = -1;

			if (objects.size() > 0)
			{
				rm = sqrt(pow((objects[0].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[0].cluster_center.y - obj.cluster_center.y), 2));
				// rm = sqrt(pow((objects[0].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[0].proposed_center().y - obj.cluster_center.y), 2));
				if (rm < rcobj * 1.0 * resolution / reduseres && rm < rcobj)
					n = 0;
			}

			for (int j = 1; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
				// float r = sqrt(pow((objects[j].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[j].proposed_center().y - obj.cluster_center.y), 2));
				if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
				{
					rm = r;
					n = j;
				}
			}

			if (n > -1)
			{
				// Update existing object
				auto& tobj = objects.at(n);
				tobj.cluster_center = obj.model_center;
				tobj.model_center = obj.model_center;
				tobj.obj_size = obj.obj_size;
				// tobj.track_points.push_back(obj.cluster_center);
				// tobj.push_track_point(obj.cluster_center);
				tobj.img = obj.img;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < id_frame && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
					, obj.obj_size.x, obj.obj_size.y});
			}
			else
			{
				assert(obj.traces.empty() && "Unexpected traces");
				obj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
					, obj.obj_size.x, obj.obj_size.y});
				objects.push_back(obj);  // New object
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

	Point2f pm;

	for (int y = 0; y < imageBGR0.rows; y++)
	{
		for (int x = 0; x < imageBGR0.cols; x++)
		{
			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
			uchar color2 = imageBGR.at<uchar>(Point(x, y));

			if (((int)color2 - (int)color1) > pft)
			{
				pm.x = x * resolution / reduseres;
				pm.y = y * resolution / reduseres;
				motion.push_back(pm);
			}
		}
	}

	Point2f pt1;
	Point2f pt2;

	for (int i = 0; i < motion.size(); i++) // visualization of the cluster_points
	{
		pt1.x = motion.at(i).x;
		pt1.y = motion.at(i).y;

		pt2.x = motion.at(i).x + resolution / reduseres;
		pt2.y = motion.at(i).y + resolution / reduseres;

		rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
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
			objects[i].cluster_points.clear();
		}

		for (int i = 0; i < motion.size(); i++)
		{
			for (int j = 0; j < objects.size(); j++)
			{
				if (objects[j].model_center.x < 0)
					continue;

				if (i < 0)
					break;

				if ((motion.at(i).x < (objects[j].model_center.x + objects[j].obj_size.x / 2)) && (motion.at(i).x > (objects[j].model_center.x - objects[j].obj_size.x / 2)) && (motion.at(i).y < (objects[j].model_center.y + objects[j].obj_size.y / 2)) && (motion.at(i).y > (objects[j].model_center.y - objects[j].obj_size.y / 2)))
				{
					objects[j].cluster_points.push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		float rm = rcobj * 1.0 * resolution / reduseres;

		for (int i = 0; i < motion.size(); i++)
		{
			rm = sqrt(pow((objects[0].cluster_center.x - motion.at(i).x), 2) + pow((objects[0].cluster_center.y - motion.at(i).y), 2));

			int n = -1;
			if (rm < rcobj * 1.0 * resolution / reduseres)
				n = 0;

			for (int j = 1; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - motion.at(i).x), 2) + pow((objects[j].cluster_center.y - motion.at(i).y), 2));
				if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
				{
					rm = r;
					n = j;
				}
			}

			if (n > -1)
			{
				objects[n].cluster_points.push_back(motion.at(i));
				motion.erase(motion.begin() + i);
				i--;
				objects[n].center_determine(id_frame, false);
				if (i < 0)
					break;
			}
		}
	}

	for (int j = 0; j < objects.size(); j++)
	{

		objects[j].center_determine(id_frame, false);

		Point2f clustercenter = objects[j].cluster_center;

		imagbuf = frame_resizing(framebuf);
		imagbuf.convertTo(imagbuf, CV_8UC3);

		if (clustercenter.y - half_imgsize < 0)
			clustercenter.y = half_imgsize + 1;

		if (clustercenter.x - half_imgsize < 0)
			clustercenter.x = half_imgsize + 1;

		if (clustercenter.y + half_imgsize > imagbuf.rows)
			clustercenter.y = imagbuf.rows - 1;

		if (clustercenter.x + half_imgsize > imagbuf.cols)
			clustercenter.x = imagbuf.cols - 1;

		img = imagbuf(cv::Range(max_u16(clustercenter.y - half_imgsize), min_u16(imagbuf.rows, clustercenter.y + half_imgsize))
			, cv::Range(max_u16(clustercenter.x - half_imgsize), min_u16(imagbuf.cols, clustercenter.x + half_imgsize)));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_8UC3);
		objects[j].img = img;
		objects[j].samples_creation();
	}
	//--------------</layout of motion points by objects>----

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj > -1 && nobj < objects.size())
		{
			pc = objects[nobj].cluster_center;
			// pc = objects[nobj].proposed_center();
			nobj++;
		}
		else
		{
			pc = motion.at(0);
			motion.erase(motion.begin());
		}

		clusters.push_back(vector<Point2f>());
		clusters[ncls].push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
			if (r < nd * 1.0 * resolution / reduseres)
			{
				Point2f cl_c = cluster_center(clusters.at(ncls));
				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
				if (r < mdist * 1.0 * resolution / reduseres)
				{
					clusters.at(ncls).push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		int newp;
		do
		{
			newp = 0;

			for (int c = 0; c < clusters[ncls].size(); c++)
			{
				pc = clusters[ncls].at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

					if (r < nd * 1.0 * resolution / reduseres)
					{
						Point2f cl_c = cluster_center(clusters.at(ncls));
						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
						if (r < mdist * 1.0 * resolution / reduseres)
						{
							clusters.at(ncls).push_back(motion.at(i));
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
	//--------------</cluster creation>----------------------

	//--------------<clusters to objects>--------------------
	for (int cls = 0; cls < ncls; cls++)
	{
		if (clusters[cls].size() > mpc) // if there are enough moving points
		{

			Point2f clustercenter = cluster_center(clusters[cls]);
			imagbuf = frame_resizing(framebuf);
			imagbuf.convertTo(imagbuf, CV_8UC3);
			img = imagbuf(cv::Range(max_u16(clustercenter.y - half_imgsize), min_u16(imagbuf.rows, clustercenter.y + half_imgsize))
				, cv::Range(max_u16(clustercenter.x - half_imgsize), min_u16(imagbuf.cols, clustercenter.x + half_imgsize)));
			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), "a", clusters[cls], img);
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				float r = sqrt(pow((objects[i].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[i].cluster_center.y - obj.cluster_center.y), 2));
				// float r = sqrt(pow((objects[i].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[i].proposed_center().y - obj.cluster_center.y), 2));
				if (r < robj * 1.0 * resolution / reduseres)
				{
					newobj = false;

					objects[i].img = obj.img;
					objects[i].cluster_points = obj.cluster_points;
					objects[i].center_determine(id_frame, true);
					break;
				}
			}

			if (newobj == true)
				objects.push_back(obj);
		}
	}
	//--------------</clusters to objects>-------------------

	for (int i = 0; i < objects.size(); i++)
		objects[i].push_track_point(objects[i].cluster_center);

	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = objects.at(i).cluster_points.at(j).x;
			pt1.y = objects.at(i).cluster_points.at(j).y;

			pt2.x = objects.at(i).cluster_points.at(j).x + resolution / reduseres;
			pt2.y = objects.at(i).cluster_points.at(j).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		if (objects.at(i).model_center.x > -1) // visualization of the classifier
		{
			pt1.x = objects.at(i).model_center.x - objects.at(i).obj_size.x / 2;
			pt1.y = objects.at(i).model_center.y - objects.at(i).obj_size.y / 2;

			pt2.x = objects.at(i).model_center.x + objects.at(i).obj_size.x / 2;
			pt2.y = objects.at(i).model_center.y + objects.at(i).obj_size.y / 2;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		for (int j = 0; j < objects.at(i).track_points.size(); j++)
			cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color(objects.at(i).id), 2);
	}
	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------
	Mat baseimag(resolution, resolution + extr, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < objects.size(); i++)
	{
		string text = objects.at(i).obj_type + " ID" + to_string(objects.at(i).id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								class_name_color(objects.at(i).id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + objects.at(i).img.cols + 1;
		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, class_name_color(objects.at(i).id), 1);
			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
		}
	}

	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(id_frame), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------

	imshow("Motion", baseimag);
	cv::waitKey(10);

	/*
		al_objs.push_back(objects.at(1));
		if (al_objs.size() > 1)
		{
			draw_compare(al_objs.at(al_objs.size() - 2), al_objs.at(al_objs.size() - 1), class_name_color(al_objs.at(0).id));
		}
	*/

	return baseimag;
}

void trackingMotV2b(Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;
	Mat framebuf = frame;

	int rows = frame.rows;
	int cols = frame.cols;

	float koef = (float)rows / (float)992;

	float mpc = 15 * koef;   // minimum number of points for a cluster (good value 15)
	float nd = 9 * koef;     //(good value 6-15)
	float rcobj = 15 * koef; //(good value 15)
	float robj = 17 * koef;  //(good value 17)
	float mdist = 12 * koef; // maximum distance from cluster center (good value 10)
	int pft = 9;             // points fixation threshold (good value 9)

	Mat img;

	//--------------------<moution detections>--------------------

	float rwsize;
	float clsize;

	imagbuf = frame;

	resolution = rows; // cancels resizing, but the image is cropped to a square

	if (rows > cols)
	{
		rwsize = (float)resolution * rows / (float)cols;
		clsize = resolution;
	}
	else
	{
		rwsize = resolution;
		clsize = (float)resolution * cols / (float)rows;
	}

	cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	cv::Rect rectb(0, 0, resolution, resolution);
	imag = imagbuf(rectb);

	cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
	imag.convertTo(imag, CV_8UC3);

	if (rows > cols)
	{
		rwsize = (float)reduseres * rows / (float)cols;
		clsize = reduseres;
	}
	else
	{
		rwsize = reduseres;
		clsize = (float)reduseres * cols / (float)rows;
	}

	cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	cv::Rect rect0(0, 0, reduseres, reduseres);
	imageBGR0 = frame0(rect0);
	imageBGR0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	cv::Rect rect(0, 0, reduseres, reduseres);
	imageBGR = frame(rect);

	imageBGR.convertTo(imageBGR, CV_8UC1);

	Point2f pm;

	for (int y = 0; y < imageBGR0.rows; y++)
	{
		for (int x = 0; x < imageBGR0.cols; x++)
		{
			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
			uchar color2 = imageBGR.at<uchar>(Point(x, y));

			if (((int)color2 - (int)color1) > pft)
			{
				pm.x = x * resolution / reduseres;
				pm.y = y * resolution / reduseres;
				motion.push_back(pm);
			}
		}
	}

	Point2f pt1;
	Point2f pt2;

	for (int i = 0; i < motion.size(); i++) // visualization of the cluster_points
	{
		pt1.x = motion.at(i).x;
		pt1.y = motion.at(i).y;

		pt2.x = motion.at(i).x + resolution / reduseres;
		pt2.y = motion.at(i).y + resolution / reduseres;

		rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
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
			objects[i].cluster_points.clear();
		}

		for (int i = 0; i < motion.size(); i++)
		{
			for (int j = 0; j < objects.size(); j++)
			{
				if (objects[j].model_center.x < 0)
					continue;

				if (i < 0)
					break;

				if ((motion.at(i).x < (objects[j].model_center.x + objects[j].obj_size.x / 2)) && (motion.at(i).x > (objects[j].model_center.x - objects[j].obj_size.x / 2)) && (motion.at(i).y < (objects[j].model_center.y + objects[j].obj_size.y / 2)) && (motion.at(i).y > (objects[j].model_center.y - objects[j].obj_size.y / 2)))
				{
					objects[j].cluster_points.push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		float rm = rcobj * 1.0 * resolution / reduseres;

		for (int i = 0; i < motion.size(); i++)
		{
			rm = sqrt(pow((objects[0].cluster_center.x - motion.at(i).x), 2) + pow((objects[0].cluster_center.y - motion.at(i).y), 2));

			int n = -1;
			if (rm < rcobj * 1.0 * resolution / reduseres)
				n = 0;

			for (int j = 1; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - motion.at(i).x), 2) + pow((objects[j].cluster_center.y - motion.at(i).y), 2));
				if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
				{
					rm = r;
					n = j;
				}
			}

			if (n > -1)
			{
				objects[n].cluster_points.push_back(motion.at(i));
				motion.erase(motion.begin() + i);
				i--;
				objects[n].center_determine(id_frame, false);
				if (i < 0)
					break;
			}
		}
	}

	for (int j = 0; j < objects.size(); j++)
	{

		objects[j].center_determine(id_frame, false);

		Point2f clustercenter = objects[j].cluster_center;

		imagbuf = frame_resizing(framebuf);
		imagbuf.convertTo(imagbuf, CV_8UC3);

		if (clustercenter.y - half_imgsize * koef < 0)
			clustercenter.y = half_imgsize * koef + 1;

		if (clustercenter.x - half_imgsize * koef < 0)
			clustercenter.x = half_imgsize * koef + 1;

		if (clustercenter.y + half_imgsize * koef > imagbuf.rows)
			clustercenter.y = imagbuf.rows - 1;

		if (clustercenter.x + half_imgsize * koef > imagbuf.cols)
			clustercenter.x = imagbuf.cols - 1;

		img = imagbuf(cv::Range(max_u16(clustercenter.y - half_imgsize * koef), min_u16(imagbuf.rows, clustercenter.y + half_imgsize * koef))
			, cv::Range(max_u16(clustercenter.x - half_imgsize * koef), min_u16(imagbuf.cols, clustercenter.x + half_imgsize * koef)));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_8UC3);
		objects[j].img = img;
		objects[j].samples_creation();
	}
	//--------------</layout of motion points by objects>----

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj > -1 && nobj < objects.size())
		{
			pc = objects[nobj].cluster_center;
			// pc = objects[nobj].proposed_center();
			nobj++;
		}
		else
		{
			pc = motion.at(0);
			motion.erase(motion.begin());
		}

		clusters.push_back(vector<Point2f>());
		clusters[ncls].push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
			if (r < nd * 1.0 * resolution / reduseres)
			{
				Point2f cl_c = cluster_center(clusters.at(ncls));
				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
				if (r < mdist * 1.0 * resolution / reduseres)
				{
					clusters.at(ncls).push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		int newp;
		do
		{
			newp = 0;

			for (int c = 0; c < clusters[ncls].size(); c++)
			{
				pc = clusters[ncls].at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

					if (r < nd * 1.0 * resolution / reduseres)
					{
						Point2f cl_c = cluster_center(clusters.at(ncls));
						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
						if (r < mdist * 1.0 * resolution / reduseres)
						{
							clusters.at(ncls).push_back(motion.at(i));
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
	//--------------</cluster creation>----------------------

	//--------------<clusters to objects>--------------------
	for (int cls = 0; cls < ncls; cls++)
	{
		if (clusters[cls].size() > mpc) // if there are enough moving points
		{

			Point2f clustercenter = cluster_center(clusters[cls]);
			imagbuf = frame_resizing(framebuf);
			imagbuf.convertTo(imagbuf, CV_8UC3);
			img = imagbuf(cv::Range(max_u16(clustercenter.y - half_imgsize * koef), min_u16(imagbuf.rows, clustercenter.y + half_imgsize * koef))
				, cv::Range(max_u16(clustercenter.x - half_imgsize * koef), min_u16(imagbuf.cols, clustercenter.x + half_imgsize * koef)));
			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), "a", clusters[cls], img);
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				float r = sqrt(pow((objects[i].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[i].cluster_center.y - obj.cluster_center.y), 2));
				// float r = sqrt(pow((objects[i].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[i].proposed_center().y - obj.cluster_center.y), 2));
				if (r < robj * 1.0 * resolution / reduseres)
				{
					newobj = false;

					objects[i].img = obj.img;
					objects[i].cluster_points = obj.cluster_points;
					objects[i].center_determine(id_frame, true);
					break;
				}
			}

			if (newobj == true)
				objects.push_back(obj);
		}
	}
	//--------------</clusters to objects>-------------------

	for (int i = 0; i < objects.size(); i++)
		objects[i].push_track_point(objects[i].cluster_center);

	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = objects.at(i).cluster_points.at(j).x;
			pt1.y = objects.at(i).cluster_points.at(j).y;

			pt2.x = objects.at(i).cluster_points.at(j).x + resolution / reduseres;
			pt2.y = objects.at(i).cluster_points.at(j).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		if (objects.at(i).model_center.x > -1) // visualization of the classifier
		{
			pt1.x = objects.at(i).model_center.x - objects.at(i).obj_size.x / 2;
			pt1.y = objects.at(i).model_center.y - objects.at(i).obj_size.y / 2;

			pt2.x = objects.at(i).model_center.x + objects.at(i).obj_size.x / 2;
			pt2.y = objects.at(i).model_center.y + objects.at(i).obj_size.y / 2;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		for (int j = 0; j < objects.at(i).track_points.size(); j++)
			cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color(objects.at(i).id), 2);
	}
	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------
	Mat baseimag(resolution, resolution + extr * koef, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 0; i < objects.size(); i++)
	{
		string text = objects.at(i).obj_type + " ID" + to_string(objects.at(i).id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								class_name_color(objects.at(i).id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + objects.at(i).img.cols + 1;
		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, class_name_color(objects.at(i).id), 1);
			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
		}
	}

	imag.copyTo(baseimag(cv::Rect(extr * koef, 0, imag.cols, imag.rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr * koef - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(id_frame), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------
	// imshow("Motion", baseimag);
	// cv::waitKey(0);
}

void OBJdetectsToObjs(vector<OBJdetect> objdetects, vector<Obj> &objs)
{
	objs.clear();
	Obj objbuf;
	for (int i = 0; i < objdetects.size(); i++)
	{
		int j = 0;
		for (j = 0; j < sizeof(class_name) / sizeof(*class_name); j++)
			if (objdetects.at(i).type == class_name[j])
				break;

		objbuf.type = j;                         // Object type
		objbuf.id = i;                           // Object id
		objbuf.x = objdetects.at(i).detect.x;    // Center x of the bounding box
		objbuf.y = objdetects.at(i).detect.y;    // Center y of the bounding box
		objbuf.w = objdetects.at(i).obj_size.x; // Width of the bounding box
		objbuf.h = objdetects.at(i).obj_size.y; // Height of the bounding box

		objs.push_back(objbuf);
	}
}

void ALObjectsToObjs(vector<ALObject> objects, vector<Obj> &objs)
{
	objs.clear();
	Obj objbuf;
	for (int i = 0; i < objects.size(); i++)
	{
		int j = 0;
		for (j = 0; j < sizeof(class_name) / sizeof(*class_name); j++)
			if (objects.at(i).obj_type == class_name[j])
				break;

		objbuf.type = j;                           // Object type
		objbuf.id = objects.at(i).id;              // Object id
		objbuf.x = objects.at(i).cluster_center.x; // Center x of the bounding box
		objbuf.y = objects.at(i).cluster_center.y; // Center y of the bounding box
		objbuf.w = 0;                              // Width of the bounding box
		objbuf.h = 0;                              // Height of the bounding box

		objs.push_back(objbuf);
	}
}

void fixIDs(const vector<vector<Obj>> &objs, vector<std::pair<uint, idFix>> &fixedIds, vector<Mat> &d_images, uint framesize)
{
	vector<ALObject> objects;
	vector<Obj> objsbuf;
	vector<vector<Obj>> fixedobjs;
	idFix idfix;

	if (framesize > 0)
	{
		for (int i = 0; i < d_images.size(); i++)
			d_images.at(i) = frame_resizingV2(d_images.at(i), framesize);
	}

	float koef = (float)d_images.at(0).rows / (float)resolution;
	const float maxr = 17.0 * koef; // set depending on the size of the ant

	fixedIds.clear();

	trackingMotV2b(d_images.at(1), d_images.at(0), objects, 0);
	ALObjectsToObjs(objects, objsbuf);
	fixedobjs.push_back(objsbuf);

	for (int i = 0; i < d_images.size() - 1; i++)
	{
		trackingMotV2b(d_images.at(i), d_images.at(i + 1), objects, i + 1);
		ALObjectsToObjs(objects, objsbuf);
		fixedobjs.push_back(objsbuf);
	}

	for (int i = 0; i < objs.size(); i++)
	{
		for (int j = 0; j < objs.at(i).size(); j++)
		{
			if (objs.at(i).at(j).type != 1)
				continue;

			float minr = sqrt(pow((float)objs.at(i).at(j).x - (float)fixedobjs.at(i).at(0).x * koef, 2) + pow((float)objs.at(i).at(j).y - (float)fixedobjs.at(i).at(0).y * koef, 2));
			int ind = 0;

			for (int n = 0; n < fixedobjs.at(i).size(); n++)
			{
				float r = sqrt(pow((float)objs.at(i).at(j).x - (float)fixedobjs.at(i).at(n).x * koef, 2) + pow((float)objs.at(i).at(j).y - (float)fixedobjs.at(i).at(n).y * koef, 2));
				if (r < minr)
				{
					minr = r;
					ind = n;
				}
			}

			if (minr < maxr && objs.at(i).at(j).id != fixedobjs.at(i).at(ind).id)
			{
				idfix.id = fixedobjs.at(i).at(ind).id;
				idfix.idOld = objs.at(i).at(j).id;
				idfix.type = objs.at(i).at(j).type; // not fixed
				fixedIds.push_back(std::make_pair((uint)i, idfix));
				fixedobjs.at(i).erase(fixedobjs.at(i).begin() + ind);
			}
		}
	}
}

bool compare(intpoint a, intpoint b)
{
	if (a.ipoint < b.ipoint)
		return 1;
	else
		return 0;
}

vector<Point2f> draw_map_prob(vector<int> npsamples, vector<Point2f> mpoints)
{
	int maxprbp = 9;

	vector<Point2f> probapoints;

	int max = npsamples.at(0);
	int min = npsamples.at(1);

	Point2f mpoint;

	vector<intpoint> top;
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

	Point2f bufmp;
	int bufnp;

	/*std::cout << "----------------------" << endl;
	for (int i = 0; i < 10; i++)
	{
		std::cout << i << ". " << top.at(i).ipoint << endl;
		std::cout << i << ". " << top.at(i).mpoint << endl;
	}

	std::cout << "min - " << min << endl;
	std::cout << "mpoint.x - " << mpoint.x << endl;
	std::cout << "mpoint.y - " << mpoint.y << endl;
	*/

	for (int i = 0; i < npsamples.size(); i++)
		npsamples.at(i) -= min;

	int color_step = (int)((max - min) / (3 * 255));

	Mat imgpmap(2 * half_imgsize, 2 * half_imgsize, CV_8UC3, Scalar(0, 0, 0));

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

	cv::circle(imgpmap, mpoint, 1, Scalar(0, 255, 0), 1);

	for (int i = 0; i < top.size(); i++)
	{

		if (i >= maxprbp)
			break;

		cv::circle(imgpmap, top.at(i).mpoint, 1, Scalar(0, 255, 255), 1);

		probapoints.push_back(top.at(i).mpoint);
	}

	cv::resize(imgpmap, imgpmap, cv::Size(imgpmap.rows * 5, imgpmap.cols * 5), cv::InterpolationFlags::INTER_CUBIC);
	imshow("imgpmap", imgpmap);
	cv::waitKey(0);

	return probapoints;
}

vector<Point2f> map_prob(vector<int> npsamples, vector<Point2f> mpoints)
{
	int maxprbp = 12;

	vector<Point2f> probapoints;

	int max = npsamples.at(0);
	int min = npsamples.at(1);

	Point2f mpoint;

	vector<intpoint> top;
	vector<intpoint> buftop;

	intpoint bufnpmp;

	for (int i = 0; i < npsamples.size(); i++)
	{
		bufnpmp.ipoint = npsamples.at(i);
		bufnpmp.mpoint = mpoints.at(i);
		top.push_back(bufnpmp);
	}

	// sort(top.begin(), top.end(), compare);
	buftop.push_back(top.at(0));

	for (int i = 1; i < top.size(); i++)
	{
		if (buftop.at(buftop.size() - 1).ipoint > top.at(i).ipoint)
		{
			if (buftop.size() < maxprbp)
				buftop.push_back(top.at(i));
			else
				buftop.at(buftop.size() - 1) = top.at(i);
		}
		sort(buftop.begin(), buftop.end(), compare);
	}

	for (int i = 0; i < buftop.size(); i++)
	{
		probapoints.push_back(buftop.at(i).mpoint);
	}

	return probapoints;
}

Mat color_correction(Mat imag)
{
	Mat imagchange;

	imag.copyTo(imagchange);

	for (int y = 0; y < imag.rows; y++)
	{
		for (int x = 0; x < imag.cols; x++)
		{
			uchar color1 = imag.at<uchar>(Point(x, y));

			if (color1 < (uchar)color_threshold) // 65-70
				imagchange.at<uchar>(Point(x, y)) = 0;
			else
				imagchange.at<uchar>(Point(x, y)) = 255;
		}
	}

	// imshow("imagchange", imagchange);
	// cv::waitKey(0);

	return imagchange;
}

void objdeterm(vector<Point2f> &cluster_points, Mat frame, ALObject &obj, size_t id_frame)
{

	int maxr = 1;       // if point is outside then ignore
	int half_range = 6; // half deviation

	int wh = 800;
	vector<Mat> cluster_samples;
	Point2f minp;
	Point2f maxp;
	Point2f bufp;

	Point2f p1;
	Point2f p2;

	minp.y = frame.rows;
	minp.x = frame.cols;

	maxp.x = 0;
	maxp.y = 0;
	Mat bufimg;

	vector<int> npforpoints;
	vector<int> npsamples;
	vector<Point2f> mpoints;

	vector<vector<int>> all_npforpoints;

	for (int i = 0; i < cluster_points.size(); i++)
	{
		if (cluster_points.at(i).y < minp.y)
			minp.y = cluster_points.at(i).y;

		if (cluster_points.at(i).x < minp.x)
			minp.x = cluster_points.at(i).x;

		if (cluster_points.at(i).y > maxp.y)
			maxp.y = cluster_points.at(i).y;

		if (cluster_points.at(i).x > maxp.x)
			maxp.x = cluster_points.at(i).x;

		bufimg = frame(cv::Range(cluster_points.at(i).y, min_u16(frame.rows, cluster_points.at(i).y + resolution / reduseres))
			, cv::Range(cluster_points.at(i).x, min_u16(frame.cols, cluster_points.at(i).x + resolution / reduseres)));
		cluster_samples.push_back(bufimg);

		npforpoints.push_back(0);
	}

	int st = 1; // step for samles compare

	Mat imag = obj.img;

	int start_y = 0;
	int start_x = 0;

	int and_y = (int)((imag.rows - resolution / reduseres) / st);
	int and_x = (int)((imag.cols - resolution / reduseres) / st);

	for (int step_x = start_x; step_x < and_x; step_x++)
	{
		for (int step_y = start_y; step_y < and_y; step_y++)
		{
			bool cont = true;
			for (int i = 0; i < obj.cluster_points.size(); i++)
			{
				if (abs(step_y - (obj.cluster_points.at(i).y - obj.cluster_center.y + half_imgsize)) < maxr && abs(step_x - (obj.cluster_points.at(i).x - obj.cluster_center.x + half_imgsize)) < maxr)
					cont = false;
			}

			if (cont == true)
				continue;

			for (int i = 0; i < cluster_samples.size(); i++)
			{
				Mat sample;
				sample = imag(cv::Range(step_y * st, min_u16(imag.rows, step_y * st + resolution / reduseres))
					, cv::Range(step_x * st, min_u16(imag.cols, step_x * st + resolution / reduseres)));
				int npbuf = samples_compV2(cluster_samples.at(i), sample);
				npforpoints.at(i) = npbuf;
			}

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

	vector<int> sample_np;
	vector<vector<Point2f>> alt_cluster_points;

	vector<intpoint> chain;
	vector<vector<intpoint>> chains;

	// std::cout << "cluster_points.size() - " << cluster_points.size() << endl;
	for (int ci = 0; ci < cluster_points.size(); ci++)
	{
		/*/------------------<TESTING>-----------------------------------
		Mat resimg = imag;
		Point2f correct;

		correct.x = 0;
		correct.y = 0;

		for (int i = 0; i < cluster_points.size(); i++)
		{
			p1.x = cluster_points.at(i).x - minp.x;
			p1.y = cluster_points.at(i).y - minp.y;
			p2.x = cluster_points.at(i).x - minp.x + resolution / reduseres;
			p2.y = cluster_points.at(i).y - minp.y + resolution / reduseres;

			p1 += correct;
			p2 += correct;

			Mat s_imag = cluster_samples.at(i);
			cv::cvtColor(s_imag, s_imag, cv::COLOR_BGR2RGB);
			s_imag.convertTo(s_imag, CV_8UC3);
			s_imag.copyTo(resimg(cv::Rect(p1.x, p1.y, resolution / reduseres, resolution / reduseres)));
			rectangle(resimg, p1, p2, Scalar(0, 255, 0), 1);


			p1.x = p1.x + (resolution / reduseres) / 2;
			p1.y = p1.y + (resolution / reduseres) / 2;

			if (i == ci)
				cv::circle(resimg, p1, 1, Scalar(0, 0, 255), 1);
		}

		cv::resize(resimg, resimg, cv::Size(resimg.cols * 5, resimg.rows * 5), cv::InterpolationFlags::INTER_CUBIC);
		imshow("resimg", resimg);
		cv::waitKey(0);

		//------------------</TESTING>-----------------------------------*/

		sample_np.clear();
		for (int i = 0; i < all_npforpoints.size(); i++)
			sample_np.push_back(all_npforpoints.at(i).at(ci));

		alt_cluster_points.push_back(map_prob(sample_np, mpoints));
	}

	for (int i = 0; i < alt_cluster_points.size(); i++)
	{
		for (int ci = 0; ci < alt_cluster_points.at(i).size(); ci++)
		{
			chain.clear();
			intpoint bufch;
			bufch.ipoint = i;
			bufch.mpoint = alt_cluster_points.at(i).at(ci);
			chain.push_back(bufch);

			for (int j = i + 1; j < alt_cluster_points.size(); j++)
			{
				Point2f dp;
				dp.x = cluster_points.at(i).x - cluster_points.at(j).x;
				dp.y = cluster_points.at(i).y - cluster_points.at(j).y;

				for (int cj = 0; cj < alt_cluster_points.at(j).size(); cj++)
				{
					if (abs(alt_cluster_points.at(i).at(ci).x - alt_cluster_points.at(j).at(cj).x - dp.x) < half_range
					&& abs(alt_cluster_points.at(i).at(ci).y - alt_cluster_points.at(j).at(cj).y - dp.y) < half_range)
					{
						bufch.ipoint = j;
						bufch.mpoint = alt_cluster_points.at(j).at(cj);
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

	obj.cluster_points.clear();
	for (int i = 0; i < chains.at(maxchains).size(); i++)
		obj.cluster_points.push_back(cluster_points.at(chains.at(maxchains).at(i).ipoint));

	/*
		vector<Point2f> cluster_points_buf;

		int maxradd = 5;
		for(int i=0; i < cluster_points.size(); i++)
		{
			 bool push = false;

				for (int j = 0; j < obj.cluster_points.size(); j++)
				{
					if(abs(obj.cluster_points.at(j).y - cluster_points.at(i).y) < maxradd && abs(obj.cluster_points.at(j).x - cluster_points.at(i).x) < maxradd)
					{
						cluster_points_buf.push_back(cluster_points.at(i));
						break;
					}
				}
		}

		for(int i=0 ; i< cluster_points_buf.size(); i++)
			obj.cluster_points.push_back(cluster_points_buf.at(i));*/

	/*/------------------<removing motion points from a cluster>-------------------
	vector<Point2f> cluster_points_buf;
	for (int i = 0; i < cluster_points.size(); i++)
	{
		bool cpy = true;
		for(int j=0; j<chains.at(maxchains).size(); j++)
		{
			if(i==chains.at(maxchains).at(j).ipoint)
			{
				cpy = false;
				break;
			}
		}

		if(cpy == true)
				cluster_points_buf.push_back(cluster_points.at(i));
	}

	cluster_points.clear();
	for(int i=0; i<cluster_points_buf.size(); i++)
	 cluster_points.push_back(cluster_points_buf.at(i));
	//------------------<removing motion points from a cluster>-------------------*/

	obj.center_determine(id_frame, false);
	frame.convertTo(bufimg, CV_8UC3);
	obj.img = bufimg(cv::Range(max_u16(obj.cluster_center.y - half_imgsize), min_u16(bufimg.rows, obj.cluster_center.y + half_imgsize))
		, cv::Range(max_u16(obj.cluster_center.x - half_imgsize), min_u16(bufimg.cols, obj.cluster_center.x + half_imgsize)));
	cv::cvtColor(obj.img, obj.img, cv::COLOR_BGR2RGB);
	obj.img.convertTo(obj.img, CV_8UC3);

	//-----------------<probe visualization>-----------------
	Mat resimg;
	imag.copyTo(resimg);

	Mat resimg2(resimg.rows, resimg.cols, CV_8UC3, Scalar(0, 0, 0));
	Mat res2x(resimg.rows, resimg.cols * 2, CV_8UC3, Scalar(0, 0, 0));

	p1.y = minp.y + (maxp.y - minp.y) / 2 - half_imgsize;
	p1.x = minp.x + (maxp.x - minp.x) / 2 - half_imgsize;

	p2.y = minp.y + (maxp.y - minp.y) / 2 + half_imgsize;
	p2.x = minp.x + (maxp.x - minp.x) / 2 + half_imgsize;

	bufimg = frame(cv::Range(max_u16(p1.y), min_u16(frame.rows, p2.y)), cv::Range(max_u16(p1.x), min_u16(frame.cols, p2.x)));

	cv::cvtColor(bufimg, bufimg, cv::COLOR_BGR2RGB);
	bufimg.convertTo(bufimg, CV_8UC3);
	bufimg.copyTo(resimg2(cv::Rect(0, 0, bufimg.cols, bufimg.rows)));

	for (int i = 0; i < cluster_points.size(); i++)
	{
		p1.x = cluster_points.at(i).x - minp.x - (maxp.x - minp.x) / 2 + half_imgsize;
		p1.y = cluster_points.at(i).y - minp.y - (maxp.y - minp.y) / 2 + half_imgsize;
		p2.x = p1.x + resolution / reduseres;
		p2.y = p1.y + resolution / reduseres;
		// rectangle(resimg2, p1, p2, Scalar(0, 0, 255), 1);
	}

	for (int i = 0; i < chains.at(maxchains).size(); i++)
	{
		size_t R = rand() % 255;
		size_t G = rand() % 255;
		size_t B = rand() % 255;

		p1.x = chains.at(maxchains).at(i).mpoint.x;
		p1.y = chains.at(maxchains).at(i).mpoint.y;
		p2.x = p1.x + resolution / reduseres;
		p2.y = p1.y + resolution / reduseres;

		rectangle(resimg, p1, p2, Scalar(R, G, B), 1);

		p1.x = cluster_points.at(chains.at(maxchains).at(i).ipoint).x - minp.x - (maxp.x - minp.x) / 2 + half_imgsize;
		p1.y = cluster_points.at(chains.at(maxchains).at(i).ipoint).y - minp.y - (maxp.y - minp.y) / 2 + half_imgsize;
		p2.x = p1.x + resolution / reduseres;
		p2.y = p1.y + resolution / reduseres;

		rectangle(resimg2, p1, p2, Scalar(R, G, B), 1);
	}

	resimg2.copyTo(res2x(cv::Rect(0, 0, resimg2.cols, resimg2.rows)));
	resimg.copyTo(res2x(cv::Rect(resimg2.cols, 0, resimg2.cols, resimg2.rows)));

	cv::resize(res2x, res2x, cv::Size(res2x.cols * 5, res2x.rows * 5), cv::InterpolationFlags::INTER_CUBIC);

	imshow("res2x", res2x);
	cv::waitKey(0);
	//-----------------</probe visualization>-----------------*/
}

bool compare_clsobj(ClsObjR a, ClsObjR b)
{
	if (a.r < b.r)
		return 1;
	else
		return 0;
}

Mat trackingMotV2_1(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, bool usedetector, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	// if (usedetector)
	//   frame = frame_resizing(frame);

	frame = frame_resizing(frame);
	frame0 = frame_resizing(frame0);

	Mat framebuf;

	frame.copyTo(framebuf);

	int mpct = 3;      // minimum number of points for a cluster (tracking object) (good value 5)
	int mpcc = 7;      // minimum number of points for a cluster (creation new object) (good value 13)
	float nd = 1.5;    //(good value 6-15)
	int rcobj = 15;    //(good value 15)
	float robj = 22.0; //(good value 17)
	float robj_k = 1.0;
	int mdist = 10; // maximum distance from cluster center (good value 10)
	int pft = 1;    // points fixation threshold (good value 9)

	Mat img;

	vector<OBJdetect> detects;

	//--------------------<detection using a classifier>----------
	if (usedetector)
	{

		detects = detectorV4(pathmodel, frame, device_type, confidence);

		for (int i = 0; i < objects.size(); i++)
		{
			objects[i].det_mc = false;
		}

		for (int i = 0; i < detects.size(); i++)
		{
			if (detects.at(i).type != "a")
			{
				detects.erase(detects.begin() + i);
				i--;
			}
		}

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			vector<Point2f> cluster_points;
			cluster_points.push_back(detects.at(i).detect);
			// imagbuf = frame_resizing(framebuf);
			// img = imagbuf(cv::Range(detects.at(i).detect.y - half_imgsize, detects.at(i).detect.y + half_imgsize), cv::Range(detects.at(i).detect.x - half_imgsize, detects.at(i).detect.x + half_imgsize));

			img = framebuf(cv::Range(max_u16(detects.at(i).detect.y - half_imgsize), min_u16(framebuf.rows, detects.at(i).detect.y + half_imgsize))
				, cv::Range(max_u16(detects.at(i).detect.x - half_imgsize), min_u16(framebuf.cols, detects.at(i).detect.x + half_imgsize)));

			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), detects.at(i).type, cluster_points, img);
			obj.model_center = detects.at(i).detect;
			obj.cluster_center = detects.at(i).detect;
			obj.obj_size = detects.at(i).obj_size;
			obj.det_mc = true;
			// obj.track_points.push_back(detects.at(i).detect);
			// obj.push_track_point(detects.at(i).detect);

			float rm = rcobj * (float)resolution / (float)reduseres;
			bool newobj = true;
			uint16_t n;

			if (objects.size() > 0)
			{
				rm = sqrt(pow((objects[0].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[0].cluster_center.y - obj.cluster_center.y), 2));
				// rm = sqrt(pow((objects[0].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[0].proposed_center().y - obj.cluster_center.y), 2));
				if (rm < rcobj * (float)resolution / (float)reduseres)
				{
					n = 0;
					newobj = false;
				}
			}

			for (uint16_t j = 1; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
				// float r = sqrt(pow((objects[j].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[j].proposed_center().y - obj.cluster_center.y), 2));
				if (r < rcobj * (float)resolution / (float)reduseres && r < rm)
				{
					rm = r;
					n = j;
					newobj = false;
				}
			}

			if (newobj == false)
			{
				auto& tobj = objects.at(n);
				// // void  showOcclusion(Mat& frameClr, const Point& o1corn1, const Point& o1corn2, const Scalar& clr1, const Point& o2corn1, const Point& o2corn2, const Scalar& clr2, const int w=1);
				// if(!tobj.traces.empty() && tobj.traces.back().frame == id_frame) {
				//   Mat  dimg(frame.rows, frame.cols, CV_8UC3);
				//   cv::cvtColor(frame, dimg, cv::COLOR_GRAY2BGR);
				//   //frame.convertTo(dimg, CV_8UC3);
				//   rectangle(dimg, Point(roundf(tobj.model_center.x - tobj.obj_size.x/2.f)-1, roundf(tobj.model_center.y - tobj.obj_size.y/2.f)-1)
				//     , Point(roundf(tobj.model_center.x + tobj.obj_size.x/2.f)+1, roundf(tobj.model_center.y + tobj.obj_size.y/2.f)+1), Scalar(255, 0, 0), 1);
				//   rectangle(dimg, Point(roundf(obj.model_center.x - obj.obj_size.x/2.f)+1, roundf(obj.model_center.y - obj.obj_size.y/2.f)+1)
				//     , Point(roundf(obj.model_center.x + obj.obj_size.x/2.f)-1, roundf(obj.model_center.y + obj.obj_size.y/2.f)-1), Scalar(0, 255, 0), 1);
				//   imshow("Occlusion", dimg);
				//   cv::waitKey(500);
				// }
				tobj.model_center = obj.model_center;
				tobj.obj_size = obj.obj_size;
				tobj.img = obj.img;
				tobj.det_mc = true;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < id_frame && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
					, obj.obj_size.x, obj.obj_size.y});
			}
			else
			{

				for (size_t j = 0; j < objects.size(); j++)
				{
					if (objects[i].det_pos == false)
						continue;

					float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
					if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
					{
						newobj = false;
						break;
					}
				}

				if (newobj == true) {
					assert(obj.traces.empty() && "Unexpected traces");
					obj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
						, obj.obj_size.x, obj.obj_size.y});
					objects.push_back(obj);
				}
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

	Point2f pm;

	imageBGR0 = color_correction(imageBGR0);
	imageBGR = color_correction(imageBGR);

	for (uint16_t y = 0; y < imageBGR0.rows; y++)
	{
		for (uint16_t x = 0; x < imageBGR0.cols; x++)
		{
			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
			uchar color2 = imageBGR.at<uchar>(Point(x, y));

			if (((int)color2 - (int)color1) > pft)
			{
				pm.x = x * resolution / reduseres;
				pm.y = y * resolution / reduseres;
				motion.push_back(pm);
			}
		}
	}

	Point2f pt1;
	Point2f pt2;

	if (2 == 3)
		for (int i = 0; i < motion.size(); i++) // visualization of the white cluster_points
		{
			pt1.x = motion.at(i).x;
			pt1.y = motion.at(i).y;

			pt2.x = motion.at(i).x + resolution / reduseres;
			pt2.y = motion.at(i).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
		}

	uint16_t ncls = 0;
	uint16_t nobj;

	if (objects.size() > 0)
		nobj = 0;
	else
		nobj = -1;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj > -1 && nobj < objects.size())
		{
			pc = objects[nobj].cluster_center;
			// pc = objects[nobj].proposed_center();
			nobj++;
		}
		else
		{
			pc = motion.at(0);
			motion.erase(motion.begin());
		}

		clusters.push_back(vector<Point2f>());
		clusters[ncls].push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
			if (r < nd * (float)resolution / (float)reduseres)
			{
				Point2f cl_c = cluster_center(clusters.at(ncls));
				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
				if (r < mdist * 1.0 * resolution / reduseres)
				{
					clusters.at(ncls).push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < clusters[ncls].size(); c++)
			{
				pc = clusters[ncls].at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

					if (r < nd * (float)resolution / (float)reduseres)
					{
						Point2f cl_c = cluster_center(clusters.at(ncls));
						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
						if (r < mdist * 1.0 * resolution / reduseres)
						{
							clusters.at(ncls).push_back(motion.at(i));
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
	//--------------</cluster creation>----------------------

	//--------------<clusters to objects>--------------------

	if (objects.size() > 0)
	{
		for (size_t i = 0; i < objects.size(); i++)
			objects[i].det_pos = false;

		vector<ClsObjR> clsobjrs;
		ClsObjR clsobjr;
		for (size_t i = 0; i < objects.size(); i++)
		{
			for (size_t cls = 0; cls < clusters.size(); cls++)
			{
				clsobjr.cls_id = cls;
				clsobjr.obj_id = i;
				Point2f clustercenter = cluster_center(clusters[cls]);
				// clsobjr.r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				clsobjr.r = sqrt(pow((objects[i].proposed_center().x - clustercenter.x), 2) + pow((objects[i].proposed_center().y - clustercenter.y), 2));
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (usedetector == true)
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				size_t obj_id = clsobjrs.at(i).obj_id;

				if (objects.at(obj_id).det_mc == true)
				{
					Point2f clustercenter = cluster_center(clusters[cls_id]);
					auto& cobj = objects[obj_id];
					pt1.x = cobj.model_center.x - cobj.obj_size.x / 2;
					pt1.y = cobj.model_center.y - cobj.obj_size.y / 2;

					pt2.x = cobj.model_center.x + cobj.obj_size.x / 2;
					pt2.y = cobj.model_center.y + cobj.obj_size.y / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{

						if (cobj.det_pos == false)
							cobj.cluster_points = clusters.at(cls_id);
						else
						{
							for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
								cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
						}

						cobj.center_determine(id_frame, false);

						for (size_t j = 0; j < clsobjrs.size(); j++)
						{
							if (clsobjrs.at(j).cls_id == cls_id)
							{
								clsobjrs.erase(clsobjrs.begin() + j);
								j--;
							}
						}
						i = 0;
					}
				}
			}
		//--</corr obj use model>---

		//---<det obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * (float)resolution / (float)reduseres && clusters.at(cls_id).size() > mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (cobj.det_pos == false)
					cobj.cluster_points = clusters.at(cls_id);
				else
				{
					continue;
					// for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					//   cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
				}

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//---</det obj>---
		//---<new obj>----
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			Point2f clustercenter = cluster_center(clusters[cls_id]);
			bool newobj = true;

			for (size_t j = 0; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - clustercenter.x), 2) + pow((objects[j].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls_id].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - half_imgsize, clustercenter.y + half_imgsize), cv::Range(clustercenter.x - half_imgsize, clustercenter.x + half_imgsize));

				img = framebuf(cv::Range(max_u16(clustercenter.y - half_imgsize), min_u16(framebuf.rows, clustercenter.y + half_imgsize))
					, cv::Range(max_u16(clustercenter.x - half_imgsize), min_u16(framebuf.cols, clustercenter.x + half_imgsize)));

				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls_id], img);
				assert(obj.traces.empty() && "Unexpected traces");
				obj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
						, obj.obj_size.x, obj.obj_size.y});
				objects.push_back(obj);
				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</new obj>--

		//--<corr obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * robj_k * (float)resolution / (float)reduseres && clusters.at(cls_id).size() > mpct / 2)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					cobj.cluster_points.push_back(clusters.at(cls_id).at(j));

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</corr obj>---
	}
	else
	{
		//--<new obj>--
		for (int cls = 0; cls < clusters.size(); cls++)
		{
			Point2f clustercenter = cluster_center(clusters[cls]);
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				float r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// magbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - half_imgsize, clustercenter.y + half_imgsize), cv::Range(clustercenter.x - half_imgsize, clustercenter.x + half_imgsize));

				img = framebuf(cv::Range(max_u16(clustercenter.y - half_imgsize), min_u16(framebuf.rows, clustercenter.y + half_imgsize))
					, cv::Range(max_u16(clustercenter.x - half_imgsize), min_u16(framebuf.cols, clustercenter.x + half_imgsize)));

				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls], img);
				assert(obj.traces.empty() && "Unexpected traces");
				obj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
						, obj.obj_size.x, obj.obj_size.y});
				objects.push_back(obj);
				clusters.erase(clusters.begin() + cls);
				cls--;

				if (cls < 0)
					cls = 0;
			}
		}
		//--</new obj>--
	}
	//--------------</clusters to objects>-------------------

	//--------------<post processing>-----------------------
	// std::cout << "<post processing>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		if (objects.at(i).det_mc == false && objects.at(i).det_pos == false)
			continue;

		// imagbuf = frame_resizing(framebuf);
		// imagbuf.convertTo(imagbuf, CV_8UC3);

		// imagbuf = frame_resizing(framebuf);
		framebuf.convertTo(imagbuf, CV_8UC3);

		if (objects[i].det_mc == false)
		{
			pt1.y = objects[i].cluster_center.y - half_imgsize;
			pt2.y = objects[i].cluster_center.y + half_imgsize;

			pt1.x = objects[i].cluster_center.x - half_imgsize;
			pt2.x = objects[i].cluster_center.x + half_imgsize;
		}
		else
		{
			pt1.y = objects[i].model_center.y - half_imgsize;
			pt2.y = objects[i].model_center.y + half_imgsize;

			pt1.x = objects[i].model_center.x - half_imgsize;
			pt2.x = objects[i].model_center.x + half_imgsize;
		}

		if (pt1.y < 0)
			pt1.y = 0;

		if (pt2.y > imagbuf.rows)
			pt2.y = imagbuf.rows;

		if (pt1.x < 0)
			pt1.x = 0;

		if (pt2.x > imagbuf.cols)
			pt2.x = imagbuf.cols;

		// std::cout << "<post processing 2>" << endl;
		// std::cout << "pt1 - " << pt1 << endl;
		// std::cout << "pt2 - " << pt2 << endl;

		img = imagbuf(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_8UC3);
		img.copyTo(objects[i].img);
		objects[i].center_determine(id_frame, true);

		if (objects[i].det_mc == false)
			objects[i].push_track_point(objects[i].cluster_center);
		else
			objects[i].push_track_point(objects[i].model_center);
	}
	// std::cout << "</post processing>" << endl;
	//--------------<visualization>--------------------------
	// std::cout << "<visualization>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = objects.at(i).cluster_points.at(j).x;
			pt1.y = objects.at(i).cluster_points.at(j).y;

			pt2.x = objects.at(i).cluster_points.at(j).x + resolution / reduseres;
			pt2.y = objects.at(i).cluster_points.at(j).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		if (objects.at(i).det_mc == true) // visualization of the classifier
		{
			pt1.x = objects.at(i).model_center.x - objects.at(i).obj_size.x / 2;
			pt1.y = objects.at(i).model_center.y - objects.at(i).obj_size.y / 2;

			pt2.x = objects.at(i).model_center.x + objects.at(i).obj_size.x / 2;
			pt2.y = objects.at(i).model_center.y + objects.at(i).obj_size.y / 2;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		for (int j = 0; j < objects.at(i).track_points.size(); j++)
			cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color(objects.at(i).id), 2);
	}
	// std::cout << "</visualization>" << endl;
	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------
	Mat baseimag(resolution, resolution + extr, CV_8UC3, Scalar(0, 0, 0));
	// std::cout << "<baseimag 1>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		string text = objects.at(i).obj_type + " ID" + to_string(objects.at(i).id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								class_name_color(objects.at(i).id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + objects.at(i).img.cols + 1;
		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, class_name_color(objects.at(i).id), 1);
			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
		}
	}
	// std::cout << "<baseimag 2>" << endl;
	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(id_frame), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------

	imshow("Motion", baseimag);
	cv::waitKey(10);

	return baseimag;
}

vector<std::pair<Point2f, uint16_t>> trackingMotV2_1_artemis(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, bool usedetector, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	if (usedetector)
		frame = frame_resizing(frame);

	Mat framebuf = frame;

	uint16_t rows = frame.rows;
	uint16_t cols = frame.cols;

	frame_resolution = rows;

	float koef = (float)rows / (float)model_resolution;

	uint8_t mpct = 3 * koef;   // minimum number of points for a cluster (tracking object) (good value 5)
	uint8_t mpcc = 7 * koef;   // minimum number of points for a cluster (creation new object) (good value 13)
	float nd = 1.5 * koef;     //(good value 6-15)
	uint8_t rcobj = 15 * koef; //(good value 15)
	float robj = 22.0 * koef;  //(good value 17)
	float robj_k = 1.0;
	uint8_t mdist = 10 * koef; // maximum distance from cluster center (good value 10)
	uint8_t pft = 1;           // points fixation threshold (good value 9)

	Mat img;

	vector<OBJdetect> detects;

	//--------------------<detection using a classifier>----------
	if (usedetector)
	{
		detects = detectorV4(pathmodel, frame, device_type, confidence);

		for (uint16_t i = 0; i < objects.size(); i++)
		{
			objects[i].det_mc = false;
		}

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			if (detects.at(i).type != "a")
			{
				detects.erase(detects.begin() + i);
				i--;
			}
		}

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			vector<Point2f> cluster_points;
			cluster_points.push_back(detects.at(i).detect);
			// imagbuf = frame_resizing(framebuf);
			// img = imagbuf(cv::Range(detects.at(i).detect.y - half_imgsize * koef, detects.at(i).detect.y + half_imgsize * koef), cv::Range(detects.at(i).detect.x - half_imgsize * koef, detects.at(i).detect.x + half_imgsize * koef));

			img = framebuf(cv::Range(max_u16(detects.at(i).detect.y - half_imgsize * koef)
				, min_u16(framebuf.rows, detects.at(i).detect.y + half_imgsize * koef))
				, cv::Range(max_u16(detects.at(i).detect.x - half_imgsize * koef)
				, min_u16(framebuf.cols, detects.at(i).detect.x + half_imgsize * koef)));

			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), detects.at(i).type, cluster_points, img);
			obj.model_center = detects.at(i).detect;
			obj.cluster_center = detects.at(i).detect;
			obj.obj_size = detects.at(i).obj_size;
			obj.det_mc = true;
			// obj.track_points.push_back(detects.at(i).detect);
			// obj.push_track_point(detects.at(i).detect);

			float rm = rcobj * (float)rows / (float)reduseres;
			bool newobj = true;
			uint16_t n;

			if (objects.size() > 0)
			{
				rm = sqrt(pow((objects[0].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[0].cluster_center.y - obj.cluster_center.y), 2));
				// rm = sqrt(pow((objects[0].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[0].proposed_center().y - obj.cluster_center.y), 2));
				if (rm < rcobj * (float)rows / (float)reduseres)
				{
					n = 0;
					newobj = false;
				}
			}

			for (uint16_t j = 1; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
				// float r = sqrt(pow((objects[j].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[j].proposed_center().y - obj.cluster_center.y), 2));
				if (r < rcobj * (float)rows / (float)reduseres && r < rm)
				{
					rm = r;
					n = j;
					newobj = false;
				}
			}

			if (newobj == false)
			{
				auto& tobj = objects.at(n);
				tobj.model_center = obj.model_center;
				tobj.obj_size = obj.obj_size;
				tobj.img = obj.img;
				tobj.det_mc = true;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < id_frame && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
					, obj.obj_size.x, obj.obj_size.y});
			}
			else
			{

				for (size_t j = 0; j < objects.size(); j++)
				{
					if (objects[i].det_pos == false)
						continue;

					float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
					if (r < (float)robj * 2.3 * (float)rows / (float)reduseres)
					{
						newobj = false;
						break;
					}
				}

				if (newobj == true)
					objects.push_back(obj);
			}
		}
	}
	//--------------------</detection using a classifier>---------

	//--------------------<moution detections>--------------------

	float rwsize;
	float clsize;

	imagbuf = frame;

	if (rows > cols)
	{
		rwsize = (float)frame_resolution * rows / (float)cols;
		clsize = (float)rows;
	}
	else
	{
		rwsize = (float)rows;
		clsize = (float)frame_resolution * cols / (float)rows;
		;
	}

	cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	cv::Rect rectb(0, 0, rows, rows);
	imag = imagbuf(rectb);

	cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
	imag.convertTo(imag, CV_8UC3);

	if (rows > cols)
	{
		rwsize = (float)reduseres * rows / (float)cols;
		clsize = reduseres;
	}
	else
	{
		rwsize = reduseres;
		clsize = (float)reduseres * cols / (float)rows;
	}

	cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	cv::Rect rect0(0, 0, reduseres, reduseres);
	imageBGR0 = frame0(rect0);
	imageBGR0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	cv::Rect rect(0, 0, reduseres, reduseres);
	imageBGR = frame(rect);

	imageBGR.convertTo(imageBGR, CV_8UC1);

	Point2f pm;

	imageBGR0 = color_correction(imageBGR0);
	imageBGR = color_correction(imageBGR);

	for (uint16_t y = 0; y < imageBGR0.rows; y++)
	{
		for (uint16_t x = 0; x < imageBGR0.cols; x++)
		{
			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
			uchar color2 = imageBGR.at<uchar>(Point(x, y));

			if (((int)color2 - (int)color1) > pft)
			{
				pm.x = x * rows / reduseres;
				pm.y = y * rows / reduseres;
				motion.push_back(pm);
			}
		}
	}

	Point2f pt1;
	Point2f pt2;

	uint16_t ncls = 0;
	uint16_t nobj;

	if (objects.size() > 0)
		nobj = 0;
	else
		nobj = -1;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj > -1 && nobj < objects.size())
		{
			pc = objects[nobj].cluster_center;
			// pc = objects[nobj].proposed_center();
			nobj++;
		}
		else
		{
			pc = motion.at(0);
			motion.erase(motion.begin());
		}

		clusters.push_back(vector<Point2f>());
		clusters[ncls].push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
			if (r < nd * (float)rows / (float)reduseres)
			{
				Point2f cl_c = cluster_center(clusters.at(ncls));
				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
				if (r < (float)mdist * (float)rows / (float)reduseres)
				{
					clusters.at(ncls).push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < clusters[ncls].size(); c++)
			{
				pc = clusters[ncls].at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

					if (r < nd * (float)rows / (float)reduseres)
					{
						Point2f cl_c = cluster_center(clusters.at(ncls));
						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
						if (r < (float)mdist * (float)rows / (float)reduseres)
						{
							clusters.at(ncls).push_back(motion.at(i));
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
	//--------------</cluster creation>----------------------

	//--------------<clusters to objects>--------------------
	if (objects.size() > 0)
	{
		for (size_t i = 0; i < objects.size(); i++)
			objects[i].det_pos = false;

		vector<ClsObjR> clsobjrs;
		ClsObjR clsobjr;
		for (size_t i = 0; i < objects.size(); i++)
		{
			for (size_t cls = 0; cls < clusters.size(); cls++)
			{
				clsobjr.cls_id = cls;
				clsobjr.obj_id = i;
				Point2f clustercenter = cluster_center(clusters[cls]);
				// clsobjr.r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				clsobjr.r = sqrt(pow((objects[i].proposed_center().x - clustercenter.x), 2) + pow((objects[i].proposed_center().y - clustercenter.y), 2));
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (usedetector == true)
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				size_t obj_id = clsobjrs.at(i).obj_id;

				if (objects.at(obj_id).det_mc == true)
				{
					Point2f clustercenter = cluster_center(clusters[cls_id]);
					auto& cobj = objects[obj_id];
					pt1.x = cobj.model_center.x - cobj.obj_size.x / 2;
					pt1.y = cobj.model_center.y - cobj.obj_size.y / 2;

					pt2.x = cobj.model_center.x + cobj.obj_size.x / 2;
					pt2.y = cobj.model_center.y + cobj.obj_size.y / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{

						if (cobj.det_pos == false)
							cobj.cluster_points = clusters.at(cls_id);
						else
						{
							for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
								cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
						}

						cobj.center_determine(id_frame, false);

						for (size_t j = 0; j < clsobjrs.size(); j++)
						{
							if (clsobjrs.at(j).cls_id == cls_id)
							{
								clsobjrs.erase(clsobjrs.begin() + j);
								j--;
							}
						}
						i = 0;
					}
				}
			}
		//--</corr obj use model>---

		//---<det obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * (float)rows / (float)reduseres && clusters.at(cls_id).size() > mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (cobj.det_pos == false)
					cobj.cluster_points = clusters.at(cls_id);
				else
				{
					continue;
					// for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					//   cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
				}

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//---</det obj>---
		//---<new obj>----
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			Point2f clustercenter = cluster_center(clusters[cls_id]);
			bool newobj = true;

			for (size_t j = 0; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - clustercenter.x), 2) + pow((objects[j].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * 2.3 * (float)rows / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls_id].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);

				framebuf.convertTo(imagbuf, CV_8UC3);

				img = imagbuf(cv::Range(max_u16(clustercenter.y - half_imgsize * koef)
					, min_u16(imagbuf.rows, clustercenter.y + half_imgsize * koef))
					, cv::Range(max_u16(clustercenter.x - half_imgsize * koef)
					, min_u16(imagbuf.cols, clustercenter.x + half_imgsize * koef)));
				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls_id], img);
				objects.push_back(obj);
				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</new obj>--

		//--<corr obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * robj_k * (float)rows / (float)reduseres && clusters.at(cls_id).size() > mpct / 2)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					cobj.cluster_points.push_back(clusters.at(cls_id).at(j));

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</corr obj>---
	}
	else
	{
		//--<new obj>--
		for (int cls = 0; cls < clusters.size(); cls++)
		{
			Point2f clustercenter = cluster_center(clusters[cls]);
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				float r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * (float)rows / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				framebuf.convertTo(imagbuf, CV_8UC3);
				img = imagbuf(cv::Range(max_u16(clustercenter.y - half_imgsize * koef)
					, min_u16(imagbuf.rows, clustercenter.y + half_imgsize * koef))
					, cv::Range(max_u16(clustercenter.x - half_imgsize * koef)
					, min_u16(imagbuf.cols, clustercenter.x + half_imgsize * koef)));
				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls], img);
				objects.push_back(obj);
				clusters.erase(clusters.begin() + cls);
				cls--;

				if (cls < 0)
					cls = 0;
			}
		}
		//--</new obj>--
	}
	//--------------</clusters to objects>-------------------

	//--------------<post processing>-----------------------
	for (int i = 0; i < objects.size(); i++)
	{
		if (objects.at(i).det_mc == false && objects.at(i).det_pos == false)
			continue;

		// imagbuf = frame_resizing(framebuf);
		// imagbuf.convertTo(imagbuf, CV_8UC3);

		framebuf.convertTo(imagbuf, CV_8UC3);

		if (objects[i].det_mc == false)
		{
			pt1.y = objects[i].cluster_center.y - half_imgsize * koef;
			pt2.y = objects[i].cluster_center.y + half_imgsize * koef;

			pt1.x = objects[i].cluster_center.x - half_imgsize * koef;
			pt2.x = objects[i].cluster_center.x + half_imgsize * koef;
		}
		else
		{
			pt1.y = objects[i].model_center.y - half_imgsize * koef;
			pt2.y = objects[i].model_center.y + half_imgsize * koef;

			pt1.x = objects[i].model_center.x - half_imgsize * koef;
			pt2.x = objects[i].model_center.x + half_imgsize * koef;
		}

		if (pt1.y < 0)
			pt1.y = 0;

		if (pt2.y > imagbuf.rows)
			pt2.y = imagbuf.rows;

		if (pt1.x < 0)
			pt1.x = 0;

		if (pt2.x > imagbuf.cols)
			pt2.x = imagbuf.cols;

		// std::cout << "<post processing 2>" << endl;
		// std::cout << "pt1 - " << pt1 << endl;
		// std::cout << "pt2 - " << pt2 << endl;

		img = imagbuf(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_8UC3);
		img.copyTo(objects[i].img);
		objects[i].center_determine(id_frame, true);

		if (objects[i].det_mc == false)
			objects[i].push_track_point(objects[i].cluster_center);
		else
			objects[i].push_track_point(objects[i].model_center);
	}

	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = objects.at(i).cluster_points.at(j).x;
			pt1.y = objects.at(i).cluster_points.at(j).y;

			pt2.x = objects.at(i).cluster_points.at(j).x + (float)rows / (float)reduseres;
			pt2.y = objects.at(i).cluster_points.at(j).y + (float)rows / (float)reduseres;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		if (objects.at(i).det_mc == true) // visualization of the classifier
		{
			pt1.x = objects.at(i).model_center.x - objects.at(i).obj_size.x / 2;
			pt1.y = objects.at(i).model_center.y - objects.at(i).obj_size.y / 2;

			pt2.x = objects.at(i).model_center.x + objects.at(i).obj_size.x / 2;
			pt2.y = objects.at(i).model_center.y + objects.at(i).obj_size.y / 2;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		for (int j = 0; j < objects.at(i).track_points.size(); j++)
			cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color(objects.at(i).id), 2);
	}
	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------
	Mat baseimag(rows, rows + extr * koef, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < objects.size(); i++)
	{
		string text = objects.at(i).obj_type + " ID" + to_string(objects.at(i).id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								class_name_color(objects.at(i).id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + objects.at(i).img.cols + 1;
		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, class_name_color(objects.at(i).id), 1);
			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
		}
	}
	imag.copyTo(baseimag(cv::Rect(extr * koef, 0, imag.cols, imag.rows)));

	Point2f p_idframe;
	p_idframe.x = rows + (extr - 95) * koef;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(id_frame), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	cv::resize(baseimag, baseimag, cv::Size(992 + extr, 992), cv::InterpolationFlags::INTER_CUBIC);
	imshow("Motion", baseimag);
	cv::waitKey(0);
	//--------------</baseimag>-------------------------------*/

	vector<std::pair<Point2f, uint16_t>> detects_P2f_id;

	for (int i = 0; i < objects.size(); i++)
	{
		if (objects[i].det_mc == true)
			detects_P2f_id.push_back(std::make_pair(objects[i].model_center, objects[i].id));
		else if (objects[i].det_pos == true)
			detects_P2f_id.push_back(std::make_pair(objects[i].cluster_center, objects[i].id));
	}

	return detects_P2f_id;
}

const float GOOD_MATCH_PERCENT = 0.99f;
int nfeatures = 20000;
float scaleFactor = 1.2f;
int nlevels = 28;
int edgeThreshold = 31;
int firstLevel = 0;
int WTA_K = 2;

int patchSize = 31;
int fastThreshold = 20;

std::tuple<vector<Point2f>, vector<Point2f>, Mat> detectORB(Mat &im1, Mat &im2, float reskoef)
{
	// Convert images to grayscale
	Mat im1Gray, im2Gray, imbuf;
	// cv::cvtColor(im1, im1Gray, cv::COLOR_BGR2GRAY);
	// cv::cvtColor(im2, im2Gray, cv::COLOR_BGR2GRAY);

	// im1.copyTo(im1Gray);
	// im2.copyTo(im2Gray);

	cv::resize(im1, im1Gray, cv::Size(im1.cols / reskoef, im1.rows / reskoef), cv::InterpolationFlags::INTER_CUBIC);
	cv::resize(im2, im2Gray, cv::Size(im2.cols / reskoef, im2.rows / reskoef), cv::InterpolationFlags::INTER_CUBIC);

	cv::resize(im2, imbuf, cv::Size(im2.cols / reskoef, im2.rows / reskoef), cv::InterpolationFlags::INTER_CUBIC);

	// Variables to store keypoints and descriptors
	vector<cv::KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	cv::Ptr<cv::Feature2D> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);

	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features.
	vector<cv::DMatch> matches;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin() + numGoodMatches, matches.end());

	// Draw top matches
	Mat imMatches;

	for (int i = 0; i < matches.size(); i++)
	{
		if (cv::norm(keypoints1[matches[i].queryIdx].pt - keypoints2[matches[i].trainIdx].pt) < 5 || cv::norm(keypoints1[matches[i].queryIdx].pt - keypoints2[matches[i].trainIdx].pt) > 20)
		{
			matches.erase(matches.begin() + i);
			i--;
		}
	}

	cv::drawMatches(im1Gray, keypoints1, im2Gray, keypoints2, matches, imMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	cv::resize(imMatches, imMatches, cv::Size(imMatches.cols, imMatches.rows), cv::InterpolationFlags::INTER_CUBIC);
	// imshow("imMatches", imMatches);
	// cv::waitKey(10);

	// Extract location of good matches
	vector<Point2f> points1, points2;

	Point2f p1, p2;
	for (size_t i = 0; i < matches.size(); i++)
	{
		p1.x = keypoints1[matches[i].queryIdx].pt.x * reskoef;
		p1.y = keypoints1[matches[i].queryIdx].pt.y * reskoef;

		p2.x = keypoints2[matches[i].trainIdx].pt.x * reskoef;
		p2.y = keypoints2[matches[i].trainIdx].pt.y * reskoef;

		points1.push_back(p1);
		points2.push_back(p2);

		cv::circle(imbuf, p2, 3, Scalar(200, 0, 200), 1);
	}
	return std::make_tuple(points1, points2, imMatches);
}

Mat trackingMotV2_2(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, bool usedetector, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	// if (usedetector)
	//   frame = frame_resizing(frame);

	frame = frame_resizing(frame);
	frame0 = frame_resizing(frame0);

	Mat framebuf;

	frame.copyTo(framebuf);

	int mpct = 3;      // minimum number of points for a cluster (tracking object) (good value 5)
	int mpcc = 7;      // minimum number of points for a cluster (creation new object) (good value 13)
	float nd = 1.5;    //(good value 6-15)
	int rcobj = 15;    //(good value 15)
	float robj = 17.0; //(good value 17)

	float rORB = 10;
	float robj_k = 1.0;
	int mdist = 10; // maximum distance from cluster center (good value 10)
	int pft = 1;    // points fixation threshold (good value 9)

	Mat img;

	vector<OBJdetect> detects;

	vector<Point2f> points1ORB;

	vector<Point2f> points2ORB;

	std::tuple<vector<Point2f>, vector<Point2f>, Mat> detectsORB;
	//--------------------<detection using a classifier or ORB detection>----------
	if (usedetector)
	{

		detects = detectorV4(pathmodel, frame, device_type, confidence);

		for (int i = 0; i < objects.size(); i++)
		{
			objects[i].det_mc = false;
		}

		for (int i = 0; i < detects.size(); i++)
		{
			if (detects.at(i).type != "a")
			{
				detects.erase(detects.begin() + i);
				i--;
			}
		}

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			vector<Point2f> cluster_points;
			cluster_points.push_back(detects.at(i).detect);
			// imagbuf = frame_resizing(framebuf);
			// img = imagbuf(cv::Range(detects.at(i).detect.y - half_imgsize, detects.at(i).detect.y + half_imgsize), cv::Range(detects.at(i).detect.x - half_imgsize, detects.at(i).detect.x + half_imgsize));

			img = framebuf(cv::Range(max_u16(detects.at(i).detect.y - half_imgsize)
				, min_u16(framebuf.rows, detects.at(i).detect.y + half_imgsize))
				, cv::Range(max_u16(detects.at(i).detect.x - half_imgsize)
				, min_u16(framebuf.cols, detects.at(i).detect.x + half_imgsize)));

			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), detects.at(i).type, cluster_points, img);
			obj.model_center = detects.at(i).detect;
			obj.cluster_center = detects.at(i).detect;
			obj.obj_size = detects.at(i).obj_size;
			obj.det_mc = true;
			// obj.track_points.push_back(detects.at(i).detect);
			// obj.push_track_point(detects.at(i).detect);

			float rm = rcobj * (float)resolution / (float)reduseres;
			bool newobj = true;
			uint16_t n;

			if (objects.size() > 0)
			{
				rm = sqrt(pow((objects[0].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[0].cluster_center.y - obj.cluster_center.y), 2));
				// rm = sqrt(pow((objects[0].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[0].proposed_center().y - obj.cluster_center.y), 2));
				if (rm < rcobj * (float)resolution / (float)reduseres)
				{
					n = 0;
					newobj = false;
				}
			}

			for (uint16_t j = 1; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
				// float r = sqrt(pow((objects[j].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[j].proposed_center().y - obj.cluster_center.y), 2));
				if (r < rcobj * (float)resolution / (float)reduseres && r < rm)
				{
					rm = r;
					n = j;
					newobj = false;
				}
			}

			if (newobj == false)
			{
				auto& tobj = objects.at(n);
				tobj.model_center = obj.model_center;
				tobj.obj_size = obj.obj_size;
				tobj.img = obj.img;
				tobj.det_mc = true;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < id_frame && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
					, obj.obj_size.x, obj.obj_size.y});
			}
			else
			{

				for (size_t j = 0; j < objects.size(); j++)
				{
					if (objects[i].det_pos == false)
						continue;

					float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
					if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
					{
						newobj = false;
						break;
					}
				}

				if (newobj == true)
					objects.push_back(obj);
			}
		}
	}
	else
	{
		vector<ClsObjR> orbobjrs;
		detectsORB = detectORB(frame0, frame, 1);

		for (size_t i = 0; i < objects.size(); i++)
			objects[i].ORB_ids.clear();

		//----------------------<points1ORB>---------------------------------------------
		points1ORB = std::get<0>(detectsORB);
		ClsObjR orbobjr;
		for (size_t i = 0; i < objects.size(); i++)
		{
			for (size_t orb = 0; orb < points1ORB.size(); orb++)
			{
				orbobjr.cls_id = orb;
				orbobjr.obj_id = i;
				orbobjr.r = sqrt(pow((objects[i].cluster_center.x - points1ORB.at(orb).x), 2) + pow((objects[i].cluster_center.y - points1ORB.at(orb).y), 2));
				orbobjrs.push_back(orbobjr);
			}
		}

		sort(orbobjrs.begin(), orbobjrs.end(), compare_clsobj);

		for (size_t i = 0; i < orbobjrs.size(); i++)
		{
			size_t orb_id = orbobjrs.at(i).cls_id;
			size_t obj_id = orbobjrs.at(i).obj_id;

			if (orbobjrs.at(i).r < (float)rORB * (float)resolution / (float)reduseres)
				objects.at(obj_id).ORB_ids.push_back(orb_id);

			for (size_t j = 0; j < orbobjrs.size(); j++)
			{
				if (orbobjrs.at(j).cls_id == orb_id)
				{
					orbobjrs.erase(orbobjrs.begin() + j);
					j--;
				}
			}
		}
		//--------------------</points1ORB>---------------------------
	}
	//--------------------</detection using a classifier or ORB detection>---------

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

	// cv::Rect rect0(0, 0, reduseres, reduseres);
	// imageBGR0 = frame0(rect0);

	frame0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	// cv::Rect rect(0, 0, reduseres, reduseres);
	// imageBGR = frame(rect);

	frame.convertTo(imageBGR, CV_8UC1);

	Point2f pm;

	imageBGR0 = color_correction(imageBGR0);
	imageBGR = color_correction(imageBGR);

	for (uint16_t y = 0; y < imageBGR0.rows; y++)
	{
		for (uint16_t x = 0; x < imageBGR0.cols; x++)
		{
			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
			uchar color2 = imageBGR.at<uchar>(Point(x, y));

			if (((int)color2 - (int)color1) > pft)
			{
				pm.x = (float)x * (float)resolution / (float)reduseres;
				pm.y = (float)y * (float)resolution / (float)reduseres;
				motion.push_back(pm);
			}
		}
	}

	Point2f pt1;
	Point2f pt2;

	if (false == true)
		for (int i = 0; i < motion.size(); i++) // visualization of the white cluster_points
		{
			pt1.x = motion.at(i).x;
			pt1.y = motion.at(i).y;

			pt2.x = motion.at(i).x + (float)resolution / (float)reduseres;
			pt2.y = motion.at(i).y + (float)resolution / (float)reduseres;

			rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
		}

	uint16_t ncls = 0;
	uint16_t nobj;

	if (objects.size() > 0)
		nobj = 0;
	else
		nobj = -1;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj > -1 && nobj < objects.size())
		{
			pc = objects[nobj].cluster_center;
			// pc = objects[nobj].proposed_center();
			nobj++;
		}
		else
		{
			pc = motion.at(0);
			motion.erase(motion.begin());
		}

		clusters.push_back(vector<Point2f>());
		clusters[ncls].push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
			if (r < (float)nd * (float)resolution / (float)reduseres)
			{
				Point2f cl_c = cluster_center(clusters.at(ncls));
				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
				if (r < (float)mdist * (float)resolution / (float)reduseres)
				{
					clusters.at(ncls).push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < clusters[ncls].size(); c++)
			{
				pc = clusters[ncls].at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

					if (r < (float)nd * (float)resolution / (float)reduseres)
					{
						Point2f cl_c = cluster_center(clusters.at(ncls));
						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
						if (r < mdist * 1.0 * resolution / reduseres)
						{
							clusters.at(ncls).push_back(motion.at(i));
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
	//--------------</cluster creation>----------------------

	//--------------<clusters to objects>--------------------
	if (objects.size() > 0)
	{
		for (size_t i = 0; i < objects.size(); i++)
			objects[i].det_pos = false;

		vector<ClsObjR> clsobjrs;
		ClsObjR clsobjr;
		for (size_t i = 0; i < objects.size(); i++)
		{
			for (size_t cls = 0; cls < clusters.size(); cls++)
			{
				clsobjr.cls_id = cls;
				clsobjr.obj_id = i;
				Point2f clustercenter = cluster_center(clusters[cls]);
				// clsobjr.r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				clsobjr.r = sqrt(pow((objects[i].proposed_center().x - clustercenter.x), 2) + pow((objects[i].proposed_center().y - clustercenter.y), 2));
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (usedetector == true)
		{
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				size_t obj_id = clsobjrs.at(i).obj_id;

				if (objects.at(obj_id).det_mc == true)
				{
					Point2f clustercenter = cluster_center(clusters[cls_id]);
					auto& cobj = objects.at(obj_id);
					pt1.x = cobj.model_center.x - cobj.obj_size.x / 2;
					pt1.y = cobj.model_center.y - cobj.obj_size.y / 2;

					pt2.x = cobj.model_center.x + cobj.obj_size.x / 2;
					pt2.y = cobj.model_center.y + cobj.obj_size.y / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{
						if (cobj.det_pos == false)
							cobj.cluster_points = clusters.at(cls_id);
						else
						{
							for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
								cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
						}

						cobj.center_determine(id_frame, false);

						for (size_t j = 0; j < clsobjrs.size(); j++)
						{
							if (clsobjrs.at(j).cls_id == cls_id)
							{
								clsobjrs.erase(clsobjrs.begin() + j);
								j--;
							}
						}
						i = 0;
					}
				}
			}
		}
		else
		{
			//----------------------<points2ORB>---------------------------------------------

			points2ORB = std::get<1>(detectsORB);

			vector<ClsObjR> orbobjrs;
			ClsObjR orbobjr;

			for (size_t i = 0; i < clusters.size(); i++)
			{
				for (size_t orb = 0; orb < points2ORB.size(); orb++)
				{
					orbobjr.cls_id = orb;
					orbobjr.obj_id = i;

					Point2f clustercenter = cluster_center(clusters[i]);
					orbobjr.r = sqrt(pow((clustercenter.x - points2ORB.at(orb).x), 2) + pow((clustercenter.y - points2ORB.at(orb).y), 2));
					orbobjrs.push_back(orbobjr);
				}
			}

			sort(orbobjrs.begin(), orbobjrs.end(), compare_clsobj);

			for (size_t i = 0; i < orbobjrs.size(); i++)
			{
				size_t orb_id = orbobjrs.at(i).cls_id; // cls_id - means orb_id!!!
				size_t cls_id = orbobjrs.at(i).obj_id; // obj_id - means cls_id!!!

				if (orbobjrs.at(i).r < (float)rORB * (float)resolution / (float)reduseres && clusters.at(cls_id).size() > mpct)
				{
					size_t obj_id;
					bool orb_in_obj = false;
					for (size_t j = 0; j < objects.size(); j++)
					{
						for (size_t o = 0; o < objects[j].ORB_ids.size(); o++)
						{
							if (objects[j].ORB_ids[o] == orb_id)
							{
								obj_id = j;
								orb_in_obj = true;
								goto jump1;
							}
						}
					}
				jump1:
					if (orb_in_obj == true)
					{
						auto& cobj = objects.at(obj_id);
						if (cobj.det_pos == false)
							cobj.cluster_points = clusters.at(cls_id);
						else
						{
							for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
								cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
						}

						cobj.center_determine(id_frame, false);

						for (size_t j = 0; j < orbobjrs.size(); j++)
						{
							if (orbobjrs.at(j).cls_id == orb_id)
							{
								orbobjrs.erase(orbobjrs.begin() + j);
								j--;
							}
						}

						for (size_t j = 0; j < clsobjrs.size(); j++)
						{
							if (clsobjrs.at(j).cls_id == cls_id)
							{
								clsobjrs.erase(clsobjrs.begin() + j);
								j--;
							}
						}
						i = 0;
					}
				}
			}
			//--------------------</points2ORB>---------------------------
		}
		//--</corr obj use model>---

		//---<det obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * (float)resolution / (float)reduseres && clusters.at(cls_id).size() > mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (cobj.det_pos == false)
					cobj.cluster_points = clusters.at(cls_id);
				else
				{
					continue;
					// for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					//   cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
				}

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//---</det obj>---
		//---<new obj>----
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			Point2f clustercenter = cluster_center(clusters[cls_id]);
			bool newobj = true;

			for (size_t j = 0; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - clustercenter.x), 2) + pow((objects[j].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls_id].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - half_imgsize, clustercenter.y + half_imgsize), cv::Range(clustercenter.x - half_imgsize, clustercenter.x + half_imgsize));

				img = framebuf(cv::Range(max_u16(clustercenter.y - half_imgsize)
					, min_u16(framebuf.rows, clustercenter.y + half_imgsize))
					, cv::Range(max_u16(clustercenter.x - half_imgsize)
					, min_u16(framebuf.cols, clustercenter.x + half_imgsize)));

				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls_id], img);
				objects.push_back(obj);
				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</new obj>--

		//--<corr obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * robj_k * (float)resolution / (float)reduseres && clusters.at(cls_id).size() > mpct / 2)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					cobj.cluster_points.push_back(clusters.at(cls_id).at(j));

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</corr obj>---
	}
	else
	{
		//--<new obj>--
		for (int cls = 0; cls < clusters.size(); cls++)
		{
			Point2f clustercenter = cluster_center(clusters[cls]);
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				float r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// magbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - half_imgsize, clustercenter.y + half_imgsize), cv::Range(clustercenter.x - half_imgsize, clustercenter.x + half_imgsize));

				img = framebuf(cv::Range(max_u16(clustercenter.y - half_imgsize)
					, min_u16(framebuf.rows, clustercenter.y + half_imgsize))
					, cv::Range(max_u16(clustercenter.x - half_imgsize)
					, min_u16(framebuf.cols, clustercenter.x + half_imgsize)));

				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls], img);
				objects.push_back(obj);
				clusters.erase(clusters.begin() + cls);
				cls--;

				if (cls < 0)
					cls = 0;
			}
		}
		//--</new obj>--
	}
	//--------------</clusters to objects>-------------------

	//--------------<post processing>-----------------------
	for (int i = 0; i < objects.size(); i++)
	{
		if (objects.at(i).det_mc == false && objects.at(i).det_pos == false)
			continue;

		// imagbuf = frame_resizing(framebuf);
		// imagbuf.convertTo(imagbuf, CV_8UC3);

		// imagbuf = frame_resizing(framebuf);
		framebuf.convertTo(imagbuf, CV_8UC3);

		if (objects[i].det_mc == false)
		{
			pt1.y = objects[i].cluster_center.y - half_imgsize;
			pt2.y = objects[i].cluster_center.y + half_imgsize;

			pt1.x = objects[i].cluster_center.x - half_imgsize;
			pt2.x = objects[i].cluster_center.x + half_imgsize;
		}
		else
		{
			pt1.y = objects[i].model_center.y - half_imgsize;
			pt2.y = objects[i].model_center.y + half_imgsize;

			pt1.x = objects[i].model_center.x - half_imgsize;
			pt2.x = objects[i].model_center.x + half_imgsize;
		}

		if (pt1.y < 0)
			pt1.y = 0;

		if (pt2.y > imagbuf.rows)
			pt2.y = imagbuf.rows;

		if (pt1.x < 0)
			pt1.x = 0;

		if (pt2.x > imagbuf.cols)
			pt2.x = imagbuf.cols;

		// std::cout << "<post processing 2>" << endl;
		// std::cout << "pt1 - " << pt1 << endl;
		// std::cout << "pt2 - " << pt2 << endl;

		img = imagbuf(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_8UC3);
		img.copyTo(objects[i].img);
		objects[i].center_determine(id_frame, true);

		if (objects[i].det_mc == false)
			objects[i].push_track_point(objects[i].cluster_center);
		else
			objects[i].push_track_point(objects[i].model_center);
	}
	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = objects.at(i).cluster_points.at(j).x;
			pt1.y = objects.at(i).cluster_points.at(j).y;

			pt2.x = objects.at(i).cluster_points.at(j).x + resolution / reduseres;
			pt2.y = objects.at(i).cluster_points.at(j).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		if (objects.at(i).det_mc == true) // visualization of the classifier
		{
			pt1.x = objects.at(i).model_center.x - objects.at(i).obj_size.x / 2;
			pt1.y = objects.at(i).model_center.y - objects.at(i).obj_size.y / 2;

			pt2.x = objects.at(i).model_center.x + objects.at(i).obj_size.x / 2;
			pt2.y = objects.at(i).model_center.y + objects.at(i).obj_size.y / 2;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		for (int j = 0; j < objects.at(i).track_points.size(); j++)
			cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color(objects.at(i).id), 2);

		for (int j = 0; j < objects[i].ORB_ids.size(); j++)
			cv::circle(imag, points2ORB.at(objects[i].ORB_ids.at(j)), 3, class_name_color(objects.at(i).id), 1);
	}

	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------

	Mat baseimag(resolution, 3 * resolution + extr, CV_8UC3, Scalar(0, 0, 0));
	// std::cout << "<baseimag 1>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		string text = objects.at(i).obj_type + " ID" + to_string(objects.at(i).id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								class_name_color(objects.at(i).id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + objects.at(i).img.cols + 1;
		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, class_name_color(objects.at(i).id), 1);
			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
		}
	}
	// std::cout << "<baseimag 2>" << endl;
	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

	cv::resize(std::get<2>(detectsORB), std::get<2>(detectsORB), cv::Size(2 * resolution, resolution), cv::InterpolationFlags::INTER_CUBIC);

	std::get<2>(detectsORB).copyTo(baseimag(cv::Rect(resolution + extr, 0, std::get<2>(detectsORB).cols, std::get<2>(detectsORB).rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(id_frame), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------

	imshow("Motion", baseimag);
	cv::waitKey(0);

	return baseimag;
}

Mat trackingMotV2_3(string pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t id_frame, bool usedetector, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	// if (usedetector)
	//   frame = frame_resizing(frame);

	frame = frame_resizing(frame);
	frame0 = frame_resizing(frame0);

	Mat framebuf;

	frame.copyTo(framebuf);

	int mpct = 3;      // minimum number of points for a cluster (tracking object) (good value 5)
	int mpcc = 7;      // minimum number of points for a cluster (creation new object) (good value 13)
	float nd = 1.5;    //(good value 6-15)
	int rcobj = 15;    //(good value 15)
	float robj = 17.0; //(good value 17)

	float rORB = 2.0;
	float robj_k = 1.0;
	int mdist = 10; // maximum distance from cluster center (good value 10)
	int pft = 1;    // points fixation threshold (good value 9)

	Mat img;

	vector<OBJdetect> detects;

	vector<Point2f> points1ORB;

	vector<Point2f> points2ORB;

	std::tuple<vector<Point2f>, vector<Point2f>, Mat> detectsORB;
	//--------------------<detection using a classifier or ORB detection>----------
	if (usedetector)
	{

		detects = detectorV4(pathmodel, frame, device_type, confidence);

		for (int i = 0; i < objects.size(); i++)
		{
			objects[i].det_mc = false;
		}

		for (int i = 0; i < detects.size(); i++)
		{
			if (detects.at(i).type != "a")
			{
				detects.erase(detects.begin() + i);
				i--;
			}
		}

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			vector<Point2f> cluster_points;
			cluster_points.push_back(detects.at(i).detect);
			// imagbuf = frame_resizing(framebuf);
			// img = imagbuf(cv::Range(detects.at(i).detect.y - half_imgsize, detects.at(i).detect.y + half_imgsize), cv::Range(detects.at(i).detect.x - half_imgsize, detects.at(i).detect.x + half_imgsize));

			img = framebuf(cv::Range(max_u16(detects.at(i).detect.y - half_imgsize)
				, min_u16(framebuf.rows, detects.at(i).detect.y + half_imgsize))
				, cv::Range(max_u16(detects.at(i).detect.x - half_imgsize)
				, min_u16(framebuf.cols, detects.at(i).detect.x + half_imgsize)));

			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), detects.at(i).type, cluster_points, img);
			obj.model_center = detects.at(i).detect;
			obj.cluster_center = detects.at(i).detect;
			obj.obj_size = detects.at(i).obj_size;
			obj.det_mc = true;
			// obj.track_points.push_back(detects.at(i).detect);
			// obj.push_track_point(detects.at(i).detect);

			float rm = rcobj * (float)resolution / (float)reduseres;
			bool newobj = true;
			uint16_t n;

			if (objects.size() > 0)
			{
				rm = sqrt(pow((objects[0].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[0].cluster_center.y - obj.cluster_center.y), 2));
				// rm = sqrt(pow((objects[0].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[0].proposed_center().y - obj.cluster_center.y), 2));
				if (rm < rcobj * (float)resolution / (float)reduseres)
				{
					n = 0;
					newobj = false;
				}
			}

			for (uint16_t j = 1; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
				// float r = sqrt(pow((objects[j].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[j].proposed_center().y - obj.cluster_center.y), 2));
				if (r < rcobj * (float)resolution / (float)reduseres && r < rm)
				{
					rm = r;
					n = j;
					newobj = false;
				}
			}

			if (newobj == false)
			{
				auto& tobj = objects.at(n);
				tobj.model_center = obj.model_center;
				tobj.obj_size = obj.obj_size;
				tobj.img = obj.img;
				tobj.det_mc = true;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < id_frame && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{id_frame, obj.cluster_center.x, obj.cluster_center.y
					, obj.obj_size.x, obj.obj_size.y});
			}
			else
			{

				for (size_t j = 0; j < objects.size(); j++)
				{
					if (objects[i].det_pos == false)
						continue;

					float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
					if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
					{
						newobj = false;
						break;
					}
				}

				if (newobj == true)
					objects.push_back(obj);
			}
		}
	}
	else
	{
		vector<ClsObjR> orbobjrs;
		detectsORB = detectORB(frame0, frame, 1);

		for (size_t i = 0; i < objects.size(); i++)
			objects[i].ORB_ids.clear();

		//----------------------<points1ORB>---------------------------------------------
		points1ORB = std::get<0>(detectsORB);
		ClsObjR orbobjr;
		for (size_t i = 0; i < objects.size(); i++)
		{
			for (size_t j = 0; j < objects[i].cluster_points.size(); j++)
			{
				for (size_t orb = 0; orb < points1ORB.size(); orb++)
				{
					float r = sqrt(pow((objects[i].cluster_points[j].x - points1ORB.at(orb).x), 2) + pow((objects[i].cluster_points[j].y - points1ORB.at(orb).y), 2));

					if (r < (float)rORB * (float)resolution / (float)reduseres)
						objects[i].ORB_ids.push_back(orb);
				}
			}
		}
		//--------------------</points1ORB>---------------------------
	}
	//--------------------</detection using a classifier or ORB detection>---------

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

	// cv::Rect rect0(0, 0, reduseres, reduseres);
	// imageBGR0 = frame0(rect0);

	frame0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	// cv::Rect rect(0, 0, reduseres, reduseres);
	// imageBGR = frame(rect);

	frame.convertTo(imageBGR, CV_8UC1);

	Point2f pm;

	imageBGR0 = color_correction(imageBGR0);
	imageBGR = color_correction(imageBGR);

	for (uint16_t y = 0; y < imageBGR0.rows; y++)
	{
		for (uint16_t x = 0; x < imageBGR0.cols; x++)
		{
			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
			uchar color2 = imageBGR.at<uchar>(Point(x, y));

			if (((int)color2 - (int)color1) > pft)
			{
				pm.x = (float)x * (float)resolution / (float)reduseres;
				pm.y = (float)y * (float)resolution / (float)reduseres;
				motion.push_back(pm);
			}
		}
	}

	Point2f pt1;
	Point2f pt2;

	if (false == true)
		for (int i = 0; i < motion.size(); i++) // visualization of the white cluster_points
		{
			pt1.x = motion.at(i).x;
			pt1.y = motion.at(i).y;

			pt2.x = motion.at(i).x + (float)resolution / (float)reduseres;
			pt2.y = motion.at(i).y + (float)resolution / (float)reduseres;

			rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
		}

	uint16_t ncls = 0;
	uint16_t nobj;

	if (objects.size() > 0)
		nobj = 0;
	else
		nobj = -1;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj > -1 && nobj < objects.size())
		{
			pc = objects[nobj].cluster_center;
			// pc = objects[nobj].proposed_center();
			nobj++;
		}
		else
		{
			pc = motion.at(0);
			motion.erase(motion.begin());
		}

		clusters.push_back(vector<Point2f>());
		clusters[ncls].push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
			if (r < (float)nd * (float)resolution / (float)reduseres)
			{
				Point2f cl_c = cluster_center(clusters.at(ncls));
				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
				if (r < (float)mdist * (float)resolution / (float)reduseres)
				{
					clusters.at(ncls).push_back(motion.at(i));
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < clusters[ncls].size(); c++)
			{
				pc = clusters[ncls].at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

					if (r < (float)nd * (float)resolution / (float)reduseres)
					{
						Point2f cl_c = cluster_center(clusters.at(ncls));
						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
						if (r < mdist * 1.0 * resolution / reduseres)
						{
							clusters.at(ncls).push_back(motion.at(i));
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
	//--------------</cluster creation>----------------------

	//--------------<clusters to objects>--------------------
	if (objects.size() > 0)
	{
		for (size_t i = 0; i < objects.size(); i++)
			objects[i].det_pos = false;

		vector<ClsObjR> clsobjrs;
		ClsObjR clsobjr;
		for (size_t i = 0; i < objects.size(); i++)
		{
			for (size_t cls = 0; cls < clusters.size(); cls++)
			{
				clsobjr.cls_id = cls;
				clsobjr.obj_id = i;
				Point2f clustercenter = cluster_center(clusters[cls]);
				// clsobjr.r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				clsobjr.r = sqrt(pow((objects[i].proposed_center().x - clustercenter.x), 2) + pow((objects[i].proposed_center().y - clustercenter.y), 2));
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (usedetector == true)
		{
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				size_t obj_id = clsobjrs.at(i).obj_id;

				if (objects.at(obj_id).det_mc == true)
				{
					Point2f clustercenter = cluster_center(clusters[cls_id]);
					auto& cobj = objects[obj_id];
					pt1.x = cobj.model_center.x - cobj.obj_size.x / 2;
					pt1.y = cobj.model_center.y - cobj.obj_size.y / 2;

					pt2.x = cobj.model_center.x + cobj.obj_size.x / 2;
					pt2.y = cobj.model_center.y + cobj.obj_size.y / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{
						if (cobj.det_pos == false)
							cobj.cluster_points = clusters.at(cls_id);
						else
						{
							for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
								cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
						}

						cobj.center_determine(id_frame, false);

						for (size_t j = 0; j < clsobjrs.size(); j++)
						{
							if (clsobjrs.at(j).cls_id == cls_id)
							{
								clsobjrs.erase(clsobjrs.begin() + j);
								j--;
							}
						}
						i = 0;
					}
				}
			}
		}
		else
		{
			//----------------------<points2ORB>---------------------------------------------
			points2ORB = std::get<1>(detectsORB);
			vector<ClsObjR> orbobjrs;
			ClsObjR orbobjr;

			bool orb_in_obj;
			size_t obj_id, cls_id;
			for (size_t i = 0; i < clusters.size(); i++)
			{
				if (clusters[i].size() <= mpct/2)
					continue;
				orb_in_obj = false;

				for (size_t j = 0; j < clusters[i].size(); j++)
				{
					for (size_t orb = 0; orb < points2ORB.size(); orb++)
					{
						float r = sqrt(pow((clusters[i][j].x - points2ORB.at(orb).x), 2) + pow((clusters[i][j].y - points2ORB.at(orb).y), 2));
						if (r < (float)rORB * (float)resolution / (float)reduseres)
						{
							orb_in_obj = false;
							for (size_t obj = 0; obj < objects.size(); obj++)
							{
								for (size_t o = 0; o < objects[obj].ORB_ids.size(); o++)
								{
									if (objects[obj].ORB_ids[o] == orb)
									{
										orb_in_obj = true;
										obj_id = obj;
										cls_id = i;
										goto jump1;
									}
								}
							}
						}
					}
				}

			jump1:
				if (orb_in_obj == true)
				{
					auto& cobj = objects.at(obj_id);
					if (cobj.det_pos == false)
						cobj.cluster_points = clusters.at(cls_id);
					else
					{
						for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
							cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
					}

					cobj.center_determine(id_frame, false);

					for (size_t j = 0; j < clsobjrs.size(); j++)
					{
						if (clsobjrs.at(j).cls_id == cls_id)
						{
							clsobjrs.erase(clsobjrs.begin() + j);
							j--;
						}
					}
				}
			}
			//--------------------</points2ORB>---------------------------
		}
		//--</corr obj use model>---

		//---<det obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * (float)resolution / (float)reduseres && clusters.at(cls_id).size() > mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (cobj.det_pos == false)
					cobj.cluster_points = clusters.at(cls_id);
				else
				{
					continue;
					// for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					//   cobj.cluster_points.push_back(clusters.at(cls_id).at(j));
				}

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//---</det obj>---
		//---<new obj>----
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			Point2f clustercenter = cluster_center(clusters[cls_id]);
			bool newobj = true;

			for (size_t j = 0; j < objects.size(); j++)
			{
				float r = sqrt(pow((objects[j].cluster_center.x - clustercenter.x), 2) + pow((objects[j].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls_id].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - half_imgsize, clustercenter.y + half_imgsize), cv::Range(clustercenter.x - half_imgsize, clustercenter.x + half_imgsize));

				img = framebuf(cv::Range(max_u16(clustercenter.y - half_imgsize)
					, min_u16(framebuf.rows, clustercenter.y + half_imgsize))
					, cv::Range(max_u16(clustercenter.x - half_imgsize)
					, min_u16(framebuf.cols, clustercenter.x + half_imgsize)));

				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls_id], img);
				objects.push_back(obj);
				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</new obj>--

		//--<corr obj>---
		for (size_t i = 0; i < clsobjrs.size(); i++)
		{
			size_t cls_id = clsobjrs.at(i).cls_id;
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * robj_k * (float)resolution / (float)reduseres && clusters.at(cls_id).size() > mpct / 2)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < clusters.at(cls_id).size(); j++)
					cobj.cluster_points.push_back(clusters.at(cls_id).at(j));

				cobj.center_determine(id_frame, false);

				for (size_t j = 0; j < clsobjrs.size(); j++)
				{
					if (clsobjrs.at(j).cls_id == cls_id)
					{
						clsobjrs.erase(clsobjrs.begin() + j);
						j--;
					}
				}
				i = 0;
			}
		}
		//--</corr obj>---
	}
	else
	{
		//--<new obj>--
		for (int cls = 0; cls < clusters.size(); cls++)
		{
			Point2f clustercenter = cluster_center(clusters[cls]);
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				float r = sqrt(pow((objects[i].cluster_center.x - clustercenter.x), 2) + pow((objects[i].cluster_center.y - clustercenter.y), 2));
				if (r < (float)robj * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// magbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - half_imgsize, clustercenter.y + half_imgsize), cv::Range(clustercenter.x - half_imgsize, clustercenter.x + half_imgsize));

				img = framebuf(cv::Range(max_u16(clustercenter.y - half_imgsize)
					, min_u16(framebuf.rows, clustercenter.y + half_imgsize))
					, cv::Range(max_u16(clustercenter.x - half_imgsize)
					, min_u16(framebuf.cols, clustercenter.x + half_imgsize)));

				cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), "a", clusters[cls], img);
				objects.push_back(obj);
				clusters.erase(clusters.begin() + cls);
				cls--;

				if (cls < 0)
					cls = 0;
			}
		}
		//--</new obj>--
	}
	//--------------</clusters to objects>-------------------

	//--------------<post processing>-----------------------
	for (int i = 0; i < objects.size(); i++)
	{
		if (objects.at(i).det_mc == false && objects.at(i).det_pos == false)
			continue;

		// imagbuf = frame_resizing(framebuf);
		// imagbuf.convertTo(imagbuf, CV_8UC3);

		// imagbuf = frame_resizing(framebuf);
		framebuf.convertTo(imagbuf, CV_8UC3);

		if (objects[i].det_mc == false)
		{
			pt1.y = objects[i].cluster_center.y - half_imgsize;
			pt2.y = objects[i].cluster_center.y + half_imgsize;

			pt1.x = objects[i].cluster_center.x - half_imgsize;
			pt2.x = objects[i].cluster_center.x + half_imgsize;
		}
		else
		{
			pt1.y = objects[i].model_center.y - half_imgsize;
			pt2.y = objects[i].model_center.y + half_imgsize;

			pt1.x = objects[i].model_center.x - half_imgsize;
			pt2.x = objects[i].model_center.x + half_imgsize;
		}

		if (pt1.y < 0)
			pt1.y = 0;

		if (pt2.y > imagbuf.rows)
			pt2.y = imagbuf.rows;

		if (pt1.x < 0)
			pt1.x = 0;

		if (pt2.x > imagbuf.cols)
			pt2.x = imagbuf.cols;

		// std::cout << "<post processing 2>" << endl;
		// std::cout << "pt1 - " << pt1 << endl;
		// std::cout << "pt2 - " << pt2 << endl;

		img = imagbuf(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x));
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		img.convertTo(img, CV_8UC3);
		img.copyTo(objects[i].img);
		objects[i].center_determine(id_frame, true);

		if (objects[i].det_mc == false)
			objects[i].push_track_point(objects[i].cluster_center);
		else
			objects[i].push_track_point(objects[i].model_center);
	}
	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = objects.at(i).cluster_points.at(j).x;
			pt1.y = objects.at(i).cluster_points.at(j).y;

			pt2.x = objects.at(i).cluster_points.at(j).x + resolution / reduseres;
			pt2.y = objects.at(i).cluster_points.at(j).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		if (objects.at(i).det_mc == true) // visualization of the classifier
		{
			pt1.x = objects.at(i).model_center.x - objects.at(i).obj_size.x / 2;
			pt1.y = objects.at(i).model_center.y - objects.at(i).obj_size.y / 2;

			pt2.x = objects.at(i).model_center.x + objects.at(i).obj_size.x / 2;
			pt2.y = objects.at(i).model_center.y + objects.at(i).obj_size.y / 2;

			rectangle(imag, pt1, pt2, class_name_color(objects.at(i).id), 1);
		}

		for (int j = 0; j < objects.at(i).track_points.size(); j++)
			cv::circle(imag, objects.at(i).track_points.at(j), 1, class_name_color(objects.at(i).id), 2);

		for (int j = 0; j < objects[i].ORB_ids.size(); j++)
			cv::circle(imag, points2ORB.at(objects[i].ORB_ids.at(j)), 3, class_name_color(objects.at(i).id), 1);
	}

	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------

	Mat baseimag(resolution, resolution + extr, CV_8UC3, Scalar(0, 0, 0));
	// std::cout << "<baseimag 1>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		string text = objects.at(i).obj_type + " ID" + to_string(objects.at(i).id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								class_name_color(objects.at(i).id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + objects.at(i).img.cols + 1;
		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, class_name_color(objects.at(i).id), 1);
			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
		}
	}
	// std::cout << "<baseimag 2>" << endl;
	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

	//cv::resize(std::get<2>(detectsORB), std::get<2>(detectsORB), cv::Size(2 * resolution, resolution), cv::InterpolationFlags::INTER_CUBIC);

	//std::get<2>(detectsORB).copyTo(baseimag(cv::Rect(resolution + extr, 0, std::get<2>(detectsORB).cols, std::get<2>(detectsORB).rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(id_frame), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------

	imshow("Motion", baseimag);
	cv::waitKey(10);

	return baseimag;
}

string dateTime()
{
	const time_t  now = time(0); 
	const struct tm  tstruct = *localtime(&now);
	char  tcstr[63];
	strftime(tcstr, sizeof(tcstr), "%Y-%m-%d_%H-%M-%S", &tstruct);
	return tcstr;
}

void traceObjects(const vector<ALObject> &objects, const string& odir)  // , const cv::Size& frame
{
	const fs::path  odp = odir;
	std::error_code  ec;
	if(!fs::exists(odp, ec)) {
		if(ec || !fs::create_directories(odp, ec))
			throw std::runtime_error("The output directory can't be created (" + to_string(ec.value()) + ": " + ec.message() + "): " + odp.string() + "\n");
	}

	for(const auto& obj: objects) {
		// cout << endl << "obj: " << obj.obj_type + to_string(obj.id) << endl;
		cout << obj.obj_type + to_string(obj.id) << endl;
		std::fstream fout((odp / obj.obj_type).string() + to_string(obj.id) + ".csv", std::ios_base::out);
		fout << "# FrameId ObjCenterX ObjCenterY ObjWidth ObjHeight\n";
		for(const auto& tr: obj.traces) {
			fout << tr.frame << ' ' << tr.center.x << ' ' << tr.center.y << ' ' << tr.size.width << ' ' << tr.size.height << endl;
			cout << "frame: " << tr.frame << ", center: " << tr.center << ", size: " << tr.size << endl;
		}
	}
}