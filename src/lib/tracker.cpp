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


const uint16_t model_resolution = 992;  // frame resizing for model (992)
uint16_t frame_resolution = model_resolution;  //frame frame_resolution
int resolution = model_resolution;  // Frame size for a model
const float kres = resolution / 1200.f;  // Note: basic motion parameters were adjusted for 1200 px

uint16_t extr = 204;  // sidebar size
uint16_t objhsz = 80; // area half size for a moving object: max(w, h) [/2];  8 .. 128

// Frame resolution for motion detection using non-adaptive thresholding:  (good value 248), 400
// Ant size should be >= 8 px (=> objhsz >= 4)
constexpr uint16_t antLenMin = 8;  // Minimal length of an ant
float evalReduseres(uint16_t objhsz) noexcept { return roundf(model_resolution * 12.f / objhsz / 32.f) * 32; }  // 480; 224;  12 is a reference length on an ant for motion processing

float reduseres = evalReduseres(objhsz);

//string objClassTitle(] = {"ta", "a", "ah", "tl", "l", "fn", "p", "b", "u"};  // objClassTitle[9)
//const char *const objClassTitle(] = {"a", "ah", "ta", "l", "tl", "fn", "p", "b", "u"};  // objClassTitle[9)

const char* objClassTitle(ObjClass objClass) noexcept {
	static const char *const titles[] = {"a", "ah", "ta", "l", "tl", "fn", "p", "b", "u"};  // objClassTitle(9)

	return titles[static_cast<uint8_t>(objClass)];
}

Scalar objClassColor(ObjClass objClass) noexcept
{
	static Scalar class_name_color[9] = {Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255), Scalar(255, 255, 0), Scalar(255, 255, 255), Scalar(200, 0, 200), Scalar(100, 0, 255)};
	return class_name_color[static_cast<uint8_t>(objClass)];
}

Scalar objColor(uint32_t id, uint8_t clrLow=32, uint8_t clrHigh=223) noexcept
{
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

float distance(const Point2f& a, const Point2f& b) noexcept
{
	const float  dx = a.x - b.x;
	const float  dy = a.y - b.y;

	return sqrt(dx*dx + dy*dy);
}

//! Harmonic mean
constexpr float hmean(float a, float b) noexcept  { return 2.f * a*b / (a+b); }
//! Geometric mean >= Harmonic mean
float gmean(float a, float b) noexcept  { return sqrtf(a*b); }

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

				// if(framebuf.channels() == 1)
				// 	cv::cvtColor(framebuf, framebuf, cv::COLOR_GRAY2BGR);
				// // framebuf.convertTo(framebuf, CV_8UC3);
				if(framebuf.channels() > 1)
					cv::cvtColor(framebuf, framebuf, cv::COLOR_BGR2GRAY);
				uint16_t res;

				if (framebuf.rows > framebuf.cols)
					res = framebuf.rows;
				else
					res = framebuf.cols;

				Mat frame(res, res, CV_8UC1, Scalar::all(0));

				if (framebuf.rows > framebuf.cols)
					framebuf.copyTo(frame(cv::Rect((framebuf.rows - framebuf.cols) / 2, 0, framebuf.cols, framebuf.rows)));
				else
					framebuf.copyTo(frame(cv::Rect(0, (framebuf.cols - framebuf.rows) / 2, framebuf.cols, framebuf.rows)));

				// cv::cvtColor(frame, frame, cv::COLOR_RGB2GRAY);

				d_images.push_back(frame);
				std::cout << "[LoadVideo]: Success to extracted the frame " << frame_count << endl;
			}
		}
	}
	return d_images;
}


// Cuts frame to rect area
Mat frame_resizing(Mat frame, uint16_t framesize=model_resolution)
{
	int rows = frame.rows;
	int cols = frame.cols;

	float rwsize = model_resolution;
	float clsize = model_resolution;

	if (rows > cols)
	{
		rwsize = framesize * float(rows) / cols;
		clsize = framesize;
	}
	else
	{
		rwsize = framesize;
		clsize = framesize * float(cols) / rows;
	}

	if (clsize != frame.cols || rwsize != frame.rows)
		cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	if(clsize != rwsize) {
		cv::Rect rect(0, 0, framesize, framesize);
		return frame(rect);
	}
	return frame;
}

// vector<OBJdetect> detectorV4_old(const string& pathmodel, Mat frame, torch::DeviceType device_type, const float confidence=dftConf)  // 0.5
// {
// 	vector<OBJdetect> obj_detects;
// 	auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
// 	torch::jit::script::Module module = torch::jit::load(pathmodel);
// 	std::cout << "Load module +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << endl;
// 	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

// 	int resolution = 992;
// 	int pointsdelta = 5;
// 	vector<Point2f> detects;
// 	vector<Point2f> detectsCent;
// 	vector<Point2f> detectsRect;
// 	vector<ObjClass> objType;
// 	Mat imageBGR;

// 	// imageBGR = frame_resizing(frame);
// 	frame.copyTo(imageBGR);
// 	// cv::resize(frame, imageBGR,cv::Size(992, 992),cv::InterpolationFlags::INTER_CUBIC);

// 	cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
// 	imageBGR.convertTo(imageBGR, CV_32FC3, 1.0f / 255.0f);
// 	auto input_tensor = torch::from_blob(imageBGR.data, {1, imageBGR.rows, imageBGR.cols, 3});
// 	input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous();
// 	input_tensor = input_tensor.to(device_type);
// 	//----------------------------------
// 	// module.to(device_type);

// 	if (device_type != torch::kCPU)
// 	{
// 		input_tensor = input_tensor.to(torch::kHalf);
// 	}
// 	//----------------------------------

// 	// std::cout<<"input_tensor.to(device_type) - OK"<<endl;
// 	vector<torch::jit::IValue> input;
// 	input.emplace_back(input_tensor);
// 	// std::cout<<"input.emplace_back(input_tensor) - OK"<<endl;

// 	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
// 	auto outputs = module.forward(input).toTuple();
// 	// std::cout << "Processing +" << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << "ms" << endl;
// 	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

// 	// std::cout<<"module.forward(input).toTuple() - OK"<<endl;
// 	torch::Tensor detections = outputs->elements()[0].toTensor();

// 	// int item_attr_size = 13;
// 	int batch_size = detections.size(0);
// 	auto num_classes = detections.size(2); // - item_attr_size;
// 	auto conf_mask = detections.select(2, 4).ge(confidence).unsqueeze(2);

// 	vector<vector<Detection>> output;
// 	output.reserve(batch_size);

// 	for (int batch_i = 0; batch_i < batch_size; batch_i++)
// 	{
// 		// apply constrains to get filtered detections for current image
// 		auto det = torch::masked_select(detections[batch_i], conf_mask[batch_i]).view({-1, num_classes});
// 		// if none detections remain then skip and start to process next image

// 		if (0 == det.size(0))
// 		{
// 			continue;
// 		}

// 		for (size_t i = 0; i < det.size(0); ++i)
// 		{
// 			const auto dcur = det[i];
// 			float x = dcur[0].item().toFloat() * imageBGR.cols / resolution;
// 			float y = dcur[1].item().toFloat() * imageBGR.rows / resolution;

// 			float h = dcur[2].item().toFloat() * imageBGR.cols / resolution;
// 			float w = dcur[3].item().toFloat() * imageBGR.rows / resolution;

// 			float wheit = 0;
// 			objType.push_back(ObjClass::UNCATECORIZED);

// 			for (int j = 5; j < det.size(1); j++)
// 			{
// 				if (dcur[j].item().toFloat() > wheit)
// 				{
// 					wheit = dcur[j].item().toFloat();
// 					objType.at(i) = ObjClass(j - 5);
// 				}
// 			}

// 			detectsCent.push_back(Point(x, y));
// 			detectsRect.push_back(Point(h, w));
// 		}
// 	}

// 	for (size_t i = 0; i < detectsCent.size(); i++)
// 	{
// 		auto& dci = detectsCent[i];
// 		if (dci.x > 0)
// 		{
// 			for (size_t j = 0; j < detectsCent.size(); j++)
// 			{
// 				if (detectsCent.at(j).x > 0 && i != j)
// 				{
// 					if (distance(dci, detectsCent[j]) < pointsdelta)
// 					{
// 						dci.x = (dci.x + detectsCent.at(j).x) * 1.0 / 2;
// 						dci.y = (dci.y + detectsCent.at(j).y) * 1.0 / 2;

// 						detectsRect.at(i).x = (detectsRect.at(i).x + detectsRect.at(j).x) * 1.0 / 2;
// 						detectsRect.at(i).y = (detectsRect.at(i).y + detectsRect.at(j).y) * 1.0 / 2;

// 						detectsCent.at(j).x = -1;
// 					}
// 				}
// 			}
// 		}
// 	}

// 	for (size_t i = 0; i < detectsCent.size(); i++)
// 	{

// 		Point2f pt1;
// 		Point2f pt2;
// 		Point2f ptext;
// 		const auto& dci = detectsCent[i];

// 		if (dci.x > -1)
// 		{

// 			OBJdetect obj_buf;

// 			obj_buf.center = dci;
// 			obj_buf.size = detectsRect.at(i);
// 			obj_buf.type = objType.at(i);
// 			obj_detects.push_back(obj_buf);

// 			detects.push_back(dci);

// 			// // Visualization of the detected objects
// 			// pt1.x = dci.x - detectsRect.at(i).x / 2;
// 			// pt1.y = dci.y - detectsRect.at(i).y / 2;

// 			// pt2.x = dci.x + detectsRect.at(i).x / 2;
// 			// pt2.y = dci.y + detectsRect.at(i).y / 2;

// 			// ptext.x = dci.x - 5;
// 			// ptext.y = dci.y + 5;

// 			// rectangle(imageBGR, pt1, pt2, objClassColor(obj_buf.type), 1);

// 			// pt1.y -= 2; // 5
// 			// cv::putText(imageBGR,                  // target image
// 			// 	objClassTitle(obj_buf.type), // text
// 			// 	pt1,                     // top-left position
// 			// 	1,
// 			// 	0.8,
// 			// 	objClassColor(obj_buf.type), // font color
// 			// 	1);
// 		}
// 	}

// 	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
// 	// imshow("Detected objects", imageBGR);
// 	return obj_detects;
// }

vector<OBJdetect> detectorV4(const string& pathmodel, Mat frame, torch::DeviceType device_type, const float confidence, const string& outfile)
{
	vector<OBJdetect> obj_detects;
	auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	static torch::jit::script::Module module = torch::jit::load(pathmodel);
	std::cout << "Model loading time: " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec
		// << "ms; " << "confidence threshold : " << confidence
		<< " ms" << endl;
	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

	const uint16_t pointsdelta = objhsz / 4 + 1;  // Max distance between points to merge them into a single cluster
	vector<Point2f> detects;
	vector<Point2f> detectsCent;
	vector<Point2f> detectsRect;
	vector<ObjClass> objType;
	vector<float> objProb;  // Object probability = avg(conf, score)
	Mat imageBGR;

	// imageBGR = frame_resizing(frame);
	// std::cout << "Frame size: " << frame.cols  << "x" << frame.rows << endl;
	//cv::resize(frame, imageBGR, cv::Size(992, 992), cv::InterpolationFlags::INTER_CUBIC);
	frame.copyTo(imageBGR);

	// Ensure that the input image size corresponds to the model size

	if(imageBGR.channels() == 1) {
		cv::cvtColor(imageBGR, imageBGR, cv::COLOR_GRAY2RGB);
	} else {
		assert(imageBGR.type() == CV_8UC3 && "Color input frames in the BGR (OpenCV-native) format is expected");
		cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2RGB);
	}
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
	std::cout << "Raw object detection time: " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec << " ms" << endl;
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
			// objType.push_back(ObjClass::UNCATECORIZED);  // Add max index
			const uint16_t  ibeg = 5;  // Index of the begining of categories
			uint16_t  iclass = -1;  // Class index
			float cscore = 0;  // Class score

			// TODO: reimplement consideing tracking history
			cout << "Top classes (conf: " << std::setprecision(2) << conf << ") for #" << batch_i * batch_size + i << ": ";
			for (uint16_t j = ibeg; j < det.size(1); j++)
			{
				const float v = dcur[j].item().toFloat();
				if (v > cscore)
				{
					// Assign a class only if it's score sufficiently differs from the closest another one
					if(gmean((v-cscore),  1 - cscore/v) >= v*confidence*confidence)
						iclass = j - ibeg;
					//objType.at(i) = ObjClass(j - ibeg);
					cscore = v;
					cout << " " << objClassTitle(ObjClass(j - ibeg)) << "=" <<  std::setprecision(3) << v;
				}
				// cout << " " << objClassTitle(ObjClass(j - ibeg)) << "=" <<  std::setprecision(3) << v;
			}
			cout << endl;

			const float  oprob = gmean(cscore, conf);
			if(iclass != static_cast<decltype(iclass)>(-1) && oprob >= confidence) {
				objType.push_back(ObjClass(iclass));
				objProb.push_back(oprob);
				detectsCent.push_back(Point2f(x, y));
				detectsRect.push_back(Point2f(w, h));
			}
		}
	}

	for (size_t i = 0; i < detectsCent.size(); i++)
	{
		if (detectsCent[i].x > -1)
		{
			auto& dci = detectsCent[i];
			for (size_t j = 0; j < detectsCent.size(); j++)
			{
				if (detectsCent[j].x > 0 && i != j)
				{
					if (distance(dci, detectsCent[j]) < pointsdelta)
					{
						dci.x = (dci.x + detectsCent.at(j).x) / 2.f;
						dci.y = (dci.y + detectsCent.at(j).y) / 2.f;

						detectsRect.at(i).x = (detectsRect.at(i).x + detectsRect.at(j).x) / 2.f;
						detectsRect.at(i).y = (detectsRect.at(i).y + detectsRect.at(j).y) / 2.f;

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
		const auto& dci = detectsCent[i];

		if (dci.x > -1)
		{
			OBJdetect obj_buf;

			obj_buf.center = dci;
			obj_buf.size = detectsRect.at(i);
			obj_buf.type = objType.at(i);
			obj_buf.prob = objProb.at(i);
			obj_detects.push_back(obj_buf);

			detects.push_back(dci);

			// Visualization of the detected objects
			pt1.x = dci.x - detectsRect.at(i).x / 2;
			pt1.y = dci.y - detectsRect.at(i).y / 2;

			pt2.x = dci.x + detectsRect.at(i).x / 2;
			pt2.y = dci.y + detectsRect.at(i).y / 2;

			rectangle(imageBGR, pt1, pt2, objClassColor(obj_buf.type), 1);

			pt1.y -= 2; // 5
			cv::putText(imageBGR,                  // target image
				objClassTitle(obj_buf.type) + to_string(i), // text
				pt1,                     // top-left position
				1,
				0.8,
				objClassColor(obj_buf.type), // font color
				1);
		}
	}
	cout << "detectorV4(), detected: " << obj_detects.size() << endl;
	imshow("Detected objects", imageBGR);
	if(!outfile.empty()) {
		if(!imwrite(outfile, imageBGR * 0xFF))  // Note: 0xFF is required to scale the output to 0..255 from 0..1
			cout << "WARNING: detectorV4(), resulting frame cannot be saved: " << outfile << endl;
	}
	// millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
	return obj_detects;
}

Point2f cluster_center(vector<Point2f> cluster_points)
{
	float powx = 0;
	float powy = 0;

	for (int i = 0; i < cluster_points.size(); i++)
	{
		const auto& cp = cluster_points[i];
		powx += cp.x * cp.x;
		powy += cp.y * cp.y;
	}

	return Point2f(sqrt(powx / cluster_points.size()), sqrt(powy / cluster_points.size()));
}

size_t samples_compV2(Mat sample1, Mat sample2)
{
	size_t npf = 0;

	for(Mat* v: {&sample1, &sample2})
		if(v->channels() != 1)
			cv::cvtColor(*v, *v, cv::COLOR_BGR2GRAY);
	// sample1.convertTo(sample1, CV_8UC1);

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

// Mat draw_object(const ALObject& obj, const ALObject& obj2, const Scalar& color)
// {
// 	int wh = 800;
// 	float hp = 1.0 * (resolution / reduseres) * wh / (2 * objhsz);

// 	Mat imag(wh, wh, CV_8UC3, Scalar(0, 0, 0));
// 	Mat imag2(wh, wh, CV_8UC3, Scalar(0, 0, 0));
// 	Mat imgres(wh, wh * 3, CV_8UC3, Scalar(0, 0, 0));
// 	Mat imgbuf;

// 	Point2f bufp;
// 	Point2f pt1;
// 	Point2f pt2;

// 	Mat imgsm;

// 	cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);

// 	imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 	cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);

// 	imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 	/*
// 		for (int i = 0; i < obj.samples.size(); i++)
// 		{
// 			imgsm = obj.samples.at(i);
// 			cv::resize(imgsm, imgsm, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

// 			//--test color---
// 			size_t rows = imgsm.rows;
// 			size_t cols = imgsm.cols;
// 			uint8_t *pixelPtr1 = (uint8_t *)imgsm.data;
// 			int cn = imgsm.channels();
// 			cv::Scalar_<uint8_t> bgrPixel1;

// 			for (size_t x = 0; x < rows; x++)
// 			{
// 				for (size_t y = 0; y < cols; y++)
// 				{
// 					bgrPixel1.val[0] = pixelPtr1[x * imgsm.cols * cn + y * cn + 0]; // B
// 					bgrPixel1.val[1] = pixelPtr1[x * imgsm.cols * cn + y * cn + 1]; // G
// 					bgrPixel1.val[2] = pixelPtr1[x * imgsm.cols * cn + y * cn + 2]; // R

// 					if (bgrPixel1.val[0] > 30)
// 					{
// 						if(bgrPixel1.val[0] > 100)
// 							pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)255;
// 						else
// 							pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)100;
// 					}
// 					else
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)0;

// 					if (bgrPixel1.val[1] > 30)
// 					{
// 						if(bgrPixel1.val[0] > 100)
// 							pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)255;
// 						else
// 							pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)100;
// 					}
// 					else
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)0;

// 					if (bgrPixel1.val[2] > 30)
// 					{
// 						if(bgrPixel1.val[0] > 100)
// 							pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)255;
// 						else
// 							pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)100;
// 					}
// 					else
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)0;
// 				}
// 			}
// 			//--test color---/

// 			bufp.x = 1.0 * obj.coords.at(i).x * wh / (2 * objhsz);
// 			bufp.y = 1.0 * obj.coords.at(i).y * wh / (2 * objhsz);
// 			imgsm.copyTo(imag(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));
// 		}

// 		for (int i = 0; i < obj.cluster_points.size(); i++)
// 		{
// 			bufp.x = ((obj.cluster_points.at(i).x - obj.cluster_center.x) * wh / (2 * objhsz) + wh / 2);
// 			bufp.y = ((obj.cluster_points.at(i).y - obj.cluster_center.y) * wh / (2 * objhsz) + wh / 2);

// 			pt1.x = bufp.x;
// 			pt1.y = bufp.y;

// 			pt2.x = bufp.x + hp;
// 			pt2.y = bufp.y + hp;

// 			rectangle(imag, pt1, pt2, color, 1);
// 		}
// 	*/
// 	cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);

// 	//---------------------------------------------

// 	cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);

// 	for (int i = 0; i < obj2.samples.size(); i++)
// 	{
// 		imgsm = obj2.samples.at(i);
// 		cv::resize(imgsm, imgsm, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

// 		/*/--test color---
// 		size_t rows = imgsm.rows;
// 		size_t cols = imgsm.cols;
// 		uint8_t *pixelPtr1 = (uint8_t *)imgsm.data;
// 		int cn = imgsm.channels();
// 		cv::Scalar_<uint8_t> bgrPixel1;

// 		for (size_t x = 0; x < rows; x++)
// 		{
// 			for (size_t y = 0; y < cols; y++)
// 			{
// 				bgrPixel1.val[0] = pixelPtr1[x * imgsm.cols * cn + y * cn + 0]; // B
// 				bgrPixel1.val[1] = pixelPtr1[x * imgsm.cols * cn + y * cn + 1]; // G
// 				bgrPixel1.val[2] = pixelPtr1[x * imgsm.cols * cn + y * cn + 2]; // R

// 				if (bgrPixel1.val[0] > 30)
// 				{
// 					if(bgrPixel1.val[0] > 100)
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)255;
// 					else
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)100;
// 				}
// 				else
// 					pixelPtr1[x * imgsm.cols * cn + y * cn + 0] = (uint8_t)0;

// 				if (bgrPixel1.val[1] > 30)
// 				{
// 					if(bgrPixel1.val[0] > 100)
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)255;
// 					else
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)100;
// 				}
// 				else
// 					pixelPtr1[x * imgsm.cols * cn + y * cn + 1] = (uint8_t)0;

// 				if (bgrPixel1.val[2] > 30)
// 				{
// 					if(bgrPixel1.val[0] > 100)
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)255;
// 					else
// 						pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)100;
// 				}
// 				else
// 					pixelPtr1[x * imgsm.cols * cn + y * cn + 2] = (uint8_t)0;
// 			}
// 		}
// 		//--test color---*/

// 		bufp.x = 1.0 * obj2.coords.at(i).x * wh / (2 * objhsz);
// 		bufp.y = 1.0 * obj2.coords.at(i).y * wh / (2 * objhsz);
// 		imgsm.copyTo(imag2(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));
// 	}

// 	for (int i = 0; i < obj2.cluster_points.size(); i++)
// 	{
// 		bufp.x = ((obj2.cluster_points.at(i).x - obj2.cluster_center.x) * wh / (2 * objhsz) + wh / 2);
// 		bufp.y = ((obj2.cluster_points.at(i).y - obj2.cluster_center.y) * wh / (2 * objhsz) + wh / 2);

// 		pt1.x = bufp.x;
// 		pt1.y = bufp.y;

// 		pt2.x = bufp.x + hp;
// 		pt2.y = bufp.y + hp;

// 		rectangle(imag2, pt1, pt2, color, 1);
// 	}

// 	cv::cvtColor(imag2, imag2, cv::COLOR_BGR2RGB);

// 	//-------using matchTemplate--bad idea---------------------
// 	/*
// 		Mat result;
// 		result.create(imag.rows, imag.cols, CV_32FC1);

// 		for (size_t s2 = 0; s2 < obj2.samples.size(); s2++)
// 		{
// 			matchTemplate(imag, obj2.samples.at(s2), result, 2);

// 			//imshow("result", result);

// 			size_t R = rand() % 255;
// 			size_t G = rand() % 255;
// 			size_t B = rand() % 255;

// 			bufp.x = 1.0 * obj2.coords.at(s2).x * wh / (2 * objhsz);
// 			bufp.y = 1.0 * obj2.coords.at(s2).y * wh / (2 * objhsz);
// 			// imgsm.copyTo(imag2(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

// 			pt1.x = bufp.x;
// 			pt1.y = bufp.y;

// 			pt2.x = bufp.x + hp;
// 			pt2.y = bufp.y + hp;

// 			rectangle(imag2, pt1, pt2, Scalar(R, G, B), 1);
// 			imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
// 			imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
// 			imshow("imgres", imgres);

// 			cv::waitKey(0);
// 		}*/
// 	//------------------------------

// 	/*imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
// 	imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
// 	imshow("imgres", imgres);*/

// 	cv::waitKey(0);

// 	do
// 	{
// 		size_t ns2 = 0;
// 		size_t ns1 = 0;
// 		size_t minnp = 0;

// 		for (size_t s1 = 0; s1 < obj.samples.size(); s1++)
// 		{
// 			for (size_t s2 = 0; s2 < obj2.samples.size(); s2++)
// 			{
// 				size_t np = samples_compV2(obj.samples.at(s1), obj2.samples.at(s2));
// 				if ((minnp > np || minnp == 0) && np > 0)
// 				{
// 					minnp = np;
// 					ns2 = s2;
// 					ns1 = s1;
// 				}
// 			}
// 		}

// 		std::cout << "minnp - " << minnp << endl;

// 		Mat imgsm = obj2.samples.at(ns2);
// 		cv::resize(imgsm, imgsm, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

// 		bufp.x = 1.0 * obj.coords.at(ns1).x * wh / (2 * objhsz);
// 		bufp.y = 1.0 * obj.coords.at(ns1).y * wh / (2 * objhsz);

// 		Point2f c_cir;

// 		c_cir.x = (objhsz - (obj2.coords.at(ns2).x - obj.coords.at(ns1).x)) * wh / (2 * objhsz);
// 		c_cir.y = (objhsz - (obj2.coords.at(ns2).y - obj.coords.at(ns1).y)) * wh / (2 * objhsz);

// 		std::cout << "c_cir.y - " << c_cir.y << endl;
// 		std::cout << "c_cir.x - " << c_cir.x << endl;
// 		std::cout << "objhsz - " << objhsz << endl;
// 		// Scalar  objClr = Scalar(rand() % 255, rand() % 255, rand() % 255);
// 		Scalar  objClr = objColor(obj.id);

// 		if (c_cir.y < 0 || c_cir.x < 0 || c_cir.y > 2 * objhsz * wh / (2 * objhsz) || c_cir.x > 2 * objhsz * wh / (2 * objhsz))
// 			goto dell;

// 		cv::circle(imag, c_cir, 3, objClr, 1);

// 		// imgsm.copyTo(imag(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

// 		pt1.x = bufp.x;
// 		pt1.y = bufp.y;

// 		pt2.x = bufp.x + hp;
// 		pt2.y = bufp.y + hp;

// 		rectangle(imag, pt1, pt2, objClr, 1);

// 		//---

// 		bufp.x = 1.0 * obj2.coords.at(ns2).x * wh / (2 * objhsz);
// 		bufp.y = 1.0 * obj2.coords.at(ns2).y * wh / (2 * objhsz);
// 		// imgsm.copyTo(imag2(cv::Rect(bufp.x, bufp.y, imgsm.cols, imgsm.rows)));

// 		pt1.x = bufp.x;
// 		pt1.y = bufp.y;

// 		pt2.x = bufp.x + hp;
// 		pt2.y = bufp.y + hp;

// 		rectangle(imag2, pt1, pt2, objClr, 1);

// 		imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
// 		imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));

// 		imshow("imgres", imgres);
// 		cv::waitKey(0);
// 	dell:
// 		obj.samples.erase(obj.samples.begin() + ns1);
// 		obj2.samples.erase(obj2.samples.begin() + ns2);

// 		obj.coords.erase(obj.coords.begin() + ns1);
// 		obj2.coords.erase(obj2.coords.begin() + ns2);

// 	} while (obj.samples.size() > 0 && obj2.samples.size() > 0);
// 	//-----------------------------------

// 	imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
// 	imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
// 	imshow("imgres", imgres);
// 	cv::waitKey(0);

// 	return imgres;
// }

// Mat draw_compare(const ALObject& obj, const ALObject& obj2, Scalar color)
// {
// 	int wh = 800;
// 	float hp = 1.0 * (resolution / reduseres) * wh / (2 * objhsz);

// 	Mat imag(wh, wh, CV_8UC3, Scalar(0, 0, 0));
// 	Mat imag2(wh, wh, CV_8UC3, Scalar(0, 0, 0));
// 	Mat imgres(wh, wh * 2, CV_8UC3, Scalar(0, 0, 0));
// 	Mat imgbuf;

// 	Mat sample;
// 	Mat sample2;

// 	Point2f bufp;
// 	Point2f pt1;
// 	Point2f pt2;

// 	vector<Point2f> center;
// 	vector<int> npsamples;

// 	cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
// 	imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 	cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
// 	imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 	int st = 5; // step for samles compare

// 	cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
// 	imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 	cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
// 	imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 	for (int step_x = -(int)(imag.cols / st) / 2; step_x < (int)(imag.cols / st) / 2; step_x++)
// 	{
// 		for (int step_y = -(int)(imag.rows / st) / 2; step_y < (int)(imag.rows / st) / 2; step_y++)
// 		{
// 			// cv::resize(obj.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
// 			// imgbuf.copyTo(imag(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 			// cv::resize(obj2.img, imgbuf, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
// 			// imgbuf.copyTo(imag2(cv::Rect(0, 0, imgbuf.cols, imgbuf.rows)));

// 			int np = 0;
// 			int ns = 0;
// 			for (int i = 0; i < obj2.samples.size(); i++)
// 			{
// 				sample2 = obj2.samples.at(i);
// 				// cv::resize(sample2, sample2, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

// 				bufp.x = 1.0 * obj2.coords.at(i).x * wh / (2 * objhsz);
// 				bufp.y = 1.0 * obj2.coords.at(i).y * wh / (2 * objhsz);

// 				if ((bufp.y + step_y * st + hp) > imag.rows || (bufp.x + step_x * st + hp) > imag.cols || (bufp.y + step_y * st) < 0 || (bufp.x + step_x * st) < 0)
// 					continue;

// 				sample = obj.img(cv::Range(obj2.coords.at(i).y + step_y * st * obj2.img.rows / imag.rows
// 					, min_u16(obj.img.rows, obj2.coords.at(i).y + step_y * st * obj2.img.rows / imag.rows + resolution / reduseres))
// 					, cv::Range(obj2.coords.at(i).x + step_x * st * obj2.img.cols / imag.cols
// 					,  min_u16(obj.img.cols, obj2.coords.at(i).x + step_x * st * obj2.img.cols / imag.cols + resolution / reduseres)));

// 				np += samples_compV2(obj2.samples.at(i), sample); // sample compare
// 				ns++;

// 				/*
// 				cv::resize(sample, sample, cv::Size(hp, hp), cv::InterpolationFlags::INTER_CUBIC);

// 				sample.copyTo(imag2(cv::Rect(bufp.x + step_x*st, bufp.y + step_y*st, sample.cols, sample.rows)));
// 				sample2.copyTo(imag(cv::Rect(bufp.x + step_x*st, bufp.y + step_y*st, sample2.cols, sample2.rows)));

// 				pt1.x = bufp.x + step_x*st-1;
// 				pt1.y = bufp.y + step_y*st-1;

// 				pt2.x = bufp.x + hp + step_x*st;
// 				pt2.y = bufp.y + hp + step_y*st;

// 				rectangle(imag, pt1, pt2, color, 1);
// 				rectangle(imag2, pt1, pt2, color, 1);
// 				*/
// 			}

// 			bufp.y = objhsz + step_y * st * obj2.img.rows / imag.rows;
// 			bufp.x = objhsz + step_x * st * obj2.img.cols / imag.cols;

// 			if (bufp.y > 0 && bufp.y < 2 * objhsz && bufp.x > 0 && bufp.x < 2 * objhsz)
// 			{
// 				npsamples.push_back(np);
// 				center.push_back(bufp);
// 			}

// 			// imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
// 			// imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));
// 			// imshow("imgres", imgres);
// 			// cv::waitKey(0);
// 		}
// 	}

// 	int max = npsamples.at(0);
// 	int min = npsamples.at(1);
// 	for (int i = 0; i < npsamples.size(); i++)
// 	{
// 		if (max < npsamples.at(i))
// 			max = npsamples.at(i);

// 		if (min > npsamples.at(i))
// 			min = npsamples.at(i);
// 	}

// 	for (int i = 0; i < npsamples.size(); i++)
// 		npsamples.at(i) -= min;

// 	int color_step = (max - min) / 255;

// 	Mat imgcenter(2 * objhsz, 2 * objhsz, CV_8UC1, Scalar(0, 0, 0));

// 	for (int i = 0; i < npsamples.size(); i++)
// 	{
// 		imgcenter.at<uchar>(center.at(i).y, center.at(i).x) = 255 - npsamples.at(i) / color_step;
// 	}

// 	cv::resize(imgcenter, imgcenter, cv::Size(wh, wh), cv::InterpolationFlags::INTER_CUBIC);
// 	imgcenter.convertTo(imgcenter, CV_8UC3);

// 	imag.copyTo(imgres(cv::Rect(0, 0, wh, wh)));
// 	imag2.copyTo(imgres(cv::Rect(wh, 0, wh, wh)));

// 	imshow("imgres", imgres);
// 	imshow("imgcenter", imgcenter);
// 	cv::waitKey(0);

// 	return imgres;
// }

// Mat trackingMotV2(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence, bool init)
// {
// 	vector<vector<Point2f>> clusters;
// 	vector<Point2f> motion;
// 	vector<Mat> imgs;

// 	Mat imageBGR0;
// 	Mat imageBGR;

// 	Mat imag;
// 	Mat imagbuf;
// 	Mat framebuf = frame;

// 	// TODO: eliminate hardcoded values like in trackingMotV2_1
// 	int mpc = 15;   // minimum number of points for a cluster (good value 15)
// 	int nd = 9;     //(good value 6-15)
// 	int rcobj = 17; //(good value 15)
// 	int robj = 17;  //(good value 17)
// 	int mdist = 10; // maximum distance from cluster center (good value 10)
// 	int pft = 9;    // points fixation threshold (good value 9)

// 	Mat img;

// 	vector<OBJdetect> detects;  //, detobjs;

// 	//--------------------<detection using a classifier>----------
// 	if (!pathmodel.empty())
// 	{

// 		detects = detectorV4(pathmodel, frame_resizing(frame), device_type, confidence);
// 		//detobjs = detects;

// 		for (int i = 0; i < objects.size(); i++)
// 		{
// 			objects[i].model_center.x = -1;
// 			objects[i].model_center.y = -1;
// 		}

// 		// Remove non-ant objects
// 		float objLinSz = init ? 0 : objhsz;
// 		unsigned  ndobjs = 0;
// 		for (int i = 0; i < detects.size(); i++)
// 		{
// 			if (ObjClass(detects[i].type) != ObjClass::ANT) {  // Note: "ta" also should be tracked
// 				detects.erase(detects.begin() + i--);
// 				++ndobjs;
// 			}
// 			else if(objLinSz) {
// 				// Update motion-related parameters
// 				objLinSz = (objLinSz * detects.size() + std::max(detects[i].size.width, detects[i].size.height)) / (detects.size() + 1);
// 			} else {
// 				// Init motion-related parameters
// 				objLinSz = (objLinSz * (i + ndobjs) + std::max(detects[i].size.width, detects[i].size.height)) / (i + 1 + ndobjs);
// 			}
// 		}
// 		if(objLinSz < antLenMin)
// 			objLinSz = antLenMin;
// 		objhsz = objLinSz / 2 + 1;
// 		reduseres = evalReduseres(objhsz);

// 		// Identify previously tracked and new objects among detections
// 		if(!detects.empty())
// 			imagbuf = frame_resizing(framebuf);
// 		for (int i = 0; i < detects.size(); i++)
// 		{
// 			vector<Point2f> cluster_points;
//			const auto& det = detects[i];
// 			cluster_points.push_back(det.center);
// 			img = imagbuf(cv::Range(max_u16(det.center.y - objhsz), min_u16(imagbuf.rows, det.center.y + objhsz))
// 				, cv::Range(max_u16(det.center.x - objhsz), min_u16(imagbuf.cols, det.center.x + objhsz)));
// 			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
// 			img.convertTo(img, CV_8UC3);

// 			ALObject obj(objects.size(), detects[i].type, det.size, det.center, cluster_points, img, detects[i].prob);
// 			float rm = rcobj * 1.0 * resolution / reduseres;
// 			uint16_t iobj = -1;

// 			if (objects.size() > 0)
// 			{
// 				rm = sqrt(pow((objects[0].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[0].cluster_center.y - obj.cluster_center.y), 2));
// 				// rm = sqrt(pow((objects[0].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[0].proposed_center().y - obj.cluster_center.y), 2));
// 				if (rm < rcobj * 1.0 * resolution / reduseres && rm < rcobj)
// 					iobj = 0;
// 			}

// 			for (int j = 1; j < objects.size(); j++)
// 			{
// 				float r = sqrt(pow((objects[j].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[j].cluster_center.y - obj.cluster_center.y), 2));
// 				// float r = sqrt(pow((objects[j].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[j].proposed_center().y - obj.cluster_center.y), 2));
// 				if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
// 				{
// 					rm = r;
// 					iobj = j;
// 				}
// 			}

// 			if (iobj != static_cast<decltype(iobj)>(-1))
// 			{
// 				// Update existing object
// 				auto& tobj = objects.at(iobj);
// 				tobj.cluster_center = obj.model_center;
// 				tobj.model_center = obj.model_center;
// 				tobj.size = obj.size;
// 				// tobj.track_points.push_back(obj.cluster_center);
// 				// tobj.push_track_point(obj.cluster_center);
// 				tobj.img = obj.img;
// 				// assert(!tobj.traces.empty() && tobj.traces.back().frame < frameId && "Unexpected frame number in the traces");
// 				tobj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
// 					, obj.size.width, obj.size.height, obj.prob});
// 			}
// 			else
// 			{
// 				assert(obj.traces.empty() && "Unexpected traces");
// 				obj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
// 					, obj.size.width, obj.size.height, obj.prob});
// 				objects.push_back(obj);  // New object
// 			}
// 		}
// 	}
// 	//--------------------</detection using a classifier>---------

// 	//--------------------<moution detections>--------------------
// 	int rows = frame.rows;
// 	int cols = frame.cols;

// 	float rwsize;
// 	float clsize;

// 	imagbuf = frame;
// 	if (rows > cols)
// 	{
// 		rwsize = resolution * rows * 1.0 / cols;
// 		clsize = resolution;
// 	}
// 	else
// 	{
// 		rwsize = resolution;
// 		clsize = resolution * cols * 1.0 / rows;
// 	}

// 	cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
// 	cv::Rect rectb(0, 0, resolution, resolution);
// 	imag = imagbuf(rectb);

// 	cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
// 	imag.convertTo(imag, CV_8UC3);

// 	if (rows > cols)
// 	{
// 		rwsize = reduseres * rows * 1.0 / cols;
// 		clsize = reduseres;
// 	}
// 	else
// 	{
// 		rwsize = reduseres;
// 		clsize = reduseres * cols * 1.0 / rows;
// 	}

// 	cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
// 	cv::Rect rect0(0, 0, reduseres, reduseres);
// 	imageBGR0 = frame0(rect0);
// 	imageBGR0.convertTo(imageBGR0, CV_8UC1);

// 	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

// 	cv::Rect rect(0, 0, reduseres, reduseres);
// 	imageBGR = frame(rect);

// 	imageBGR.convertTo(imageBGR, CV_8UC1);

// 	Point2f pm;

// 	for (int y = 0; y < imageBGR0.rows; y++)
// 	{
// 		for (int x = 0; x < imageBGR0.cols; x++)
// 		{
// 			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
// 			uchar color2 = imageBGR.at<uchar>(Point(x, y));

// 			if (((int)color2 - (int)color1) > pft)
// 			{
// 				pm.x = x * resolution / reduseres;
// 				pm.y = y * resolution / reduseres;
// 				motion.push_back(pm);
// 			}
// 		}
// 	}

// 	Point2f pt1;
// 	Point2f pt2;

// 	for (int i = 0; i < motion.size(); i++) // visualization of the cluster_points
// 	{
// 		pt1.x = motion.at(i).x;
// 		pt1.y = motion.at(i).y;

// 		pt2.x = motion.at(i).x + resolution / reduseres;
// 		pt2.y = motion.at(i).y + resolution / reduseres;

// 		rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
// 	}

// 	uint16_t ncls = 0;
// 	uint16_t nobj = objects.empty() ? -1 : 0;
// 	//--------------</moution detections>--------------------

// 	//--------------<layout of motion points by objects>-----

// 	if (objects.size() > 0)
// 	{
// 		for (int i = 0; i < objects.size(); i++)
// 		{
// 			objects[i].cluster_points.clear();
// 		}

// 		for (int i = 0; i < motion.size(); i++)
// 		{
// 			for (int j = 0; j < objects.size(); j++)
// 			{
// 				if (objects[j].model_center.x < 0)
// 					continue;

// 				if (i < 0)
// 					break;

// 				if ((motion.at(i).x < (objects[j].model_center.x + objects[j].size.width / 2)) && (motion.at(i).x > (objects[j].model_center.x - objects[j].size.width / 2)) && (motion.at(i).y < (objects[j].model_center.y + objects[j].size.height / 2)) && (motion.at(i).y > (objects[j].model_center.y - objects[j].size.height / 2)))
// 				{
// 					objects[j].cluster_points.push_back(motion.at(i));
// 					motion.erase(motion.begin() + i);
// 					i--;
// 				}
// 			}
// 		}

// 		float rm = rcobj * 1.0 * resolution / reduseres;

// 		for (int i = 0; i < motion.size(); i++)
// 		{
// 			rm = sqrt(pow((objects[0].cluster_center.x - motion.at(i).x), 2) + pow((objects[0].cluster_center.y - motion.at(i).y), 2));

// 			uint16_t iobj = -1;
// 			if (rm < rcobj * 1.0 * resolution / reduseres)
// 				iobj = 0;

// 			for (int j = 1; j < objects.size(); j++)
// 			{
// 				float r = sqrt(pow((objects[j].cluster_center.x - motion.at(i).x), 2) + pow((objects[j].cluster_center.y - motion.at(i).y), 2));
// 				if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
// 				{
// 					rm = r;
// 					iobj = j;
// 				}
// 			}

// 			if (iobj != static_cast<decltype(iobj)>(-1))
// 			{
// 				objects[iobj].cluster_points.push_back(motion.at(i));
// 				motion.erase(motion.begin() + i);
// 				i--;
// 				objects[iobj].center_determine(frameId, false);
// 				if (i < 0)
// 					break;
// 			}
// 		}
// 	}

// 	for (int j = 0; j < objects.size(); j++)
// 	{

// 		objects[j].center_determine(frameId, false);

// 		Point2f clustercenter = objects[j].cluster_center;

// 		imagbuf = frame_resizing(framebuf);
// 		imagbuf.convertTo(imagbuf, CV_8UC3);

// 		if (clustercenter.y - objhsz < 0)
// 			clustercenter.y = objhsz;

// 		if (clustercenter.x - objhsz < 0)
// 			clustercenter.x = objhsz;

// 		if (clustercenter.y + objhsz >= imagbuf.rows)
// 			clustercenter.y = imagbuf.rows - 1;

// 		if (clustercenter.x + objhsz >= imagbuf.cols)
// 			clustercenter.x = imagbuf.cols - 1;

// 		img = imagbuf(cv::Range(max_u16(clustercenter.y - objhsz), min_u16(imagbuf.rows, clustercenter.y + objhsz))
// 			, cv::Range(max_u16(clustercenter.x - objhsz), min_u16(imagbuf.cols, clustercenter.x + objhsz)));
// 		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
// 		img.convertTo(img, CV_8UC3);
// 		objects[j].img = img;
// 		// objects[j].samples_creation();
// 	}
// 	//--------------</layout of motion points by objects>----

// 	//--------------<cluster creation>-----------------------

// 	while (motion.size() > 0)
// 	{
// 		Point2f pc;

// 		if (nobj != static_cast<decltype(nobj)>(-1) && nobj < objects.size())
// 		{
// 			pc = objects[nobj].cluster_center;
// 			// pc = objects[nobj].proposed_center();
// 			nobj++;
// 		}
// 		else
// 		{
// 			pc = motion.at(0);
// 			motion.erase(motion.begin());
// 		}

// 		clusters.push_back(vector<Point2f>());
// 		clusters[ncls].push_back(pc);

// 		for (int i = 0; i < motion.size(); i++)
// 		{
// 			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
// 			if (r < nd * 1.0 * resolution / reduseres)
// 			{
// 				Point2f cl_c = cluster_center(clusters.at(ncls));
// 				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
// 				if (r < mdist * 1.0 * resolution / reduseres)
// 				{
// 					clusters.at(ncls).push_back(motion.at(i));
// 					motion.erase(motion.begin() + i);
// 					i--;
// 				}
// 			}
// 		}

// 		int newp;
// 		do
// 		{
// 			newp = 0;

// 			for (int c = 0; c < clusters[ncls].size(); c++)
// 			{
// 				pc = clusters[ncls].at(c);
// 				for (int i = 0; i < motion.size(); i++)
// 				{
// 					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

// 					if (r < nd * 1.0 * resolution / reduseres)
// 					{
// 						Point2f cl_c = cluster_center(clusters.at(ncls));
// 						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
// 						if (r < mdist * 1.0 * resolution / reduseres)
// 						{
// 							clusters.at(ncls).push_back(motion.at(i));
// 							motion.erase(motion.begin() + i);
// 							i--;
// 							newp++;
// 						}
// 					}
// 				}
// 			}
// 		} while (newp > 0 && motion.size() > 0);

// 		ncls++;
// 	}
// 	//--------------</cluster creation>----------------------

// 	//--------------<clusters to objects>--------------------
// 	for (int cls = 0; cls < ncls; cls++)
// 	{
// 		if (clusters[cls].size() > mpc) // if there are enough moving points
// 		{

// 			Point2f clustercenter = cluster_center(clusters[cls]);
// 			imagbuf = frame_resizing(framebuf);
// 			imagbuf.convertTo(imagbuf, CV_8UC3);
// 			img = imagbuf(cv::Range(max_u16(clustercenter.y - objhsz), min_u16(imagbuf.rows, clustercenter.y + objhsz))
// 				, cv::Range(max_u16(clustercenter.x - objhsz), min_u16(imagbuf.cols, clustercenter.x + objhsz)));
// 			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
// 			img.convertTo(img, CV_8UC3);

// 			ALObject obj(objects.size(), ObjClass::ANT, clusters[cls], img);
// 			bool newobj = true;

// 			for (int i = 0; i < objects.size(); i++)
// 			{
// 				float r = sqrt(pow((objects[i].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[i].cluster_center.y - obj.cluster_center.y), 2));
// 				// float r = sqrt(pow((objects[i].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[i].proposed_center().y - obj.cluster_center.y), 2));
// 				if (r < robj * 1.0 * resolution / reduseres)
// 				{
// 					newobj = false;

// 					objects[i].img = obj.img;
// 					objects[i].cluster_points = obj.cluster_points;
// 					objects[i].center_determine(frameId, true);
// 					break;
// 				}
// 			}

// 			if (newobj == true)
// 				objects.push_back(obj);
// 		}
// 	}
// 	//--------------</clusters to objects>-------------------

// 	for (int i = 0; i < objects.size(); i++)
// 		objects[i].push_track_point(objects[i].cluster_center);

// 	// //--------------<visualization>--------------------------
// 	// for (const auto& obj: detobjs) {
// 	// 	pt1.x = obj.center.x - obj.size.width / 2;
// 	// 	pt1.y = obj.center.y - obj.size.height / 2;

// 	// 	pt2.x = obj.center.x + obj.size.width / 2;
// 	// 	pt2.y = obj.center.y + obj.size.height / 2;

// 	// 	const auto  clr = objClassColor(obj.type);
// 	// 	rectangle(imag, pt1, pt2, clr, 1);
// 	// 	const string caption = string(objClassTitle(obj.type)) + to_string(obj.id);
// 	// 	pt1.y -= 2;
// 	// 	cv::putText(imag, // target image
// 	// 				caption,     // text
// 	// 				pt1,    // top-left position
// 	// 				1,
// 	// 				1,
// 	// 				clr, // font color
// 	// 				1);
// 	// }

// 	for (int i = 0; i < objects.size(); i++)
// 	{
// 		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
// 		{
// 			pt1.x = objects.at(i).cluster_points.at(j).x;
// 			pt1.y = objects.at(i).cluster_points.at(j).y;

// 			pt2.x = objects.at(i).cluster_points.at(j).x + resolution / reduseres;
// 			pt2.y = objects.at(i).cluster_points.at(j).y + resolution / reduseres;

// 			rectangle(imag, pt1, pt2, objColor(objects.at(i).id), 1);
// 		}

// 		if (objects.at(i).model_center.x > -1) // visualization of the classifier
// 		{
// 			pt1.x = objects.at(i).model_center.x - objects.at(i).size.width / 2;
// 			pt1.y = objects.at(i).model_center.y - objects.at(i).size.height / 2;

// 			pt2.x = objects.at(i).model_center.x + objects.at(i).size.width / 2;
// 			pt2.y = objects.at(i).model_center.y + objects.at(i).size.height / 2;

// 			rectangle(imag, pt1, pt2, objColor(objects.at(i).id), 1);
// 		}

// 		for (int j = 0; j < objects.at(i).track_points.size(); j++)
// 			cv::circle(imag, objects.at(i).track_points.at(j), 1, objColor(objects.at(i).id), 2);
// 	}
// 	//--------------</visualization>-------------------------

// 	//--------------<baseimag>-------------------------------
// 	Mat baseimag(resolution, resolution + extr, CV_8UC3, Scalar(0, 0, 0));

// 	for (int i = 0; i < objects.size(); i++)
// 	{
// 		string text = objClassTitle(objects.at(i).type) + to_string(objects.at(i).id);

// 		Point2f ptext;
// 		ptext.x = 20;
// 		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

// 		cv::putText(baseimag, // target image
// 					text,     // text
// 					ptext,    // top-left position
// 					1,
// 					1,
// 					objColor(objects.at(i).id), // font color
// 					1);

// 		pt1.x = ptext.x - 1;
// 		pt1.y = ptext.y - 1 + 10;

// 		pt2.x = ptext.x + objects.at(i).img.cols + 1;
// 		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

// 		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
// 		{
// 			rectangle(baseimag, pt1, pt2, objColor(objects.at(i).id), 1);
// 			ptext = pt1;
// 			ptext.y -= 2;
// 			cv::putText(baseimag, // target image
// 						text,     // text
// 						ptext,    // top-left position
// 						1,
// 						1,  // 0.8
// 						objColor(objects.at(i).id), // font color
// 						1);

// 			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
// 		}
// 	}

// 	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

// 	Point2f p_idframe;
// 	p_idframe.x = resolution + extr - 95;
// 	p_idframe.y = 50;
// 	cv::putText(baseimag, to_string(frameId), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
// 	cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
// 	//--------------</baseimag>-------------------------------

// 	imshow("Tracking", baseimag);
// 	cv::waitKey(10);

// 	/*
// 		al_objs.push_back(objects.at(1));
// 		if (al_objs.size() > 1)
// 		{
// 			draw_compare(al_objs.at(al_objs.size() - 2), al_objs.at(al_objs.size() - 1), objColor(al_objs.at(0).id));
// 		}
// 	*/

// 	return baseimag;
// }

// void trackingMotV2b(Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId)
// {
// 	vector<vector<Point2f>> clusters;
// 	vector<Point2f> motion;
// 	vector<Mat> imgs;

// 	Mat imageBGR0;
// 	Mat imageBGR;

// 	Mat imag;
// 	Mat imagbuf;
// 	Mat framebuf = frame;

// 	int rows = frame.rows;
// 	int cols = frame.cols;

// 	float koef = (float)rows / (float)992;

// 	// TODO: eliminate hardcoded values like in trackingMotV2_1
// 	float mpc = 15 * koef;   // minimum number of points for a cluster (good value 15)
// 	float nd = 9 * koef;     //(good value 6-15)
// 	float rcobj = 15 * koef; //(good value 15)
// 	float robj = 17 * koef;  //(good value 17)
// 	float mdist = 12 * koef; // maximum distance from cluster center (good value 10)
// 	int pft = 9;             // points fixation threshold (good value 9)

// 	Mat img;

// 	//--------------------<moution detections>--------------------

// 	float rwsize;
// 	float clsize;

// 	imagbuf = frame;

// 	resolution = rows; // cancels resizing, but the image is cropped to a square

// 	if (rows > cols)
// 	{
// 		rwsize = (float)resolution * rows / (float)cols;
// 		clsize = resolution;
// 	}
// 	else
// 	{
// 		rwsize = resolution;
// 		clsize = (float)resolution * cols / (float)rows;
// 	}

// 	cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
// 	cv::Rect rectb(0, 0, resolution, resolution);
// 	imag = imagbuf(rectb);

// 	cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
// 	imag.convertTo(imag, CV_8UC3);

// 	if (rows > cols)
// 	{
// 		rwsize = (float)reduseres * rows / (float)cols;
// 		clsize = reduseres;
// 	}
// 	else
// 	{
// 		rwsize = reduseres;
// 		clsize = (float)reduseres * cols / (float)rows;
// 	}

// 	cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
// 	cv::Rect rect0(0, 0, reduseres, reduseres);
// 	imageBGR0 = frame0(rect0);
// 	imageBGR0.convertTo(imageBGR0, CV_8UC1);

// 	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

// 	cv::Rect rect(0, 0, reduseres, reduseres);
// 	imageBGR = frame(rect);

// 	imageBGR.convertTo(imageBGR, CV_8UC1);

// 	Point2f pm;

// 	for (int y = 0; y < imageBGR0.rows; y++)
// 	{
// 		for (int x = 0; x < imageBGR0.cols; x++)
// 		{
// 			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
// 			uchar color2 = imageBGR.at<uchar>(Point(x, y));

// 			if (((int)color2 - (int)color1) > pft)
// 			{
// 				pm.x = x * resolution / reduseres;
// 				pm.y = y * resolution / reduseres;
// 				motion.push_back(pm);
// 			}
// 		}
// 	}

// 	Point2f pt1;
// 	Point2f pt2;

// 	for (int i = 0; i < motion.size(); i++) // visualization of the cluster_points
// 	{
// 		pt1.x = motion.at(i).x;
// 		pt1.y = motion.at(i).y;

// 		pt2.x = motion.at(i).x + resolution / reduseres;
// 		pt2.y = motion.at(i).y + resolution / reduseres;

// 		rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
// 	}

// 	uint16_t ncls = 0;
// 	uint16_t nobj = objects.empty() ? -1 : 0;
// 	//--------------</moution detections>--------------------

// 	//--------------<layout of motion points by objects>-----

// 	if (objects.size() > 0)
// 	{
// 		for (int i = 0; i < objects.size(); i++)
// 		{
// 			objects[i].cluster_points.clear();
// 		}

// 		for (int i = 0; i < motion.size(); i++)
// 		{
// 			for (int j = 0; j < objects.size(); j++)
// 			{
// 				if (objects[j].model_center.x < 0)
// 					continue;

// 				if (i < 0)
// 					break;

// 				if ((motion.at(i).x < (objects[j].model_center.x + objects[j].size.width / 2)) && (motion.at(i).x > (objects[j].model_center.x - objects[j].size.width / 2)) && (motion.at(i).y < (objects[j].model_center.y + objects[j].size.height / 2)) && (motion.at(i).y > (objects[j].model_center.y - objects[j].size.height / 2)))
// 				{
// 					objects[j].cluster_points.push_back(motion.at(i));
// 					motion.erase(motion.begin() + i);
// 					i--;
// 				}
// 			}
// 		}

// 		float rm = rcobj * 1.0 * resolution / reduseres;

// 		for (int i = 0; i < motion.size(); i++)
// 		{
// 			rm = sqrt(pow((objects[0].cluster_center.x - motion.at(i).x), 2) + pow((objects[0].cluster_center.y - motion.at(i).y), 2));

// 			uint16_t iobj = -1;
// 			if (rm < rcobj * 1.0 * resolution / reduseres)
// 				iobj = 0;

// 			for (int j = 1; j < objects.size(); j++)
// 			{
// 				float r = sqrt(pow((objects[j].cluster_center.x - motion.at(i).x), 2) + pow((objects[j].cluster_center.y - motion.at(i).y), 2));
// 				if (r < rcobj * 1.0 * resolution / reduseres && r < rm)
// 				{
// 					rm = r;
// 					iobj = j;
// 				}
// 			}

// 			if (iobj != static_cast<decltype(iobj)>(-1))
// 			{
// 				objects[iobj].cluster_points.push_back(motion.at(i));
// 				motion.erase(motion.begin() + i--);
// 				objects[iobj].center_determine(frameId, false);
// 				if (i < 0)
// 					break;
// 			}
// 		}
// 	}

// 	for (int j = 0; j < objects.size(); j++)
// 	{

// 		objects[j].center_determine(frameId, false);

// 		Point2f clustercenter = objects[j].cluster_center;

// 		imagbuf = frame_resizing(framebuf);
// 		imagbuf.convertTo(imagbuf, CV_8UC3);

// 		if (clustercenter.y - objhsz * koef < 0)
// 			clustercenter.y = objhsz * koef + 1;

// 		if (clustercenter.x - objhsz * koef < 0)
// 			clustercenter.x = objhsz * koef + 1;

// 		if (clustercenter.y + objhsz * koef > imagbuf.rows)
// 			clustercenter.y = imagbuf.rows - 1;

// 		if (clustercenter.x + objhsz * koef > imagbuf.cols)
// 			clustercenter.x = imagbuf.cols - 1;

// 		img = imagbuf(cv::Range(max_u16(clustercenter.y - objhsz * koef), min_u16(imagbuf.rows, clustercenter.y + objhsz * koef))
// 			, cv::Range(max_u16(clustercenter.x - objhsz * koef), min_u16(imagbuf.cols, clustercenter.x + objhsz * koef)));
// 		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
// 		img.convertTo(img, CV_8UC3);
// 		objects[j].img = img;
// 		// objects[j].samples_creation();
// 	}
// 	//--------------</layout of motion points by objects>----

// 	//--------------<cluster creation>-----------------------

// 	while (motion.size() > 0)
// 	{
// 		Point2f pc;

// 		if (nobj != static_cast<decltype(nobj)>(-1) && nobj < objects.size())
// 		{
// 			pc = objects[nobj].cluster_center;
// 			// pc = objects[nobj].proposed_center();
// 			nobj++;
// 		}
// 		else
// 		{
// 			pc = motion.at(0);
// 			motion.erase(motion.begin());
// 		}

// 		clusters.push_back(vector<Point2f>());
// 		clusters[ncls].push_back(pc);

// 		for (int i = 0; i < motion.size(); i++)
// 		{
// 			float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));
// 			if (r < nd * 1.0 * resolution / reduseres)
// 			{
// 				Point2f cl_c = cluster_center(clusters.at(ncls));
// 				r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
// 				if (r < mdist * 1.0 * resolution / reduseres)
// 				{
// 					clusters.at(ncls).push_back(motion.at(i));
// 					motion.erase(motion.begin() + i);
// 					i--;
// 				}
// 			}
// 		}

// 		int newp;
// 		do
// 		{
// 			newp = 0;

// 			for (int c = 0; c < clusters[ncls].size(); c++)
// 			{
// 				pc = clusters[ncls].at(c);
// 				for (int i = 0; i < motion.size(); i++)
// 				{
// 					float r = sqrt(pow((pc.x - motion.at(i).x), 2) + pow((pc.y - motion.at(i).y), 2));

// 					if (r < nd * 1.0 * resolution / reduseres)
// 					{
// 						Point2f cl_c = cluster_center(clusters.at(ncls));
// 						r = sqrt(pow((cl_c.x - motion.at(i).x), 2) + pow((cl_c.y - motion.at(i).y), 2));
// 						if (r < mdist * 1.0 * resolution / reduseres)
// 						{
// 							clusters.at(ncls).push_back(motion.at(i));
// 							motion.erase(motion.begin() + i);
// 							i--;
// 							newp++;
// 						}
// 					}
// 				}
// 			}
// 		} while (newp > 0 && motion.size() > 0);

// 		ncls++;
// 	}
// 	//--------------</cluster creation>----------------------

// 	//--------------<clusters to objects>--------------------
// 	for (int cls = 0; cls < ncls; cls++)
// 	{
// 		if (clusters[cls].size() > mpc) // if there are enough moving points
// 		{

// 			Point2f clustercenter = cluster_center(clusters[cls]);
// 			imagbuf = frame_resizing(framebuf);
// 			imagbuf.convertTo(imagbuf, CV_8UC3);
// 			img = imagbuf(cv::Range(max_u16(clustercenter.y - objhsz * koef), min_u16(imagbuf.rows, clustercenter.y + objhsz * koef))
// 				, cv::Range(max_u16(clustercenter.x - objhsz * koef), min_u16(imagbuf.cols, clustercenter.x + objhsz * koef)));
// 			cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
// 			img.convertTo(img, CV_8UC3);

// 			ALObject obj(objects.size(), ObjClass::ANT, clusters[cls], img);
// 			bool newobj = true;

// 			for (int i = 0; i < objects.size(); i++)
// 			{
// 				float r = sqrt(pow((objects[i].cluster_center.x - obj.cluster_center.x), 2) + pow((objects[i].cluster_center.y - obj.cluster_center.y), 2));
// 				// float r = sqrt(pow((objects[i].proposed_center().x - obj.cluster_center.x), 2) + pow((objects[i].proposed_center().y - obj.cluster_center.y), 2));
// 				if (r < robj * 1.0 * resolution / reduseres)
// 				{
// 					newobj = false;

// 					objects[i].img = obj.img;
// 					objects[i].cluster_points = obj.cluster_points;
// 					objects[i].center_determine(frameId, true);
// 					break;
// 				}
// 			}

// 			if (newobj == true)
// 				objects.push_back(obj);
// 		}
// 	}
// 	//--------------</clusters to objects>-------------------

// 	for (int i = 0; i < objects.size(); i++)
// 		objects[i].push_track_point(objects[i].cluster_center);

// 	//--------------<visualization>--------------------------
// 	for (int i = 0; i < objects.size(); i++)
// 	{
// 		for (int j = 0; j < objects.at(i).cluster_points.size(); j++) // visualization of the cluster_points
// 		{
// 			pt1.x = objects.at(i).cluster_points.at(j).x;
// 			pt1.y = objects.at(i).cluster_points.at(j).y;

// 			pt2.x = objects.at(i).cluster_points.at(j).x + resolution / reduseres;
// 			pt2.y = objects.at(i).cluster_points.at(j).y + resolution / reduseres;

// 			rectangle(imag, pt1, pt2, objColor(objects.at(i).id), 1);
// 		}

// 		if (objects.at(i).model_center.x > -1) // visualization of the classifier
// 		{
// 			pt1.x = objects.at(i).model_center.x - objects.at(i).size.width / 2;
// 			pt1.y = objects.at(i).model_center.y - objects.at(i).size.height / 2;

// 			pt2.x = objects.at(i).model_center.x + objects.at(i).size.width / 2;
// 			pt2.y = objects.at(i).model_center.y + objects.at(i).size.height / 2;

// 			rectangle(imag, pt1, pt2, objColor(objects.at(i).id), 1);
// 		}

// 		for (int j = 0; j < objects.at(i).track_points.size(); j++)
// 			cv::circle(imag, objects.at(i).track_points.at(j), 1, objColor(objects.at(i).id), 2);
// 	}
// 	//--------------</visualization>-------------------------

// 	//--------------<baseimag>-------------------------------
// 	Mat baseimag(resolution, resolution + extr * koef, CV_8UC3, Scalar(0, 0, 0));

// 	for (int i = 0; i < objects.size(); i++)
// 	{
// 		string text = objClassTitle(objects.at(i).type) + to_string(objects.at(i).id);

// 		Point2f ptext;
// 		ptext.x = 20;
// 		ptext.y = (30 + objects.at(i).img.cols) * objects.at(i).id + 20;

// 		cv::putText(baseimag, // target image
// 								text,     // text
// 								ptext,    // top-left position
// 								1,
// 								1,
// 								objColor(objects.at(i).id), // font color
// 								1);

// 		pt1.x = ptext.x - 1;
// 		pt1.y = ptext.y - 1 + 10;

// 		pt2.x = ptext.x + objects.at(i).img.cols + 1;
// 		pt2.y = ptext.y + objects.at(i).img.rows + 1 + 10;

// 		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
// 		{
// 			rectangle(baseimag, pt1, pt2, objColor(objects.at(i).id), 1);
// 			objects.at(i).img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, objects.at(i).img.cols, objects.at(i).img.rows)));
// 		}
// 	}

// 	imag.copyTo(baseimag(cv::Rect(extr * koef, 0, imag.cols, imag.rows)));

// 	Point2f p_idframe;
// 	p_idframe.x = resolution + extr * koef - 95;
// 	p_idframe.y = 50;
// 	cv::putText(baseimag, to_string(frameId), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
// 	cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
// 	//--------------</baseimag>-------------------------------
// 	// imshow("Tracking", baseimag);
// 	// cv::waitKey(0);
// }

void OBJdetectsToObjs(vector<OBJdetect> objdetects, vector<Obj> &objs)
{
	objs.clear();
	objs.reserve(objdetects.size());
	Obj objbuf;
	for (int i = 0; i < objdetects.size(); i++)
	{
		objbuf.type = static_cast<uint8_t>(objdetects.at(i).type);     // Object type
		objbuf.id = i;                           // Object id
		objbuf.x = objdetects.at(i).center.x;    // Center x of the bounding box
		objbuf.y = objdetects.at(i).center.y;    // Center y of the bounding box
		objbuf.w = objdetects.at(i).size.width; // Width of the bounding box
		objbuf.h = objdetects.at(i).size.height; // Height of the bounding box

		objs.push_back(objbuf);
	}
}

void ALObjectsToObjs(vector<ALObject> objects, vector<Obj> &objs)
{
	objs.clear();
	objs.reserve(objects.size());
	Obj objbuf;
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects.at(i);
		objbuf.type = static_cast<uint8_t>(oi.type);      // Object type
		objbuf.id = oi.id;              // Object id
		objbuf.x = oi.cluster_center.x; // Center x of the bounding box
		objbuf.y = oi.cluster_center.y; // Center y of the bounding box
		objbuf.w = 0;                              // Width of the bounding box
		objbuf.h = 0;                              // Height of the bounding box

		objs.push_back(objbuf);
	}
}

void fixIDs(const vector<vector<Obj>> &objs, vector<std::pair<uint, IdFix>> &fixedIds, vector<Mat> &d_images, float confidence, uint16_t framesize, const string& pathmodel, torch::DeviceType device)
{
	if(d_images.empty())
		return;

	vector<ALObject> objects;
	vector<Obj> objsbuf;
	vector<vector<Obj>> fixedobjs;
	IdFix idfix;

	if (framesize > 0)
	{
		for (int i = 0; i < d_images.size(); i++)
			d_images.at(i) = frame_resizing(d_images.at(i), framesize);
	}

	float koef = (float)d_images.at(0).rows / (float)resolution;
	const float maxr = 17.0 * koef; // set depending on the size of the ant

	fixedIds.clear();

	// trackingMotV2b(d_images.at(1), d_images.at(0), objects, 0);
	trackingMotV2_1(pathmodel, device, d_images.front(), d_images.front(), objects, 0, confidence);
	ALObjectsToObjs(objects, objsbuf);
	fixedobjs.push_back(objsbuf);

	for (int i = 0; i < d_images.size() - 1; i++)
	{
		// trackingMotV2b(d_images.at(i), d_images.at(i + 1), objects, i + 1);
		trackingMotV2_1(pathmodel, device, d_images[i], d_images[i + 1], objects, i + 1, confidence);
		ALObjectsToObjs(objects, objsbuf);
		fixedobjs.push_back(objsbuf);
	}

	for (int i = 0; i < objs.size(); i++)
	{
		const auto& oi = objs[i];
		auto& fxo = fixedobjs.at(i);
		for (int j = 0; j < oi.size(); j++)
		{
			if (ObjClass(oi.at(j).type) != ObjClass::ANT)
				continue;

			const auto& oij = oi[j];
			float minr = sqrt(pow((float)oij.x - (float)fxo.at(0).x * koef, 2) + pow((float)oij.y - (float)fxo.at(0).y * koef, 2));
			int ind = 0;

			for (int iobj = 0; iobj < fxo.size(); iobj++)
			{
				float r = sqrt(pow((float)oij.x - (float)fxo.at(iobj).x * koef, 2) + pow((float)oij.y - (float)fxo.at(iobj).y * koef, 2));
				if (r < minr)
				{
					minr = r;
					ind = iobj;
				}
			}

			if (minr < maxr && oij.id != fxo.at(ind).id)
			{
				idfix.id = fxo.at(ind).id;
				idfix.idOld = oij.id;
				idfix.type = oij.type; // not fixed
				fixedIds.push_back(std::make_pair((uint)i, idfix));
				fxo.erase(fxo.begin() + ind);
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

	Mat imgpmap(2 * objhsz, 2 * objhsz, CV_8UC3, Scalar(0, 0, 0));

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
	Mat imagchange(imag.rows, imag.cols, imag.type());
	cv::adaptiveThreshold(imag, imagchange, 0xFF, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY
		, std::max(2.f, roundf(objhsz / 12.f)) * 2 + 1, -1);  // 15, -2; 7, 1; 11, 2; roundf(objhsz/12.f) * 2 + 1

	// // constexpr uint16_t color_threshold = 80; // 65-70, 80
	// // imag.copyTo(imagchange);
	// // for (int y = 0; y < imag.rows; y++)
	// // {
	// // 	for (int x = 0; x < imag.cols; x++)
	// // 	{
	// // 		uchar color1 = imag.at<uchar>(Point(x, y));
	// //
	// // 		if (color1 < (uchar)color_threshold) // 65-70
	// // 			imagchange.at<uchar>(Point(x, y)) = 0;
	// // 		else
	// // 			imagchange.at<uchar>(Point(x, y)) = 255;
	// // 	}
	// // }

	// imshow("imagchange", imagchange);
	// cv::waitKey(500);

	return imagchange;
}

// void objdeterm(vector<Point2f> &cluster_points, Mat frame, ALObject &obj, size_t frameId)
// {

// 	int maxr = 1;       // if point is outside then ignore
// 	int half_range = 6; // half deviation

// 	int wh = 800;
// 	vector<Mat> cluster_samples;
// 	Point2f minp;
// 	Point2f maxp;
// 	Point2f bufp;

// 	Point2f p1;
// 	Point2f p2;

// 	minp.y = frame.rows;
// 	minp.x = frame.cols;

// 	maxp.x = 0;
// 	maxp.y = 0;
// 	Mat bufimg;

// 	vector<int> npforpoints;
// 	vector<int> npsamples;
// 	vector<Point2f> mpoints;

// 	vector<vector<int>> all_npforpoints;

// 	for (int i = 0; i < cluster_points.size(); i++)
// 	{
// 		if (cluster_points.at(i).y < minp.y)
// 			minp.y = cluster_points.at(i).y;

// 		if (cluster_points.at(i).x < minp.x)
// 			minp.x = cluster_points.at(i).x;

// 		if (cluster_points.at(i).y > maxp.y)
// 			maxp.y = cluster_points.at(i).y;

// 		if (cluster_points.at(i).x > maxp.x)
// 			maxp.x = cluster_points.at(i).x;

// 		bufimg = frame(cv::Range(cluster_points.at(i).y, min_u16(frame.rows, cluster_points.at(i).y + resolution / reduseres))
// 			, cv::Range(cluster_points.at(i).x, min_u16(frame.cols, cluster_points.at(i).x + resolution / reduseres)));
// 		cluster_samples.push_back(bufimg);

// 		npforpoints.push_back(0);
// 	}

// 	int st = 1; // step for samles compare

// 	Mat imag = obj.img;

// 	int start_y = 0;
// 	int start_x = 0;

// 	int and_y = (int)((imag.rows - resolution / reduseres) / st);
// 	int and_x = (int)((imag.cols - resolution / reduseres) / st);

// 	for (int step_x = start_x; step_x < and_x; step_x++)
// 	{
// 		for (int step_y = start_y; step_y < and_y; step_y++)
// 		{
// 			bool cont = true;
// 			for (int i = 0; i < obj.cluster_points.size(); i++)
// 			{
// 				if (abs(step_y - (obj.cluster_points.at(i).y - obj.cluster_center.y + objhsz)) < maxr && abs(step_x - (obj.cluster_points.at(i).x - obj.cluster_center.x + objhsz)) < maxr)
// 					cont = false;
// 			}

// 			if (cont == true)
// 				continue;

// 			for (int i = 0; i < cluster_samples.size(); i++)
// 			{
// 				Mat sample;
// 				sample = imag(cv::Range(step_y * st, min_u16(imag.rows, step_y * st + resolution / reduseres))
// 					, cv::Range(step_x * st, min_u16(imag.cols, step_x * st + resolution / reduseres)));
// 				int npbuf = samples_compV2(cluster_samples.at(i), sample);
// 				npforpoints.at(i) = npbuf;
// 			}

// 			bufp.y = step_y * st;
// 			bufp.x = step_x * st;

// 			if (bufp.y > 0 && bufp.y < imag.rows && bufp.x > 0 && bufp.x < imag.cols)
// 			{
// 				// npsamples.push_back(np);
// 				mpoints.push_back(bufp);
// 				all_npforpoints.push_back(npforpoints);
// 			}
// 		}
// 	}

// 	vector<int> sample_np;
// 	vector<vector<Point2f>> alt_cluster_points;

// 	vector<intpoint> chain;
// 	vector<vector<intpoint>> chains;

// 	// std::cout << "cluster_points.size() - " << cluster_points.size() << endl;
// 	for (int ci = 0; ci < cluster_points.size(); ci++)
// 	{
// 		/*/------------------<TESTING>-----------------------------------
// 		Mat resimg = imag;
// 		Point2f correct;

// 		correct.x = 0;
// 		correct.y = 0;

// 		for (int i = 0; i < cluster_points.size(); i++)
// 		{
// 			p1.x = cluster_points.at(i).x - minp.x;
// 			p1.y = cluster_points.at(i).y - minp.y;
// 			p2.x = cluster_points.at(i).x - minp.x + resolution / reduseres;
// 			p2.y = cluster_points.at(i).y - minp.y + resolution / reduseres;

// 			p1 += correct;
// 			p2 += correct;

// 			Mat s_imag = cluster_samples.at(i);
// 			cv::cvtColor(s_imag, s_imag, cv::COLOR_BGR2RGB);
// 			s_imag.convertTo(s_imag, CV_8UC3);
// 			s_imag.copyTo(resimg(cv::Rect(p1.x, p1.y, resolution / reduseres, resolution / reduseres)));
// 			rectangle(resimg, p1, p2, Scalar(0, 255, 0), 1);


// 			p1.x = p1.x + (resolution / reduseres) / 2;
// 			p1.y = p1.y + (resolution / reduseres) / 2;

// 			if (i == ci)
// 				cv::circle(resimg, p1, 1, Scalar(0, 0, 255), 1);
// 		}

// 		cv::resize(resimg, resimg, cv::Size(resimg.cols * 5, resimg.rows * 5), cv::InterpolationFlags::INTER_CUBIC);
// 		imshow("resimg", resimg);
// 		cv::waitKey(0);

// 		//------------------</TESTING>-----------------------------------*/

// 		sample_np.clear();
// 		for (int i = 0; i < all_npforpoints.size(); i++)
// 			sample_np.push_back(all_npforpoints.at(i).at(ci));

// 		alt_cluster_points.push_back(map_prob(sample_np, mpoints));
// 	}

// 	for (int i = 0; i < alt_cluster_points.size(); i++)
// 	{
// 		for (int ci = 0; ci < alt_cluster_points.at(i).size(); ci++)
// 		{
// 			chain.clear();
// 			intpoint bufch;
// 			bufch.ipoint = i;
// 			bufch.mpoint = alt_cluster_points.at(i).at(ci);
// 			chain.push_back(bufch);

// 			for (int j = i + 1; j < alt_cluster_points.size(); j++)
// 			{
// 				Point2f dp;
// 				dp.x = cluster_points.at(i).x - cluster_points.at(j).x;
// 				dp.y = cluster_points.at(i).y - cluster_points.at(j).y;

// 				for (int cj = 0; cj < alt_cluster_points.at(j).size(); cj++)
// 				{
// 					if (abs(alt_cluster_points.at(i).at(ci).x - alt_cluster_points.at(j).at(cj).x - dp.x) < half_range
// 					&& abs(alt_cluster_points.at(i).at(ci).y - alt_cluster_points.at(j).at(cj).y - dp.y) < half_range)
// 					{
// 						bufch.ipoint = j;
// 						bufch.mpoint = alt_cluster_points.at(j).at(cj);
// 						chain.push_back(bufch);
// 						break;
// 					}
// 				}
// 			}
// 			chains.push_back(chain);
// 		}
// 	}

// 	int maxchains = 0;
// 	for (int i = 0; i < chains.size(); i++)
// 	{
// 		if (maxchains < chains.at(i).size())
// 			maxchains = i;
// 	}

// 	obj.cluster_points.clear();
// 	for (int i = 0; i < chains.at(maxchains).size(); i++)
// 		obj.cluster_points.push_back(cluster_points.at(chains.at(maxchains).at(i).ipoint));

// 	/*
// 		vector<Point2f> cluster_points_buf;

// 		int maxradd = 5;
// 		for(int i=0; i < cluster_points.size(); i++)
// 		{
// 			 bool push = false;

// 				for (int j = 0; j < obj.cluster_points.size(); j++)
// 				{
// 					if(abs(obj.cluster_points.at(j).y - cluster_points.at(i).y) < maxradd && abs(obj.cluster_points.at(j).x - cluster_points.at(i).x) < maxradd)
// 					{
// 						cluster_points_buf.push_back(cluster_points.at(i));
// 						break;
// 					}
// 				}
// 		}

// 		for(int i=0 ; i< cluster_points_buf.size(); i++)
// 			obj.cluster_points.push_back(cluster_points_buf.at(i));*/

// 	/*/------------------<removing motion points from a cluster>-------------------
// 	vector<Point2f> cluster_points_buf;
// 	for (int i = 0; i < cluster_points.size(); i++)
// 	{
// 		bool cpy = true;
// 		for(int j=0; j<chains.at(maxchains).size(); j++)
// 		{
// 			if(i==chains.at(maxchains).at(j).ipoint)
// 			{
// 				cpy = false;
// 				break;
// 			}
// 		}

// 		if(cpy == true)
// 				cluster_points_buf.push_back(cluster_points.at(i));
// 	}

// 	cluster_points.clear();
// 	for(int i=0; i<cluster_points_buf.size(); i++)
// 	 cluster_points.push_back(cluster_points_buf.at(i));
// 	//------------------<removing motion points from a cluster>-------------------*/

// 	obj.center_determine(frameId, false);
// 	frame.convertTo(bufimg, CV_8UC3);
// 	obj.img = bufimg(cv::Range(max_u16(obj.cluster_center.y - objhsz), min_u16(bufimg.rows, obj.cluster_center.y + objhsz))
// 		, cv::Range(max_u16(obj.cluster_center.x - objhsz), min_u16(bufimg.cols, obj.cluster_center.x + objhsz)));
// 	cv::cvtColor(obj.img, obj.img, cv::COLOR_BGR2RGB);
// 	obj.img.convertTo(obj.img, CV_8UC3);

// 	//-----------------<probe visualization>-----------------
// 	Mat resimg;
// 	imag.copyTo(resimg);

// 	Mat resimg2(resimg.rows, resimg.cols, CV_8UC3, Scalar(0, 0, 0));
// 	Mat res2x(resimg.rows, resimg.cols * 2, CV_8UC3, Scalar(0, 0, 0));

// 	p1.y = minp.y + (maxp.y - minp.y) / 2 - objhsz;
// 	p1.x = minp.x + (maxp.x - minp.x) / 2 - objhsz;

// 	p2.y = minp.y + (maxp.y - minp.y) / 2 + objhsz;
// 	p2.x = minp.x + (maxp.x - minp.x) / 2 + objhsz;

// 	bufimg = frame(cv::Range(max_u16(p1.y), min_u16(frame.rows, p2.y)), cv::Range(max_u16(p1.x), min_u16(frame.cols, p2.x)));

// 	cv::cvtColor(bufimg, bufimg, cv::COLOR_BGR2RGB);
// 	bufimg.convertTo(bufimg, CV_8UC3);
// 	bufimg.copyTo(resimg2(cv::Rect(0, 0, bufimg.cols, bufimg.rows)));

// 	for (int i = 0; i < cluster_points.size(); i++)
// 	{
// 		p1.x = cluster_points.at(i).x - minp.x - (maxp.x - minp.x) / 2 + objhsz;
// 		p1.y = cluster_points.at(i).y - minp.y - (maxp.y - minp.y) / 2 + objhsz;
// 		p2.x = p1.x + resolution / reduseres;
// 		p2.y = p1.y + resolution / reduseres;
// 		// rectangle(resimg2, p1, p2, Scalar(0, 0, 255), 1);
// 	}

// 	for (int i = 0; i < chains.at(maxchains).size(); i++)
// 	{
// 		size_t R = rand() % 255;
// 		size_t G = rand() % 255;
// 		size_t B = rand() % 255;

// 		p1.x = chains.at(maxchains).at(i).mpoint.x;
// 		p1.y = chains.at(maxchains).at(i).mpoint.y;
// 		p2.x = p1.x + resolution / reduseres;
// 		p2.y = p1.y + resolution / reduseres;

// 		rectangle(resimg, p1, p2, Scalar(R, G, B), 1);

// 		p1.x = cluster_points.at(chains.at(maxchains).at(i).ipoint).x - minp.x - (maxp.x - minp.x) / 2 + objhsz;
// 		p1.y = cluster_points.at(chains.at(maxchains).at(i).ipoint).y - minp.y - (maxp.y - minp.y) / 2 + objhsz;
// 		p2.x = p1.x + resolution / reduseres;
// 		p2.y = p1.y + resolution / reduseres;

// 		rectangle(resimg2, p1, p2, Scalar(R, G, B), 1);
// 	}

// 	resimg2.copyTo(res2x(cv::Rect(0, 0, resimg2.cols, resimg2.rows)));
// 	resimg.copyTo(res2x(cv::Rect(resimg2.cols, 0, resimg2.cols, resimg2.rows)));

// 	cv::resize(res2x, res2x, cv::Size(res2x.cols * 5, res2x.rows * 5), cv::InterpolationFlags::INTER_CUBIC);

// 	imshow("res2x", res2x);
// 	cv::waitKey(0);
// 	//-----------------</probe visualization>-----------------*/
// }

bool compare_clsobj(ClsObjR a, ClsObjR b)
{
	if (a.r < b.r)
		return 1;
	else
		return 0;
}

Mat trackingMotV2_1(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence, const string& outfileBase)
{
	const float koef = (float)frame0.rows / (float)model_resolution;
	// Tracking parameters
	const uint8_t mpct = 5 * koef;      // minimum number of points for a cluster (tracking object) (good value 5); 3
	const uint8_t mpcc = 13 * koef;      // minimum number of points for a cluster (creation new object) (good value 13); 7
	const float nd = 3 * koef;    //(good value 6-15); max distance ratio between motion points to join them
	//float nd = 1.5;    //(good value 6-15); max distance ration between motion points to join them
	const uint8_t rcobj = 15 * koef;    //(good value 15);  mean radius of a cluster
	//float robj = 22.0; //(good value 17)
	const float rclgap = 1.75f;  // ratio of a cluster gap on subsequent frames: 1.2 .. 1.8;  1.42f
	const float rclmv = 0.5f;  // ratio of a cluster movement: 0.2 .. 0.8
	//float robj_k = 1.0;
	//int mdist = 10; // maximum distance from cluster center (good value 10)
	//int pft = 1;    // points fixation threshold (good value 9)

	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	// if (!pathmodel.empty())
	//   frame = frame_resizing(frame);

	const bool init = frame.data == frame0.data;
	if(!pathmodel.empty()) {
		frame0 = frame_resizing(frame0);
		frame = init ? frame0 : frame_resizing(frame);
	}

	// Initialize framebuf for the visual output of tracking results
	Mat framebuf;
	//if(frame.type() != CV_8UC3)
	if(frame.channels() != 3) {
		// framebuf = Mat(frame.rows, frame.cols, CV_8UC3);
		cv::cvtColor(frame, framebuf, cv::COLOR_GRAY2BGR);
		// cv::cvtColor(framebuf, framebuf, cv::COLOR_BGR2RGB);
		// framebuf.convertTo(framebuf, CV_8UC3);
	} else framebuf = frame.clone();

	Mat img;

	vector<OBJdetect> detects;

	//--------------------<detection using a classifier>----------
	if (!pathmodel.empty())
	{
		string outfile = outfileBase;
		if(!outfile.empty())
			outfile.append("_").append(to_string(frameId)).append(".jpg");
		detects = detectorV4(pathmodel, frame, device_type, confidence, outfile);

		// Reset object detection flag for already existing objects
		for (int i = 0; i < objects.size(); i++)
			objects[i].det_mc = false;

		// for (unsigned i = 0; i < detects.size(); i++)
		// {
		// 	if (ObjClass(detects[i].type) != ObjClass::ANT)
		// 		detects.erase(detects.begin() + i--);
		// }
		//
		// Remove non-ant objects
		float objLinSz = init ? 0 : objhsz;
		unsigned  ndobjs = 0;
		for (int i = 0; i < detects.size(); i++)
		{
			// TODO: consider tracking of non-ant objects
			if (ObjClass(detects.at(i).type) != ObjClass::ANT) {  // Note: "ta" also should be tracked
				detects.erase(detects.begin() + i--);
				++ndobjs;
			}
			else if(objLinSz) {
				// Update motion-related parameters
				objLinSz = (objLinSz * detects.size() + std::max(detects.at(i).size.width, detects.at(i).size.height)) / (detects.size() + 1);
			} else {
				// Init motion-related parameters
				objLinSz = (objLinSz * (i + ndobjs) + std::max(detects.at(i).size.width, detects.at(i).size.height)) / (i + 1 + ndobjs);
			}
		}
		if(objLinSz < antLenMin)
			objLinSz = antLenMin;
		objhsz = objLinSz / 2 + 1;
		reduseres = evalReduseres(objhsz);
		cout << "objhsz: " << objhsz << ", reduseres: " << std::fixed << reduseres << endl;

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			vector<Point2f> cluster_points;
			const auto& det = detects[i];
			cluster_points.push_back(det.center);
			img = framebuf(cv::Range(max_u16(det.center.y - det.size.height/2), min_u16(framebuf.rows, det.center.y + det.size.height/2))
				, cv::Range(max_u16(det.center.x - det.size.width/2), min_u16(framebuf.cols, det.center.x + det.size.width/2)));
			// cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			// img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), det.type, det.size, det.center, cluster_points, img, det.prob);
			// obj.track_points.push_back(det.center);
			// obj.push_track_point(det.center);

			// float rm = rcobj * (float)resolution / (float)reduseres;
			float rm = std::max(obj.size.width, obj.size.height) * rclgap;  // Object movement distance on subsequent frames
			bool newobj = true;
			uint16_t iobj;  // Index of a new detected object that is closest to the existing object to be updated

			for (uint16_t j = 0; j < objects.size(); j++)
			{
				const float r = distance(objects[j].cluster_center, obj.cluster_center);
				// float r = distance(objects[j].proposed_center(), obj.cluster_center);
				if (r < rcobj * (float)framebuf.rows / (float)reduseres && r < rm)
				{
					rm = r;
					iobj = j;
					//if(r < objhsz)
					newobj = false;
				}
			}

			if (newobj == false)
			{
				auto& tobj = objects.at(iobj);
				// // void  showOcclusion(Mat& frameClr, const Point& o1corn1, const Point& o1corn2, const Scalar& clr1, const Point& o2corn1, const Point& o2corn2, const Scalar& clr2, const int w=1);
				// if(!tobj.traces.empty() && tobj.traces.back().frame == frameId) {
				//   Mat  dimg(frame.rows, frame.cols, CV_8UC3);
				//   cv::cvtColor(frame, dimg, cv::COLOR_GRAY2BGR);
				//   //frame.convertTo(dimg, CV_8UC3);
				//   rectangle(dimg, Point(roundf(tobj.model_center.x - tobj.size.width/2.f)-1, roundf(tobj.model_center.y - tobj.size.height/2.f)-1)
				//     , Point(roundf(tobj.model_center.x + tobj.size.width/2.f)+1, roundf(tobj.model_center.y + tobj.size.height/2.f)+1), Scalar(255, 0, 0), 1);
				//   rectangle(dimg, Point(roundf(obj.model_center.x - obj.size.width/2.f)+1, roundf(obj.model_center.y - obj.size.height/2.f)+1)
				//     , Point(roundf(obj.model_center.x + obj.size.width/2.f)-1, roundf(obj.model_center.y + obj.size.height/2.f)-1), Scalar(0, 255, 0), 1);
				//   imshow("Occlusion", dimg);
				//   cv::waitKey(500);
				// }

				// Resolve possible overlapping objects

				tobj.model_center = obj.model_center;
				tobj.size = obj.size;
				tobj.img = obj.img;
				tobj.det_mc = obj.det_mc;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < frameId && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
					, obj.size.width, obj.size.height, obj.prob});
			}
			else
			{
				for (size_t j = 0; j < objects.size(); j++)
				{
					if (!objects[j].det_pos)
						continue;

					// if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
					if (distance(objects[j].cluster_center, obj.cluster_center) < rm)  // rclmv * objhsz
					{
						newobj = false;
						break;
					}
				}

				if (newobj == true) {
					assert(obj.traces.empty() && "Unexpected traces");
					obj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
						, obj.size.width, obj.size.height, obj.prob});
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
		rwsize = roundf(resolution * float(rows) / cols);
		clsize = resolution;
	}
	else
	{
		rwsize = resolution;
		clsize = roundf(resolution * float(cols) / rows);
	}

	cv::resize(imagbuf, imagbuf, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	cv::Rect rectb(0, 0, resolution, resolution);

	imag = imagbuf(rectb);

	//- cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
	//- imag.convertTo(imag, CV_8UC3);

	if (rows > cols)
	{
		rwsize = roundf(reduseres * float(rows) / cols);
		clsize = reduseres;
	}
	else
	{
		rwsize = reduseres;
		clsize = roundf(reduseres * float(cols) / rows);
	}

	cv::resize(frame0, frame0, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);
	cv::Rect rect0(0, 0, reduseres, reduseres);
	imageBGR0 = frame0(rect0);
	if(imageBGR0.channels() != 1)
		cv::cvtColor(imageBGR0, imageBGR0, cv::COLOR_BGR2GRAY);
	// imageBGR0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	cv::Rect rect(0, 0, reduseres, reduseres);
	imageBGR = frame(rect);
	if(imageBGR.channels() != 1)
		cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2GRAY);
	// imageBGR.convertTo(imageBGR, CV_8UC1);

	Point2f pm;

	imageBGR0 = color_correction(imageBGR0);
	imageBGR = color_correction(imageBGR);

	for (uint16_t y = 0; y < imageBGR0.rows; y++)
	{
		for (uint16_t x = 0; x < imageBGR0.cols; x++)
		{
			uchar color1 = imageBGR0.at<uchar>(Point(x, y));
			uchar color2 = imageBGR.at<uchar>(Point(x, y));

			// if (((int)color2 - (int)color1) > pft)
			if (color2 > color1)
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
	uint16_t nobj = objects.empty() ? -1 : 0;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj != static_cast<decltype(nobj)>(-1) && nobj < objects.size())
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
		auto& cn = clusters.at(ncls);
		cn.push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			if (distance(pc, motion.at(i)) < nd * (float)resolution / reduseres)
			{
				// if (r < mdist * (float)resolution / reduseres)
				if (distance(cluster_center(cn), motion.at(i)) < objhsz)
				{
					cn.push_back(motion.at(i));
					motion.erase(motion.begin() + i--);
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < cn.size(); c++)
			{
				pc = cn.at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					if (distance(pc, motion.at(i)) < nd * (float)resolution / (float)reduseres)
					{
						// if (r < mdist * 1.0 * resolution / reduseres)
						if (distance(cluster_center(cn), motion.at(i)) < objhsz)
						{
							cn.push_back(motion.at(i));
							motion.erase(motion.begin() + i--);
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
				const Point2f clustercenter = cluster_center(clusters[cls]);
				// clsobjr.r = distance(objects[i].cluster_center, clustercenter)
				clsobjr.r = distance(objects[i].proposed_center(), clustercenter);
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (!pathmodel.empty())
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				size_t obj_id = clsobjrs.at(i).obj_id;

				if (objects.at(obj_id).det_mc)
				{
					const auto& cl = clusters.at(cls_id);
					Point2f clustercenter = cluster_center(cl);
					auto& cobj = objects[obj_id];
					pt1.x = cobj.model_center.x - cobj.size.width / 2;
					pt1.y = cobj.model_center.y - cobj.size.height / 2;

					pt2.x = cobj.model_center.x + cobj.size.width / 2;
					pt2.y = cobj.model_center.y + cobj.size.height / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{

						if (!cobj.det_pos)
							cobj.cluster_points = cl;
						else
						{
							for (size_t j = 0; j < cl.size(); j++)
								cobj.cluster_points.push_back(cl.at(j));
						}

						cobj.center_determine(frameId, false);

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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			// if (clsobjrs.at(i).r < (float)robj * (float)resolution / (float)reduseres && cl.size() >= mpct)
			if (clsobjrs.at(i).r < rclgap * objhsz && cl.size() >= mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (!cobj.det_pos)
					cobj.cluster_points = cl;
				else
				{
					continue;
					// for (size_t j = 0; j < cl.size(); j++)
					//   cobj.cluster_points.push_back(cl.at(j));
				}

				cobj.center_determine(frameId, false);

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
		// Don't add motion-based object when model-based detection is applied
		if (pathmodel.empty())
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				const auto& cl = clusters.at(cls_id);
				size_t obj_id = clsobjrs.at(i).obj_id;

				Point2f clustercenter = cluster_center(cl);
				bool newobj = true;

				for (size_t j = 0; j < objects.size(); j++)
				{
					// if (r < (float)robj * 2.3 * (float)resolution / (float)reduseres)
					if (distance(objects[j].cluster_center, clustercenter) < rclmv * objhsz)
					{
						newobj = false;
						break;
					}
				}

				if (cl.size() > mpcc && newobj == true) // if there are enough moving points
				{
					img = framebuf(cv::Range(max_u16(clustercenter.y - objhsz), min_u16(framebuf.rows, clustercenter.y + objhsz))
						, cv::Range(max_u16(clustercenter.x - objhsz), min_u16(framebuf.cols, clustercenter.x + objhsz)));
					// cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
					// img.convertTo(img, CV_8UC3);

					ALObject obj(objects.size(), ObjClass::ANT, cl, img);
					assert(obj.traces.empty() && "Unexpected traces");
					obj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
							, obj.size.width, obj.size.height, obj.prob});
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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			// if (clsobjrs.at(i).r < (float)robj * robj_k * (float)resolution / (float)reduseres && cl.size() >= mpct / 2)
			if (clsobjrs.at(i).r < rclgap * objhsz && cl.size() >= mpct / 2 + 1)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < cl.size(); j++)
					cobj.cluster_points.push_back(cl.at(j));

				cobj.center_determine(frameId, false);

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
	else if (pathmodel.empty())
	{
		//--<new obj>--
		for (int cls = 0; cls < clusters.size(); cls++)
		{
			Point2f clustercenter = cluster_center(clusters.at(cls));
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				//if (r < (float)robj * (float)resolution / (float)reduseres)
				if (distance(objects[i].cluster_center, clustercenter) < rclmv * objhsz)
				{
					newobj = false;
					break;
				}
			}

			if (clusters.at(cls).size() > mpcc && newobj == true) // if there are enough moving points
			{
				img = framebuf(cv::Range(max_u16(clustercenter.y - objhsz), min_u16(framebuf.rows, clustercenter.y + objhsz))
					, cv::Range(max_u16(clustercenter.x - objhsz), min_u16(framebuf.cols, clustercenter.x + objhsz)));
				// cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				// img.convertTo(img, CV_8UC3);

				ALObject obj(objects.size(), ObjClass::ANT, clusters.at(cls), img);
				assert(obj.traces.empty() && "Unexpected traces");
				obj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
						, obj.size.width, obj.size.height, obj.prob});
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
		auto& obj = objects[i];
		if (!obj.det_mc && !obj.det_pos)
			continue;

		// framebuf.convertTo(imagbuf, CV_8UC3);
		imagbuf = framebuf;
		if (!obj.det_mc)
		{
			cv::Size sz = obj.size;
			if(sz.width <= 0)
				sz = cv::Size(objhsz, objhsz);
			pt1.y = obj.cluster_center.y - objhsz;
			pt2.y = obj.cluster_center.y + objhsz;

			pt1.x = obj.cluster_center.x - objhsz;
			pt2.x = obj.cluster_center.x + objhsz;

			if (pt1.y < 0)
				pt1.y = 0;

			if (pt2.y > imagbuf.rows)
				pt2.y = imagbuf.rows;

			if (pt1.x < 0)
				pt1.x = 0;

			if (pt2.x > imagbuf.cols)
				pt2.x = imagbuf.cols;
		}
		else
		{
			pt1.y = obj.model_center.y - roundf(obj.size.height/2.f);
			pt2.y = obj.model_center.y + roundf(obj.size.height/2.f);

			pt1.x = obj.model_center.x - roundf(obj.size.width/2.f);
			pt2.x = obj.model_center.x + roundf(obj.size.width/2.f);
		}

		// std::cout << "<post processing 2>" << endl;
		// std::cout << "pt1 - " << pt1 << endl;
		// std::cout << "pt2 - " << pt2 << endl;

		img = imagbuf(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x));
		//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		//- img.convertTo(img, CV_8UC3);
		img.copyTo(obj.img);
		obj.center_determine(frameId, true);
		obj.push_track_point(obj.det_mc ? obj.model_center : obj.cluster_center);
	}
	// std::cout << "</post processing>" << endl;
	//--------------<visualization>--------------------------
	// std::cout << "<visualization>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects.at(i);
		for (int j = 0; j < oi.cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = oi.cluster_points.at(j).x;
			pt1.y = oi.cluster_points.at(j).y;

			pt2.x = oi.cluster_points.at(j).x + resolution / reduseres;
			pt2.y = oi.cluster_points.at(j).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		if (oi.det_mc) // visualization of the classifier
		{
			pt1.x = oi.model_center.x - oi.size.width / 2;
			pt1.y = oi.model_center.y - oi.size.height / 2;

			pt2.x = oi.model_center.x + oi.size.width / 2;
			pt2.y = oi.model_center.y + oi.size.height / 2;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		for (int j = 0; j < oi.track_points.size(); j++)
			cv::circle(imag, oi.track_points.at(j), 1, objColor(oi.id), 2);
	}
	// std::cout << "</visualization>" << endl;
	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------
	Mat baseimag(resolution, resolution + extr, CV_8UC3, Scalar(0, 0, 0));
	// std::cout << "<baseimag 1>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects.at(i);
		string text = objClassTitle(oi.type) + to_string(oi.id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + oi.img.cols) * oi.id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								objColor(oi.id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + oi.img.cols + 1;
		pt2.y = ptext.y + oi.img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, objColor(oi.id), 1);
			oi.img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, oi.img.cols, oi.img.rows)));
		}
	}
	// std::cout << "<baseimag 2>" << endl;
	if(baseimag.type() != imag.type())
		for(auto* fr: {&baseimag, &imag})
			if(fr->channels() == 1) {
				cv::cvtColor(*fr, *fr, cv::COLOR_GRAY2BGR);
				break;
			}
	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(frameId), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------

	imshow("Tracking", baseimag);
	cv::waitKey(10);  // Required to visualize window

	return baseimag;
}

vector<std::pair<Point2f, uint16_t>> trackingMotV2_1_artemis(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	const bool init = frame.data == frame0.data;
	if(!pathmodel.empty()) {
		frame0 = frame_resizing(frame0);
		frame = init ? frame0 : frame_resizing(frame);
	}

	Mat framebuf = frame;

	uint16_t rows = frame.rows;
	uint16_t cols = frame.cols;

	frame_resolution = rows;

	float koef = (float)rows / (float)model_resolution;

	// TODO: eliminate hardcoded values like in trackingMotV2_1
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
	if (!pathmodel.empty())
	{
		detects = detectorV4(pathmodel, frame, device_type, confidence);

		for (uint16_t i = 0; i < objects.size(); i++)
			objects[i].det_mc = false;

		// for (uint16_t i = 0; i < detects.size(); i++)
		// {
		// 	if (ObjClass(detects[i].type) != ObjClass::ANT)
		// 		detects.erase(detects.begin() + i--);
		// }
		//
		// Remove non-ant objects
		float objLinSz = init ? 0 : objhsz;
		unsigned  ndobjs = 0;
		for (int i = 0; i < detects.size(); i++)
		{
			if (ObjClass(detects.at(i).type) != ObjClass::ANT) {  // Note: "ta" also should be tracked
				detects.erase(detects.begin() + i--);
				++ndobjs;
			}
			else if(objLinSz) {
				// Update motion-related parameters
				objLinSz = (objLinSz * detects.size() + std::max(detects.at(i).size.width, detects.at(i).size.height)) / (detects.size() + 1);
			} else {
				// Init motion-related parameters
				objLinSz = (objLinSz * (i + ndobjs) + std::max(detects.at(i).size.width, detects.at(i).size.height)) / (i + 1 + ndobjs);
			}
		}
		if(objLinSz < antLenMin)
			objLinSz = antLenMin;
		objhsz = objLinSz / 2 + 1;
		reduseres = evalReduseres(objhsz);

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			const auto& det = detects[i];
			vector<Point2f> cluster_points;
			cluster_points.push_back(det.center);
			// imagbuf = frame_resizing(framebuf);
			// img = imagbuf(cv::Range(det.center.y - objhsz * koef, det.center.y + objhsz * koef), cv::Range(det.center.x - objhsz * koef, det.center.x + objhsz * koef));

			img = framebuf(cv::Range(max_u16(det.center.y - objhsz * koef)
				, min_u16(framebuf.rows, det.center.y + objhsz * koef))
				, cv::Range(max_u16(det.center.x - objhsz * koef)
				, min_u16(framebuf.cols, det.center.x + objhsz * koef)));

			//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			//- img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), det.type, det.size, det.center, cluster_points, img, det.prob);
			// obj.track_points.push_back(det.center);
			// obj.push_track_point(det.center);

			float rm = rcobj * (float)rows / (float)reduseres;
			bool newobj = true;
			uint16_t iobj;

			if (objects.size() > 0)
			{
				rm = distance(objects.front().cluster_center, obj.cluster_center);
				// rm = distance(objects.front().proposed_center(), obj.cluster_center)
				if (rm < rcobj * (float)rows / (float)reduseres)
				{
					iobj = 0;
					newobj = false;
				}
			}

			for (uint16_t j = 1; j < objects.size(); j++)
			{
				float r = distance(objects[j].cluster_center, obj.cluster_center);
				// float r = distance(objects[j].proposed_center(), obj.cluster_center);
				if (r < rcobj * (float)rows / (float)reduseres && r < rm)
				{
					rm = r;
					iobj = j;
					newobj = false;
				}
			}

			if (newobj == false)
			{
				auto& tobj = objects.at(iobj);
				tobj.model_center = obj.model_center;
				tobj.size = obj.size;
				tobj.img = obj.img;
				tobj.det_mc = true;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < frameId && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
					, obj.size.width, obj.size.height, obj.prob});
			}
			else
			{

				for (size_t j = 0; j < objects.size(); j++)
				{
					if (!objects[j].det_pos)
						continue;

					if (distance(objects[j].cluster_center, obj.cluster_center) < (float)robj * 2.3 * (float)rows / (float)reduseres)
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

	//- cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
	//- imag.convertTo(imag, CV_8UC3);

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
	if(imageBGR0.channels() != 1)
		cv::cvtColor(imageBGR0, imageBGR0, cv::COLOR_BGR2GRAY);
	// imageBGR0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	cv::Rect rect(0, 0, reduseres, reduseres);
	imageBGR = frame(rect);
	if(imageBGR.channels() != 1)
		cv::cvtColor(imageBGR, imageBGR, cv::COLOR_BGR2GRAY);
	// imageBGR.convertTo(imageBGR, CV_8UC1);

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
	uint16_t nobj = objects.empty() ? -1 : 0;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj != static_cast<decltype(nobj)>(-1) && nobj < objects.size())
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
		auto& cn = clusters.at(ncls);
		cn.push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			if (distance(pc, motion.at(i)) < nd * (float)rows / (float)reduseres)
			{
				if (distance(cluster_center(cn), motion.at(i)) < (float)mdist * (float)rows / (float)reduseres)
				{
					cn.push_back(motion.at(i));
					motion.erase(motion.begin() + i--);
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < cn.size(); c++)
			{
				pc = cn.at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					if (distance(pc, motion.at(i)) < nd * (float)rows / (float)reduseres)
					{
						if (distance(cluster_center(cn), motion.at(i)) < (float)mdist * (float)rows / (float)reduseres)
						{
							cn.push_back(motion.at(i));
							motion.erase(motion.begin() + i--);
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
				// clsobjr.r = distance(objects[i].cluster_center, clustercenter);
				clsobjr.r = distance(objects[i].proposed_center(), clustercenter);
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (!pathmodel.empty())
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				size_t obj_id = clsobjrs.at(i).obj_id;

				if (objects.at(obj_id).det_mc)
				{
					const auto& cl = clusters.at(cls_id);
					Point2f clustercenter = cluster_center(cl);
					auto& cobj = objects[obj_id];
					pt1.x = cobj.model_center.x - cobj.size.width / 2;
					pt1.y = cobj.model_center.y - cobj.size.height / 2;

					pt2.x = cobj.model_center.x + cobj.size.width / 2;
					pt2.y = cobj.model_center.y + cobj.size.height / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{

						if (!cobj.det_pos)
							cobj.cluster_points = cl;
						else
						{
							for (size_t j = 0; j < cl.size(); j++)
								cobj.cluster_points.push_back(cl[j]);
						}

						cobj.center_determine(frameId, false);

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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * (float)rows / (float)reduseres && cl.size() > mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (!cobj.det_pos)
					cobj.cluster_points = cl;
				else
				{
					continue;
					// for (size_t j = 0; j < cl.size(); j++)
					//   cobj.cluster_points.push_back(cl.at(j));
				}

				cobj.center_determine(frameId, false);

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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			Point2f clustercenter = cluster_center(cl);
			bool newobj = true;

			for (size_t j = 0; j < objects.size(); j++)
			{
				if (distance(objects[j].cluster_center, clustercenter) < (float)robj * 2.3 * (float)rows / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (cl.size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);

				// framebuf.convertTo(imagbuf, CV_8UC3);
				// framebuf.copyTo(imagbuf);
				imagbuf = framebuf;

				img = imagbuf(cv::Range(max_u16(clustercenter.y - objhsz * koef)
					, min_u16(imagbuf.rows, clustercenter.y + objhsz * koef))
					, cv::Range(max_u16(clustercenter.x - objhsz * koef)
					, min_u16(imagbuf.cols, clustercenter.x + objhsz * koef)));
				//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				//- img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), ObjClass::ANT, cl, img);
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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * robj_k * (float)rows / (float)reduseres && cl.size() > mpct / 2)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < cl.size(); j++)
					cobj.cluster_points.push_back(cl.at(j));

				cobj.center_determine(frameId, false);

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
			Point2f clustercenter = cluster_center(clusters.at(cls));
			bool newobj = true;

			for (int i = 0; i < objects.size(); i++)
			{
				if (distance(objects[i].cluster_center, clustercenter) < (float)robj * (float)rows / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters.at(cls).size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// framebuf.convertTo(imagbuf, CV_8UC3);
				// framebuf.copyTo(imagbuf);
				imagbuf = framebuf;
				img = imagbuf(cv::Range(max_u16(clustercenter.y - objhsz * koef)
					, min_u16(imagbuf.rows, clustercenter.y + objhsz * koef))
					, cv::Range(max_u16(clustercenter.x - objhsz * koef)
					, min_u16(imagbuf.cols, clustercenter.x + objhsz * koef)));
				//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				//- img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), ObjClass::ANT, clusters.at(cls), img);
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
		auto& oi = objects[i];
		if (!oi.det_mc && !oi.det_pos)
			continue;

		// imagbuf = frame_resizing(framebuf);
		// imagbuf.convertTo(imagbuf, CV_8UC3);
		// framebuf.convertTo(imagbuf, CV_8UC3);
		// framebuf.copyTo(imagbuf);
		imagbuf = framebuf;

		if (!oi.det_mc)
		{
			Size2f  sz(oi.size);
			if (oi.size.width <= 0)
				sz = Size2f(objhsz * 2, objhsz * 2);

			pt1.y = oi.cluster_center.y - sz.height / 2 * koef;
			pt2.y = oi.cluster_center.y + sz.height / 2 * koef;

			pt1.x = oi.cluster_center.x - sz.width / 2 * koef;
			pt2.x = oi.cluster_center.x + sz.width / 2 * koef;
		}
		else
		{
			pt1.y = oi.model_center.y - oi.size.height / 2 * koef;
			pt2.y = oi.model_center.y + oi.size.height / 2 * koef;

			pt1.x = oi.model_center.x - oi.size.width / 2 * koef;
			pt2.x = oi.model_center.x + oi.size.width / 2 * koef;
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
		//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		//- img.convertTo(img, CV_8UC3);
		img.copyTo(oi.img);
		oi.center_determine(frameId, true);
		oi.push_track_point(oi.det_mc ? oi.model_center : oi.cluster_center);
	}

	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects[i];
		for (int j = 0; j < oi.cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = oi.cluster_points.at(j).x;
			pt1.y = oi.cluster_points.at(j).y;

			pt2.x = oi.cluster_points.at(j).x + (float)rows / (float)reduseres;
			pt2.y = oi.cluster_points.at(j).y + (float)rows / (float)reduseres;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		if (oi.det_mc) // visualization of the classifier
		{
			pt1.x = oi.model_center.x - oi.size.width / 2;
			pt1.y = oi.model_center.y - oi.size.height / 2;

			pt2.x = oi.model_center.x + oi.size.width / 2;
			pt2.y = oi.model_center.y + oi.size.height / 2;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		for (int j = 0; j < oi.track_points.size(); j++)
			cv::circle(imag, oi.track_points.at(j), 1, objColor(oi.id), 2);
	}
	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------
	Mat baseimag(rows, rows + extr * koef, CV_8UC3, Scalar(0, 0, 0));
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects[i];
		string text = objClassTitle(oi.type) + to_string(oi.id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + oi.img.cols) * oi.id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								objColor(oi.id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + oi.img.cols + 1;
		pt2.y = ptext.y + oi.img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, objColor(oi.id), 1);
			oi.img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, oi.img.cols, oi.img.rows)));
		}
	}
	if(baseimag.type() != imag.type())
		for(auto* fr: {&baseimag, &imag})
			if(fr->channels() == 1) {
				cv::cvtColor(*fr, *fr, cv::COLOR_GRAY2BGR);
				break;
			}
	imag.copyTo(baseimag(cv::Rect(extr * koef, 0, imag.cols, imag.rows)));

	Point2f p_idframe;
	p_idframe.x = rows + (extr - 95) * koef;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(frameId), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	cv::resize(baseimag, baseimag, cv::Size(992 + extr, 992), cv::InterpolationFlags::INTER_CUBIC);
	imshow("Tracking", baseimag);
	cv::waitKey(0);
	//--------------</baseimag>-------------------------------*/

	vector<std::pair<Point2f, uint16_t>> detects_P2f_id;

	for (int i = 0; i < objects.size(); i++) {
		const auto& oi = objects[i];
		detects_P2f_id.push_back(oi.det_mc ? std::make_pair(oi.model_center, oi.id) : std::make_pair(oi.cluster_center, oi.id));
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
	// cv::waitKey(10);  // Required to visualize window

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

Mat trackingMotV2_2(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	// if (!pathmodel.empty())
	//   frame = frame_resizing(frame);

	const bool init = frame.data == frame0.data;
	if(!pathmodel.empty()) {
		frame0 = frame_resizing(frame0);
		frame = init ? frame0 : frame_resizing(frame);
	}
	Mat framebuf;

	frame.copyTo(framebuf);

	// TODO: eliminate hardcoded values like in trackingMotV2_1
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
	if (!pathmodel.empty())
	{

		detects = detectorV4(pathmodel, frame, device_type, confidence);

		for (int i = 0; i < objects.size(); i++)
			objects[i].det_mc = false;

		for (int i = 0; i < detects.size(); i++)
		{
			if (ObjClass(detects[i].type) != ObjClass::ANT)
				detects.erase(detects.begin() + i--);
		}

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			vector<Point2f> cluster_points;
			const auto& det = detects[i];
			cluster_points.push_back(det.center);
			// imagbuf = frame_resizing(framebuf);
			// img = imagbuf(cv::Range(det.center.y - objhsz, det.center.y + objhsz), cv::Range(det.center.x - objhsz, det.center.x + objhsz));

			img = framebuf(cv::Range(max_u16(det.center.y - objhsz)
				, min_u16(framebuf.rows, det.center.y + objhsz))
				, cv::Range(max_u16(det.center.x - objhsz)
				, min_u16(framebuf.cols, det.center.x + objhsz)));

			//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			//- img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), det.type, det.size, det.center, cluster_points, img, det.prob);
			// obj.track_points.push_back(det.center);
			// obj.push_track_point(det.center);

			float rm = rcobj * (float)resolution / (float)reduseres;
			bool newobj = true;
			uint16_t iobj;

			if (objects.size() > 0)
			{
				rm = distance(objects.front().cluster_center, obj.cluster_center);
				// rm = distance(objects.front().proposed_center(), obj.cluster_center);
				if (rm < rcobj * (float)resolution / (float)reduseres)
				{
					iobj = 0;
					newobj = false;
				}
			}

			for (uint16_t j = 1; j < objects.size(); j++)
			{
				float r = distance(objects[j].cluster_center, obj.cluster_center);
				// float r = distance(objects[j].proposed_center(), obj.cluster_center);;
				if (r < rcobj * (float)resolution / (float)reduseres && r < rm)
				{
					rm = r;
					iobj = j;
					newobj = false;
				}
			}

			if (!newobj)
			{
				auto& tobj = objects.at(iobj);
				tobj.model_center = obj.model_center;
				tobj.size = obj.size;
				tobj.img = obj.img;
				tobj.det_mc = true;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < frameId && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
					, obj.size.width, obj.size.height, obj.prob});
			}
			else
			{

				for (size_t j = 0; j < objects.size(); j++)
				{
					if (!objects[j].det_pos)
						continue;

					if (distance(objects[j].cluster_center, obj.cluster_center) < (float)robj * 2.3 * (float)resolution / (float)reduseres)
					{
						newobj = false;
						break;
					}
				}

				if (newobj)
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
				orbobjr.r = distance(objects[i].cluster_center, points1ORB[orb]);
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

	//- cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
	//- imag.convertTo(imag, CV_8UC3);

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

	if(frame0.channels() != 1)
		cv::cvtColor(frame0, imageBGR0, cv::COLOR_BGR2GRAY);
	// frame0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	// cv::Rect rect(0, 0, reduseres, reduseres);
	// imageBGR = frame(rect);

	if(frame.channels() != 1)
		cv::cvtColor(frame, imageBGR, cv::COLOR_BGR2GRAY);
	// frame.convertTo(imageBGR, CV_8UC1);

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
	uint16_t nobj = objects.empty() ? -1 : 0;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj != static_cast<decltype(nobj)>(-1) && nobj < objects.size())
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
		auto& cn = clusters.at(ncls);
		cn.push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			if (distance(pc, motion.at(i)) < (float)nd * (float)resolution / (float)reduseres)
			{
				if (distance(cluster_center(cn), motion[i]) < (float)mdist * (float)resolution / (float)reduseres)
				{
					cn.push_back(motion[i]);
					motion.erase(motion.begin() + i);
					i--;
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < cn.size(); c++)
			{
				pc = cn.at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					if (distance(pc, motion.at(i)) < (float)nd * (float)resolution / (float)reduseres)
					{
						if (distance(cluster_center(cn), motion[i]) < mdist * 1.0 * resolution / reduseres)
						{
							cn.push_back(motion[i]);
							motion.erase(motion.begin() + i--);
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
				// clsobjr.r = distance(objects[i].cluster_center, clustercenter);
				clsobjr.r = distance(objects[i].proposed_center(), clustercenter);
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (!pathmodel.empty())
		{
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs.at(i).cls_id;
				const auto& cl = clusters.at(cls_id);
				size_t obj_id = clsobjrs.at(i).obj_id;

				if (objects.at(obj_id).det_mc)
				{
					Point2f clustercenter = cluster_center(cl);
					auto& cobj = objects.at(obj_id);
					pt1.x = cobj.model_center.x - cobj.size.width / 2;
					pt1.y = cobj.model_center.y - cobj.size.height / 2;

					pt2.x = cobj.model_center.x + cobj.size.width / 2;
					pt2.y = cobj.model_center.y + cobj.size.height / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{
						if (!cobj.det_pos)
							cobj.cluster_points = cl;
						else
						{
							for (size_t j = 0; j < cl.size(); j++)
								cobj.cluster_points.push_back(cl.at(j));
						}

						cobj.center_determine(frameId, false);

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
					orbobjr.r = distance(clustercenter, points2ORB[orb]);
					orbobjrs.push_back(orbobjr);
				}
			}

			sort(orbobjrs.begin(), orbobjrs.end(), compare_clsobj);

			for (size_t i = 0; i < orbobjrs.size(); i++)
			{
				size_t orb_id = orbobjrs[i].cls_id; // cls_id - means orb_id!!!
				size_t cls_id = orbobjrs[i].obj_id; // obj_id - means cls_id!!!
				const auto& cl = clusters.at(cls_id);

				if (orbobjrs[i].r < (float)rORB * (float)resolution / (float)reduseres && cl.size() > mpct)
				{
					size_t obj_id;
					bool orb_in_obj = false;
					for (size_t j = 0; j < objects.size(); j++)
					{
						const auto& oj = objects[j];
						for (size_t o = 0; o < oj.ORB_ids.size(); o++)
						{
							if (oj.ORB_ids[o] == orb_id)
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
						if (!cobj.det_pos)
							cobj.cluster_points = cl;
						else
						{
							for (size_t j = 0; j < cl.size(); j++)
								cobj.cluster_points.push_back(cl.at(j));
						}

						cobj.center_determine(frameId, false);

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
			size_t cls_id = clsobjrs[i].cls_id;
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs[i].obj_id;

			if (clsobjrs[i].r < (float)robj * (float)resolution / (float)reduseres && cl.size() > mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (!cobj.det_pos)
					cobj.cluster_points = cl;
				else
				{
					continue;
					// for (size_t j = 0; j < cl.size(); j++)
					//   cobj.cluster_points.push_back(cl.at(j));
				}

				cobj.center_determine(frameId, false);

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
			size_t cls_id = clsobjrs[i].cls_id;
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs[i].obj_id;

			Point2f clustercenter = cluster_center(cl);
			bool newobj = true;

			for (size_t j = 0; j < objects.size(); j++)
			{
				if (distance(objects[j].cluster_center, clustercenter) < (float)robj * 2.3 * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (cl.size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - objhsz, clustercenter.y + objhsz), cv::Range(clustercenter.x - objhsz, clustercenter.x + objhsz));

				img = framebuf(cv::Range(max_u16(clustercenter.y - objhsz)
					, min_u16(framebuf.rows, clustercenter.y + objhsz))
					, cv::Range(max_u16(clustercenter.x - objhsz)
					, min_u16(framebuf.cols, clustercenter.x + objhsz)));

				//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				//- img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), ObjClass::ANT, cl, img);
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
			size_t cls_id = clsobjrs[i].cls_id;
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs[i].obj_id;

			if (clsobjrs[i].r < (float)robj * robj_k * (float)resolution / (float)reduseres && cl.size() > mpct / 2)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < cl.size(); j++)
					cobj.cluster_points.push_back(cl.at(j));

				cobj.center_determine(frameId, false);

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
				if (distance(objects[i].cluster_center, clustercenter) < (float)robj * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// magbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - objhsz, clustercenter.y + objhsz), cv::Range(clustercenter.x - objhsz, clustercenter.x + objhsz));

				img = framebuf(cv::Range(max_u16(clustercenter.y - objhsz)
					, min_u16(framebuf.rows, clustercenter.y + objhsz))
					, cv::Range(max_u16(clustercenter.x - objhsz)
					, min_u16(framebuf.cols, clustercenter.x + objhsz)));

				//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				//- img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), ObjClass::ANT, clusters[cls], img);
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
		auto& oi = objects[i];
		if (!oi.det_mc && !oi.det_pos)
			continue;

		// imagbuf = frame_resizing(framebuf);
		// imagbuf.convertTo(imagbuf, CV_8UC3);
		// imagbuf = frame_resizing(framebuf);
		// framebuf.convertTo(imagbuf, CV_8UC3);
		// framebuf.copyTo(imagbuf);
		imagbuf = framebuf;

		if (!oi.det_mc)
		{
			Size2f  sz(oi.size);
			if (oi.size.width <= 0)
				sz = Size2f(objhsz * 2, objhsz * 2);

			pt1.y = oi.cluster_center.y - sz.height / 2;
			pt2.y = oi.cluster_center.y + sz.height / 2;

			pt1.x = oi.cluster_center.x - sz.width / 2;
			pt2.x = oi.cluster_center.x + sz.width / 2;
		}
		else
		{
			pt1.y = oi.model_center.y - oi.size.height / 2;
			pt2.y = oi.model_center.y + oi.size.height / 2;

			pt1.x = oi.model_center.x - oi.size.width / 2;
			pt2.x = oi.model_center.x + oi.size.width / 2;
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
		//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		//- img.convertTo(img, CV_8UC3);
		img.copyTo(oi.img);
		oi.center_determine(frameId, true);
		oi.push_track_point(oi.det_mc ? oi.model_center : oi.cluster_center);
	}
	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects[i];
		for (int j = 0; j < oi.cluster_points.size(); j++) // visualization of the cluster_points
		{
			const auto& cp = oi.cluster_points[j];
			pt1.x = cp.x;
			pt1.y = cp.y;

			pt2.x = cp.x + resolution / reduseres;
			pt2.y = cp.y + resolution / reduseres;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		if (oi.det_mc) // visualization of the classifier
		{
			pt1.x = oi.model_center.x - oi.size.width / 2;
			pt1.y = oi.model_center.y - oi.size.height / 2;

			pt2.x = oi.model_center.x + oi.size.width / 2;
			pt2.y = oi.model_center.y + oi.size.height / 2;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		for (int j = 0; j < oi.track_points.size(); j++)
			cv::circle(imag, oi.track_points.at(j), 1, objColor(oi.id), 2);

		for (int j = 0; j < oi.ORB_ids.size(); j++)
			cv::circle(imag, points2ORB.at(oi.ORB_ids[j]), 3, objColor(oi.id), 1);
	}

	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------

	Mat baseimag(resolution, 3 * resolution + extr, CV_8UC3, Scalar(0, 0, 0));
	// std::cout << "<baseimag 1>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects[i];
		string text = objClassTitle(oi.type) + to_string(oi.id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + oi.img.cols) * oi.id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								objColor(oi.id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + oi.img.cols + 1;
		pt2.y = ptext.y + oi.img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, objColor(oi.id), 1);
			oi.img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, oi.img.cols, oi.img.rows)));
		}
	}
	// std::cout << "<baseimag 2>" << endl;
	Mat& imgOrb = std::get<2>(detectsORB);
	if(baseimag.type() != imag.type())
		for(auto* fr: {&baseimag, &imag, &imgOrb})
			if(fr->channels() == 1) {
				cv::cvtColor(*fr, *fr, cv::COLOR_GRAY2BGR);
				break;
			}
	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

	cv::resize(imgOrb, imgOrb, cv::Size(2 * resolution, resolution), cv::InterpolationFlags::INTER_CUBIC);

	imgOrb.copyTo(baseimag(cv::Rect(resolution + extr, 0, imgOrb.cols, imgOrb.rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(frameId), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------

	imshow("Tracking", baseimag);
	cv::waitKey(0);

	return baseimag;
}

Mat trackingMotV2_3(const string& pathmodel, torch::DeviceType device_type, Mat frame0, Mat frame, vector<ALObject> &objects, size_t frameId, float confidence)
{
	vector<vector<Point2f>> clusters;
	vector<Point2f> motion;
	vector<Mat> imgs;

	Mat imageBGR0;
	Mat imageBGR;

	Mat imag;
	Mat imagbuf;

	// if (!pathmodel.empty())
	//   frame = frame_resizing(frame);

	const bool init = frame.data == frame0.data;
	if(!pathmodel.empty()) {
		frame0 = frame_resizing(frame0);
		frame = init ? frame0 : frame_resizing(frame);
	}

	Mat framebuf;

	frame.copyTo(framebuf);

	// TODO: eliminate hardcoded values like in trackingMotV2_1
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
	if (!pathmodel.empty())
	{

		detects = detectorV4(pathmodel, frame, device_type, confidence);

		for (int i = 0; i < objects.size(); i++)
			objects[i].det_mc = false;

		for (int i = 0; i < detects.size(); i++)
		{
			if (ObjClass(detects[i].type) != ObjClass::ANT)
				detects.erase(detects.begin() + i--);
		}

		for (uint16_t i = 0; i < detects.size(); i++)
		{
			vector<Point2f> cluster_points;
			const auto& det = detects[i];
			cluster_points.push_back(det.center);
			// imagbuf = frame_resizing(framebuf);
			// img = imagbuf(cv::Range(det.center.y - objhsz, det.center.y + objhsz), cv::Range(det.center.x - objhsz, det.center.x + objhsz));

			img = framebuf(cv::Range(max_u16(det.center.y - objhsz)
				, min_u16(framebuf.rows, det.center.y + objhsz))
				, cv::Range(max_u16(det.center.x - objhsz)
				, min_u16(framebuf.cols, det.center.x + objhsz)));

			//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
			//- img.convertTo(img, CV_8UC3);

			ALObject obj(objects.size(), det.type, det.size, det.center, cluster_points, img, det.prob);
			// obj.track_points.push_back(det.center);
			// obj.push_track_point(det.center);

			float rm = rcobj * (float)resolution / (float)reduseres;
			bool newobj = true;
			uint16_t iobj;

			if (!objects.empty())
			{
				rm = distance(objects.front().cluster_center, obj.cluster_center);
				// rm = distance(objects.front().proposed_center(), obj.cluster_center);
				if (rm < rcobj * (float)resolution / (float)reduseres)
				{
					iobj = 0;
					newobj = false;
				}
			}

			for (uint16_t j = 1; j < objects.size(); j++)
			{
				const float r = distance(objects[j].cluster_center, obj.cluster_center);
				// const float r = distance(objects[j].proposed_center(), obj.cluster_center);
				if (r < rcobj * (float)resolution / (float)reduseres && r < rm)
				{
					rm = r;
					iobj = j;
					newobj = false;
				}
			}

			if (!newobj)
			{
				auto& tobj = objects.at(iobj);
				tobj.model_center = obj.model_center;
				tobj.size = obj.size;
				tobj.img = obj.img;
				tobj.det_mc = true;
				// assert(!tobj.traces.empty() && tobj.traces.back().frame < frameId && "Unexpected frame number in the traces");
				tobj.traces.push_back(Trace{frameId, obj.cluster_center.x, obj.cluster_center.y
					, obj.size.width, obj.size.height, obj.prob});
			}
			else
			{

				for (size_t j = 0; j < objects.size(); j++)
				{
					if (!objects[j].det_pos)
						continue;

					if (distance(objects[j].cluster_center, obj.cluster_center) < (float)robj * 2.3 * (float)resolution / (float)reduseres)
					{
						newobj = false;
						break;
					}
				}

				if (newobj)
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
			auto& oi = objects[i];
			for (size_t j = 0; j < oi.cluster_points.size(); j++)
			{
				for (size_t orb = 0; orb < points1ORB.size(); orb++)
					if (distance(oi.cluster_points[j], points1ORB.at(orb)) < (float)rORB * (float)resolution / (float)reduseres)
						oi.ORB_ids.push_back(orb);
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

	//- cv::cvtColor(imag, imag, cv::COLOR_BGR2RGB);
	//- imag.convertTo(imag, CV_8UC3);

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

	if(frame0.channels() != 1)
		cv::cvtColor(frame0, imageBGR0, cv::COLOR_BGR2GRAY);
	//frame0.convertTo(imageBGR0, CV_8UC1);

	cv::resize(frame, frame, cv::Size(clsize, rwsize), cv::InterpolationFlags::INTER_CUBIC);

	// cv::Rect rect(0, 0, reduseres, reduseres);
	// imageBGR = frame(rect);

	if(frame.channels() != 1)
		cv::cvtColor(frame, imageBGR, cv::COLOR_BGR2GRAY);
	// frame.convertTo(imageBGR, CV_8UC1);

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
			pt1.x = motion[i].x;
			pt1.y = motion[i].y;

			pt2.x = motion[i].x + (float)resolution / (float)reduseres;
			pt2.y = motion[i].y + (float)resolution / (float)reduseres;

			rectangle(imag, pt1, pt2, Scalar(255, 255, 255), 1);
		}

	uint16_t ncls = 0;
	uint16_t nobj = objects.empty() ? -1 : 0;
	//--------------</moution detections>--------------------

	//--------------<cluster creation>-----------------------

	while (motion.size() > 0)
	{
		Point2f pc;

		if (nobj != static_cast<decltype(nobj)>(-1) && nobj < objects.size())
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
		auto& cn = clusters.at(ncls);
		cn.push_back(pc);

		for (int i = 0; i < motion.size(); i++)
		{
			if (distance(pc, motion[i]) < (float)nd * (float)resolution / (float)reduseres)
			{
				if (distance(cluster_center(cn), motion[i]) < (float)mdist * (float)resolution / (float)reduseres)
				{
					cn.push_back(motion.at(i));
					motion.erase(motion.begin() + i--);
				}
			}
		}

		uint16_t newp;
		do
		{
			newp = 0;

			for (uint16_t c = 0; c < cn.size(); c++)
			{
				pc = cn.at(c);
				for (int i = 0; i < motion.size(); i++)
				{
					if (distance(pc, motion[i]) < (float)nd * (float)resolution / (float)reduseres)
					{
						if (distance(cluster_center(cn), motion[i]) < mdist * 1.0 * resolution / reduseres)
						{
							cn.push_back(motion.at(i));
							motion.erase(motion.begin() + i--);
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
				clsobjr.r = distance(objects[i].proposed_center(), clustercenter);  // objects[i].cluster_center
				clsobjrs.push_back(clsobjr);
			}
		}

		sort(clsobjrs.begin(), clsobjrs.end(), compare_clsobj);

		//--<corr obj use model>---
		if (!pathmodel.empty())
		{
			for (size_t i = 0; i < clsobjrs.size(); i++)
			{
				size_t cls_id = clsobjrs[i].cls_id;
				const auto& cl = clusters.at(cls_id);
				size_t obj_id = clsobjrs[i].obj_id;

				if (objects.at(obj_id).det_mc)
				{
					Point2f clustercenter = cluster_center(cl);
					auto& cobj = objects[obj_id];
					pt1.x = cobj.model_center.x - cobj.size.width / 2;
					pt1.y = cobj.model_center.y - cobj.size.height / 2;

					pt2.x = cobj.model_center.x + cobj.size.width / 2;
					pt2.y = cobj.model_center.y + cobj.size.height / 2;

					if (pt1.x < clustercenter.x && clustercenter.x < pt2.x && pt1.y < clustercenter.y && clustercenter.y < pt2.y)
					{
						if (!cobj.det_pos)
							cobj.cluster_points = cl;
						else
						{
							for (size_t j = 0; j < cl.size(); j++)
								cobj.cluster_points.push_back(cl.at(j));
						}

						cobj.center_determine(frameId, false);

						for (size_t j = 0; j < clsobjrs.size(); j++)
						{
							if (clsobjrs[j].cls_id == cls_id)
								clsobjrs.erase(clsobjrs.begin() + j--);
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
						if (distance(clusters[i][j], points2ORB.at(orb)) < (float)rORB * (float)resolution / (float)reduseres)
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
				if (orb_in_obj)
				{
					auto& cobj = objects.at(obj_id);
					const auto& cl = clusters.at(cls_id);
					if (!cobj.det_pos)
						cobj.cluster_points = cl;
					else
					{
						for (size_t j = 0; j < cl.size(); j++)
							cobj.cluster_points.push_back(cl.at(j));
					}

					cobj.center_determine(frameId, false);

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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * (float)resolution / (float)reduseres && cl.size() > mpct)
			{
				auto& cobj = objects.at(obj_id);
				if (!cobj.det_pos)
					cobj.cluster_points = cl;
				else
				{
					continue;
					// for (size_t j = 0; j < cl.size(); j++)
					//   cobj.cluster_points.push_back(cl.at(j));
				}

				cobj.center_determine(frameId, false);

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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			Point2f clustercenter = cluster_center(cl);
			bool newobj = true;

			for (size_t j = 0; j < objects.size(); j++)
			{
				if (distance(objects[j].cluster_center, clustercenter) < (float)robj * 2.3 * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (cl.size() > mpcc && newobj == true) // if there are enough moving points
			{
				// imagbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - objhsz, clustercenter.y + objhsz), cv::Range(clustercenter.x - objhsz, clustercenter.x + objhsz));

				img = framebuf(cv::Range(max_u16(clustercenter.y - objhsz)
					, min_u16(framebuf.rows, clustercenter.y + objhsz))
					, cv::Range(max_u16(clustercenter.x - objhsz)
					, min_u16(framebuf.cols, clustercenter.x + objhsz)));

				//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				//- img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), ObjClass::ANT, cl, img);
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
			const auto& cl = clusters.at(cls_id);
			size_t obj_id = clsobjrs.at(i).obj_id;

			if (clsobjrs.at(i).r < (float)robj * robj_k * (float)resolution / (float)reduseres && cl.size() > mpct / 2)
			{
				auto& cobj = objects.at(obj_id);
				for (size_t j = 0; j < cl.size(); j++)
					cobj.cluster_points.push_back(cl.at(j));

				cobj.center_determine(frameId, false);

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
				if (distance(objects[i].cluster_center, clustercenter) < (float)robj * (float)resolution / (float)reduseres)
				{
					newobj = false;
					break;
				}
			}

			if (clusters[cls].size() > mpcc && newobj == true) // if there are enough moving points
			{
				// magbuf = frame_resizing(framebuf);
				// imagbuf.convertTo(imagbuf, CV_8UC3);
				// img = imagbuf(cv::Range(clustercenter.y - objhsz, clustercenter.y + objhsz), cv::Range(clustercenter.x - objhsz, clustercenter.x + objhsz));

				img = framebuf(cv::Range(max_u16(clustercenter.y - objhsz)
					, min_u16(framebuf.rows, clustercenter.y + objhsz))
					, cv::Range(max_u16(clustercenter.x - objhsz)
					, min_u16(framebuf.cols, clustercenter.x + objhsz)));

				//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
				//- img.convertTo(img, CV_8UC3);
				ALObject obj(objects.size(), ObjClass::ANT, clusters[cls], img);
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
		auto& oi = objects[i];
		if (!oi.det_mc && !oi.det_pos)
			continue;

		// imagbuf = frame_resizing(framebuf);
		// imagbuf.convertTo(imagbuf, CV_8UC3);

		// imagbuf = frame_resizing(framebuf);
		if(framebuf.channels() == 1)
			cv::cvtColor(framebuf, imagbuf, cv::COLOR_GRAY2BGR);
		else imagbuf = framebuf;
		// framebuf.convertTo(imagbuf, CV_8UC3);

		if (!oi.det_mc)
		{
			Size2f  sz(oi.size);
			if (oi.size.width <= 0)
				sz = Size2f(objhsz * 2, objhsz * 2);

			pt1.y = oi.cluster_center.y - sz.height / 2;
			pt2.y = oi.cluster_center.y + sz.height / 2;

			pt1.x = oi.cluster_center.x - sz.width / 2;
			pt2.x = oi.cluster_center.x + sz.width / 2;
		}
		else
		{
			pt1.y = oi.model_center.y - oi.size.height / 2;
			pt2.y = oi.model_center.y + oi.size.height / 2;

			pt1.x = oi.model_center.x - oi.size.width / 2;
			pt2.x = oi.model_center.x + oi.size.width / 2;
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
		//- cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		//- img.convertTo(img, CV_8UC3);
		img.copyTo(oi.img);
		oi.center_determine(frameId, true);
		oi.push_track_point(oi.det_mc ? oi.model_center : oi.cluster_center);
	}
	//--------------<visualization>--------------------------
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects[i];
		for (int j = 0; j < oi.cluster_points.size(); j++) // visualization of the cluster_points
		{
			pt1.x = oi.cluster_points.at(j).x;
			pt1.y = oi.cluster_points.at(j).y;

			pt2.x = oi.cluster_points.at(j).x + resolution / reduseres;
			pt2.y = oi.cluster_points.at(j).y + resolution / reduseres;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		if (oi.det_mc) // visualization of the classifier
		{
			pt1.x = oi.model_center.x - oi.size.width / 2;
			pt1.y = oi.model_center.y - oi.size.height / 2;

			pt2.x = oi.model_center.x + oi.size.width / 2;
			pt2.y = oi.model_center.y + oi.size.height / 2;

			rectangle(imag, pt1, pt2, objColor(oi.id), 1);
		}

		for (int j = 0; j < oi.track_points.size(); j++)
			cv::circle(imag, oi.track_points.at(j), 1, objColor(oi.id), 2);

		for (int j = 0; j < oi.ORB_ids.size(); j++)
			cv::circle(imag, points2ORB.at(oi.ORB_ids[j]), 3, objColor(oi.id), 1);
	}

	//--------------</visualization>-------------------------

	//--------------<baseimag>-------------------------------

	Mat baseimag(resolution, resolution + extr, CV_8UC3, Scalar(0, 0, 0));
	// std::cout << "<baseimag 1>" << endl;
	for (int i = 0; i < objects.size(); i++)
	{
		const auto& oi = objects[i];
		string text = objClassTitle(oi.type) + to_string(oi.id);

		Point2f ptext;
		ptext.x = 20;
		ptext.y = (30 + oi.img.cols) * oi.id + 20;

		cv::putText(baseimag, // target image
								text,     // text
								ptext,    // top-left position
								1,
								1,
								objColor(oi.id), // font color
								1);

		pt1.x = ptext.x - 1;
		pt1.y = ptext.y - 1 + 10;

		pt2.x = ptext.x + oi.img.cols + 1;
		pt2.y = ptext.y + oi.img.rows + 1 + 10;

		if (pt2.y < baseimag.rows && pt2.x < baseimag.cols)
		{
			rectangle(baseimag, pt1, pt2, objColor(oi.id), 1);
			oi.img.copyTo(baseimag(cv::Rect(pt1.x + 1, pt1.y + 1, oi.img.cols, oi.img.rows)));
		}
	}
	// std::cout << "<baseimag 2>" << endl;
	if(baseimag.type() != imag.type())
		for(auto* fr: {&baseimag, &imag})
			if(fr->channels() == 1) {
				cv::cvtColor(*fr, *fr, cv::COLOR_GRAY2BGR);
				break;
			}
	imag.copyTo(baseimag(cv::Rect(extr, 0, imag.cols, imag.rows)));

	//cv::resize(std::get<2>(detectsORB), std::get<2>(detectsORB), cv::Size(2 * resolution, resolution), cv::InterpolationFlags::INTER_CUBIC);

	//std::get<2>(detectsORB).copyTo(baseimag(cv::Rect(resolution + extr, 0, std::get<2>(detectsORB).cols, std::get<2>(detectsORB).rows)));

	Point2f p_idframe;
	p_idframe.x = resolution + extr - 95;
	p_idframe.y = 50;
	cv::putText(baseimag, to_string(frameId), p_idframe, 1, 3, Scalar(255, 255, 255), 2);
	// cv::cvtColor(baseimag, baseimag, cv::COLOR_BGR2RGB);
	//--------------</baseimag>-------------------------------

	imshow("Tracking", baseimag);
	cv::waitKey(10);  // Required to visualize window

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

	cout << std::fixed << std::setprecision(2);  // Fixed floating point with specified precision
	for(const auto& obj: objects) {
		// cout << endl << "obj: " << objClassTitle(obj.type) + to_string(obj.id) << endl;
		cout << objClassTitle(obj.type) + to_string(obj.id) << endl;
		std::fstream fout((odp / objClassTitle(obj.type)).string() + to_string(obj.id) + ".csv", std::ios_base::out);
		fout << "# FrameId ObjCenterX ObjCenterY ObjWidth ObjHeight Likelihood\n" << std::fixed << std::setprecision(2);
		for(const auto& tr: obj.traces) {
			fout << tr.frame << ' ' << tr.center.x << ' ' << tr.center.y << ' ' << tr.size.width << ' ' << tr.size.height << ' ' << tr.prob << endl;
			cout << "frame: " << tr.frame << ", center: " << tr.center << ", size: " << tr.size << ", likelihood: " << tr.prob << endl;
		}
	}
}