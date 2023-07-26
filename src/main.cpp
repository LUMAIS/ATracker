#include <iostream>
#include <filesystem>
#include <ctime>
#include <unistd.h>
#include <torch/script.h> // One-stop header.
#include "lib/tracker.h"
#include "lib/utils.h"
#include "autogen/cmdline.h"

using std::cout;
using std::to_string;
using std::endl;
using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::system_clock;
namespace fs = std::filesystem;

struct ArgParser: gengetopt_args_info {
	ArgParser(int argc, char **argv) {
		auto  err = cmdline_parser(argc, argv, this);
		if(err)
			throw std::invalid_argument("Arguments parsing failed: " + to_string(err));
	}

	~ArgParser() {
		cmdline_parser_free(this);
	}
};

// Rescale canvas of the frames
void rescaleCanvas(cv::Mat& frame, float scale)
{
	if(scale <= 0 || scale >= 1) {
		assert(scale > 0 && scale < 1 && "Unexpected size of the canvas scale parameter");
		return;
	}
	Mat tmp = cv::Mat::zeros(roundf(frame.rows/scale), roundf(frame.cols/scale), frame.type());
	frame.copyTo(tmp(cv::Rect(0, 0, frame.cols, frame.rows)));
	frame = tmp;
}

int main(int argc, char **argv)
{
	// testtorch();
	//./detector ../data/modules/traced_resnet_model.pt
	//./detector ../data/modules/AnTroD_resnet50.pt
	// testmodule(argv[1]);
	// 
	// // Show help
	// if (argc <= 2 || string(argv[1]) == string("-h")) {
	//     cout << "Usage: app [-h | (MODEL_PATH INPUT_DATA [COMP_DEV={CUDA, CPU}])]\n\n"
	//         << "-h  - show help,\n"
	//         << "MODEL_PATH  - path to the object detector (PyTorch ML model)\n"
	//         << "INPUT_DATA  - path to the input image or .mp4 video file\n"
	//         << "COMP_DEV=CPU  - computational device for the object detector {CUDA, CPU}\n";
	//     cout << "argc: " << argc << endl;
	//     return 0;
	// }
	// torch::DeviceType device=torch::kCPU, kCUDA

	ArgParser  args_info(argc, argv);

	torch::jit::script::Module module;
	vector<ALObject> objects;

	string pathmodel;
	if(args_info.model_given) {
		pathmodel = args_info.model_arg;  // argv[1];
		cout << "Object detector: " << pathmodel << endl;

		try
		{
			// Deserialize the ScriptModule from a file using torch::jit::load().
			module = torch::jit::load(pathmodel);
		}
		catch (const c10::Error &e)
		{
			std::cerr << "error loading the model\n";
			return -1;
		}
	} else {
		assert(args_info.ant_length_given && "ant_length should be provided when an ML-based detector is not applied");
		objhsz = args_info.ant_length_arg / 2 + 1;
	}

	torch::DeviceType device_type = torch::kCPU;

	//if (argc >= 4 && std::strstr(argv[3], "CUDA") != NULL)
	if (args_info.cuda_flag)
		device_type = torch::kCUDA;

	//if (std::strstr(argv[2], ".mp4") != NULL)
	if (args_info.video_given)
	{
		// const odir = fs::path(args_info.output_arg);
		// const string fname = fs::path(args_info.video_arg).stem();
		uint16_t start = args_info.frame_start_arg;  // 460
		uint16_t nfram = args_info.frame_num_arg;  // 30; -1

		vector<vector<Obj>> objs;
		vector<std::pair<uint, IdFix>> fixedIds;

		//---TEST---
		// vector<cv::Mat> d_images;
		// d_images = LoadVideo(args_info.video_arg, start, nfram);  // argv[2], start, nfram

		cv::VideoWriter writer;
		int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');  // 'm', 'p', '4', 'v';  'h', '2', '6', '4'

		// Ensure that output dir exists
		const fs::path  odp = args_info.output_arg;
		std::error_code  ec;
		if(!fs::exists(odp, ec)) {
			if(ec || !fs::create_directories(odp, ec))
				throw std::runtime_error("The output directory can't be created (" + to_string(ec.value()) + ": " + ec.message() + "): " + odp.string() + "\n");
		}

		const float confidence = args_info.confidence_arg;  // dftConf
		string filename = (fs::path(args_info.output_arg) / fs::path(args_info.video_arg).stem()).string()
			+ "_i" + to_string(start) + "-" + to_string(nfram) + (args_info.model_given
				? "_c" + to_string(confidence).substr(0, 4) : "_a" + to_string(args_info.ant_length_arg));  // + dateTime();
		if(args_info.rescale_arg < 1)
			filename += string("_r") + to_string(args_info.rescale_arg).substr(0, 4);
		if(args_info.fout_suffix_given)
			filename += string("_") + args_info.fout_suffix_arg;
// GIT_SRC_VERSION (a custom macro definition) might be defined by CMAKE or another build tool
#ifdef GIT_SRC_VERSION
		filename += string("_") + GIT_SRC_VERSION;
#endif // GIT_SRC_VERSION

		const double fps = 1.0;  // FPS of the forming video

		// Capture video in the streaming mode
		cv::VideoCapture cap(args_info.video_arg);
		if (!cap.isOpened()) {
			std::cout << "Cannot open the video file" << endl;
			return 1;
		}
		Mat framebuf;
		cap.set(cv::CAP_PROP_POS_FRAMES, start);

		vector<ALObject> objects;
		bool initOutp = true;
		const size_t  framesNum = cap.get(cv::CAP_PROP_FRAME_COUNT);
		cv::Mat framePrev, frame;  // = d_images.at(1);
		for (size_t ifr = 0; ifr < framesNum; ifr++) {
			if (ifr > nfram) {
				std::cout << "nfram:  " << nfram << endl;
				break;
			}

			if (!cap.read(framebuf)) {
				std::cout << "Failed to extract the frame " << ifr << endl;
			} else {
				// Mat frame;
				// cv::cvtColor(framebuf, frame, cv::COLOR_RGB2GRAY);
				// cv::cvtColor(framebuf, framebuf, cv::COLOR_RGB2GRAY);

				// if(framebuf.channels() == 1)
				// 	cv::cvtColor(framebuf, framebuf, cv::COLOR_GRAY2BGR);
				// // framebuf.convertTo(framebuf, CV_8UC3);
				if(framebuf.channels() > 1)
					cv::cvtColor(framebuf, framebuf, cv::COLOR_BGR2GRAY);

				// Ensure that the frame is square
				if(framebuf.rows != framebuf.cols) {
					uint16_t res = framebuf.cols;
					if (framebuf.rows > res)
						res = framebuf.rows;

					Mat frame(res, res, CV_8UC1, Scalar::all(0));
					if (framebuf.rows > framebuf.cols)
						framebuf.copyTo(frame(cv::Rect((framebuf.rows - framebuf.cols) / 2, 0, framebuf.cols, framebuf.rows)));
					else
						framebuf.copyTo(frame(cv::Rect(0, (framebuf.cols - framebuf.rows) / 2, framebuf.cols, framebuf.rows)));
					framebuf = frame;
				}
				
				// Frame processing ----------------
				framePrev = frame; frame = framebuf;
				if(framePrev.empty())
					framePrev = frame;

				// cv::Size sizeFrame(992+extr,992);
				// cv::Mat testimg = std::get<2>(detectORB(d_images.at(0), d_images.at(1), 2.5));

				cout << "[Frame: " << start + ifr << "]" << endl;
				auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
				// Initial call is required to initialize motion processing
				// trackingMotV2_1  - Detector + motion
				// trackingMotV2_2  - Interactive ORB descriptors for the whole frame
				// trackingMotV2_3  - Detector + ORB descriptors for the whole frame
				if(args_info.rescale_arg < 1) {
					rescaleCanvas(framePrev, args_info.rescale_arg);
					if(frame.data != framePrev.data)
						rescaleCanvas(frame, args_info.rescale_arg);
				}
				cv::Mat res = trackingMotV2_1(pathmodel, device_type, framePrev, frame, objects, start + ifr, confidence, initOutp ? filename: string());
				// cv::Mat testimg = trackingMotV2_3(pathmodel, device_type, framePrev, framePrev, objects, start, confidence);
				std::cout << "Tracking time: " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec
					// << "ms; " << "confidence threshold : " << confidence
					<< " ms" << endl;

				cv::Size sizeFrame(res.cols, res.rows);
				if(initOutp) {
					writer.open(filename + "_demo.mp4", codec, fps, sizeFrame, true);
					initOutp = false;
				}
				writer.write(res);
				// trackingMotV2_1_artemis(pathmodel, device_type, d_images.at(ifr), d_images.at(ifr + 1), objects, start + ifr, confidence);
				// writer.write(trackingMotV2_3(pathmodel, device_type, d_images.at(ifr), d_images.at(ifr + 1), objects, start + ifr, confidence));
				// writer.write(std::get<2>(detectORB(d_images.at(ifr), d_images.at(ifr + 1), 2.5)));
			}
		}

		// for (int i = 0; i < d_images.size() - 1; i++)
		// {
		// 	cout << "[Frame: " << start + i << "]" << endl;
		// 	millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		// 	framePrev = d_images.at(i);
		// 	frame = d_images.at(i + 1);
		// 	if(args_info.rescale_arg < 1) {
		// 		rescaleCanvas(framePrev, args_info.rescale_arg);
		// 		rescaleCanvas(frame, args_info.rescale_arg);
		// 	}
		// 	cv::Mat res = trackingMotV2_1(pathmodel, device_type, framePrev, frame, objects, start + i, confidence);
		// 	writer.write(res);
		// 	std::cout << "Tracking time: " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec
		// 		// << "ms; " << "confidence threshold : " << confidence
		// 		<< " ms" << endl;
		// 	// trackingMotV2_1_artemis(pathmodel, device_type, d_images.at(i), d_images.at(i + 1), objects, start + i, confidence);
		// 	// writer.write(trackingMotV2_3(pathmodel, device_type, d_images.at(i), d_images.at(i + 1), objects, start + i, confidence));
		// 	// writer.write(std::get<2>(detectORB(d_images.at(i), d_images.at(i + 1), 2.5)));
		// }

		writer.release();
		traceObjects(objects, filename);
		return 0;

		// Test Id fixing
		vector<cv::Mat> d_images;
		d_images = LoadVideo(argv[2], start, nfram);

		vector<OBJdetect> obj_detects;
		vector<Obj> objbuf;
		for (int i = 0; i < d_images.size(); i++)
		{
			obj_detects.clear();
			obj_detects = detectorV4(pathmodel, d_images.at(i), device_type);
			OBJdetectsToObjs(obj_detects, objbuf);
			objs.push_back(objbuf);
		}

		fixIds(objs, fixedIds, d_images);

		// // Pairwise application
		// fixIds(objs, fixedIds, frames[0], frames[0], 0);
		// for (int i = 0; i < frames.size() - 1; i++)
		// 	fixIds(objs, fixedIds, frames[i], frames[i + 1], i + 1);
	}
	else
	{
		cv::Mat img = cv::imread(args_info.img_arg);  // , cv::ImreadModes::IMREAD_COLOR
		detectorV4(pathmodel, img, device_type);
	}

	return 0;
}