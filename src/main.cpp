#include <iostream>
#include <filesystem>
#include <ctime>
#include <unistd.h>
#include <torch/script.h> // One-stop header.
#include "lib/tracker.hpp"
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
		objhsz = args_info.ant_length_arg;
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
		vector<cv::Mat> d_images;

		//---TEST---
		d_images = LoadVideo(args_info.video_arg, start, nfram);  // argv[2], start, nfram
		cv::VideoWriter writer;
		int codec = cv::VideoWriter::fourcc('m', 'p', '4', 'v');  // 'm', 'p', '4', 'v';  'h', '2', '6', '4'

		const float confidence = args_info.confidence_arg;  // dftConf
		string filename = (fs::path(args_info.output_arg) / fs::path(args_info.video_arg).stem()).string()
			+ "_" + to_string(start) + "-" + to_string(nfram) + "_c" + to_string(confidence).substr(0, 4);  // + dateTime();
		if(args_info.fout_suffix_given)
			filename += string(" ") + args_info.fout_suffix_arg;

		const double fps = 1.0;  // FPS of the forming video
		cv::Mat frame;

		// cv::Size sizeFrame(992+extr,992);
		// cv::Mat testimg = std::get<2>(detectORB(d_images.at(0), d_images.at(1), 2.5));

		auto millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
		// trackingMotV2_1  - Detector + motion
		// trackingMotV2_2  - Interactive ORB descriptors for the whole frame
		// trackingMotV2_3  - Detector + ORB descriptors for the whole frame
		cv::Mat testimg = trackingMotV2_1(pathmodel, device_type, d_images.at(0), d_images.at(1), objects, 0, args_info.model_given, confidence);
		// cv::Mat testimg = trackingMotV2_3(pathmodel, device_type, d_images.at(0), d_images.at(1), objects, 0, args_info.model_given, confidence);
		std::cout << "Tracking time: " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec
			// << "ms; " << "confidence threshold : " << confidence
			<< " ms" << endl;

		cv::Size sizeFrame(testimg.cols, testimg.rows);

		writer.open(filename + "_demo.mp4", codec, fps, sizeFrame, true);
		vector<ALObject> objects;

		for (int i = 0; i < d_images.size() - 1; i++)
		{
			cout << "[Frame: " << start + i << "]" << endl;
			millisec = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
			writer.write(trackingMotV2_1(pathmodel, device_type, d_images.at(i), d_images.at(i + 1), objects, start + i, args_info.model_given, confidence));
			std::cout << "Tracking time: " << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() - millisec
				// << "ms; " << "confidence threshold : " << confidence
				<< " ms" << endl;
			// trackingMotV2_1_artemis(pathmodel, device_type, d_images.at(i), d_images.at(i + 1), objects, start + i, args_info.model_given, confidence);
			// writer.write(trackingMotV2_3(pathmodel, device_type, d_images.at(i), d_images.at(i + 1), objects, start + i, args_info.model_given, confidence));
			// writer.write(std::get<2>(detectORB(d_images.at(i), d_images.at(i + 1), 2.5)));
		}
		writer.release();
		traceObjects(objects, filename);
		return 0;
		//---TEST---*/

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

		fixIDs(objs, fixedIds, d_images);
	}
	else
	{
		cv::Mat imageBGR = cv::imread(args_info.img_arg, cv::ImreadModes::IMREAD_COLOR);  // argv[2]
		detectorV4(pathmodel, imageBGR, device_type);
	}

	return 0;
}