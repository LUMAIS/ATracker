#include <iostream>
#include "trophallaxis/antdetect.hpp"
#include <torch/script.h> // One-stop header.

int main(int argc, char** argv) {

    //testtorch();
    //./detector ../data/modules/traced_resnet_model.pt
    //./detector ../data/modules/AnTroD_resnet50.pt
    //testmodule(argv[1]);

    torch::jit::script::Module module;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    torch::DeviceType device_type = torch::kCPU;
    if(std::strstr(argv[2],".mp4") != NULL)
    {
        std::vector<cv::Mat> d_images;
        d_images = LoadVideo(argv[2]);
        for(int i=0; i<d_images.size(); i++)
            detectorT3(module, d_images.at(i), device_type);
       
    }
    else
    {
        cv::Mat imageBGR = cv::imread(argv[2], cv::ImreadModes::IMREAD_COLOR);
        detectorT(module, imageBGR, device_type);
    }

	return 0;
}