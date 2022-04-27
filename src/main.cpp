#include <iostream>
#include "trophallaxis/antdetect.hpp"
#include <torch/script.h> // One-stop header.
#include <unistd.h>

int main(int argc, char** argv) {

    //testtorch();
    //./detector ../data/modules/traced_resnet_model.pt
    //./detector ../data/modules/AnTroD_resnet50.pt
    //testmodule(argv[1]);

    torch::jit::script::Module module;
    std::string pathmodel;
    std::vector<ALObject> objects;

    pathmodel = argv[1];
    std::cout<<"pathmodel - "<<pathmodel<<std::endl;

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(pathmodel);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    torch::DeviceType device_type = torch::kCPU;

    if(std::strstr(argv[3],"CUDA") != NULL)
        device_type = torch::kCUDA;

    if(std::strstr(argv[2],".mp4") != NULL)
    {
        int start = 135;
        int nfram = 15;
       
        std::vector<std::vector<Obj>> objs;
        std::vector<std::pair<uint,idFix>> fixedIds;
        std::vector<cv::Mat> d_images;

        //---TEST---
        d_images = LoadVideo(argv[2],start,nfram);
        std::vector<ALObject> objects;
        for(int i=0; i<d_images.size()-1; i++)
        {
            DetectorMotionV3(d_images.at(i), d_images.at(i + 1), objects, i);
        }
        return 0;
        //---TEST---*/


        d_images = LoadVideo(argv[2],start,nfram);

        std::vector<OBJdetect> obj_detects;
        std::vector<Obj> objbuf;
        for(int i=0; i<d_images.size(); i++)
        {
            obj_detects.clear();
            obj_detects = detectorV4(pathmodel, d_images.at(i), device_type);
            OBJdetectsToObjs(obj_detects,objbuf);
            objs.push_back(objbuf);
        }

        fixIDs(objs,fixedIds,d_images);
    }
    else
    {
        cv::Mat imageBGR = cv::imread(argv[2], cv::ImreadModes::IMREAD_COLOR);
        detectorV4(pathmodel, imageBGR, device_type);
    }

	return 0;
}