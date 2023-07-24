#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>

#include <torch/script.h> // One-stop header.

#include "tracker.h"
#include "utils.h"


float reduseres = evalReduseres(objhsz);

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


// ORB detector -----------------------------------------------------------------------------------
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
