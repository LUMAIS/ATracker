#include "tracker.h"
#include "utils.h"

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
