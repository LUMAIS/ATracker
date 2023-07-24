#include "tracker.h"
#include "utils.h"

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
