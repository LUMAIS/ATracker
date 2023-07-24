#include <tuple>
//#include <sys/time.h>
#include <ctime>

#include <torch/script.h> // One-stop header.

#include "utils.h"


// General Definitions -------------------------------------------------------------
const uint16_t model_resolution = 992;  // frame resizing for model (992)
uint16_t frame_resolution = model_resolution;  //frame frame_resolution
int resolution = model_resolution;  // Frame size for a model
const float kres = resolution / 1200.f;  // Note: basic motion parameters were adjusted for 1200 px

uint16_t extr = 204;  // sidebar size
uint16_t objhsz = 80; // area half size for a moving object: max(w, h) [/2];  8 .. 128

// Local declarations --------------------------------------------------------------

// class IMGsamples
// {
// public:
// 	uint16_t num;
// 	vector<Mat> samples;
// 	vector<Point2f> coords; // coordinates relative to the center of the cluster
//
// 	IMGsamples(vector<Mat> samples, vector<Point2f> coords)
// 	{
// 		this->samples = samples;
// 		this->coords = coords;
// 		num = samples.size();
// 	}
// };


// ALObject definitions ------------------------------------------------------------
void ALObject::center_determine(size_t frame_id, bool samplescreation)
{
	vector<Point2f> cp;  // pairwise cluster points
	vector<float> l;  // distances

	for(int i=0; i < cluster_points.size(); i++)
	{
		const auto& pi = cluster_points[i];
		for(int j=i; j< cluster_points.size(); j++)
		{
			const auto& pj = cluster_points[j];
			cp.push_back(Point2f((pi.x + pj.x) / 2, (pi.y + pj.y) / 2));
			l.push_back(distance(pi, pj));
		}
	}

	Point2f sumcp;
	float suml = 0;

	sumcp.x = 0;
	sumcp.y = 0;

	for(int i =0; i< cp.size(); i++)
	{
		sumcp.x += cp.at(i).x*l.at(i);
		sumcp.y += cp.at(i).y*l.at(i);
		suml += l.at(i);
	}

	cluster_center.x = sumcp.x/suml;
	cluster_center.y = sumcp.y/suml;

	if(frame_id != static_cast<decltype(frame_id)>(-1)) {
		if(!traces.empty() && traces.back().frame == frame_id) {
			traces.back().center.x = roundf(cluster_center.x);
			traces.back().center.y = roundf(cluster_center.y);
		} else {
			assert((traces.empty() || traces.back().frame < frame_id) && "Unexpected frame number in the traces");
			traces.push_back(Trace{frame_id, cluster_center.x, cluster_center.y, size.width, size.height, prob});
		}
	}

	det_pos = true;
}

// void ALObject::samples_creation()
// {
// 	Mat sample;
// 	Point2f coord;
//
// 	samples.clear();
// 	coords.clear();
//
// 	for (int i = 0; i < cluster_points.size(); i++)
// 	{
// 		// coord.x = cluster_points.at(i).x - cluster_center.x + objhsz;
// 		// coord.y = cluster_points.at(i).y - cluster_center.y + objhsz;
// 		coord.x = min_u16(cluster_points.at(i).x - roundf(frame_resolution / reduseres / 2.f));
// 		coord.y = min_u16(cluster_points.at(i).y - roundf(frame_resolution / reduseres / 2.f));
//
// 		sample = img(cv::Range(coord.y, min_u16(img.rows, coord.y + frame_resolution / reduseres))
// 			, cv::Range(coord.x, min_u16(img.cols, coord.x + frame_resolution / reduseres)));
// 		samples.push_back(sample);
// 		coords.push_back(coord);
// 	}
//
// 	IMGsamples buf(samples, coords);
// 	moution_samples.push_back(buf);
// }

void ALObject::push_track_point(Point2f track_point)
{
	track_points.push_back(track_point);

	if (track_points.size() > trackptsMax)
		track_points.erase(track_points.begin());
}

//! Expected center of the object based on it's movement in the opposite direction to the previous position
Point2f ALObject::proposed_center()
{
	Point2f proposed(cluster_center);

	if (track_points.size() > 1)
	{
		proposed.x += 0.5f * (cluster_center.x - track_points.at(track_points.size() - 2).x);
		proposed.y += 0.5f * (cluster_center.y - track_points.at(track_points.size() - 2).y);
	}

	return proposed;
}

// General operations --------------------------------------------------------------
float distance(const Point2f& a, const Point2f& b) noexcept
{
	const float  dx = a.x - b.x;
	const float  dy = a.y - b.y;

	return sqrt(dx*dx + dy*dy);
}

// Torch-related operations --------------------------------------------------------
void testtorch()
{
	torch::Tensor tensor = torch::eye(3);
	std::cout << tensor << endl;
	std::cout << "testtorch() - OK!!" << endl;
}

int testmodule(const string& strpath)
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

// IO-related operations -----------------------------------------------------------
string dateTime()
{
	const time_t  now = time(0); 
	const struct tm  tstruct = *localtime(&now);
	char  tcstr[63];
	strftime(tcstr, sizeof(tcstr), "%Y-%m-%d_%H-%M-%S", &tstruct);
	return tcstr;
}

Mat frame_resizing(Mat frame, uint16_t framesize)
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

	sort(top.begin(), top.end(), cmpIPt);

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

// Accessory operations ------------------------------------------------------------
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

// Object-related operations -------------------------------------------------------

//string objClassTitle(] = {"ta", "a", "ah", "tl", "l", "fn", "p", "b", "u"};  // objClassTitle[9)
//const char *const objClassTitle(] = {"a", "ah", "ta", "l", "tl", "fn", "p", "b", "u"};  // objClassTitle[9)

const char* objClassTitle(ObjClass objClass) noexcept
{
	static const char *const titles[] = {"a", "ah", "ta", "l", "tl", "fn", "p", "b", "u"};  // objClassTitle(9)

	return titles[static_cast<uint8_t>(objClass)];
}

Scalar objClassColor(ObjClass objClass) noexcept
{
	static Scalar class_name_color[9] = {Scalar(255, 0, 0), Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 255), Scalar(0, 255, 255), Scalar(255, 255, 0), Scalar(255, 255, 255), Scalar(200, 0, 200), Scalar(100, 0, 255)};
	return class_name_color[static_cast<uint8_t>(objClass)];
}

Scalar objColor(uint32_t id, uint8_t clrLow, uint8_t clrHigh) noexcept
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

bool cmpIPt(intpoint a, intpoint b) noexcept
{
	return a.ipoint < b.ipoint;
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

bool compare_clsobj(const ClsObjR& a, const ClsObjR& b) noexcept
{
	return a.r < b.r;
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

	// sort(top.begin(), top.end(), cmpIPt);
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
		sort(buftop.begin(), buftop.end(), cmpIPt);
	}

	for (int i = 0; i < buftop.size(); i++)
	{
		probapoints.push_back(buftop.at(i).mpoint);
	}

	return probapoints;
}

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
