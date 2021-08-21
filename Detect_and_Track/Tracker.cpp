#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp> // contrib yüklenmeli !!!
#include <opencv2/core/ocl.hpp>
#include <opencv2/gapi/core.hpp> // GPU API library

using namespace cv;
using namespace std; 

#define val 4
#define frame_ratio 15 // i� boxun ROI ye oran�
#define min_box_size 64

#define frame_ratio 15 // iç boxun ROI ye oraný
#define min_box_size 30

#define PID_KP  2.0f
#define PID_KI  0.5f
#define PID_KD  0.25f

#define PID_TAU 0.02f

#define PID_LIM_MIN   -1000.0f
#define PID_LIM_MAX   1000.0f

#define PID_LIM_MIN_T   0.0f
#define PID_LIM_MAX_T   1000.0f

#define PID_LIM_MIN_INT -5.0f
#define PID_LIM_MAX_INT  5.0f

#define SAMPLE_TIME_S 0.01f

typedef struct {

	/* Controller gains */
	float Kp;
	float Ki;
	float Kd;

	/* Derivative low-pass filter time constant */
	float tau;

	/* Output limits */
	float limMin;
	float limMax;

	/* Integrator limits */
	float limMinInt;
	float limMaxInt;

	/* Sample time (in seconds) */
	float T;

	/* Controller "memory" */
	float integrator;
	float prevError;			/* Required for integrator */
	float differentiator;
	float prevMeasurement;		/* Required for differentiator */

	/* Controller output */
	float out;

} PID;

float PID_update(PID* pid, float setpoint, float measurement) {

	float error = setpoint - measurement;

	/*
	* Proportional
	*/
	float proportional = pid->Kp * error;

	/*
	* Integral
	*/
	pid->integrator = pid->integrator + 0.5f * pid->Ki * pid->T * (error + pid->prevError);

	/* Anti-wind-up via integrator clamping */
	if (pid->integrator > pid->limMaxInt) {
		pid->integrator = pid->limMaxInt;
	}
	else if (pid->integrator < pid->limMinInt) {
		pid->integrator = pid->limMinInt;
	}

	/*
	* Derivative (band-limited differentiator)
	*/

	pid->differentiator = -(2.0f * pid->Kd * (measurement - pid->prevMeasurement)	/* Note: derivative on measurement, therefore minus sign in front of equation! */
		+ (2.0f * pid->tau - pid->T) * pid->differentiator)
		/ (2.0f * pid->tau + pid->T);

	/*
	* Compute output and apply limits
	*/
	pid->out = proportional + pid->integrator + pid->differentiator;

	if (pid->out > pid->limMax) {
		pid->out = pid->limMax;
	}
	else if (pid->out < pid->limMin) {
		pid->out = pid->limMin;
	}

	/* Store error and measurement for later use */
	pid->prevError = error;
	pid->prevMeasurement = measurement;

	/* Return controller output */
	return pid->out;
}

void PID_init(PID* pid) {
	pid->integrator = 0.0f;
	pid->prevError = 0.0f;
	pid->differentiator = 0.0f;
	pid->prevMeasurement = 0.0f;
	pid->out = 0.0f;
}

#include "model.hpp"
#include "track_utils.hpp"

static float scale_h, scale_w; //scaling for convenient box size in tracking
const float ext_size = 5; // extra required size 

const char* winname = "Takip ekrani";
int mode = 1; // player modes --> play - 1 : stop - 0   || tuþlar:  esc --> çýk , p --> pause , r--> return  
int win_size_h = 608, win_size_w = 608; // fixed win sizes

std::string keys =
"{ help  h     | | Print help message. }"
"{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
"{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
"{ device      |  0 | camera device number. }"
"{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
"{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
"{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
"{ thr         | .5 | Confidence threshold. }"
"{ nms         | .4 | Non-maximum suppression threshold. }"
"{ backend     |  0 | Choose one of computation backends: "
"0: automatically (by default), "
"1: Halide language (http://halide-lang.org/), "
"2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
"3: OpenCV implementation, "
"4: VKCOM, "
"5: CUDA }"
"{ target      | 0 | Choose one of target computation devices: "
"0: CPU target (by default), "
"1: OpenCL, "
"2: OpenCL fp16 (half-float precision), "
"3: VPU, "
"4: Vulkan, "
"6: CUDA, "
"7: CUDA fp16 (half-float preprocess) }"
"{ async       | 0 | Number of asynchronous forwards at the same time. "
"Choose 0 for synchronous mode }";

int main(int argc, char** argv)
{
	CommandLineParser parser(argc, argv, keys);

	const std::string modelName = parser.get<String>("@alias");
	const std::string zooFile = parser.get<String>("zoo");
	keys += genPreprocArguments(modelName, zooFile);

	parser = CommandLineParser(argc, argv, keys);

	CV_Assert(parser.has("model"));
	std::string modelPath = findFile(parser.get<String>("model"));
	std::string configPath = findFile(parser.get<String>("config"));
	
	// model object definitons
	model_param param = {modelName, modelPath, configPath, parser.get<String>("framework"), parser.get<int>("backend"), 
						parser.get<int>("target"), parser.get<int>("async")};
	model yolov4(param);
	yolov4.confThreshold = parser.get<float>("thr");
	yolov4.nmsThreshold = parser.get<float>("nms");
	yolov4.scale = parser.get<float>("scale");
	yolov4.swapRB = parser.get<bool>("rgb");
	yolov4.mean = parser.get<float>("mean");
	win_size_h = parser.get<int>("height");
	win_size_w = parser.get<int>("width");
	yolov4.inpHeigth = win_size_h;
	yolov4.inpWidth = win_size_w;

	PID manouver_control = { PID_KP, PID_KI, PID_KD, PID_TAU,PID_LIM_MIN, 
		                  PID_LIM_MAX,PID_LIM_MIN_INT, PID_LIM_MAX_INT,SAMPLE_TIME_S };

	PID_init(&manouver_control);

	if (parser.has("classes"))
		yolov4.get_classes(parser.get<string>("classes"));
	
	string filename;
	if(parser.has("input"))
		filename = parser.get<String>("input");

	Ptr<Tracker>tracker = TrackerMOSSE::create();//Tracker declaration
	scaleBox<Rect2d> scbox;

	VideoCapture video;
	if (!filename.empty())
	{
		video.open(filename);
		video.set(CAP_PROP_FRAME_WIDTH, win_size_w); // resize the screen
		video.set(CAP_PROP_FRAME_HEIGHT, win_size_h);
		cout << "file founded!!!" << endl;
	}
	else
		video.open(0);
	// Exit if video is not opened
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
		waitKey(10);
		return 1;
	}

	cout << cv::getBuildInformation << endl; // get build inf - contrib is installed ?

	Mat frame, t_frame; // frame storages
	Rect2d bbox, exp_bbox; // selected bbox ROI / resized bbox
	
	bool track_or_detect = false; 
	while (true)
	{
		if (mode)
		{
			double timer = (double)getTickCount(); // start FPS timer
			if (!video.read(frame)) // frame read control
				break; // if frame error occurs

			resize(frame, frame, Size(win_size_w, win_size_h), 0.0, 0.0, INTER_CUBIC); // frame boyutlar�n� ayarla 	
			//cvtColor(frame, grayFrame, COLOR_BGR2GRAY); // mosse takes single channel img
			t_frame = frame.clone();

			
			resize(frame, frame, Size(win_size_w, win_size_h), 0.0, 0.0, INTER_CUBIC); // frame boyutlarýný ayarla 
		//	cvtColor(frame, grayFrame, COLOR_BGR2GRAY); // mosse takes single channel img
			
			if (!track_or_detect) // detection mode
			{
				// get bbox from model...
				float confidence = yolov4.getObject<Rect2d>(frame, bbox);
				CV_Assert(confidence > 0);
				cout << "model initiated..." << endl;
				
				//grayROI = selectROI(frame); // manuel select 
				exp_bbox = bbox; // stored original box in printable exp_bbox
				scale_h = min_box_size / bbox.height; // calculated scale to adjust frame according to predefined size 
				scale_w = min_box_size / bbox.width;

				cout << "scale = " << scale_h << "frame_size" << win_size_h << endl;
				win_size_h *= scale_h;
				win_size_w *= scale_w;

				resize(t_frame, t_frame, Size(win_size_w, win_size_h), 0.0, 0.0, INTER_CUBIC);
				bbox = Rect(bbox.x * scale_w - ext_size, bbox.y * scale_h-ext_size, bbox.width * scale_w+2*ext_size, bbox.height * scale_h+2*ext_size);
				
				scbox.init(t_frame, bbox);
				tracker->init(t_frame, bbox); // initialize tracker
				//rectangle(t_frame, bbox,Scalar(0,250,0));
				//imshow("resized frame",t_frame);
				track_or_detect = true; // tracking mode'a gecis yapiliyor ...
			}
			else // tracking 
			{ 
				//rectangle(t_frame, bbox, Scalar(0, 250, 0));
				//imshow("resized frame", grayFrame);
				if (tracker->update(t_frame, bbox)) // tracking check
				{
					float fps = getTickFrequency() / ((double)getTickCount() - timer); // sayacý al
					
					bbox = Rect((bbox.x + ext_size) / scale_w, (bbox.y + ext_size) / scale_h, (bbox.width - 2 * ext_size) / scale_w, (bbox.height - 2 * ext_size) / scale_h);
					exp_bbox = bbox; 
					//scbox.updateSize(t_frame, bbox);

					int center_x = win_size_w / scale_w / 2;
					int center_y = win_size_h / scale_h / 2;

					Point center_box = Center(bbox);
					
					signed int error_x = center_x - center_box.x; // -x error sağa yani + x yönüne git
					signed int error_y = center_y - center_box.y; // +x error sola yani - x yönüne git
					                                              // -y error aşağı yani - y yönüne git
					                                              // +y error yukarı yani + y yönüne git
					
					float speed_x = PID_update(&manouver_control, center_box.x, center_x);
					float speed_y = PID_update(&manouver_control, center_box.y, center_y);

					cout << "x_cmd" << " " << speed_x;
					cout << "y_cmd" << " " << speed_y;
					
					resize(frame, frame, Size(win_size_w/scale_w, win_size_h/scale_h), 0.0, 0.0, INTER_CUBIC);

					drawMarker(frame, Center(bbox), Scalar(0, 255, 0)); //mark the center 

					drawMarker(frame, Center(bbox), Scalar(0, 255, 255)); //mark the center 
					drawMarker(frame, Point(center_x, center_y), Scalar(0, 255, 255)); //mark the center

					putText(frame, "FPS : " + SSTR(int(fps)), Point(100, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
					putText(frame, "x_cmd : " + SSTR(float(speed_x)), Point(100, 70), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);
					putText(frame, "y_cmd : " + SSTR(float(speed_y)), Point(100, 100), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

				}
				else
				{
					// Tracking failure detected.
					resize(frame, frame, Size(win_size_w/scale_w, win_size_h/scale_h), 0.0, 0.0, INTER_CUBIC);
					putText(frame, "Tracking lost...", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 50, 200), 2);
					win_size_h = parser.get<int>("height");
					win_size_w = parser.get<int>("width");
					tracker->clear();
					tracker = TrackerMOSSE::create();
					track_or_detect = false; // return to the detection mode ...
					cout << "yeni";
				}
			}
		}

		rectangle(frame, exp_bbox, Scalar(255, 0, 0), 2, 1);
		imshow(winname, frame);// show final result ...
		moveWindow(winname,50,50);
		//waitKey(0); // to move frame by frame -- REMOVE BEFORE FLIGHT !!!

		int keyboard = waitKey(5); // kullanýcýdan kontrol tuþu al 
		if (keyboard == 'q' || keyboard == 27) // quit
			break;
		else if (keyboard == 'p' || keyboard == 112) // pause
			mode = 0;
		else if (keyboard == 'r' || keyboard == 114) // return
			mode = 1;
	}
	return 0;
}