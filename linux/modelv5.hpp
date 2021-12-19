#pragma once
#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>

using namespace dnn;
using namespace std;

struct model_param
{
	const std::string modelName;
	std::string modelPath;
	std::string configPath;
	std::string framework;
	int backend;
	int target; // hedef
	bool is_gpu;
};

struct objectInfo 
{
	String outNames;
	cv::Rect boxes;
	int classIds;
	float confidences;
};


class modelv5 {

public:
	model_param mparam;
	int input_width; // preprocess sizes
	int input_heigth;
	
	modelv5(model_param &param): mparam(param)
	{
	std::cout << "Input height = " << input_height << "\n";
   	std::cout << "Input width = " << input_width << "\n\n"; 
	
	 try {
      		std::cout << "[ObjectDetector()] torch::jit::load( " << model_filename << " ); ... \n";
      		model = torch::jit::load(model_filename);
      		std::cout << "[ObjectDetector()] " << model_filename << " has been loaded \n\n";
    	 }
   	 catch (const c10::Error& e) {
    		  std::cerr << e.what() << "\n";
     		  std::exit(EXIT_FAILURE);
   	 }
   	 catch (...) {
     		 std::cerr << "[ObjectDetector()] Exception: Could not load " << model_filename << "\n";
      	 	 std::exit(EXIT_FAILURE);
         }
         
         
         mparam.is_gpu = (mparam.is_gpu && torch::cuda::is_available());
         if(mparam.is_gpu)
         {
         	model.to(torch::kCUDA);
         	model.to(torch::kHalf);
         	std::cout<<" GPU activated \n\n";
         }
         else
         {
         	std::cout<<"Using CPU... \n\n";
         	model.to(torch::kCPU);
         }
         
         model.eval();
	}
	~modelv5() {};
	
	template <typename T>
	float getObject(Mat frame, T& bbox);
	
	protected:
	void inference(float confidence_threshold, float iou_threshold);
	void preprocess(const cv::Mat& input_image,std::vector<torch::jit::IValue>& inputs,Size inpSize,float scale);
private:
	torch::jit::script::Module model;
	vector<objectInfo> object;

};


template <typename T>
float modelv5::getObject(Mat frame, T& bbox)
{
	torch::NoGradGuard no_grad_guard;
	

}

void preprocess(const cv::Mat& input_image,std::vector<torch::jit::IValue>& inputs,Size inpSize,float scale=1.0)
{
	  Mat blob_image;
	  cv::resize(input_image, blob_image, cv::Size(), scale, scale);
	  
	  // 0 ~ 255 ---> 0.0 ~ 1.0
	  cv::cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);
	  blob_image.convertTo(blob_image, CV_32FC3, 1.0 / 255.0);
	  
	  at::Tensor input_tensor = torch::from_blob(blob_image.data,{1, input_height, input_width, 3});
	  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous(); // {Batch=1, Height, Width, Channel=3} -> {Batch=1, Channel=3, Height, Width}
	  
	  
	if (is_gpu_) {
   		 input_tensor = input_tensor.to(torch::kCUDA);
   		 input_tensor = input_tensor.to(torch::kHalf);
  	  } else {
   		 input_tensor = input_tensor.to(torch::kCPU);
	}
	
	inputs.clear();
  	inputs.emplace_back(input_tensor);
}


void modelv5::inference(float confidence_threshold, float iou_threshold)
{
	 std::cout << "=== Empty inferences to warm up === \n\n";
	 for (std::size_t i = 0; i < 3; ++i) {
    	cv::Mat tmp_image = cv::Mat::zeros(input_height_, input_width_, CV_32FC3);
    	std::vector<ObjectInfo> tmp_results;
    	Detect(tmp_image, 1.0, 1.0, tmp_results);
  	}
  	std::cout << "=== Warming up is done === \n\n\n";



}
