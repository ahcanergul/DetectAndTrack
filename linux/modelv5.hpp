#pragma once
#include <dirent.h>
#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>
#define clock


#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

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
	
	float confThreshold;
	float nmsThreshold; //iouThreshold
	
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
	void postprocess(const at::Tensor& output_tensor, float confThreshold, float nmsThreshold, std::vector<objectInfo>& results);
	void modelv5::XcenterYcenterWidthHeight2TopLeftBottomRight(const at::Tensor& xywh_bbox_tensor,at::Tensor& tlbr_bbox_tensor);
private:
	torch::jit::script::Module model_;
	vector<objectInfo> object;

};


template <typename T>
float modelv5::getObject(Mat frame, T& bbox)
{
	torch::NoGradGuard no_grad_guard;
	std::vector<torch::jit::IValue> inputs;
	#ifdef clock
	auto start_preprocess = std::chrono::high_resolution_clock::now();
	PreProcess(input_image, inputs);
	auto end_preprocess = std::chrono::high_resolution_clock::now();
	
	auto duration_preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end_preprocess - start_preprocess);
	std::cout << "Pre-processing: " << duration_preprocess.count() << " [ms] \n";
	
	auto start_inference = std::chrono::high_resolution_clock::now();
	at::Tensor output_tensor = model_.forward(inputs).toTuple()->elements()[0].toTensor();
	auto end_inference = std::chrono::high_resolution_clock::now();

	auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
	std::cout << "Inference: " << duration_inference.count() << " [ms] \n";

	#elif
	PreProcess(input_image, inputs);
	at::Tensor output_tensor = model_.forward(inputs).toTuple()->elements()[0].toTensor();
	#endif
	
}

void modelv5::preprocess(const cv::Mat& input_image,std::vector<torch::jit::IValue>& inputs,Size inpSize,float scale=1.0)
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
void modelv5::postprocess(const at::Tensor& output_tensor, float confThreshold, float nmsThreshold, std::vector<objectInfo>& results)
{
	// output_tensor ... {Batch=1, Num of max bbox=25200, 85}
	// 25200 ... {(640[px]/32[stride])^2 + (640[px]/16[stride])^2 + (640[px]/8[stride])^2} x 3[layer]
	// 85 ... 0: center x, 1: center y, 2: width, 3: height, 4: obj conf, 5~84: class conf 
  
	int batch_size = output_tensor.size(0);
	  if (batch_size != 1) {
    std::cerr << "[ObjectDetector::PostProcess()] Error: Batch size of output tensor is not 1 \n";
    return;
	}
	
	// 85 ... 0: center x, 1: center y, 2: width, 3: height, 4: obj conf, 5~84: class conf
	int num_bbox_confidence_class_idx = output_tensor.size(2);
	int num_bbox_confidence_idx = num_bbox_confidence_class_idx - class_names_.size();
	int num_bbox_idx = num_bbox_confidence_idx - 1;
	
	// output_tensor ... {Batch=1, Num of max bbox=25200, 85}
	// --->
	// candidate_object_mask ... {Batch=1, Num of max bbox=25200, 1}
	int object_confidence_idx=4; //take object confidence!!!
	at::Tensor candidate_object_mask = output_tensor.select(-1,
                                                          object_confidence_idx);
														  
	candidate_object_mask = candidate_object_mask.gt(confidence_threshold);
	candidate_object_mask = candidate_object_mask.unsqueeze(-1);
	
	// candidate_object_tensor ... {Num of candidate bbox*85}
	at::Tensor candidate_object_tensor = torch::masked_select(output_tensor[0],
                                                            candidate_object_mask[0]);
	candidate_object_tensor = candidate_object_tensor.view({-1, num_bbox_confidence_class_idx});
	
	 if (candidate_object_tensor.size(0) == 0) {
    ?????
  }
  
	at::Tensor xywh_bbox_tensor = candidate_object_tensor.slice(-1,
                                                              0, num_bbox_idx);
	 at::Tensor bbox_tensor;
	XcenterYcenterWidthHeight2TopLeftBottomRight(xywh_bbox_tensor, bbox_tensor);
	
	at::Tensor object_confidence_tensor = candidate_object_tensor.slice(-1,num_bbox_idx, num_bbox_confidence_idx);
	at::Tensor class_confidence_tensor = candidate_object_tensor.slice(-1, num_bbox_confidence_idx);
	at::Tensor class_score_tensor = class_confidence_tensor * object_confidence_tensor; //class score* class confidence
	std::tuple<at::Tensor, at::Tensor> max_class_score_tuple = torch::max(class_score_tensor,-1); // assign classes to confidence
	  
	at::Tensor max_class_score = std::get<0>(max_class_score_tuple).to(torch::kFloat);
	max_class_score = max_class_score.unsqueeze(-1);

	at::Tensor max_class_id = std::get<1>(max_class_score_tuple).to(torch::kFloat);
	max_class_id = max_class_id.unsqueeze(-1); // divide tuple
	
	// result_tensor ... {Num of candidate bbox, 6}
	// 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: class score, 5: class id
	 at::Tensor result_tensor = torch::cat({bbox_tensor, max_class_score, max_class_id},-1);

	
	at::Tensor class_id_tensor = result_tensor.slice(-1, -1);
	
	 at::Tensor class_offset_bbox_tensor = result_tensor.slice(-1, 0, num_bbox_idx)
                                          + nms_max_bbox_size_ * class_id_tensor; //offset by +4096 * class id
										  
	// Copies tensor to CPU to access tensor elements efficiently with TensorAccessor
 
	at::Tensor class_offset_bbox_tensor_cpu = class_offset_bbox_tensor.cpu();
	at::Tensor result_tensor_cpu = result_tensor.cpu();
	auto class_offset_bbox_tensor_accessor = class_offset_bbox_tensor_cpu.accessor<float, 2>();
	auto result_tensor_accessor = result_tensor_cpu.accessor<float, 2>();
	
	std::vector<cv::Rect> offset_bboxes;
  std::vector<float> class_scores;
  offset_bboxes.reserve(result_tensor_accessor.size(0));
  class_scores.reserve(result_tensor_accessor.size(0));
  
  for (std::size_t i = 0; i < result_tensor_accessor.size(0); ++i) {
    float class_offset_top_left_x = class_offset_bbox_tensor_accessor[i][0];
    float class_offset_top_left_y = class_offset_bbox_tensor_accessor[i][1];
    float class_offset_bottom_right_x = class_offset_bbox_tensor_accessor[i][2];
    float class_offset_bottom_right_y = class_offset_bbox_tensor_accessor[i][3];

    offset_bboxes.emplace_back(cv::Rect(cv::Point(class_offset_top_left_x, class_offset_top_left_y),
                                        cv::Point(class_offset_bottom_right_x, class_offset_bottom_right_y)));

    class_scores.emplace_back(result_tensor_accessor[i][4]);
  }
  
  std::vector<int> nms_indecies;
  cv::dnn::NMSBoxes(offset_bboxes, class_scores, conftTreshold, nmsThreshold, nms_indecies);
  
  ///////////////////// kontrol ET ve düazelt !!!! ///////////
	std::vector<ObjectInfo> object_infos;
  for (const auto& nms_idx : nms_indecies) {
    float top_left_x = result_tensor_accessor[nms_idx][0];
    float top_left_y = result_tensor_accessor[nms_idx][1];
    float bottom_right_x = result_tensor_accessor[nms_idx][2];
    float bottom_right_y = result_tensor_accessor[nms_idx][3];

    ObjectInfo object_info;
    object_info.bbox_rect = cv::Rect(cv::Point(top_left_x, top_left_y),
                                      cv::Point(bottom_right_x, bottom_right_y));
    object_info.class_score = result_tensor_accessor[nms_idx][4];
    object_info.class_id = result_tensor_accessor[nms_idx][5];

    object_infos.emplace_back(object_info);
  }

  RestoreBoundingboxSize(object_infos, letterbox_info, results);
 /////////////////////////////////////////////////////////////////
}

void modelv5::XcenterYcenterWidthHeight2TopLeftBottomRight(const at::Tensor& xywh_bbox_tensor,
                                                                  at::Tensor& tlbr_bbox_tensor) {
  tlbr_bbox_tensor = torch::zeros_like(xywh_bbox_tensor);

  int bbox_dim = -1;  // the last dimension

  int x_center_idx = 0;
  int y_center_idx = 1;
  int width_idx = 2;
  int height_idx = 3;

  tlbr_bbox_tensor.select(bbox_dim, 0) = xywh_bbox_tensor.select(bbox_dim, x_center_idx)
                                           - xywh_bbox_tensor.select(bbox_dim, width_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 1) = xywh_bbox_tensor.select(bbox_dim, y_center_idx)
                                           - xywh_bbox_tensor.select(bbox_dim, height_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 2) = xywh_bbox_tensor.select(bbox_dim, x_center_idx)
                                           + xywh_bbox_tensor.select(bbox_dim, width_idx).div(2.0);
  tlbr_bbox_tensor.select(bbox_dim, 3) = xywh_bbox_tensor.select(bbox_dim, y_center_idx)
                                           + xywh_bbox_tensor.select(bbox_dim, height_idx).div(2.0);
  
  return;
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
