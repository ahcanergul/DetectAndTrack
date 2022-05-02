#pragma once
#include <dirent.h>
#include <opencv2/core.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

#define clock

struct model_param
{
	std::string modelfile;
	std::string class_names_file;
	int nms_max_bbox_size; // default 4096
	bool is_gpu;
	float confThreshold;
	float nmsThreshold; //iouThreshold
};

struct objectInfo 
{
	cv::Rect bbox;
	int classId;
	float confidence;
};


class modelv5 {
public:
	model_param mparam;
	int input_width; // preprocess sizes
	int input_heigth;
	
	float scale; //optional param, unity if not defined 
	
	modelv5(model_param &param): mparam(param)
	{
		std::cout << "Input height rescaled to " << input_heigth << "...\n";
   		std::cout << "Input width rescaled to " << input_width << "...\n\n"; 
	
		loadModel();
		std::cout << "Model loaded successfully... \n";
		
		model.eval();
	}
	~modelv5() {};
	
	template <typename T>
	float getObject(cv::Mat frame, T& bbox);
	
	protected:
	void loadModel(void);
	void preprocess(const cv::Mat& input_image,std::vector<torch::jit::IValue>& inputs,cv::Size inpSize,float scale = 1);
	void postprocess(const at::Tensor& output_tensor, float confThreshold, float nmsThreshold, std::vector<objectInfo>& results);
	void XcenterYcenterWidthHeight2TopLeftBottomRight(const at::Tensor& xywh_bbox_tensor,at::Tensor& tlbr_bbox_tensor);
private:
	torch::jit::script::Module model;
	std::vector<objectInfo> objects;
	std::vector<std::string> class_names_;
	
	std::vector<cv::Rect> boxes;
	std::vector<float> confidences;
};


void modelv5::loadModel(void)
{

	std::ifstream load_class_names(mparam.class_names_file);
	if(load_class_names.is_open())
	{
		std::string class_names;
		while(std::getline(load_class_names, class_names))
			class_names_.emplace_back(class_names);
		load_class_names.close(); 
		
		if(class_names_.size()==0)
		{
			std::cerr << "[ObjectDetector::LoadClassNames()] Error: labe names are empty \n";
    			std::exit(EXIT_FAILURE);
    		}
	}
	else {
    		std::cerr << "[ObjectDetector::LoadClassNames()] Error: Could not open "<< mparam.class_names_file << "\n";
    		std::exit(EXIT_FAILURE);
  	}
	/// loading model weights ................
	try {
      		std::cout << "[ObjectDetector()] torch::jit::load( " << mparam.modelfile << " ); ... \n"; 
      		model = torch::jit::load(mparam.modelfile);
      		std::cout << "[ObjectDetector()] " << mparam.modelfile << " has been loaded \n\n";
    	 }
   	 catch (const c10::Error& e) {
    		  std::cerr << e.what() << "\n";
     		  std::exit(EXIT_FAILURE);
   	 }
   	 catch (...) {
     		 std::cerr << "[ObjectDetector()] Exception: Could not load " << mparam.modelfile << "\n";
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
}


template <typename T>
float modelv5::getObject(cv::Mat frame, T& bbox)
{
	torch::NoGradGuard no_grad_guard;
	std::vector<torch::jit::IValue> inputs;
	std::vector<objectInfo> results;
	float confidence;
	
	#ifdef clock
	auto start_preprocess = std::chrono::high_resolution_clock::now();
	preprocess(frame, inputs, cv::Size(input_width,input_heigth), scale);
	auto end_preprocess = std::chrono::high_resolution_clock::now();
	
	auto duration_preprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end_preprocess - start_preprocess);
	std::cout << "Pre-processing: " << duration_preprocess.count() << " [ms] \n";
	
	auto start_inference = std::chrono::high_resolution_clock::now();
	at::Tensor output_tensor = model.forward(inputs).toTuple()->elements()[0].toTensor();
	auto end_inference = std::chrono::high_resolution_clock::now();

	auto duration_inference = std::chrono::duration_cast<std::chrono::milliseconds>(end_inference - start_inference);
	std::cout << "Inference: " << duration_inference.count() << " [ms] \n";
	
	auto start_postprocess = std::chrono::high_resolution_clock::now();
	postprocess(output_tensor, this->mparam.confThreshold, this->mparam.nmsThreshold, results);
	auto end_postprocess = std::chrono::high_resolution_clock::now();
	auto duration_postprocess = std::chrono::duration_cast<std::chrono::milliseconds>(end_postprocess - start_postprocess);
	std::cout << "Post-processing: " << duration_preprocess.count() << " [ms] \n";
	
	#elif
	PreProcess(input_image, inputs);
	at::Tensor output_tensor = model.forward(inputs).toTuple()->elements()[0].toTensor();
	postprocess(output_tensor, mparam.confThreshold, mparam.nmsThreshold, results);
	#endif
	
	//select routine
	if(this->objects.size()>0)
	{
	 for(auto& obj : this->objects)
	 {
	 	cv::Rect box = obj.bbox;
		std::string label = cv::format("%.2f", obj.confidence);
	 	int classId = obj.classId;
		
		CV_Assert( classId < (int)class_names_.size());
		label = std::to_string(classId) + ":" + (std::string)(class_names_.at(classId)) + "-" + (std::string)(label);
		
		int baseLine;
		cv::Size labelSize = getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		box.y = MAX(box.y, labelSize.height);
		
		cv::rectangle(frame, cv::Point(box.x, box.y - labelSize.height),
						cv::Point(box.x + labelSize.width, box.y + baseLine), cv::Scalar::all(255), cv::FILLED);
		cv::putText(frame, label, cv::Point(box.x, box.y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar());
	 }

	
		cv::putText(frame, "hedeflerden bir tanesini secin", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 10), 2);
		cv::imshow("detections", frame);
		int keyboard=0;
		int num;
		while(true)
		{
			num = 0;
			do
			{
				keyboard = cv::waitKey(0)%128;
				if(keyboard>47)
				{
					num+=(int)(keyboard-48);
					num*=10;
				}
				cv::putText(frame, "Hedef:"+std::to_string(int(num/10)), cv::Point(100, 110), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 50, 200), 2);
				cv::imshow("detections", frame);
				} while (keyboard!=13);
					num/=10;
					if(num > this->objects.size()){
						std::cout<<"Maksimum kare sayisindan fazla girdiniz. Lütfen tekrar giriniz..."<< num<<std::endl;
					continue;
					}
					break;
			}
		std::cout<<"num:"<<num<<std::endl;
		bbox = this->boxes.at(num);
		confidence = this->confidences.at(num);
	}
	else if(this->boxes.size()==1)
	{
		bbox = this->boxes.back();
		confidence = this->confidences.back();
	}
	else
	{
		cv::putText(frame, "hedef bulunamadi", cv::Point(100, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 255), 2);
		cv::imshow("detections", frame);
		cv::waitKey(10000);
		return 0;
	}
		this->boxes.clear();
		this->confidences.clear();
		this->objects.clear();
return confidence;
}

void modelv5::preprocess(const cv::Mat& input_image,std::vector<torch::jit::IValue>& inputs,cv::Size inpSize,float scale)
{
	  cv::Mat blob_image;
	  cv::resize(input_image, blob_image, cv::Size(), scale, scale);
	  
	  // 0 ~ 255 ---> 0.0 ~ 1.0
	  cv::cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);
	  blob_image.convertTo(blob_image, CV_32FC3, 1.0 / 255.0);
	  
	  at::Tensor input_tensor = torch::from_blob(blob_image.data,{1, input_heigth, input_width, 3});
	  input_tensor = input_tensor.permute({0, 3, 1, 2}).contiguous(); // {Batch=1, Height, Width, Channel=3} -> {Batch=1, Channel=3, Height, Width}
	  
	  
	if (this->mparam.is_gpu) {
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
                                                          object_confidence_idx); //slice with object conf(why so 4 is used) 
														  
	candidate_object_mask = candidate_object_mask.gt(mparam.confThreshold);
	candidate_object_mask = candidate_object_mask.unsqueeze(-1); //retrieve last dimension again, dimension has changed when use select 
	
	// mask with candidate_object_mask[0] ... {Num of max bbox=25200, 1} --> [num_of_box,true/false] 
	at::Tensor candidate_object_tensor = torch::masked_select(output_tensor[0],
                                                            candidate_object_mask[0]);
	// candidate_object_tensor ... {Num of candidate bbox*85} --> {Num of candidate bbox, 85}
	candidate_object_tensor = candidate_object_tensor.view({-1, num_bbox_confidence_class_idx}); 
	
	 if (candidate_object_tensor.size(0) == 0) {
           CV_Assert(0);
  }
  
	at::Tensor xywh_bbox_tensor = candidate_object_tensor.slice(-1,
                                                              0, num_bbox_idx);
	 at::Tensor bbox_tensor;
	XcenterYcenterWidthHeight2TopLeftBottomRight(xywh_bbox_tensor, bbox_tensor);
	
	at::Tensor object_confidence_tensor = candidate_object_tensor.slice(-1,num_bbox_idx, num_bbox_confidence_idx);
	at::Tensor class_confidence_tensor = candidate_object_tensor.slice(-1, num_bbox_confidence_idx);
	at::Tensor class_score_tensor = class_confidence_tensor * object_confidence_tensor; //class score * class confidence
	std::tuple<at::Tensor, at::Tensor> max_class_score_tuple = torch::max(class_score_tensor,-1); // max class scores and indices
	  
	at::Tensor max_class_score = std::get<0>(max_class_score_tuple).to(torch::kFloat);
	max_class_score = max_class_score.unsqueeze(-1);

	at::Tensor max_class_id = std::get<1>(max_class_score_tuple).to(torch::kFloat);
	max_class_id = max_class_id.unsqueeze(-1); // divide tuple
	
	// result_tensor ... {Num of candidate bbox, 6}
	// 6 ... 0: top-left x, 1: top-left y, 2: bottom-right x, 3: bottom-right y, 4: class score, 5: class id
	 at::Tensor result_tensor = torch::cat({bbox_tensor, max_class_score, max_class_id},-1); // max_class_id == which class belongs to

	 //Non Maximum Suppression
	at::Tensor class_id_tensor = result_tensor.slice(-1, -1);
	
	 at::Tensor class_offset_bbox_tensor = result_tensor.slice(-1, 0, num_bbox_idx)
                                          + this->mparam.nms_max_bbox_size * class_id_tensor; //offset by +4096 * class id
										  
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

    class_scores.emplace_back(result_tensor_accessor[i][4]); // get class scores from result tensor
  }
  
  std::vector<int> nms_indecies;
  cv::dnn::NMSBoxes(offset_bboxes, class_scores, confThreshold, nmsThreshold, nms_indecies);
  
  //////////////////--------/// final ///-------//////////////////////

	this->objects.reserve(result_tensor_accessor.size(0));
  for (const auto& nms_idx : nms_indecies) {
  	float top_left_x = result_tensor_accessor[nms_idx][0];
  	float top_left_y = result_tensor_accessor[nms_idx][1];
    	float bottom_right_x = result_tensor_accessor[nms_idx][2];
    	float bottom_right_y = result_tensor_accessor[nms_idx][3];
    	objects.emplace_back(objectInfo{cv::Rect(top_left_x, top_left_y, bottom_right_x, bottom_right_y),result_tensor_accessor[nms_idx][5],result_tensor_accessor[nms_idx][4]});    
  }
 //////////////////////////////////////////////////////////////////////
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
