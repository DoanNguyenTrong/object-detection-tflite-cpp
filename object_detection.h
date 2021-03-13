#ifndef OBJECT_DETECTOR_H
#define OBJECT_DETECTOR_H

#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <cmath>

#include <opencv2/opencv.hpp>

#include <tensorflow/lite/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include <ft2build.h>
#include <freetype/freetype.h>
#include <freetype/ftoutln.h>


// // doan 202010227: add hexagon delegate
#include <tensorflow/lite/delegates/hexagon/hexagon_delegate.h>
#include <tensorflow/lite/delegates/gpu/delegate.h>

// inlcude filessystem to check the existance of file
#ifndef __has_include
  static_assert(false, "__has_include not supported");
#else
#  if __has_include(<filesystem>)
#    include <filesystem>
     namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
     namespace fs = std::experimental::filesystem;
#  elif __has_include(<boost/filesystem.hpp>)
#    include <boost/filesystem.hpp>
     namespace fs = boost::filesystem;
#  endif
#endif


struct Object{
  cv::Rect rec;
  int      class_id;
  float    score;
};


float expit(float x) {
  return 1.f / (1.f + expf(-x));
}

//nms
float iou(cv::Rect& rectA, cv::Rect& rectB){
    int x1 = std::max(rectA.x, rectB.x);
    int y1 = std::max(rectA.y, rectB.y);
    int x2 = std::min(rectA.x + rectA.width, rectB.x + rectB.width);
    int y2 = std::min(rectA.y + rectA.height, rectB.y + rectB.height);
    int w = std::max(0, (x2 - x1 + 1));
    int h = std::max(0, (y2 - y1 + 1));
    float inter = w * h;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float o = inter / (areaA + areaB - inter);
    return (o >= 0) ? o : 0;
}

void nms(std::vector<Object>& boxes,  const double nms_threshold)
{
		std::vector<int> scores;
    for(int i = 0; i < boxes.size();i++){
			scores.push_back(boxes[i].score);
		}

		std::vector<int> index;
    for(int i = 0; i < scores.size(); ++i){
        index.push_back(i);
    }
		std::sort(index.begin(), index.end(), [&](int a, int b){
        return scores[a] > scores[b]; }); 
    std::vector<bool> del(scores.size(), false);
		for(size_t i = 0; i < index.size(); i++){
        if( !del[index[i]]){
            for(size_t j = i+1; j < index.size(); j++){
                if(iou(boxes[index[i]].rec, boxes[index[j]].rec) > nms_threshold){
                    del[index[j]] = true;
                }
            }
        }
    }
		std::vector<Object> new_obj;
    for(const auto i : index){
				Object obj;
				if(!del[i])
				{
					obj.class_id = boxes[i].class_id;
					obj.rec.x =  boxes[i].rec.x;
					obj.rec.y =  boxes[i].rec.y;
					obj.rec.width =  boxes[i].rec.width;
					obj.rec.height =  boxes[i].rec.height;
					obj.score =  boxes[i].score;
				}
				new_obj.push_back(obj);

        
    }
    boxes = new_obj;  
}

template<typename T>
void fill(T *in, cv::Mat& src) {
  int n = 0, nc = src.channels(), ne = src.elemSize();
  for (int y = 0; y < src.rows; ++y)
    for (int x = 0; x < src.cols; ++x)
      for (int c = 0; c < nc; ++c)
        in[n++] = src.data[y * src.step + x * ne + c];
}


class ObjectDetector{
public:
    std::string                           model_file_;
    std::string                           label_file_;
    std::vector<std::string>                    labels_;


    std::unique_ptr<tflite::FlatBufferModel>    model_;
    std::unique_ptr<tflite::Interpreter>        interpreter_;
    tflite::StderrReporter                      error_reporter_;

    // parameters of interpreter's input
    int input_;
    int in_height_;
    int in_width_;
    int in_channels_;
    int in_type_;

    // parameters of original image
    int img_height_;
    int img_width_;

    // Input of the interpreter
    uint8_t *input_8_  ;
    float   *input_16_ ;
    
    int delegate_opt_;
    TfLiteDelegate * delegate_;


    ObjectDetector();
    ObjectDetector(const std::string , const std::string );
    ~ObjectDetector();


    int readLabels();

    int readModel();

    int configure(const std::uint8_t , const std::uint8_t );

    int inference(cv::Mat );
    
    std::vector<Object> * extractObjects(const float, const float );
};


ObjectDetector::ObjectDetector(std::string model_file, std::string label_file){
    model_file_ = model_file;
    label_file_ = label_file;
  
  // check for existance
  if (fs::exists(model_file_)){
    std::cout << "EXISTS: "<< model_file_ << std::endl;
  }
  else{
    std::cout << "INEXISTS: "<< model_file_ << std::endl;
  }
  if (fs::exists(label_file_)){
    std::cout << "EXISTS: "<< label_file_ << std::endl;
  }
  else{
    std::cout << "INEXISTS: "<< label_file_ << std::endl;
  }
}


ObjectDetector::~ObjectDetector(){
    // Do any needed cleanup and delete 'delegate'.
    if (delegate_opt_){
        // TfLiteHexagonDelegateDelete( delegate);
        // TfLiteHexagonTearDown();
    }
    else{
        TfLiteGpuDelegateV2Delete( delegate_);
    }
}


int ObjectDetector::readLabels(void){
    std::cout << "Loading: " << label_file_ << std::endl;
    std::ifstream file(label_file_);
    if (!file) {
        std::cerr << "Failed to read " << label_file_ << "." << std::endl;
        return -1;
    }
    std::string               line;
    while (std::getline(file, line))
        labels_.push_back(line);
    while (labels_.size() % 16)
        labels_.emplace_back();
    std::cout << "Done!!\n";
    return 0;
}


int ObjectDetector::readModel(void){

    std::cout << "Loading: " << model_file_ << std::endl;
    model_ = tflite::FlatBufferModel::BuildFromFile(model_file_.c_str(), &error_reporter_);
    if (!model_) {
        std::cerr << "Failed to load the model." << std::endl;
        return -1;
    }else{
        std::cout << "Done!!\n";
    }

    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model_, resolver)(&interpreter_);
    
    TfLiteStatus status = interpreter_->AllocateTensors();
    if (status != kTfLiteOk) {
        std::cerr << "Failed to allocate the memory for tensors." << std::endl;
        return -1;
    }else{
        std::cout << "Done!!\n";
    }
    return 0;
}

int ObjectDetector::configure(const std::uint8_t nthreads, const std::uint8_t delegate_opt){
    // input information
    input_                  = interpreter_->inputs()[0];
    TfLiteIntArray* dims    = interpreter_->tensor(input_)->dims;
    in_height_              = dims->data[1];
    in_width_               = dims->data[2];
    in_channels_            = dims->data[3];
    in_type_                = interpreter_->tensor(input_)->type;

    // Link input to interpreter
    if (in_type_ == kTfLiteFloat32) {
        std::cout << "Model has not been quantized! Could not use Hexagon Delegate!\n";
        input_16_ = interpreter_->typed_tensor<float>(input_);
    } 
    else if (in_type_ == kTfLiteUInt8) {
        std::cout << "Quantized model! Can use Hexagon Delegate!\n";
        input_8_ = interpreter_->typed_tensor<uint8_t>(input_);
    }


    interpreter_->SetNumThreads(nthreads);
    // interpreter->UseNNAPI(1);
    delegate_opt_ = delegate_opt;

    if (delegate_opt_ == 1){
        const char* path = "hexagon/hexagon_nn_skel_v1.20.0.1/";
        std::cout << "Enabling Hexagon Delegate!\n";
        // TfLiteHexagonInitWithPath(path);
        // TfLiteHexagonDelegateOptions * params = {0};
        // delegate = TfLiteHexagonDelegateCreate(params);
    }
    else if (delegate_opt_ == 0){
        std::cout << "Enabling GPU Delegate!\n";
        // GPU delegate
        TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();
        gpu_options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
        delegate_ = TfLiteGpuDelegateV2Create(&gpu_options);
    }
    else{
        std::cout << "Unable to construct delegate option " << delegate_opt << std::endl;
    }

    if (interpreter_->ModifyGraphWithDelegate(delegate_) != kTfLiteOk){
        std::cerr << "Failed to use GPU/Hexagon Delegate" << std::endl;
        return -1;
    }else{
        std::cout << "Done!\n";
    }
    return 0;
}

int ObjectDetector::inference(cv::Mat frame){

    img_height_ = frame.rows;
    img_width_  = frame.cols;

    // Resize to fit the NN
    cv::Mat resized(in_height_, in_width_, frame.type());
    cv::resize(frame, resized, resized.size(), cv::INTER_CUBIC);

    // Overwrite input of the NN
    if (in_type_ == kTfLiteFloat32) {
      fill(input_16_, resized);
    } 
    else if (in_type_ == kTfLiteUInt8) {
      fill(input_8_, resized);
    }

    // Inference
    TfLiteStatus status = interpreter_->Invoke();
    if (status != kTfLiteOk) {
      std::cout << "Failed to run inference!!\n";
      return -1;
    }
    return 0;
}


std::vector<Object> *ObjectDetector::extractObjects(const float score_thres, const float nms_thres){
    // Bounding box coordinates of detected objects
    TfLiteTensor* bboxes    = interpreter_->tensor(interpreter_->outputs()[0]);
    // Class index of detected objects
    TfLiteTensor* classes   = interpreter_->tensor(interpreter_->outputs()[1]);
    // Confidence of detected objects
    TfLiteTensor* scores    = interpreter_->tensor(interpreter_->outputs()[2]);
    // Total number of detected objects (inaccurate and not needed)
    TfLiteTensor* num_detec = interpreter_->tensor(interpreter_->outputs()[3]);
    
    auto          bboxes_   = bboxes->data.f;
    auto          classes_  = classes->data.f;
    auto          scores_   = scores->data.f;
    
    auto bboxes_size        = bboxes->dims->data[bboxes->dims->size - 1]; 
    auto classes_size       = classes->dims->data[classes->dims->size - 1];
    auto scores_size        = scores->dims->data[scores->dims->size - 1];
    
    if (bboxes_size != 4){
        std::cerr << "Incorrect bbox size: " << bboxes_size << std::endl;
        return nullptr;
    }
    if (classes_size != scores_size){
        std::cerr << "Number of classes and scores does not match: " << classes_size << " " << scores_size << std::endl;
        return nullptr;
    }

    std::cout << "bboxes: " << bboxes_size << "," << bboxes->dims->size << std::endl;
    std::cout << "classes: " << classes_size << "," << classes->dims->size << std::endl;
    std::cout << "scores: " << scores_size << "," << scores->dims->size << std::endl;
    
    // std::cout << "Output size: " << interpreter->outputs().size() << std::endl;
    
    // int output_type = bboxes->type;


    std::vector<float> locations;
    std::vector<float> cls;

    for (int i = 0; i < bboxes_size * classes_size; i++){
        locations.push_back(bboxes_[i]);
    }

    for (int i = 0; i < classes_size; i++){
        cls.push_back(classes_[i]);
    }
    
    int count=0;
    std::vector<Object> objects;
    for(int j = 0; j <locations.size(); j+=4){
        float score = scores_[count];
        if (score < score_thres) {
            count++;
            continue;
        }

        auto ymin=locations[j]  *img_height_;
        auto xmin=locations[j+1]*img_width_;
        auto ymax=locations[j+2]*img_height_;
        auto xmax=locations[j+3]*img_width_;
        auto width= xmax - xmin;
        auto height= ymax - ymin;

        std::cout << labels_[cls[count]] << std::endl;
        std::cout << cls[count] << " score: "<< score << " (" << xmin << "," << ymin << "," << width << "," << height << ")"<< std::endl;

        // auto rec = Rect(xmin, ymin, width, height)
        
        // auto id=outputClasses;
        Object object;
        object.class_id = cls[count];
        object.rec.x = xmin;
        object.rec.y = ymin;
        object.rec.width = width;
        object.rec.height = height;
        object.score = score;
        objects.push_back(object);

        count++;
    }

    nms(objects, nms_thres);
    return &objects;
}



#endif // OBJECT_DETECTOR_H