#include <cstdint>
#include <vector>
#include <chrono>
#include <iostream>
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

// To put text to image
class ft_renderer {
private:
  FT_Library lib_;
  FT_Face    face_;

  void stoc(int &c, int &i, const std::string &text) {
    unsigned char t1 = (unsigned char) text[i];
    if (t1 >= 0xF0) {
      c  = (((unsigned char)text[i] << 18) & 0x70000) |
        (((unsigned char)text[i+1] << 12) & 0xF000) |
        (((unsigned char)text[i+2] << 6) & 0x0FC0) |
        (((unsigned char)text[i+3] << 0) & 0x003F) ;
      i += 4;
    } else if (t1 >= 0xE0) {
      c  = (((unsigned char)text[i] << 12) & 0xF000) |
        (((unsigned char)text[i+1] << 6) & 0x0FC0) |
        (((unsigned char)text[i+2] << 0) & 0x003F) ;
      i += 3;
    } else if (t1 >= 0xC2) {
      c  = (((unsigned char)text[i] << 6) & 0x0FC0) |
        (((unsigned char)text[i+1] << 0) & 0x003F) ;
      i += 2;
    } else if (t1 > 0) {
      c  = text[i];
      i += 1;
    } else {
      c = '?';
      i += 1;
    }
  }

public:
  ft_renderer(std::vector<uint8_t>& fontdata) {
    FT_Init_FreeType(&lib_);
    FT_New_Memory_Face(lib_, 
        fontdata.data(), 
        fontdata.size(), 0, &face_); 
  }
  ~ft_renderer() {
    FT_Done_FreeType(lib_);
  }

public:
  void putText(cv::InputOutputArray _img, const std::string& text, cv::Point pos,
    double fontScale, cv::Scalar color, bool bottomLeftOrigin) {

    if (text.empty())
      return;

    FT_Set_Pixel_Sizes(face_, fontScale, fontScale);

    cv::Mat img = _img.getMat();

    for (int i = 0; text[i] != 0;) {
      int c;

      stoc(c, i, text);
      FT_Load_Char(face_, (FT_Long)c, 0); 
      FT_Render_Glyph(face_->glyph, FT_RENDER_MODE_MONO);
      FT_Bitmap *bmp = &(face_->glyph->bitmap);

      cv::Point loc = pos;
      loc.y = loc.y - (face_->glyph->metrics.horiBearingY >> 6) ;
      loc.x = loc.x + (face_->glyph->metrics.horiBearingX >> 6) ;
      int row, col, bit, cl;

      for (row = 0; row < bmp->rows; ++row) {
        if (loc.y + row > img.rows)
          continue;
        cv::Vec3b* ptr = img.ptr<cv::Vec3b>(loc.y + row);
        for (col = 0; col < bmp->pitch; ++col) {
          cl = bmp->buffer[ row * bmp->pitch + col];
          for (bit = 7; bit >= 0; --bit) {
            if (loc.x + col * 8 + (7 - bit) > img.cols)
              continue;
            if (((cl >> bit) & 0x01) == 1) {
              ptr[loc.x + col * 8 + (7 - bit)][0] = color[0];
              ptr[loc.x + col * 8 + (7 - bit)][1] = color[1];
              ptr[loc.x + col * 8 + (7 - bit)][2] = color[2];
            }
          }
        }
      }
      pos.x += (face_->glyph->advance.x) >> 6;
      pos.y += (face_->glyph->advance.y) >> 6;
    }
  }
};

template<typename T>
void
fill(T *in, cv::Mat& src) {
  int n = 0, nc = src.channels(), ne = src.elemSize();
  for (int y = 0; y < src.rows; ++y)
    for (int x = 0; x < src.cols; ++x)
      for (int c = 0; c < nc; ++c)
        in[n++] = src.data[y * src.step + x * ne + c];
}


int
main(int argc, char const * argv[]) {
  
  // @doan 20210226: change file paths
  std::string modelfile = "models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite";
  std::string labelfile = "models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt";
  
  bool delegate_option = false;
  if (argc == 2){
    delegate_option = std::atoi(argv[1]);
    std::cout <<  ((delegate_option != 0)? "Hexagon" : "GPU") << " Delegate!\n";
  }
  else if (argc == 4) {
    modelfile = argv[1];
    labelfile = argv[2];
    delegate_option = std::atoi(argv[3]);
  } 
  else if (argc != 1) {
    std::cerr << "Usage of " << argv[0] << " [modelfile] [labelfile] [delegate_option]" << std::endl;
    return -1;
  }
  
  // @doan 20210226: check for existance
  if (fs::exists(modelfile)){
    std::cout << "EXISTS: "<< modelfile << std::endl;
  }
  else{
    std::cout << "INEXISTS: "<< modelfile << std::endl;
    return -1;
  }
  if (fs::exists(labelfile)){
    std::cout << "EXISTS: "<< labelfile << std::endl;
  }
  else{
    std::cout << "INEXISTS: "<< labelfile << std::endl;
    return -1;
  }


  TfLiteStatus                              status;
  std::unique_ptr<tflite::FlatBufferModel>  model;
  std::unique_ptr<tflite::Interpreter>      interpreter;
  tflite::StderrReporter                    error_reporter;
  
  std::cout << "Loading: " << modelfile << std::endl;
  model = tflite::FlatBufferModel::BuildFromFile(modelfile.c_str(), &error_reporter);
  if (!model) {
    std::cerr << "Failed to load the model." << std::endl;
    return -1;
  }else{
    std::cout << "Done!!\n";
  }

  std::cout << "Loading: " << labelfile << std::endl;
  std::ifstream file(labelfile);
  if (!file) {
    std::cerr << "Failed to read " << labelfile << "." << std::endl;
    return -1;
  }else{
    std::cout << "Done!!\n";
  }
  std::vector<std::string>  labels;
  std::string               line;
  while (std::getline(file, line))
    labels.push_back(line);
  while (labels.size() % 16)
    labels.emplace_back();
  


  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  
  status = interpreter->AllocateTensors();
  if (status != kTfLiteOk) {
    std::cerr << "Failed to allocate the memory for tensors." << std::endl;
    return -1;
  }else{
    std::cout << "Done!!\n";
  }


  cv::Scalar white(255, 255, 255);
  int input = interpreter->inputs()[0];

  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  int wanted_type = interpreter->tensor(input)->type;


  // Input of the interpreter
  uint8_t *in8 = nullptr;
  float *in16 = nullptr;

  if (wanted_type == kTfLiteFloat32) {
    std::cout << "Model has not been quantized! Could not use Hexagon Delegate!\n";
    in16 = interpreter->typed_tensor<float>(input);
  } else if (wanted_type == kTfLiteUInt8) {
    std::cout << "Quantized model! Can use Hexagon Delegate!\n";
    in8 = interpreter->typed_tensor<uint8_t>(input);
  }




  interpreter->SetNumThreads(4);
  // interpreter->UseNNAPI(1);

  // doan 20210227: Hexagon V66Q
  // https://developer.qualcomm.com/qualcomm-robotics-rb5-kit/software-reference-manual/
  
  TfLiteDelegate * delegate = nullptr;
  // Initializes the DSP connection.

  if (delegate_option){
    const char* path = "hexagon/hexagon_nn_skel_v1.20.0.1/";
    std::cout << "Enabling Hexagon Delegate!\n";
    // TfLiteHexagonInitWithPath(path);
    // TfLiteHexagonDelegateOptions * params = {0};
    // delegate = TfLiteHexagonDelegateCreate(params);
  }
  else{
    std::cout << "Enabling GPU Delegate!\n";
    // GPU delegate
    TfLiteGpuDelegateOptionsV2 gpu_options = TfLiteGpuDelegateOptionsV2Default();
    gpu_options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;
    delegate = TfLiteGpuDelegateV2Create(&gpu_options);
    
  }

  if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk){
    std::cerr << "Failed to use GPU/Hexagon Delegate" << std::endl;
    return -1;
  }else{
    std::cout << "Done!\n";
  }

  // Renderer to put text on frame
  // std::cout << "Loading: mplus-1c-thin.ttf" << std::endl;
  // std::ifstream fontfile("mplus-1c-thin.ttf", std::ios::in | std::ios::binary);
  // if (!fontfile) {
  //   std::cerr << "Failed to read font file" << std::endl;
  //   return -1;
  // }
  // else{
  //   std::cout << "Done!\n";
  // }
  // std::vector<uint8_t> fontdata(
  //     (std::istreambuf_iterator<char>(fontfile)),
  //     std::istreambuf_iterator<char>());
  // ft_renderer ftw(fontdata);
  
  
  
  // std::cout << cv::getBuildInformation() << std::endl;


  // camx-hal3-test > A:id=0,psize=3840x2160,pformat=yuv420,ssize=3840x2160,sformat=jpeg,zsl=1
  // v4l2-ctl --list-devices
  // const char *pipeline = "qtiqmmfsrc device-name=/dev/video32 ! video/x-raw,format=NV12,framerate=30/1,width=1920,height=1080 ! appsink";
  const char *pipeline = "qtiqmmfsrc ! video/x-h264,format=NV12,width=1920,height=1080,framerate=30/1 ! h264parse ! mp4mux ! queue ! appsink";
  cv::VideoCapture cap("images/grace_hopper.bmp");
  // cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open VideoCapture." << std::endl;
    return -1;
  }else{
    std::cout << "Opened videocapture stream!!!\n";
  }
  int nFrames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
  int counter = 0;
  while (true) {
    cv::Mat frame;

    cap >> frame;
    // int key = cv::waitKey(1);
    // if (key == 27)
    //   break;

    auto img_height=frame.rows;
    auto img_width =frame.cols;

    // Resize to fit the NN
    cv::Mat resized(wanted_height, wanted_width, frame.type());
    cv::resize(frame, resized, resized.size(), cv::INTER_CUBIC);

    int n = 0;
    // Overwrite input of the NN
    if (wanted_type == kTfLiteFloat32) {
      fill(in16, resized);
    } else if (wanted_type == kTfLiteUInt8) {
      fill(in8, resized);
    }

    // Inference
    status = interpreter->Invoke();
    if (status != kTfLiteOk) {
      // cv::imshow("window", frame);
      std::cout << "Failed to run inference!!\n";
      continue;
    }


    TfLiteTensor* bboxes    = interpreter->tensor(interpreter->outputs()[0]);
    TfLiteTensor* classes   = interpreter->tensor(interpreter->outputs()[1]);
    TfLiteTensor* scores    = interpreter->tensor(interpreter->outputs()[2]);
    TfLiteTensor* num_detec = interpreter->tensor(interpreter->outputs()[3]);
    auto          nums_     = num_detec->data.f;
    auto          bboxes_   = bboxes->data.f;
    auto          classes_  = classes->data.f;
    auto          scores_   = scores->data.f;

    std::cout << "bboxes: " << bboxes_ << std::endl;
    std::cout << "classes: " << classes_ << std::endl;
    std::cout << "scores: " << scores_ << std::endl;
    std::cout << "num_detec: " << nums_ << std::endl;

    // std::cout << "Output size: " << interpreter->outputs().size() << std::endl;
    
    // Extract output
    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    auto output_size = output_dims->data[output_dims->size - 1];
    int output_type = interpreter->tensor(output)->type;

    std::cout << "output size: " << output_dims->size -1 << std::endl;


    if (wanted_type == kTfLiteFloat32){
      for (int i = 0; i < output_size; i++){
        float value = (scores_[i] - 127) / 127.0;
        std::cout << value << std::endl;
      }
    }
    else if (wanted_type == kTfLiteUInt8){
      for (int i = 0; i < output_size; i++){
        float value = (float)scores[i] / 255.0; 
        std::cout << value << std::endl;
      }
    }
    else{
      std::cout << "wanted_type is not kTfLiteFloat32 or kTfLiteUInt8\n";
    }
    // std::vector<std::pair<float, int>> results;

    // if (wanted_type == kTfLiteFloat32) {
    //   float *scores = interpreter->typed_output_tensor<float>(0);
    //   for (int i = 0; i < output_size; ++i) {
    //     float value = (scores[i] - 127) / 127.0;
    //     if (value < 0.1)
    //       continue;
    //     results.push_back(std::pair<float, int>(value, i));
    //   }
    // } else if (wanted_type == kTfLiteUInt8) {
    //   uint8_t *scores = interpreter->typed_output_tensor<uint8_t>(0);
    //   for (int i = 0; i < output_size; ++i) {
    //     float value = (float)scores[i] / 255.0;
    //     if (value < 0.2)
    //       continue;
    //     results.push_back(std::pair<float, int>(value, i));
    //   }
    // }

    // std::sort(results.begin(), results.end(),
    //   [](std::pair<float, int>& x, std::pair<float, int>& y) -> int {
    //     return x.first > y.first;
    //   }
    // );

    // std::cout << "Results size: " << results.size() << std::endl;
    // for (int i = 0; i < results.size(); i++){
    //   std::cout << results[i].first << ", " << labels[results[i].second] << std::endl;
    // }

    // // Put text to frame
    // n = 0;
    // for (const auto& result : results) {
    //   std::stringstream ss;
    //   ss << result.first << ": " << labels[result.second];
    //   ftw.putText(frame, ss.str(),  cv::Point(50, 50 + 50 * n), 16,
    //       cv::Scalar(255,255,255), false);
    //   if (++n >= 3)
    //     break;
    // }

    // cv::imshow("window", frame);
  }
  cv::destroyAllWindows();

  // Do any needed cleanup and delete 'delegate'.
  if (delegate_option){
    // TfLiteHexagonDelegateDelete( delegate);
    // TfLiteHexagonTearDown();
  }
  else{
    TfLiteGpuDelegateV2Delete( delegate);
  }
  return 0;
}

