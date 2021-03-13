#include <iostream>
#include <stdlib.h>     /* srand, rand */

#include "object_detection.h"


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



int main(int argc, char const * argv[]) {
  
  // @doan 20210226: change file paths
  std::string modelfile = "models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite";
  std::string labelfile = "models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labels.txt";
  
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
  
  
  ObjectDetector detector(modelfile, labelfile);
  detector.readLabels();
  detector.readModel();
  detector.configure(4, delegate_option);


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
  // cv::VideoCapture cap("images/grace_hopper.bmp");
  // cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
  cv::VideoCapture cap(0);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open VideoCapture." << std::endl;
    return -1;
  }else{
    std::cout << "Opened videocapture stream!!!\n";
  }

  // width & height
  int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  // Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
  cv::VideoWriter writer("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(frame_width,frame_height));

  while (true) {
    cv::Mat frame;

    cap >> frame;

    int state = detector.inference(frame);
    if (state != 0){
      exit(0);
    }

    std::vector<Object> *objects = detector.extractObjects(0.3f, 0.5f);


    std::cout << "size: "<<objects->size() << std::endl;

    for (int l = 0; l < objects->size(); l++){
      Object object = objects->at(l);
      
      auto cls = object.class_id;
      auto score =object.score;
      cv::Scalar color = cv::Scalar (rand() %255, rand() %255, rand() %255);
      cv::Mat frame_cp = frame.clone();
      cv::rectangle(frame_cp, object.rec, color, 1);
      cv::putText(frame_cp, detector.labels[cls+1], cv::Point(object.rec.x, object.rec.y - 5),
      cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
      std::cout << cls << std::endl;

    }
    
    writer.write(frame_cp);
    // cv::imshow("window", frame);
    
    
    // Press  ESC on keyboard to  exit
    int c = cv::waitKey(1);
    if( c == 27 )
      break;
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}

