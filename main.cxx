#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* compute fps*/ 

#include "object_detection.h"
#include "freetype_renderer.h"




int main(int argc, char const * argv[]) {
  
  // @doan 20210226: change file paths
  std::string modelfile = "models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite";
  std::string labelfile = "models/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labels.txt";
  
  bool delegate_option = false;
  std::string video_src;
  if (argc == 2){
    delegate_option = std::atoi(argv[1]);
    std::cout <<  ((delegate_option != 0)? "Hexagon" : "GPU") << " Delegate!\n";
  }
  else if (argc == 3){
    delegate_option = std::atoi(argv[1]);
    std::cout <<  ((delegate_option != 0)? "Hexagon" : "GPU") << " Delegate!\n";
    video_src = argv[2];
    std::cout << "Video source: " << video_src << std::endl;
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
  
  
  
  // std::cout << cv::getBuildInformation() << std::endl;


  // camx-hal3-test > A:id=0,psize=3840x2160,pformat=yuv420,ssize=3840x2160,sformat=jpeg,zsl=1
  // v4l2-ctl --list-devices
  // const char *pipeline = "qtiqmmfsrc device-name=/dev/video32 ! video/x-raw,format=NV12,framerate=30/1,width=1920,height=1080 ! appsink";
  const char *pipeline = "qtiqmmfsrc ! video/x-h264,format=NV12,width=1920,height=1080,framerate=30/1 ! h264parse ! mp4mux ! queue ! appsink";
  cv::VideoCapture cap;
  if (argc == 3){
    cap.open(video_src);
  }
  else{
    cap.open("/dev/video0");
  }
  // cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
  // cv::VideoCapture cap("/dev/video0");
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

    std::vector<Object> objects = detector.extractObjects(0.3f, 0.5f);

    // std::cout << "Start drawing to object...\n";
    std::cout << "size: "<< objects.size() << std::endl;


    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Frames per second using video.get(CAP_PROP_FPS) : " << fps << std::endl;

    cv::Mat frame_cp = frame.clone();
    for (int l = 0; l < objects.size(); l++){
      Object object = objects.at(l);
      
      auto cls = object.class_id;
      auto score =object.score;
      cv::Scalar color = cv::Scalar (rand() %255, rand() %255, rand() %255);
      
      std::ostringstream fps_str;
      fps_str.width(5);
      fps_str.precision(3);
      fps_str << fps;
      cv::putText(frame_cp, fps_str.str(), cv::Point(5, 5),
      cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));


      cv::rectangle(frame_cp, object.rec, color, 1);
      cv::putText(frame_cp, detector.labels_[cls], cv::Point(object.rec.x, object.rec.y - 5),
      cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
      // std::cout << cls << std::endl;

    }
    
    writer.write(frame_cp);
    // cv::imshow("window", frame);
    
    // require GTK+ (install libgtk2.0-dev and pkg-config)
    // // Press  ESC on keyboard to  exit
    // int c = cv::waitKey(1);
    // if( c == 27 )
    //   break;
  }

  cap.release();
  cv::destroyAllWindows();

  return 0;
}

