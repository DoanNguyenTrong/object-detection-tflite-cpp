# Inference on edge: tensorflow lite C++ and GPU/CPU/Hexagon DSP

## Introduction

As one of 10 winning startups of the Qualcomm Vietnam Innovative Challenge 2020, I have the chance to use the Qualcomm RB5 board to deploy our autonomous driving system for Automated Guided Vehicles (AGV).
If you don't know about AGV and the challenge, you can quickly found a good explanation using Google search.

The main reason which urges me to write this post is that there is a limited number of tutorials you can found online on how to use TensorFlow lite C++ even the homepage of the TensorFlow package.
I hope this post can save you some time.

In this post, I will show you:
- How to set up a necessary environment to deploy your neural network inference on a small, low computing power board (like this RB5).
 It would also be useful for one who wants to use TensorFlow lite on any other boards like Jetson Nano (TX2/Xavier of course) or Raspberry Pi.
- Especially, I will use C++, not Python. Though Python is much easier (there are some Python scripts in the GitHub repo I will share with you in this post), C++ is the language you MUST use on a commercial product as it guarantees you better performance. Also, if you want to enable GPU/Hexagon DSP acceleration, you need to use C++. Python only allows us to use GPU and Google's homemade TPU inference.
- Moreover, I will walk you through the needed steps with an explanation so that you can understand the reason behind it, not just throw the code to you.
- And most important, I wrapped my code into a class so that you can use it on your project instantly. Here, the class is a header only. It means that you can integrate it into your work without any modification in compile it.

## Setup environment

Assume that you have installed the OS into your board and can boot into it, will need to do the below steps:

- [Install tensorflow](https://www.tensorflow.org/lite/guide/build_arm64) into your board and your host computer. You should have tensorflow on your computer to compile needed libraries that will be used by tensorflow lite. Do it on the board is not a good choice although you definitely can.

```bash
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
# Use the latest
#git branch r2.1 remotes/origin/r2.1
#git checkout r2.1
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_aarch64_lib.sh

#./tensorflow/lite/tools/make/build_generic_aarch64_lib.sh
```
If you are using an RPi, you should run the sh file tailored for it (which is placed in `tensorflow/lite/tools/make/`).


- [Install opencv](https://developer.qualcomm.com/project/object-detection-tensorflow-lite) onto the board. Pay attention to the comments I place above each command!!!

```bash
sudo apt-get install build-essential curl unzip
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
sudo apt-get install libjpeg-dev libpng-dev
sudo apt-get install python-numpy libxkbcommon-dev 
sudo apt install -y g++ wget unzip
# If you use wayland to display opencv output
sudo apt-get install libwayland-client0 libwayland-dev

# Go to your working directory
# You dont need to strictly follow the below step
mkdir /home/src
cd /home/src

# If you use wayland
git clone https://github.com/pfpacket/opencv-wayland.git
cd opencv-wayland/
# Or you use lastest opencv
# Download and unpack sources
wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/master.zip
unzip opencv.zip
unzip opencv_contrib.zip

# Create build directory and switch into it
mkdir -p build && cd build

# Note: You can modify the flags as you want
# Build opencv (latest version)
cmake -D CMAKE_BUILD_TYPE=Debug -D CMAKE_INSTALL_PREFIX=/usr/local -DWITH_IPP=OFF -DWITH_WAYLAND=ON -DWITH_GTK=OFF -DWITH_GSTREAMER=ON -DBUILD_opencv_python3=yes ..

make -j7 && make install

# Build opencv (latest version)
cmake -DCMAKE_INSTALL_PREFIX=/usr/local -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-master/modules -DWITH_IPP=OFF -DWITH_GSTREAMER=ON -DWITH_GTK=OFF -DWITH_WAYLAND=ON -DBUILD_opencv_python3=yes ..

make -j7 && make install
```
the library is installed to `/usr/local/include/opencv/`


- Extra: Install freetype2, weston/wayland (in my case)
```bash
sudo apt-get install -y freetype2-demos
sudo apt-get update && sudo apt-get install weston xwayland
```
**Note**: You don't need to install reetype2 and weston/wayland if you are working on Jetson or RPi boards.

## Compile needed libraries

- Build needed libraries on the host machine
```
bazel build --config=elinux_aarch64 -c opt --copt -DMESA_EGL_NO_X11_HEADERS --copt -DEGL_NO_X11 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so

bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/delegates/hexagon:hexagon_delegate_kernel
```
You will need to build multiple libraries as tensorflow lite separated into various components.
Here, I have compiled and placed the libraries into `WORKING_DIR/tflite/` so that you can use them right away. 

If you get into compiling problem due to missing a library, you need to find and compile the corresponding one. E.g,

- If you get this error:
```cpp
[100%] Linking CXX executable webcam-detector
../tflite/libhexagon_delegate.so: undefined reference to `vtable for tflite::HexagonDelegateKernel'
```
- Copy the function `HexagonDelegateKernel()` into your `main.cxx`. Right click -> Go to Definition (using VSCode IDE in my case). You will find that this function is defined in `tensorflow/tensorflow/lite/delegates/hexagon/hexagon_delegate_kernel.h`. You need to open the `BUILD` file in the same repo (`tensorflow/tensorflow/lite/delegates/hexagon/BUILD`) and find the `cc_library` that linked with the file and compile it.
```bash
bazel build --config=elinux_aarch64 -c opt //tensorflow/lite/delegates/hexagon:hexagon_delegate_kernel
```
- Structure of the command:
```bash
bazel build --config=elinux_aarch64 -c opt TO_REPO:LIBRARY_NAME
```

- Clone this repo, compile and run
```bash
git clone https://github.com/DoanNguyenTrong/object-detection-tflite-cpp.git
cd object-detection-tflite-cpp/
./compile.sh
./object-detector 0 dev/video0
```

## API explanation

- Initiallize: 
```cpp
ObjectDetector detector(modelfile, labelfile);
detector.readLabels();
detector.readModel();
detector.configure(4, delegate_option);
```
`modelfile` and `labelfile` are two strings that point to a delegated model and a label file. `4` is the number of threads will be used and `delegate_option` defines which hardware will be used for acceleration(`0`-GPU and `1`-Hexagon DSP);
- Inference and extract objects using NMS method

```cpp
int state = detector.inference(frame);
if (state != 0){
    exit(0);
}

std::vector<Object> objects = detector.extractObjects(0.3f, 0.5f);
```
`0.3f` is the score threshold and `0.5f` is the NMS's threshold.
- For more information, please take a look into my code.


## Reference/Useful links

- RB5 object detector with c++,flite, CPU
https://developer.qualcomm.com/project/object-detection-tensorflow-lite

- tflite, GPU
https://developer.qualcomm.com/blog/tensorflow-lite-inference-edge

- Hexagon delegate
https://www.tensorflow.org/lite/performance/hexagon_delegate

- User guide
https://www.devever.net/~hl/f/80-VB419-108_Hexagon_DSP_User_Guide.pdf

- https://developer.qualcomm.com/qualcomm-robotics-rb5-kit/software-reference-manual/machine-learning/tensorflow

- ROS 1, hexagon, tflite
https://github.com/quic/sample-apps-for-Qualcomm-Robotics-RB5-platform/tree/master/HexgonSDK-Image-classification

- Gstreamer RB5 extensions: https://source.codeaurora.org/quic/le/platform/vendor/qcom-opensource/gst-plugins-qti-oss/tree/gst-plugin-mle?h=LE.UM.4.4.1.r2-01600-QRB5165.0
