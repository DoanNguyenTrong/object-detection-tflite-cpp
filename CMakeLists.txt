cmake_minimum_required(VERSION 3.0)
project(webcam-detector)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2 -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lX11 -ldl -lrt")
set(CMAKE_BUILD_TYPE "Debug")

message(STATUS "optional:-std=c++17")



set(OpenCV_DIR "/root/doan/test_ws/opencv-master/build")

include_directories(/usr/include/freetype2/)
include_directories(/root/doan/test_ws/tensorflow/)
include_directories(/root/doan/test_ws/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include)


find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS " version: ${OpenCV_VERSION}")
message(STATUS " libraries: ${OpenCV_LIBS}")
message(STATUS " include path: ${OpenCV_INCLUDE_DIRS}")
include_directories( ${OpenCV_INCLUDE_DIRS} )

# tflite
ADD_LIBRARY(TFlite_LIB STATIC IMPORTED)
SET_TARGET_PROPERTIES(TFlite_LIB PROPERTIES IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libtensorflow-lite-latest.a)

# EGL
ADD_LIBRARY(EGL_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(EGL_LIB PROPERTIES IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/libEGL.so)
# GL
ADD_LIBRARY(GL_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(GL_LIB PROPERTIES IMPORTED_LOCATION /usr/lib/aarch64-linux-gnu/libGL.so)

# tflite GPU
ADD_LIBRARY(TFlite_GPU_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(TFlite_GPU_LIB PROPERTIES 
IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libtensorflowlite_gpu_delegate.so)
# tflite Hexagon
ADD_LIBRARY(TFlite_Hexagon_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(TFlite_Hexagon_LIB PROPERTIES 
IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libhexagon_delegate.so)

ADD_LIBRARY(TFlite_Hex_Ker_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(TFlite_Hex_Ker_LIB PROPERTIES 
IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libhexagon_delegate_kernel.so)

ADD_LIBRARY(Hexagon_Imp_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(Hexagon_Imp_LIB PROPERTIES 
IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libhexagon_implementation.so)

ADD_LIBRARY(Op_Builder_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(Op_Builder_LIB PROPERTIES 
IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libop_builder.so)

ADD_LIBRARY(Simple_Delegate_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(Simple_Delegate_LIB PROPERTIES 
IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libsimple_delegate.so)

ADD_LIBRARY(Utils_LIB SHARED IMPORTED )
SET_TARGET_PROPERTIES(Utils_LIB PROPERTIES 
IMPORTED_LOCATION /root/doan/test_ws/object-detection-rb5/tflite/libutils.so)


set(SOURCE_FILES src/main.cxx )
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} )

target_link_libraries(${PROJECT_NAME}
    freetype
    pthread
    dl
)

target_link_libraries(${PROJECT_NAME} TFlite_GPU_LIB TFlite_Hexagon_LIB TFlite_Hex_Ker_LIB Hexagon_Imp_LIB Op_Builder_LIB Simple_Delegate_LIB Utils_LIB)
target_link_libraries(${PROJECT_NAME} EGL_LIB GL_LIB)
target_link_libraries(${PROJECT_NAME} TFlite_LIB ${CMAKE_DL_LIBS} stdc++fs rt)