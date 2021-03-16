cmake_minimum_required(VERSION 2.8)
project(yolov5s)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_PREFIX_PATH "/home/fanhc/code/libtorch")
#set(OpenCV_DIR "/home/zherlock/opencv-2.4.9/build/")
find_package(OpenCV 3.0 QUIET)
find_package(TorchVision REQUIRED)
#include_directories("/home/fanhc/code/vision/include")
include_directories("/home/fanhc/code/libtorch/include/torch/csrc/api/include/")
include_directories("/home/fanhc/code/libtorch/include/")
#include_directories("/home/fanhc/code/anaconda3/envs/detectron2/include/python3.7m/")

find_package(Torch REQUIRED)

#add_library(${PROJECT_NAME} SHARED
#/home/zherlock/torchvision-abi/include/torchvision/nms.cpp
#)

add_library(yoloso useso.cpp)
target_link_libraries(yoloso
        ${OpenCV_LIBS}
        ${TORCH_LIBRARIES}
        TorchVision::TorchVision)
add_executable(yolosoe use.cpp)
target_link_libraries(yolosoe yoloso)

add_executable(yolo src.cpp)
target_link_libraries(yolo 
#${PROJECT_NAME}
${OpenCV_LIBS}
${TORCH_LIBRARIES}
        TorchVision::TorchVision)

#***zh
set(CMAKE_CXX_STANDARD 14)
add_definitions(-std=c++14)
