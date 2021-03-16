//
// Created by 范宏昌 on 2020/7/13.
//

#ifndef ORB_SLAM2_USE_H
#define ORB_SLAM2_USE_H
//yolov5s的调用
//删除冗余注释，优化代码,记录下时间消耗
//原始，load:129;pre_process:1;model:365;nms:2
//init_threads:365->313
//jit::设置，365->249
//似乎是jit的设置比较有用，两个同时设置后，也没有新的提升，和设置jit的性能一样
#include <memory>
#include <torch/script.h>
#include "torchvision/vision.h"
#include "torch/torch.h"
#include "torchvision/nms.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <time.h>
#include "sys/time.h"
#include <unistd.h>
#include <mutex>

using namespace std;
using namespace cv;
using namespace torch::indexing;

long time_in_ms();

at::Tensor box_area(at::Tensor box);

at::Tensor box_iou(at::Tensor box1, at::Tensor box2);

//Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
at::Tensor xywh2xyxy(at::Tensor x);


at::Tensor non_max_suppression(at::Tensor pred, std::vector<string> labels, float conf_thres = 0.1, float iou_thres = 0.6,
                               bool merge=false,  bool agnostic=false);

cv::Mat letterbox(Mat img, int new_height = 640, int new_width = 640, Scalar color = (114,114,114), bool autos = true, bool scaleFill=false, bool scaleup=true);
#endif //ORB_SLAM2_USE_H
