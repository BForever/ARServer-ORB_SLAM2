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
#include "use.h"
using namespace std;
using namespace cv;
using namespace torch::indexing;

long time_in_ms(){
    struct timeval t;
    gettimeofday(&t, NULL);
    long time_ms = ((long)t.tv_sec)*1000+(long)t.tv_usec/1000;
    return time_ms;
}

at::Tensor box_area(at::Tensor box)
{
    return (box.index({2}) - box.index({0})) * (box.index({3}) - box.index({1}));
}

at::Tensor box_iou(at::Tensor box1, at::Tensor box2)
{
    at::Tensor area1 = box_area(box1.t());
    at::Tensor area2 = box_area(box2.t());

    at::Tensor inter = ( torch::min( box1.index({Slice(), {None}, Slice(2,None)}), box2.index({Slice(), Slice(2,None)}))
                         - torch::max( box1.index({Slice(), {None}, Slice(None,2)}), box2.index({Slice(), Slice(None,2)})) ).clamp(0).prod(2);
    return inter / (area1.index({Slice(), {None}}) + area2 - inter);

}

//Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
at::Tensor xywh2xyxy(at::Tensor x)
{
    at::Tensor y = torch::zeros_like(x);
    y.index({Slice(), 0}) = x.index({Slice(), 0}) - x.index({Slice(), 2}) / 2; // # top left x
    y.index({Slice(), 1}) = x.index({Slice(), 1}) - x.index({Slice(), 3}) / 2;
    y.index({Slice(), 2}) = x.index({Slice(), 0}) + x.index({Slice(), 2}) / 2;
    y.index({Slice(), 3}) = x.index({Slice(), 1}) + x.index({Slice(), 3}) / 2;

    return y;
}



at::Tensor non_max_suppression(at::Tensor pred, std::vector<string> labels, float conf_thres = 0.1, float iou_thres = 0.6,
                               bool merge=false,  bool agnostic=false)
{
    int nc = pred.sizes()[1] - 5;
    at::Tensor xc = pred.index({Slice(None), 4}) > conf_thres; // # candidates,

    int min_wh = 2;
    int max_wh = 4096;
    int max_det = 300;
    float time_limit = 10.0;
    bool redundant = true;  //require redundant detections
    bool multi_label = true;
    at::Tensor output;

    at::Tensor x;
    x = pred.index({xc});

    try{
        at::Tensor temp  = x.index({0});
    }catch(...){
        cout << "no objects, line 138 " << endl;
        at::Tensor temp;
        return temp;
    }

    x.index({Slice(),Slice(5,None)}) *= x.index({Slice(), Slice(4,5)});
    at::Tensor box = xywh2xyxy(x.index({Slice(), Slice(None,4)}));
    if(true){ //here mulit label in python code
        //i, j = (x[:, 5:] > conf_thres).nonzero().t()
        auto temp = (x.index({Slice(), Slice(5,None)}) > conf_thres).nonzero().t();
        at::Tensor i = temp[0];
        at::Tensor j = temp[1];
        x = torch::cat({  box.index({i}), x.index({i,j+5,{None}}), j.index({Slice(),{None}}).toType(torch::kFloat)  }, 1);
    }

    try{
        at::Tensor temp  = x.index({0});
    }catch(...){
        cout << "no objects, line 187 " << endl;
        at::Tensor temp;
        return temp;
    }

    int n = x.sizes()[0];
    at::Tensor c = x.index({Slice(), Slice(5,6)}) * max_wh;
    at::Tensor boxes = x.index({Slice(), Slice(None, 4)}) + c;
    at::Tensor scores = x.index({Slice(),4});

    at::Tensor i = nms(boxes, scores, iou_thres);

    if(i.sizes()[0] > max_det){
        i = i.index({Slice(0,max_det)});
    }

    if(merge){
        if( n > 1 && n < 3000){
            at::Tensor iou = box_iou(boxes.index({i}), boxes) > iou_thres;
            at::Tensor weights = iou * scores.index({ {None} });
            at::Tensor temp1 = torch::mm(weights, x.index({Slice(), Slice(None, 4)})).toType(torch::kFloat);
            at::Tensor temp2 = weights.sum(1, true);
            at::Tensor tempres = temp1 / temp2;
            x.index_put_({i, Slice(None, 4)}, tempres);
            if(redundant){
                i = i.index({iou.sum(1) > 1});
            }
        }
    }
    output = x.index({i});
    return output;
}


cv::Mat letterbox(Mat img, int new_height = 640, int new_width = 640, Scalar color = (114,114,114), bool autos = true, bool scaleFill=false, bool scaleup=true){
    int width = img.cols;
    int height = img.rows;
    float rh = float(new_height) / height;
    float rw = float(new_width) / width;
    float ratio;
    if(rw < rh){
        ratio = rw;}
    else{
        ratio = rh;}
    if (!scaleup){
        if(ratio >= 1.0){
            ratio = 1.0;
        }
    }
    int new_unpad_h = int(round(height * ratio));
    int new_unpad_w = int(round(width * ratio));
    int dh = new_height - new_unpad_h;
    int dw = new_width - new_unpad_w;

    if(autos){
        dw = dw % 64;
        dh = dh % 64;
    }

    dw /= 2;
    dh /= 2;//默认被二整除吧
    if( height != new_height or width != new_width){
        resize(img, img, Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    }
    int top = int(round(dh - 0.1));
    int bottom = int(round(dh + 0.1));
    int left = int(round(dw - 0.1));
    int right = int(round(dw + 0.1));

    cv::copyMakeBorder(img, img, top, bottom, left, right, cv::BORDER_CONSTANT, Scalar(114,114,114));

    return img;
}
#endif //ORB_SLAM2_USE_H