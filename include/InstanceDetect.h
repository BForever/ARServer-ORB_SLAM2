#ifndef INSTANCEDETECT_H
#define INSTANCEDETECT_H

#include <memory>
#include <torch/script.h>
#include "torchvision/vision.h"
#include "torch/torch.h"
#include "torchvision/nms.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <time.h>
#include "sys/time.h"
#include <unistd.h>
#include <mutex>
#include <iostream>
#include "Map.h"


namespace ORB_SLAM2 {

class Map;

class InstanceDetect {
public:
    InstanceDetect(Map *pMap);

    void detect(cv::Mat *im);

    long time_in_ms();

    std::vector<string> labels;

    void Run();

    void InsertFrame(const cv::Mat &im);

    void RequestFinish();

    torch::jit::script::Module model;
    string modelpath;
    string labelpath;
protected:
    Map *mpMap;

    std::list<cv::Mat *> midFrameQueue;
    std::mutex mMutexInstanceDetectQueue;
public:
    bool CheckNewDetectFrames();

private:
    at::Tensor anchor_grid;
protected:

    void ResetIfRequested();

    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();

    void SetFinish();

    bool mbFinishRequested;
    bool mbFinished;
    std::mutex mMutexFinish;
};
}
#endif // INSTANCEDETECT_H
