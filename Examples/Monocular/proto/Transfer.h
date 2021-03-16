//
// Created by 范宏昌 on 2020/4/5.
//

#ifndef ORB_SLAM2_TRANSFER_H
#define ORB_SLAM2_TRANSFER_H

#include "ARConnection.grpc.pb.h"
#include "ARConnection.pb.h"
#include "System.h"
#include <opencv2/core.hpp>
using namespace grpc;
using namespace ARConnection;
using namespace cv;

class Transfer final: public ARConnectionService::Service {
public:
//    ORB_SLAM3::System* mSystem;
    Mat frame;
    double tframe;
    Mat Twc;
    mutex updatedMutex,newFrameMutex,mMutexCamera;
    bool newFrame = false;
    bool updated = false;
    bool reset = false;

    Transfer();
//    Transfer(ORB_SLAM3::System* system);

    void SetCameraPose(const cv::Mat &Tcw);
    Status GetViewMatrix(ServerContext* context, const Request* request, MatrixBlock* response) override ;
    Status UploadImage(ServerContext* context, ServerReader< ImageBlock>* reader, Response* response) override;
    Status RequestReset(ServerContext* context, const Request* request, Response* response);
//    Status GetObjectList(ServerContext* context, const Request* request, ObjectInfoList* response);
    static int count;
    static clock_t last_time;
};


#endif //ORB_SLAM2_TRANSFER_H
