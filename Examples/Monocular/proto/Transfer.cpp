//
// Created by 范宏昌 on 2020/4/5.
//

#include "Transfer.h"
#include <opencv2/core/core.hpp>
#include <ctime>

int Transfer::count = 0;
clock_t  Transfer::last_time = 0;

Transfer::Transfer() {
    Twc = Mat::eye(4, 4, CV_32F);
}

void Transfer::SetCurrentCameraPose(const cv::Mat &Tcw) {
    unique_lock<mutex> lock(mMutexCamera);
    if (!Tcw.empty()) {
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);
        {
            Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * Tcw.rowRange(0, 3).col(3);
        }

        Twc.at<float>(0, 0) = Rwc.at<float>(0, 0);
        Twc.at<float>(1, 0) = Rwc.at<float>(1, 0);
        Twc.at<float>(2, 0) = Rwc.at<float>(2, 0);
        Twc.at<float>(3, 0) = 0.0;

        Twc.at<float>(0, 1) = Rwc.at<float>(0, 1);
        Twc.at<float>(1, 1) = Rwc.at<float>(1, 1);
        Twc.at<float>(2, 1) = Rwc.at<float>(2, 1);
        Twc.at<float>(3, 1) = 0.0;

        Twc.at<float>(0, 2) = Rwc.at<float>(0, 2);
        Twc.at<float>(1, 2) = Rwc.at<float>(1, 2);
        Twc.at<float>(2, 2) = Rwc.at<float>(2, 2);
        Twc.at<float>(3, 2) = 0.0;

        Twc.at<float>(0, 3) = twc.at<float>(0);
        Twc.at<float>(1, 3) = twc.at<float>(1);
        Twc.at<float>(2, 3) = twc.at<float>(2);
        Twc.at<float>(3, 3) = 1.0;
        updated = true;
        this->mCondVar.notify_one();
    }
}

Status Transfer::GetViewMatrix(ServerContext *context, const Request *request, MatrixBlock *response) {
    unique_lock<mutex> lock(mMutexCamera);
    this->mCondVar.wait_for(lock,chrono::microseconds(15));
    if (updated) {
        response->set_r0c0(Twc.at<float>(0, 0));
        response->set_r0c1(Twc.at<float>(0, 1));
        response->set_r0c2(Twc.at<float>(0, 2));
        response->set_r0c3(Twc.at<float>(0, 3));
        response->set_r1c0(Twc.at<float>(1, 0));
        response->set_r1c1(Twc.at<float>(1, 1));
        response->set_r1c2(Twc.at<float>(1, 2));
        response->set_r1c3(Twc.at<float>(1, 3));
        response->set_r2c0(Twc.at<float>(2, 0));
        response->set_r2c1(Twc.at<float>(2, 1));
        response->set_r2c2(Twc.at<float>(2, 2));
        response->set_r2c3(Twc.at<float>(2, 3));
        response->set_r3c0(Twc.at<float>(3, 0));
        response->set_r3c1(Twc.at<float>(3, 1));
        response->set_r3c2(Twc.at<float>(3, 2));
        response->set_r3c3(Twc.at<float>(3, 3));
        response->set_update(true);
//        cout<<"success: transfering pose "<<Tcw<<endl;
    } else {
//        cout<<"error: pose not available!"<<endl;
        response->set_update(false);
    }

    return Status::OK;
}

Status Transfer::UploadImage(ServerContext *context, ServerReader<ImageBlock> *reader, Response *response) {
    auto *imageBlock = new ImageBlock();
    bool has_next = true;
    cout << "UploadImage connected." << endl;
    while (has_next) {
//        auto decode_time_start = chrono::steady_clock::now();
        has_next = reader->Read(imageBlock);
//        auto time_used = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - decode_time_start);
//        cout << "read one frame: " << time_used.count() / 1000. << " ms." << endl;

//        cout<<"image received, size: "<<imageBlock->size()<<endl;
//        decode_time_start = chrono::steady_clock::now();
        std::vector<uchar> data(imageBlock->image().begin(), imageBlock->image().end());
//        time_used = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - decode_time_start);
//        cout << "copy one frame: " << time_used.count() / 1000. << " ms." << endl;
//        cout<<"image copied, size: "<<data.size()<<endl;

//        decode_time_start = chrono::steady_clock::now();
        cv::imdecode(data, CV_LOAD_IMAGE_UNCHANGED, &frame);
        {
            unique_lock<mutex> lock(bufferMutex);
            updated = true;
        }
//        time_used = chrono::duration_cast<chrono::microseconds>(chrono::steady_clock::now() - decode_time_start);
//        cout << "decode one frame: " << time_used.count() / 1000. << " ms." << endl;


//        cout<<"image decoded, rows: "<<frame.rows<<"cols: "<<frame.cols<<endl;
//        imshow("test",frame);
//        cout<<"image showed"<<endl;
    }
    response->set_status("ok");
    return Status::OK;
}

Status Transfer::RequestReset(ServerContext *context, const Request *request, Response *response) {
    reset = true;
    response->set_status("ok");
    return Status::OK;
}
