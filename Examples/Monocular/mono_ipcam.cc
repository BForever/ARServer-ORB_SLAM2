/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<ctime>
#include "Osmap.h"
#include "TwoLineIn3D.h"
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<System.h>
#include "ARConnection.pb.h"
#include "ARConnection.grpc.pb.h"
#include "Examples/Monocular/proto/Transfer.h"
#include <grpcpp/grpcpp.h>
#include <mutex>

//using grpc::ServerBuilder ;

using namespace std;
using namespace cv;

float fx;
float fy;
float cx;
float cy;
float F = 10;
string server_address = "0.0.0.0:50051";

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << endl << "Usage: ./mono_tum path_to_vocabulary path_to_settings" << endl;
        return 1;
    }
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << argv[2] << endl;
        exit(-1);
    }
    fx = fsSettings["Camera.fx"];
    fy = fsSettings["Camera.fy"];
    cx = fsSettings["Camera.cx"];
    cy = fsSettings["Camera.cy"];
    bool finish = false;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], \
    ORB_SLAM2::System::MONOCULAR, false);

    mutex bufferMutex;

    Transfer service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    unique_ptr<grpc::Server> server(builder.BuildAndStart());
    cout << "Server listening on " << server_address << endl;

    std::thread serverthread([&]() {
        server->Wait();
    });

    VideoCapture cap;

    chrono::steady_clock::time_point track_time_start;
    int track_count = 0;
//    std::thread runthread([&]() {  // Start in new thread
        Mat C_Tcw;

        struct timespec tn;
        double tframe;

        cout << endl << "-------" << endl;
        cout << "Start processing sequence ..." << endl;
        int inited = 0;
        track_time_start = chrono::steady_clock::now();
        while (!finish) {
            if(service.reset){
                SLAM.Reset();
                service.reset= false;
            }
            clock_gettime(CLOCK_REALTIME, &tn);
            tframe = double(tn.tv_sec) + double(tn.tv_nsec) / 1e9;
            if (service.newFrame) {

//                imshow("im",frame);
                {
                    unique_lock<mutex> lock(service.newFrameMutex);
//                    service.updated = true;
                    service.newFrame = false;
                }
                C_Tcw = SLAM.TrackMonocular(service.frame, tframe);
                service.SetCameraPose(C_Tcw);
                {
                    unique_lock<mutex> lock(service.updatedMutex);
                    service.updated = true;
//                    service.newFrame = false;
                }
                track_count += 1;
                auto time_used = chrono::duration_cast<chrono::duration<double>>(
                        chrono::steady_clock::now() - track_time_start);
                if (time_used.count()>1) {
                    double frame_rate = track_count / time_used.count();
                    cout << frame_rate << endl;
                    track_time_start = chrono::steady_clock::now();
                    track_count = 0;
                }
            }
        }
//    });
    // Start the visualization thread
//    SLAM.StartViewer();

    finish = true;
    server->Shutdown();
//    runthread.join();
//    bufferthread.join();
    serverthread.join();
    cout << "Tracking thread joined..." << endl;
    // Stop all threads
    SLAM.Shutdown();
    return 0;
}
//    chrono::steady_clock::time_point receive_time_start;
//    int receive_count = 0;
//    std::thread bufferthread([&]() {
//        int key = 0;
//        while (!finish) {
//            if (cap.open(string(argv[3]))) {
//                cout << "video stream reconnected!" << endl;
////                cap.set(CV_CAP_PROP_FPS, 30);
//                cap.set(CV_CAP_PROP_BUFFERSIZE, 2);
//
//                while (true) {
//                    bool has_image = cap.grab();
//                    if (has_image){
////                        receive_time_start = chrono::steady_clock::now();
//                        cap.retrieve(frame);
////                        auto time_used = chrono::duration_cast<chrono::microseconds>(
////                            chrono::steady_clock::now() - receive_time_start);
////                        cout<<"decode one frame: "<<time_used.count()<<"."<<endl;
//                    }
//
////                    receive_count += 1;
////
////                    if (time_used.count()>1) {
////                        double frame_rate = receive_count / time_used.count();
////                        cout << frame_rate << endl;
////                        receive_time_start = chrono::steady_clock::now();
////                        receive_count = 0;
////                    }
//                    if (!has_image) {
//                        key++;
//                        if (key >= 25) {
//                            key = 0;
//                            break;
//                        } else {
//                            continue;
//                        }
//                    } else {
//                        key = 0;
//                    }
//                }
//            }
//            cout << "video stream disconnected! sleep 1s." << endl;
//            sleep(1);
//        }
//    });