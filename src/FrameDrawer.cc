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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame()
{
    if (mvmatId == lastdraw_id){
      return lastdraw_mat;
    }
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<bool> vbVO, vbMap;// Tracked MapPoints in current frame
    vector<int> vbLabel;
    vector<bool> vbUselessPoint; // Useless MapPoints in current frame(outliers)
    int state; // Tracking state



    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
            vbLabel = mvbLabel;
            vbUselessPoint = mvbUselessPoint;
            //codes for storing infomation about unmatched feature points,***zh
          //  ifstream f;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
            vbLabel = mvbLabel;
            vbUselessPoint = mvbUselessPoint;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,CV_GRAY2BGR);

    // Now draw the good and bad descriptors
    // *** comment by ZH
    /*
    int nGoodDescriptor = mvGoodDescriptor.size();
    for(int i=0; i<nGoodDescriptor; i++){
        cv::Point2f pt1,pt2;
        pt1.x=mvGoodDescriptor[i].pt.x-mvGoodDescriptorRadius[i];
        pt1.y=mvGoodDescriptor[i].pt.y-mvGoodDescriptorRadius[i];
        pt2.x=mvGoodDescriptor[i].pt.x+mvGoodDescriptorRadius[i];
        pt2.y=mvGoodDescriptor[i].pt.y+mvGoodDescriptorRadius[i];
        cv::rectangle(im,pt1,pt2,cv::Scalar(255,255,0));  //蓝绿色，good descriptors, 半径每个不一样
        cv::circle(im,mvGoodDescriptor[i].pt,1,cv::Scalar(255,255,0),-1);
    }
    int nBadDescriptor = mvBadDescriptor.size();
    for(int i=0; i<nBadDescriptor; i++){
        cv::Point2f pt1,pt2;
        pt1.x=mvBadDescriptor[i].pt.x-mvBadDescriptorRadius[i];
        pt1.y=mvBadDescriptor[i].pt.y-mvBadDescriptorRadius[i];
        pt2.x=mvBadDescriptor[i].pt.x+mvBadDescriptorRadius[i];
        pt2.y=mvBadDescriptor[i].pt.y+mvBadDescriptorRadius[i];
        cv::rectangle(im,pt1,pt2,cv::Scalar(0,0,128));   // 红色，bad descriptors
        cv::circle(im,mvBadDescriptor[i].pt,1,cv::Scalar(0,0,128),-1);
    }
    */

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }
    }
    else if(state==Tracking::OK) //TRACKING
    {   //cout << "line 130 into good drawing " << endl;
        mnTracked=0;
        mnTrackedVO=0;
        mnUselessPoint=0; // For debug use
        mnDiscardedPoint=0; // For debug use
        const float r = 5;
        const int n = vCurrentKeys.size();
        int temp[8] =  {1,1,1,1,1,1,1,1};
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                if(vbMap[i] != 0)
                {

                  if(vbLabel[i]== -1){//background 红色
                    //cv::rectangle(im,pt1,pt2,cv::Scalar(0,0,255));
                    //cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,0,255),-1);
                    //if (temp[0] == 0){
                    //cv::putText(im,"This is background",vCurrentKeys[i].pt,1,1.0,cv::Scalar(0,255,0),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    //temp[0] = 1;
                    //}
                  }
                  if(vbLabel[i]== 0){//chair绿色
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,255,0),-1);
                    //if (temp[0] == 0){
                    //cv::putText(im,"This is chair",vCurrentKeys[i].pt,1,1.0,cv::Scalar(0,255,0),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    //temp[0] = 1;
                    //}
                  }
                  if(vbLabel[i]== 1){
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,0,255));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,0,255),-1);
                    //if (temp[0] == 0){
                    //cv::putText(im,"This is PC1",vCurrentKeys[i].pt,1,1.0,cv::Scalar(0,255,0),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    //temp[0] = 1;
                    //}
                  }
                  if(vbLabel[i]== 2){//laptop,红绿
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,128,128));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,128,128),-1);
                    if (temp[2] == 0){
                    cv::putText(im,"This is laptop",vCurrentKeys[i].pt,1,1.0,cv::Scalar(0,128,0),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    temp[2] = 1;
                    }
                  }
                  if(vbLabel[i]== 3){//book1,绿色
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,0),-1);
                    if (temp[3] == 0){
                    cv::putText(im,"This is book1",vCurrentKeys[i].pt,1,1.0,cv::Scalar(0,255,0),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    temp[3] = 1;
                    }
                  }
                  if(vbLabel[i]== 4){//bottle1,蓝红
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,255));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,255),-1);
                    if (temp[4] == 0){
                    cv::putText(im,"This is bottle1",vCurrentKeys[i].pt,1,1.0,cv::Scalar(255,0,255),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    temp[4] = 1;
                  }
                  }
                  if(vbLabel[i]== 5){//键盘，蓝色
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,0),-1);
                    if (temp[5] == 0){
                    cv::putText(im,"This is keyboard",vCurrentKeys[i].pt,1,1.0,cv::Scalar(0,0,255),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    temp[5] = 1;
                  }
                  }
                  if(vbLabel[i]== 6){//book2,蓝绿
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,255));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,255,255),-1);
                    if (temp[6] == 0){
                    cv::putText(im,"This is book2",vCurrentKeys[i].pt,1,1.0,cv::Scalar(255,0,255),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    temp[6] = 1;
                  }
                  }
                  if(vbLabel[i]== 7){
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,255));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,255),-1);
                  }
                  if(vbLabel[i]== 8){
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,255,0),-1);
                  }
                  if(vbLabel[i]== 9){
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,255,0),-1);
                  }
                  if(vbLabel[i]== 10){
                  //  cv::rectangle(im,pt1,pt2,cv::Scalar(64,64,64));
                  //  cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(64,64,64),-1);
                  }
                  if(vbLabel[i]== 11){//蓝
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,255,0),-1);
                    if (temp[11] == 0){
                    cv::putText(im,"This is bottle2",vCurrentKeys[i].pt,1,1.0,cv::Scalar(255,255,0),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    temp[11] = 1;}
                  }
                  if(vbLabel[i]== 12){
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,128,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,128,0),-1);
                  }
                  if(vbLabel[i]== 13){
                  }
                  if(vbLabel[i]== 14){//黑色
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,255,255));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,255,255),-1);
                    if (temp[6] == 0){
                    cv::putText(im,"This is cellphone",vCurrentKeys[i].pt,1,1.0,cv::Scalar(255,0,0),2);//参数一目标图像，参数二文本，参数三文本位置，参数四字体类型， 参数五字体大小，参数六字体颜色，参数七文本厚度
                    temp[6] = 1;}
                  }
                  if(vbLabel[i]== 15){
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,0),-1);
                  }
                  if(vbLabel[i]== 16){
                  }
                  if(vbLabel[i]== 17){

                  }
                  if(vbLabel[i]== 18){
                  }
                //std::cout << "vbmap[i]" << vbMap[i] << std::endl;
                    //cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0)); //BGR
                    //cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(1,1,1));//nothing showed, 这里就没东西了,啊哈，再一次实验还是没东西
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(1,1,1),-1);
                    mnTrackedVO++;
                }
            }
            else{
                cv::circle(im,vCurrentKeys[i].pt,0,cv::Scalar(0,0,255),-1); //BGR,red circle,this is useful,没有被匹配成功的点，实际上挺多的，
                mnUselessPoint++;
            }
            if(mvbDiscardedPoint[i]){ // For debug use
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;
                //cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,255));  //zise, purple
                //cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,255),-1);
                mnDiscardedPoint++;
            }
        }
    }
    else if(mState==Tracking::LOST){
        mnTracked=0;
        mnTrackedVO=0;
        mnUselessPoint=0; // For debug use
        mnDiscardedPoint=0; // For debug use
        const float r = 5;
        const int n = vCurrentKeys.size();

        // Now draw the good and bad descriptors
        int nGoodDescriptor = mvGoodDescriptor.size();
        for(int i=0; i<nGoodDescriptor; i++){
            cv::Point2f pt1,pt2;
            pt1.x=mvGoodDescriptor[i].pt.x-mvGoodDescriptorRadius[i];
            pt1.y=mvGoodDescriptor[i].pt.y-mvGoodDescriptorRadius[i];
            pt2.x=mvGoodDescriptor[i].pt.x+mvGoodDescriptorRadius[i];
            pt2.y=mvGoodDescriptor[i].pt.y+mvGoodDescriptorRadius[i];
            cv::rectangle(im,pt1,pt2,cv::Scalar(255,255,0));
            cv::circle(im,mvGoodDescriptor[i].pt,1,cv::Scalar(255,255,0),-1);
        }
        int nBadDescriptor = mvBadDescriptor.size();
        for(int i=0; i<nBadDescriptor; i++){
            cv::Point2f pt1,pt2;
            pt1.x=mvBadDescriptor[i].pt.x-mvBadDescriptorRadius[i];
            pt1.y=mvBadDescriptor[i].pt.y-mvBadDescriptorRadius[i];
            pt2.x=mvBadDescriptor[i].pt.x+mvBadDescriptorRadius[i];
            pt2.y=mvBadDescriptor[i].pt.y+mvBadDescriptorRadius[i];
            cv::rectangle(im,pt1,pt2,cv::Scalar(0,0,128));
            cv::circle(im,mvBadDescriptor[i].pt,1,cv::Scalar(0,0,128),-1);
        }

        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(0,255,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(0,255,0),-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0));
                    cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
            else{
                cv::circle(im,vCurrentKeys[i].pt,0,cv::Scalar(0,0,255),-1);
                mnUselessPoint++;
            }
            if(mvbDiscardedPoint[i]){ // For debug use
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;
                cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,255));
                cv::circle(im,vCurrentKeys[i].pt,1,cv::Scalar(255,0,255),-1);
                mnDiscardedPoint++;
            }
        }

    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);
    imWithInfo.copyTo(lastdraw_mat);//***zzh
    lastdraw_id = mvmatId;//***zzh
    return imWithInfo;
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
        if(mnUselessPoint>0)
            s << ", # of outliers: " << mnUselessPoint;
        if(mnDiscardedPoint>=0)
            s << ", # of discared matches: " << mnDiscardedPoint;
    }
    else if(nState==Tracking::LOST)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
        if(mnUselessPoint>0)
            s << ", # of outliers: " << mnUselessPoint;
        if(mnDiscardedPoint>=0)
            s << ", # of discared matches: " << mnDiscardedPoint;

        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

}

void FrameDrawer::Update(Tracking *pTracker)
{
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
    N = mvCurrentKeys.size();
    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mvbLabel = vector<int>(N,false); // ***zh
    mvbUselessPoint = vector<bool>(N,false);
    mvbDiscardedPoint = vector<bool>(N,false); // For debug use
    mbOnlyTracking = pTracker->mbOnlyTracking;
    mvBadDescriptor = pTracker->mCurrentFrame.mvBadDescriptor;  // For debug use, I'm trying to draw those points on the FrameDrawer that does not fit the DescriptorDistance requirement.
    mvBadDescriptorRadius = pTracker->mCurrentFrame.mvBadDescriptorRadius;  // For debug use, I'm trying to draw those points on the FrameDrawer that does not fit the DescriptorDistance requirement.
    mvGoodDescriptor = pTracker->mCurrentFrame.mvGoodDescriptor;  // For debug use.
    mvGoodDescriptorRadius = pTracker->mCurrentFrame.mvGoodDescriptorRadius;  // For debug use.
    mvmatId = pTracker->mCurrentFrame.matId;//***zh
    mvframe = pTracker->mCurrentFrame;//***zh


    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
        for(int i=0;i<N;i++) // For debug use
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                mvbUselessPoint[i]=true;
            }
        }
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;
                    //***zh
                    if(pMP->label != ""){
                      if (pMP->label == "background")
                      mvbLabel[i] = -1;
                      if (pMP->label == "fridge")
                      mvbLabel[i] = 0;
                      if (pMP->label == "yellow bag")
                      mvbLabel[i] = 1;
                      if (pMP->label == "water")
                      mvbLabel[i] = 2;
                      if (pMP->label == "printer")
                      mvbLabel[i] = 3;
                      //if (pMP->label == "keyboard")
                      //mvbLabel[i] = 5;
                      //if (pMP->label == "book2")
                      //mvbLabel[i] = 6;//7;
                      //if (pMP->label == "bottle2")
                      //mvbLabel[i] = 11;//8;
                      //if (pMP->label == "cellphone")
                      //mvbLabel[i] = 14;//9;
                      //if (pMP->label == "diningtable1")
                      //mvbLabel[i] = 0;//10;
                      //if (pMP->label == "book1")
                      //mvbLabel[i] = 0;//11;
                      //if (pMP->label == "book2")
                      //mvbLabel[i] = 12;//12;
                      //if (pMP->label == "book3")
                      //mvbLabel[i] = 13;//13;
                      //if (pMP->label == "book4")
                      //mvbLabel[i] = 14;//14;
                      //if (pMP->label == "keyboard1")
                      //mvbLabel[i] = 15;//15;
                      //if (pMP->label == "chair2")
                      //mvbLabel[i] = 16;//16;
                      //if (pMP->label == "chair3")
                      //mvbLabel[i] = 16;//17;
                      //if (pMP->label == "cup1")
                      //mvbLabel[i] = 18;//18;
                    }else{
                      mvbLabel[i] = -2;
                    }

                }
                else{
                    mvbUselessPoint[i]=true;  // For debug use
                }
                if(pTracker->mCurrentFrame.mvbDiscarded[i]){ // For debug use
                    mvbDiscardedPoint[i]=true;
                }
            }
        }
    }
    else{
        for(int i=0;i<N;i++)
        {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(pTracker->mCurrentFrame.mvbOutlier[i])  // For debug use
                {
                    mvbUselessPoint[i]=true;
                }

                if(pTracker->mCurrentFrame.mvbDiscarded[i]){ // For debug use
                    mvbDiscardedPoint[i]=true;
                }
            }
        }
    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
}

} //namespace ORB_SLAM
