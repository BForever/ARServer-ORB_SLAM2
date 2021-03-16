#ifndef OBJECT_H
#define OBJECT_H

#include"Map.h"
#include "MapPoint.h"
#include <mutex>

namespace ORB_SLAM2
{

class MapPoint;

class Object
{
public:
  //Object(string name, MapPoint* mp);
  Object();
  string objname;
  std::vector<MapPoint*> mpofobj;
  //std::set<MapPoint*> mpofobj;
  std::vector<int> IndexInFrame; //pnp时候用，保存一个地图点在原本frame里面的index
  void addmp(MapPoint* mp);
  void deletemp(MapPoint* mp);
  //cv::Point3f obj_location();//返回object的专属location
  int mpnums();//返回有多少个点对应于这个object
  int inliner; //在pnp里统计多少点被计算为inliner
  int inview;//在当前帧里有多少理论上被看见
  int seen;//在当前帧里有多少实际上被看见
  float seenrate; //实际看见的占理论应见的比例。
  cv::Mat mWorldPos;//1*3,xyz
  //Object(const Object&) = delete;

protected:
  //std::mutex mMutexObj;

};

}//namespace orbslam2

#endif // OBJECT_H
