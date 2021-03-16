#include"Object.h"

namespace ORB_SLAM2
{
/*
Object::Object(string name, MapPoint* mp){
  objname = name;
  mpofobj.push_back(mp);
}
*/
Object::Object(){

}



void Object::addmp(MapPoint* mp){
  //std::cout << "here 1" << std::endl;
  //unique_lock<mutex> lock(mMutexObj);
  //std::cout << "here 2" << std::endl;
  mpofobj.push_back(mp);
  //mpofobj.insert(mp);
  //std::cout << "here 3" << std::endl;
}

//这个删除函数很可能有问题，参考orbmatcher里面的写法
void Object::deletemp(MapPoint* mp){
  //unique_lock<mutex> lock(mMutexObj);
  //for(vector<MapPoint*>::const_iterator vit=mpofobj.begin(), vend=mpofobj.end(); vit!=vend; vit++){
  //  if (*vit == mp){
  //    mpofobj.erase(vit);
  //  }
  //}
}

int Object::mpnums(){
  //unique_lock<mutex> lock(mMutexObj);
  return mpofobj.size();
}






}//namespace ORB_SLAM
