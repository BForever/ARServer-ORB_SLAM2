echo "Configuring and building ORB_SLAM2 ..."
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

