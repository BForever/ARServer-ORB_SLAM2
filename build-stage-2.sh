cd Thirdparty/Pangolin
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../..

echo "Configuring and building Thirdparty/DBoW2 ..."
cd Thirdparty/DBoW2
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

cd ../../..

echo "Configuring and building Thirdparty/g2o ..."
cd Thirdparty/g2o
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
echo "Configuring and building protobuf ..."
