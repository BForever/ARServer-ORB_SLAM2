apt-get install unzip

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

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8

echo "Configuring and building Thirdparty/opencv ..."
cd ../..
unzip opencv-3.4.10.zip
cd opencv-3.4.10
mkdir build
cd build
cmake ..
make -j8
make install


cd ../../../



echo "Configuring and building protobuf ..."
cd grpc/third_party/protobuf/
make -j8        #从Makefile读取指令，然后编译
make install
sudo ldconfig       #更新共享库缓存
which protoc        #查看软件的安装位置
protoc --version

echo "Configuring and building grpc ..."
cd ../..
rm -r build
mkdir build
cd build
cmake -DgRPC_INSTALL=ON -DCARES_SHARED=ON -DgRPC_ZLIB_PROVIDER=package \
-DgRPC_ABSL_PROVIDER=package -DgRPC_CARES_PROVIDER=package \
-DgRPC_PROTOBUF_PROVIDER=package -DgRPC_SSL_PROVIDER=package ..
make -j8
make install
cd ../..

echo "Configuring and building ORB_SLAM2 ..."
rm -r build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
