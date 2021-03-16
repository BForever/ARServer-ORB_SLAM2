cd Thirdparty/grpc/third_party/protobuf/
make -j8        #从Makefile读取指令，然后编译
make install
ldconfig       #更新共享库缓存
which protoc        #查看软件的安装位置
protoc --version
cd ../..

cd third_party/abseil-cpp
mkdir build
cd build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..
make -j8
make install
cd ../../..

cd third_party/cares/cares
mkdir build
cd build
cmake ..
make -j8
make install
cd ../../../..


echo "Configuring and building grpc ..."
rm -r build
mkdir build
cd build
cmake -DgRPC_INSTALL=ON -DCARES_SHARED=ON -DgRPC_ZLIB_PROVIDER=module \
-DgRPC_ABSL_PROVIDER=package -DgRPC_CARES_PROVIDER=package \
-DgRPC_PROTOBUF_PROVIDER=package -DgRPC_SSL_PROVIDER=module ..
make -j8
make install
