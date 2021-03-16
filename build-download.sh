echo "Install dependency: cmake Eigen3 Pangolin "
apt-get update
apt-get install -y git cmake libeigen3-dev build-essential libgl1-mesa-dev libglew-dev unzip

echo "Configuring and building protobuf ..."
apt-get -y install build-essential autoconf libtool pkg-config

#cd Thirdparty
#git clone -b v1.28.1 https://github.com/grpc/grpc
#cd grpc
#git submodule update --init
#
#cd third_party/protobuf/
#git submodule update --init --recursive #确保克隆子模块，更新第三方源码
cd Thirdparty
unzip grpc.zip
cd grpc/third_party/protobuf
./autogen.sh   #生成配置脚本
./configure    #生成Makefile文件，为下一步的编译做准备，可以加上安装路径：--prefix=path ，默认路径为/usr/local/

cd ../../../..
echo "Uncompress vocabulary ..."
cd Vocabulary
unzip ORBvoc.bin.zip
cd ..

