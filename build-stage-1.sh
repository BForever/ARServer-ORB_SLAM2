apt-get install unzip

echo "Configuring and building Thirdparty/opencv ..."
cd Thirdparty
unzip opencv-3.4.10.zip
cd opencv-3.4.10
mkdir build
cd build
cmake ..
make -j8
make install
