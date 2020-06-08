FROM ubuntu:latest

COPY cmake_modules /arserver/cmake_modules
COPY Thirdparty /arserver/Thirdparty
COPY Vocabulary /arserver/Vocabulary

COPY CMakeLists.txt /arserver/CMakeLists.txt

RUN  sed -i s@/archive.ubuntu.com/@/ftp.sjtu.edu.cn/@g /etc/apt/sources.list
RUN  apt-get clean

ENV DEBIAN_FRONTEND=noninteractive

COPY build-download.sh /arserver/build-download.sh
RUN cd /arserver && sh build-download.sh
COPY build-stage-1.sh /arserver/build-stage-1.sh
RUN cd /arserver && sh build-stage-1.sh
COPY build-stage-2.sh /arserver/build-stage-2.sh
RUN cd /arserver && sh build-stage-2.sh
COPY build-stage-3.sh /arserver/build-stage-3.sh
RUN cd /arserver && sh build-stage-3.sh

COPY Examples /arserver/Examples
COPY include /arserver/include
COPY src /arserver/src
COPY build-stage-4.sh /arserver/build-stage-4.sh
RUN cd /arserver && sh build-stage-4.sh

ENTRYPOINT ["./arserver/Examples/Monocular/mono_ipcam", "/arserver/Vocabulary/ORBvoc.bin", "/arserver/Examples/Monocular/ipcam.yaml"]
