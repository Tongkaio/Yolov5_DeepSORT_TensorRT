#!/bin/bash
set -e

# 1. 安装依赖项
apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    mlocate \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python-dev \
    python-numpy \
    libtbb2 \
    libeigen3-dev \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libcanberra-gtk-module \
    libdc1394-22-dev

# 2. 下载源码并编译安装opencv-4.5.4
wget https://github.com/opencv/opencv/archive/4.5.4.zip
unzip ./4.5.4.zip
cd opencv-4.5.4 && mkdir -p build && cd build
cmake -D CMAKE_INSTALL_PREFIX=/usr/local -D CMAKE_BUILD_TYPE=Release -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_ENABLE_NONFREE=True ..
make -j4
make install
cd ../../
rm -rf ./4.5.4.zip ./opencv-4.5.4

# 3. 设置环境变量
echo 'PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig' >> /etc/bash.bashrc
echo 'export PKG_CONFIG_PATH' >> /etc/bash.bashrc
updatedb
source /etc/bash.bashrc

# 4. 清理
apt-get clean
rm -rf /var/lib/apt/lists/*
