#!/bin/bash
set -e

# ref: https://blog.csdn.net/Djsnxbjans/article/details/131904947
# 1. 安装依赖项
apt-get update && apt-get install -y \
    autoconf \
    automake \
    libtool \
    curl \
    make \
    g++ \
    unzip

# 2. 下载源码并编译安装opencv-4.5.4
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.4/protobuf-cpp-3.11.4.tar.gz
tar -xzvf protobuf-cpp-3.11.4.tar.gz
rm protobuf-cpp-3.11.4.tar.gz
cd protobuf-3.11.4
./configure --prefix=/usr/local/protobuf
make -j4
make check -j4
make install
cd ..

# 3. 设置环境变量
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/protobuf/lib' >> ~/.bashrc
echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/protobuf/lib' >> ~/.bashrc
echo 'export PATH=$PATH:/usr/local/protobuf/bin' >> ~/.bashrc
echo 'export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/protobuf/include' >> ~/.bashrc
echo 'export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/protobuf/include' >> ~/.bashrc
echo 'PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/protobuf/lib/pkgconfig' >> ~/.bashrc
echo 'export PKG_CONFIG_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. 清理
apt-get clean
rm protobuf-cpp-3.11.4.tar.gz
rm -rf protobuf-3.11.4
rm -rf /var/lib/apt/lists/*
