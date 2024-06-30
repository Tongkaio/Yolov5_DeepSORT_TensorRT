#!/bin/bash
set -e

# 1.下载并解压 TensorRT 到 /usr/local
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
tar -xzvf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -C /usr/local/
rm TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz

# 2. 设置环境变量
echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/TensorRT-8.6.1.6/lib' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/TensorRT-8.6.1.6/lib' >> ~/.bashrc
echo 'export C_INCLUDE_PATH=$C_INCLUDE_PATH:/usr/local/TensorRT-8.6.1.6/include' >> ~/.bashrc
echo 'export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/usr/local/TensorRT-8.6.1.6/include' >> ~/.bashrc
# ref: https://blog.csdn.net/lzy2253428273/article/details/132257315
echo 'export PATH=$PATH:/usr/local/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin' >> ~/.bashrc
source ~/.bashrc

# 3. 安装 TensorRT
pip install /usr/local/TensorRT-8.6.1.6/python/tensorrt-8.6.1-cp36-none-linux_x86_64.whl