构建 docker 镜像：
```shell
cd docker
docker build -t tongkai2023/cuda11.2.2-torch1.9.1-py38-trt8.6.1-opencv4.5.4:0.1 .
```

打开x服务器访问控制：
```shell
xhost +
```

进入目录：
```
cd /your/workspace/
```

启动 docker 容器：
```shell
docker run -it \
--name trt8.6-cuda11.2.2 \
-v `pwd`:/workspace \
--workdir=/workspace \
--hostname=$HOSTNAME \
--ipc=host \
--env="DISPLAY" \
--gpus=all \
-p "8888:8888" \
-e PYTHONUNBUFFERED=1 \
-e QT_X11_NO_MITSHM=1 \
-e PYTHONIOENCODING=utf-8 \
--mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
tongkai2023/cuda11.2.2-torch1.9.1-py38-trt8.6.1-opencv4.5.4:0.1 \
bash
```

查看版本：
```shell
# 查看trt版本
python -c "import tensorrt;print(tensorrt.__version__)"

# 查看cuda版本
nvcc -V
```
