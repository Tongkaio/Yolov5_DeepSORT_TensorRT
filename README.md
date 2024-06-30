# Yolov5-DeepSort-TensorRT
![Alt text](assets/demo.gif)

# 快速使用
## 💻 环境配置
推荐配置：
- python: 3.8
- cuda: 11.2
- cudnn: 8.2.2.26
- tensorRT: 8.0.3.4
- protobuf: 3.11.4

## 📥 下载视频&模型
创建工作目录:
```shell
mkdir workspace
```

下载文件到 workspace 目录下：
|     文件      |                             链接                             |
| :-----------: | :----------------------------------------------------------: |
| yolov5s.onnx  | [下载](https://pan.baidu.com/s/1RLFFuATbg9MkqLLBd3Nzdw) (提取码: tg42) |
| deepsort.onnx | [下载](https://pan.baidu.com/s/1kmDId6lzpCN50xH7e1t8BA) (提取码: iyms) |
| test.mp4      | [下载](https://pan.baidu.com/s/1dnPyUtfWupk6YTUOKj7Rxg) (提取码: vatx)

## 🏃‍ 运行
修改 MakeFile 中的相关头文件和库文件路径，然后执行：
```shell
make run
```

# 参考
- https://github.com/GesilaA/deepsort_tensorrt