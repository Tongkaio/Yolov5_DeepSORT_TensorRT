# Yolov5-DeepSORT-TensorRT
简体中文 | [English](./README_en.md)

<div align=center>
<img src="./assets/demo.gif"/>
</div>

# 引言

- 本项目是 [Yolo-DeepSORT](https://github.com/ZQPei/deep_sort_pytorch) 的 C++ 实现，使用 TensorRT 进行推理；
- 提供了 dockerfile 以快速搭建开发环境；
- 只需要提供 onnx 文件，在创建模型实例时会自动解析 onnx 并序列化出 engine 文件（*.trtmodel）到 workspace 目录下；
- 我的另一个 PyTorch 版本的实现，含撞线检测，可对行人进行计数：[Yolov5_Deepsort_Person_Count](https://github.com/Tongkaio/Yolov5_Deepsort_Person_Count)

# 快速使用
## 💻 环境配置
参考 [README](docker/README.md) 使用Docker容器，或参考下方自行配置：
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
| *.onnx, test.mp4 | [下载](https://pan.baidu.com/s/1HNveFo1S4RgXx1JlXMaSwA) (提取码: zxao) |

- yolov5s.onnx 导出自 [yolov5-6.0](https://github.com/ultralytics/yolov5/tree/v6.0)，deepsort 的 onnx 导出自 [deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)，可参考脚本 [exportOnnx.py](https://github.com/GesilaA/deepsort_tensorrt/blob/master/exportOnnx.py)
- ptq_yolov5s.onnx 是量化模型，参考 https://github.com/Tongkaio/yolov5_quant

## 🏃‍ 运行
修改 MakeFile 中的相关头文件和库文件路径（若使用本项目提供的 docker 则不需要），然后执行：
```shell
make run
```

项目运行时将显示推理结果，按 ESC 退出。

目前在 GeForce RTX 2060 上，推理 test.mp4 的速度约为 35 ms/帧。

# 文件说明

- Infer、Yolo 和 DeepSORT 使用接口模式和 RAII 进行封装：
  - [infer.h](src/infer/infer.h)， [yolo.h](src/yolo/yolo.h)，[deepsort.h](src/deepsort/include/deepsort.h) 仅暴露 `create_*` 和**推理**接口
  - 使用 `create_*` 创建对象实例，将自动解析 onnx 文件，生成 engine 并加载

- [infer.cpp](src/infer/infer.cpp): 分四个线程，两两之间为**生产者-消费者**关系：

![Alt text](assets/thread.png)

# 参考
- https://github.com/GesilaA/deepsort_tensorrt
- https://github.com/onnx/onnx-tensorrt/tree/release/8.0
- https://github.com/shouxieai/tensorRT_Pro