#ifndef YOLO_H
#define YOLO_H

#include <string>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <NvInferRuntime.h>
#include "utils.h"
#include "datatype.h"
#include "deepsort.h"

extern const char* cocolabels[];

class Yolo {
public:    
    Yolo(std::string modelPath, int input_batch, int input_channel, int input_height, int input_width, TRTLogger& logger);
    ~Yolo();

public:
    void detect(cv::Mat& image, std::vector<DetectBox>& allDetections);

private:
    int input_batch;
    int input_channel;
    int input_height;
    int input_width;
    int input_numel;
    float* input_data_host;
    float* input_data_device;
    

private:
    // std::vector<DetectBox> allDetections;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> execution_context;
    cudaStream_t stream;
    TRTLogger logger;
};

bool build_yolov5_model();

#endif  // YOLO_H