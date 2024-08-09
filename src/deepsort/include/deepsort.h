#ifndef DEEPSORT_H
#define DEEPSORT_H

#include <opencv2/opencv.hpp>
#include "datatype.h"
#include <vector>
#include <NvInfer.h>

class DeepSort {
public:
    virtual void sort(cv::Mat& frame, std::vector<DetectBox>& dets) = 0;
};

std::shared_ptr<DeepSort> create_deepsort(char* onnxFile,
                                          char* engineFile,
                                          bool useInt8,
                                          int batchSize,
                                          int featureDim,
                                          int gpuID,
                                          nvinfer1::ILogger* logger);

#endif  // DEEPSORT_H