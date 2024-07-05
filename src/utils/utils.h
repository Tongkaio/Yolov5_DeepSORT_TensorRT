#ifndef UTILS_H
#define UTILS_H

#include <cuda_runtime.h>
#include <NvInferRuntime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include "deepsort.h"

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

inline const char* severity_string(nvinfer1::ILogger::Severity t) {
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;
};

template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr);

bool exists(const std::string& path);

std::vector<unsigned char> load_file(const std::string& file);

void draw_bboxs(cv::Mat& image, std::vector<DetectBox>& allDetections);

#endif // UTILS_H