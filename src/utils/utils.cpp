#include "utils.h"

#include <onnx-tensorrt/NvOnnxParser.h>
#include <NvInfer.h>
#include <stdio.h>
#include <fstream>
#include <unistd.h>
#include "yolo.h"

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

void TRTLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
    if(severity <= Severity::kWARNING){
        // ref: https://blog.csdn.net/ericbar/article/details/79652086
        if(severity == Severity::kWARNING){
            printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else if(severity <= Severity::kERROR){
            printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
        }
        else{
            printf("%s: %s\n", severity_string(severity), msg);
        }
    }
}

// destroy automatically
template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr) {
    return std::shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

template std::shared_ptr<nvinfer1::IBuilder> make_nvshared<nvinfer1::IBuilder>(nvinfer1::IBuilder* ptr);
template std::shared_ptr<nvinfer1::IRuntime> make_nvshared<nvinfer1::IRuntime>(nvinfer1::IRuntime* ptr);
template std::shared_ptr<nvinfer1::IHostMemory> make_nvshared<nvinfer1::IHostMemory>(nvinfer1::IHostMemory* ptr);
template std::shared_ptr<nvinfer1::ICudaEngine> make_nvshared<nvinfer1::ICudaEngine>(nvinfer1::ICudaEngine* ptr);
template std::shared_ptr<nvinfer1::IBuilderConfig> make_nvshared<nvinfer1::IBuilderConfig>(nvinfer1::IBuilderConfig* ptr);
template std::shared_ptr<nvinfer1::IExecutionContext> make_nvshared<nvinfer1::IExecutionContext>(nvinfer1::IExecutionContext* ptr);
template std::shared_ptr<nvinfer1::INetworkDefinition> make_nvshared<nvinfer1::INetworkDefinition>(nvinfer1::INetworkDefinition* ptr);
template std::shared_ptr<nvonnxparser::IParser> make_nvshared<nvonnxparser::IParser>(nvonnxparser::IParser* ptr);

bool exists(const std::string& path) {
#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

std::vector<unsigned char> load_file(const std::string& file) {
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

// hsv to bgr
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

void draw_bboxs(cv::Mat& image, std::vector<DetectBox>& allDetections) {
    for (auto box : allDetections) {
        float left = box.x1;
        float top = box.y1;
        float right = box.x2;
        float bottom = box.y2;
        int class_label = box.classID;
        int track_label = box.trackID;
        float confidence = box.confidence;
        cv::Scalar color;
        std::tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name      = cocolabels[class_label];
        auto caption   = cv::format("%s ID: %d", name, track_label);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
}