#include "utils.h"

#include <onnx-tensorrt/NvOnnxParser.h>
#include <NvInfer.h>
#include <stdio.h>
#include <fstream>
#include <unistd.h>

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