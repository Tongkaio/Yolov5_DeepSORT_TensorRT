#include <NvInfer.h>

// onnxparser
#include <onnx-tensorrt/NvOnnxParser.h>

// for inference
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <opencv2/opencv.hpp>

// deepsort
#include "deepsort.h"

// openmp
#include <omp.h>

using namespace std;

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

// ref: https://cocodataset.org/#home
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

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
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
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
} logger;

// destroy automatically
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool exists(const string& path){
#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}

// build yolov5s engine, save to workspace/yolov5s.trtmodel
bool build_yolov5_model(){
    if(exists("yolov5s.trtmodel")){
        printf("yolov5s.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // make_nvshared, destroy automatically
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // parse network data from onnx file to `network`
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("yolov5s.onnx", 1)){
        printf("Failed to parse yolov5s.onnx\n");
        return false;
    }
    
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    
    // configure minimum, optimal, and maximum ranges
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // serialize model to engine file
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("yolov5s.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    printf("Build Done.\n");
    return true;
}

// build deepsort engine, save to workspace/deepsort.trtmodel
bool build_deepsort_model() {
    if(exists("deepsort.trtmodel")) {
        printf("deepsort.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // make_nvshared, destroy automatically
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // parse network data from onnx file to `network`
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("deepsort.onnx", 1)) {
        printf("Failed to parse deepsort.onnx\n");
        return false;
    }
    
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    printf("input->name = %s\n", input_tensor->getName());

    const int IMG_HEIGHT = 128;
    const int IMG_WIDTH = 64;
    const int MAX_BATCH_SIZE = 128;
    nvinfer1::Dims dims = nvinfer1::Dims4{1, 3, IMG_HEIGHT, IMG_WIDTH};

    // configure minimum, optimal, and maximum ranges
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, 
                            nvinfer1::Dims4{1, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, 
                            nvinfer1::Dims4{MAX_BATCH_SIZE, dims.d[1], dims.d[2], dims.d[3]});
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, 
                            nvinfer1::Dims4{MAX_BATCH_SIZE, dims.d[1], dims.d[2], dims.d[3]});
    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr) {
        printf("Build engine failed.\n");
        return false;
    }

    // serialize model to engine file
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("deepsort.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    printf("Build Done.\n");
    return true;
}


vector<unsigned char> load_file(const string& file) {
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

void inference(const string& file) {
    TRTLogger logger;
    auto yolov5_engine_data = load_file("yolov5s.trtmodel");
    DeepSort* DS = new DeepSort("deepsort.trtmodel", 128, 256, 0, &logger);
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(yolov5_engine_data.data(), yolov5_engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }

    if (engine->getNbBindings() != 2) {
        printf("ONNX export error: Must have exactly 1 input and "
               "1 output, but you have: %d outputs.\n", 
               engine->getNbBindings() - 1);
        return;
    }

    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());

    int input_batch = 1;
    int input_channel = 3;
    int input_height = 640;
    int input_width = 640;
    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    cv::VideoCapture cap(file);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    // std::string outputVideo = "output_test.avi";
    // cv::VideoWriter outputVideoWriter;
    // outputVideoWriter.open(outputVideo, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 25, cv::Size(1920, 1080));

    cv::Mat image;
    while (true) {
        if (!cap.read(image)) {
            // std::cerr << "Error reading frame from video stream" << std::endl;
            // outputVideoWriter.release();
            break;
        }
        // Resize the image using bilinear interpolation
        float scale_x = input_width / (float)image.cols;
        float scale_y = input_height / (float)image.rows;
        float scale = std::min(scale_x, scale_y);
        float i2d[6], d2i[6];

        // Resize the image while aligning the geometric centers of the source and target images
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = (-scale * image.cols + input_width + scale  - 1) * 0.5;
        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;

        cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);  // image to dst(network), 2x3 matrix
        cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);  // dst to image, 2x3 matrix
        cv::invertAffineTransform(m2x3_i2d, m2x3_d2i);  // Compute an inverse affine transformation

        cv::Mat input_image(input_height, input_width, CV_8UC3);
        cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));  // Translate, scale, and rotate

        int image_area = input_image.cols * input_image.rows;
        unsigned char* pimage = input_image.data;
        float* phost_b = input_data_host + image_area * 0;
        float* phost_g = input_data_host + image_area * 1;
        float* phost_r = input_data_host + image_area * 2;
        for(int i = 0; i < image_area; ++i, pimage += 3){
            // bgr -> rgb
            *phost_r++ = pimage[0] / 255.0f;
            *phost_g++ = pimage[1] / 255.0f;
            *phost_b++ = pimage[2] / 255.0f;
        }
        ///////////////////////////////////////////////////
        checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

        // 3x3 input -> 3x3 output
        auto output_dims = engine->getBindingDimensions(1);
        int output_numbox = output_dims.d[1];
        int output_numprob = output_dims.d[2];
        int num_classes = output_numprob - 5;
        int output_numel = input_batch * output_numbox * output_numprob;
        float* output_data_host = nullptr;
        float* output_data_device = nullptr;
        checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
        checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

        // Specify the data input size used during the current inference
        auto input_dims = engine->getBindingDimensions(0);
        input_dims.d[0] = input_batch;

        execution_context->setBindingDimensions(0, input_dims);
        float* bindings[] = {input_data_device, output_data_device};
        bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
        checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
        checkRuntime(cudaStreamSynchronize(stream));

        // decode box：Restore predictions from different scales 
        // to the original input image(bbox, probability, confidence）
        vector<vector<float>> bboxes;
        float confidence_threshold = 0.25;
        float nms_threshold = 0.5;
        for(int i = 0; i < output_numbox; ++i){
            float* ptr = output_data_host + i * output_numprob;
            float objness = ptr[4];
            if(objness < confidence_threshold)
                continue;

            float* pclass = ptr + 5;
            int label     = std::max_element(pclass, pclass + num_classes) - pclass;
            if (cocolabels[label] != "person")  // only detect and track person
                continue;

            float prob    = pclass[label];
            float confidence = prob * objness;
            if(confidence < confidence_threshold)
                continue;

            // center(cx, cy), width, height
            float cx     = ptr[0];
            float cy     = ptr[1];
            float width  = ptr[2];
            float height = ptr[3];

            // bbox
            float left   = cx - width * 0.5;
            float top    = cy - height * 0.5;
            float right  = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // bbox position on image
            float image_base_left   = d2i[0] * left   + d2i[2];
            float image_base_right  = d2i[0] * right  + d2i[2];
            float image_base_top    = d2i[0] * top    + d2i[5];
            float image_base_bottom = d2i[0] * bottom + d2i[5];
            bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence});
        }

        // nms
        std::sort(bboxes.begin(), bboxes.end(), [](vector<float>& a, vector<float>& b){return a[5] > b[5];});
        std::vector<bool> remove_flags(bboxes.size());
        std::vector<vector<float>> box_result;
        box_result.reserve(bboxes.size());

        auto iou = [](const vector<float>& a, const vector<float>& b){
            float cross_left   = std::max(a[0], b[0]);
            float cross_top    = std::max(a[1], b[1]);
            float cross_right  = std::min(a[2], b[2]);
            float cross_bottom = std::min(a[3], b[3]);

            float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
            float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) 
                            + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
            if(cross_area == 0 || union_area == 0) return 0.0f;
            return cross_area / union_area;
        };

        for(int i = 0; i < bboxes.size(); ++i){
            if(remove_flags[i]) continue;

            auto& ibox = bboxes[i];
            box_result.emplace_back(ibox);
            for(int j = i + 1; j < bboxes.size(); ++j){
                if(remove_flags[j]) continue;

                auto& jbox = bboxes[j];
                if(ibox[4] == jbox[4]){
                    // class matched
                    if(iou(ibox, jbox) >= nms_threshold)
                        remove_flags[j] = true;
                }
            }
        }

        vector<DetectBox> allDetections;
        
        for(int i = 0; i < box_result.size(); ++i){
            auto& ibox = box_result[i];
            float left = ibox[0];
            float top = ibox[1];
            float right = ibox[2];
            float bottom = ibox[3];
            int class_label = ibox[4];
            float confidence = ibox[5];
            DetectBox dd(left, top, right, bottom, confidence, class_label);
            allDetections.push_back(dd);
        }

        // deepsort
        DS->sort(image, allDetections);
        for (auto box : allDetections) {
            float left = box.x1;
            float top = box.y1;
            float right = box.x2;
            float bottom = box.y2;
            int class_label = box.classID;
            int track_label = box.trackID;
            float confidence = box.confidence;
            cv::Scalar color;
            tie(color[0], color[1], color[2]) = random_color(class_label);
            cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 3);

            auto name      = cocolabels[class_label];
            auto caption   = cv::format("%s ID: %d", name, track_label);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
            cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        // show
        cv::imshow("Frame", image);
        // outputVideoWriter.write(image);
        // press `Esc` to quit
        if (cv::waitKey(1) == 27) {
            std::cout << "Esc key is pressed by user. Stopping the video." << std::endl;
            break;
        }
        checkRuntime(cudaFreeHost(output_data_host));
        checkRuntime(cudaFree(output_data_device));
    }

    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    
}

int main(){
    omp_set_num_threads(8);
    // 1. build yolov5s engine, save to workspace/yolov5s.trtmodel
    if(!build_yolov5_model()){
        return -1;
    }

    // 2. build deepsort engine, save to workspace/deepsort.trtmodel
    if(!build_deepsort_model()){
        return -1;
    }

    // 3. inference
    inference("test.mp4");
    return 0;
}