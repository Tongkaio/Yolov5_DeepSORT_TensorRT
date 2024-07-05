#include "yolo.h"

// ref: https://cocodataset.org/#home
const char* cocolabels[] = {
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

Yolo::Yolo(std::string modelPath, int input_batch, int input_channel, int input_height, int input_width, TRTLogger& logger) {
    auto yolov5_engine_data = load_file(modelPath);
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
    this->engine = make_nvshared(runtime->deserializeCudaEngine(yolov5_engine_data.data(), yolov5_engine_data.size()));
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
    this->stream = nullptr;
    checkRuntime(cudaStreamCreate(&this->stream));
    execution_context = make_nvshared(engine->createExecutionContext());
    this->input_batch = input_batch;
    this->input_channel = input_channel;
    this->input_height = input_height;
    this->input_width = input_width;
    this->input_numel = input_batch * input_channel * input_height * input_width;
    checkRuntime(cudaMallocHost(&this->input_data_host, this->input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&this->input_data_device, this->input_numel * sizeof(float)));
}

Yolo::~Yolo() {
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
}

void Yolo::detect(cv::Mat& image, std::vector<DetectBox>& allDetections) {
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

    checkRuntime(cudaMemcpyAsync(this->input_data_device, this->input_data_host, this->input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

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
        if (std::string(cocolabels[label]) != "person") {  // only detect and track person
            continue;
        }
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

    allDetections.clear();
    allDetections.reserve(bboxes.size());

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
        // box_result.emplace_back(ibox);
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        DetectBox dd(left, top, right, bottom, confidence, class_label);
        allDetections.emplace_back(dd);
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

    allDetections.shrink_to_fit();
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(output_data_device));
}

// build yolov5s engine, save to workspace/yolov5s.trtmodel
bool build_yolov5_model() {
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