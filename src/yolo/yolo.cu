#include "yolo.h"

#include <onnx-tensorrt/NvOnnxParser.h>
#include <cuda_runtime.h>

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

class YoloImpl : public Yolo {
public:    
    YoloImpl(char* modelPath, int input_batch, int input_channel, int input_height, int input_width, TRTLogger* logger);
    ~YoloImpl();

public:
    void detect(cv::Mat& image, std::vector<DetectBox>& allDetections) override;
    bool build_model();
    bool load_model();

private:
    void postprocess(const std::string& device,
                     const int& output_numel,
                     const int& output_numbox,
                     float* output_data_device,
                     const int& num_classes,
                     float* d2i,
                     std::vector<DetectBox>& allDetections);
    void postprocess_cpu(const int& output_numel,
                         const int& output_numbox,
                         float* output_data_device,
                         const int& num_classes,
                         float* d2i,
                         std::vector<DetectBox>& allDetections);
    void postprocess_gpu(const int& output_numbox,
                         float* output_data_device,
                         const int& num_classes,
                         float* d2i,
                         std::vector<DetectBox>& allDetections);

private:
    const char* modelPath;
    int input_batch;
    int input_channel;
    int input_height;
    int input_width;
    int input_numel;
    float* input_data_host;
    float* input_data_device;

private:
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> execution_context;
    cudaStream_t stream;
    TRTLogger* logger;
};

YoloImpl::YoloImpl(char* modelPath, int input_batch, int input_channel, int input_height, int input_width, TRTLogger* logger) {
    this->modelPath = modelPath;
    this->input_batch = input_batch;
    this->input_channel = input_channel;
    this->input_height = input_height;
    this->input_width = input_width;
    this->input_numel = input_batch * input_channel * input_height * input_width;
    this->logger = logger;
}

YoloImpl::~YoloImpl() {
    checkRuntime(cudaStreamDestroy(stream));
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
}

// build yolov5s engine, save to workspace/modelPath
bool YoloImpl::build_model() {

    if(exists(this->modelPath)){
        printf("%s has exists.\n", this->modelPath);
        return true;
    }

    // make_nvshared, destroy automatically
    auto builder = make_nvshared(nvinfer1::createInferBuilder(*logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // parse network data from onnx file to `network`
    auto parser = make_nvshared(nvonnxparser::createParser(*network, *logger));
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
    FILE* f = fopen(this->modelPath, "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    printf("Build Done.\n");
    return true;
}

bool YoloImpl::load_model() {

    auto yolov5_engine_data = load_file(this->modelPath);
    auto runtime   = make_nvshared(nvinfer1::createInferRuntime(*(this->logger)));
    this->engine = make_nvshared(runtime->deserializeCudaEngine(yolov5_engine_data.data(), yolov5_engine_data.size()));
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return false;
    }
    if (engine->getNbBindings() != 2) {
        printf("ONNX export error: Must have exactly 1 input and "
            "1 output, but you have: %d outputs.\n", 
            engine->getNbBindings() - 1);
        return false;
    }
    this->stream = nullptr;
    checkRuntime(cudaStreamCreate(&this->stream));
    this->execution_context = make_nvshared(engine->createExecutionContext());
    checkRuntime(cudaMallocHost(&this->input_data_host, this->input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&this->input_data_device, this->input_numel * sizeof(float)));
    return true;
}

void YoloImpl::detect(cv::Mat& image, std::vector<DetectBox>& allDetections) {
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
    // Translate, scale
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));

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

    checkRuntime(cudaMemcpyAsync(this->input_data_device, this->input_data_host, this->input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    // 3x3 input -> 3x3 output
    auto output_dims = engine->getBindingDimensions(1);
    int output_numbox = output_dims.d[1];
    int output_numprob = output_dims.d[2];
    int num_classes = output_numprob - 5;
    int output_numel = input_batch * output_numbox * output_numprob;

    float* output_data_device = nullptr;
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));

    // Specify the data input size used during the current inference
    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch;

    execution_context->setBindingDimensions(0, input_dims);

    float* bindings[] = {input_data_device, output_data_device};
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    checkRuntime(cudaStreamSynchronize(stream));

    this->postprocess("gpu",
                      output_numel,
                      output_numbox,
                      output_data_device,
                      num_classes,
                      d2i,
                      allDetections);
    checkRuntime(cudaFree(output_data_device));
}

void YoloImpl::postprocess(const std::string& device,
                           const int& output_numel,
                           const int& output_numbox,
                           float* output_data_device,
                           const int& num_classes,
                           float* d2i,
                           std::vector<DetectBox>& allDetections) {
    if (device == "cpu") {
        postprocess_cpu(output_numel,
                        output_numbox,
                        output_data_device,
                        num_classes,
                        d2i,
                        allDetections);
    }
    else if (device == "gpu") {
        postprocess_gpu(output_numbox,
                        output_data_device,
                        num_classes,
                        d2i,
                        allDetections);        
    } else {
        printf("Device `%s` is not supported\n", device.c_str());
    }
}

void YoloImpl::postprocess_cpu(const int& output_numel,
                               const int& output_numbox,
                               float* output_data_device,
                               const int& num_classes,
                               float* d2i,
                               std::vector<DetectBox>& allDetections) {

    float* output_data_host = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel));
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // decode box：Restore predictions from different scales 
    // to the original input image(bbox, probability, confidence）
    std::vector<std::vector<float>> bboxes;
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;
    for(int i = 0; i < output_numbox; ++i){
        float* ptr = output_data_host + i * (num_classes + 5);
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
        bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, confidence, (float)label});
    }

    // nms
    std::sort(bboxes.begin(), bboxes.end(), [](std::vector<float>& a, std::vector<float>& b){return a[4] > b[4];});
    std::vector<bool> remove_flags(bboxes.size());

    allDetections.clear();
    allDetections.reserve(bboxes.size());

    auto iou = [](const std::vector<float>& a, const std::vector<float>& b){
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
        float confidence = ibox[4];
        int class_label = ibox[5];
        DetectBox dd(left, top, right, bottom, confidence, class_label);
        allDetections.emplace_back(dd);
        for(int j = i + 1; j < bboxes.size(); ++j){
            if(remove_flags[j]) continue;

            auto& jbox = bboxes[j];
            if(ibox[5] == jbox[5]){
                // class matched
                if(iou(ibox, jbox) >= nms_threshold)
                    remove_flags[j] = true;
            }
        }
    }

    allDetections.shrink_to_fit();
    checkRuntime(cudaFreeHost(output_data_host));
}

static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(float* predict, 
                                     int num_bboxes,
                                     int num_classes,
                                     float confidence_threshold, 
                                     float* invert_affine_matrix,
                                     float* parray,
                                     int max_objects,
                                     int NUM_BOX_ELEMENT) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem     = predict + (5 + num_classes) * position;
    float objectness = pitem[4];
    if(objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5;
    float confidence        = *class_confidence++;
    int label               = 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence) {
            confidence = *class_confidence;
            label      = i;
        }
    }

    if (label != 0) {  // only detect and track person(cocolabels[0])
        return;
    }

    confidence *= objectness;
    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;

    float cx         = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // left, top, right, bottom, confidence, class, keepflag
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(float aleft, float atop, float aright, float abottom, 
                                float bleft, float btop, float bright, float bbottom) {

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT) {

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

static void decode_kernel_invoker(float* predict,
                                  int num_bboxes,
                                  int num_classes,
                                  float confidence_threshold,
                                  float nms_threshold,
                                  float* invert_affine_matrix,
                                  float* parray,
                                  int max_objects,
                                  int NUM_BOX_ELEMENT,
                                  cudaStream_t stream) {
    
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;

    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold, 
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    );

    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}

void YoloImpl::postprocess_gpu(const int& output_numbox,
                               float* predict_device,
                               const int& num_classes,
                               float* d2i,
                               std::vector<DetectBox>& allDetections) {
    // decode box：Restore predictions from different scales 
    // to the original input image(bbox, probability, confidence)
    float* output_device = nullptr;
    float* output_host = nullptr;
    const int max_objects = 1000;
    const int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag
    float confidence_threshold = 0.25;
    float nms_threshold = 0.5;

    float* d2i_device = nullptr;
    checkRuntime(cudaMalloc(&d2i_device, 6 * sizeof(float)));
    checkRuntime(cudaMemcpyAsync(d2i_device, d2i, 6 * sizeof(float), cudaMemcpyHostToDevice, this->stream));
    // [count, box1, box2, ...]
    checkRuntime(cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    // [count=0, box1, box2, ...]
    checkRuntime(cudaMemset(output_device, 0, sizeof(float)));
    checkRuntime(cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));

    decode_kernel_invoker(
        predict_device, output_numbox, num_classes, confidence_threshold, 
        nms_threshold, d2i_device, output_device, max_objects, NUM_BOX_ELEMENT, this->stream
    );
    checkRuntime(cudaMemcpyAsync(output_host, output_device, 
        sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, this->stream
    ));
    checkRuntime(cudaStreamSynchronize(this->stream));

    int num_boxes = min((int)output_host[0], max_objects);

    allDetections.clear();
    allDetections.reserve(num_boxes);

    for(int i = 0; i < num_boxes; ++i){
        float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if(keep_flag){
            // left, top, right, bottom, confidence, class
            DetectBox dd(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]);
            allDetections.emplace_back(dd);
        }
    }
    allDetections.shrink_to_fit();
    checkRuntime(cudaFree(d2i_device));
    checkRuntime(cudaFree(output_device));
    checkRuntime(cudaFreeHost(output_host));
    
}

std::shared_ptr<Yolo> create_yolo(char* modelPath, 
                                  int input_batch,
                                  int input_channel,
                                  int input_height,
                                  int input_width,
                                  TRTLogger* logger) {

    std::shared_ptr<YoloImpl> instance(new YoloImpl(modelPath, input_batch, input_channel, input_height, input_width, logger));
    if (!instance->build_model()) {
        instance.reset();
        return instance;
    }
    if (!instance->load_model()) {
        instance.reset();
        return instance;
    }
    return instance;
}