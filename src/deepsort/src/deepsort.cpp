#include "deepsort.h"
#include "model.hpp"
#include "utils.h"
#include "tracker.h"
#include "featuretensor.h"

using std::vector;
using nvinfer1::ILogger;

class DeepSortImpl : public DeepSort {
public:    
    DeepSortImpl(char* onnxFile,
                 char* engineFile,
                 bool useInt8,
                 int batchSize,
                 int featureDim,
                 int gpuID,
                 ILogger* gLogger);
    ~DeepSortImpl();

public:
    bool build_model();
    bool load_model();
    void sort(cv::Mat& frame, vector<DetectBox>& dets) override;

private:
    void sort(cv::Mat& frame, DETECTIONS& detections);
    void sort(cv::Mat& frame, DETECTIONSV2& detectionsv2);    
    void sort(vector<DetectBox>& dets);
    void sort(DETECTIONS& detections);
    

private:
    char* onnxFile;
    char* engineFile;
    bool useInt8;
    int batchSize;
    int featureDim;
    cv::Size imgShape;
    float confThres;
    float nmsThres;
    int maxBudget;
    float maxCosineDist;

private:
    vector<RESULT_DATA> result;
    vector<std::pair<CLSCONF, DETECTBOX>> results;
    tracker* objTracker;
    FeatureTensor* featureExtractor;
    ILogger* gLogger;
    int gpuID;
};

DeepSortImpl::DeepSortImpl(char* onnxFile,
                           char* engineFile,
                           bool useInt8,
                           int batchSize,
                           int featureDim,
                           int gpuID,
                           ILogger* gLogger) {
    this->onnxFile = onnxFile;
    this->engineFile = engineFile;
    this->useInt8 = useInt8;
    this->batchSize = batchSize;
    this->featureDim = featureDim;
    this->gpuID = gpuID;
    this->imgShape = cv::Size(64, 128);
    this->maxBudget = 100;
    this->maxCosineDist = 0.2;
    this->gLogger = gLogger;
}

DeepSortImpl::~DeepSortImpl() {
    delete objTracker;
}

bool DeepSortImpl::load_model() {
    objTracker = new tracker(maxCosineDist, maxBudget);
    featureExtractor = new FeatureTensor(batchSize, imgShape, featureDim, gpuID, gLogger);
    featureExtractor->loadEngine(engineFile);
    return true;
}

void DeepSortImpl::sort(cv::Mat& frame, vector<DetectBox>& dets) {
    // preprocess Mat -> DETECTION
    DETECTIONS detections;
    vector<CLSCONF> clsConf;
    
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
        clsConf.push_back(CLSCONF((int)i.classID, i.confidence));
    }
    result.clear();
    results.clear();
    if (detections.size() > 0) {
        DETECTIONSV2 detectionsv2 = make_pair(clsConf, detections);
        sort(frame, detectionsv2);
    }
    // postprocess DETECTION -> Mat
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2)+i(0), i(3)+i(1), 1.);
        b.trackID = (float)r.first;
        dets.push_back(b);
    }
    for (int i = 0; i < results.size(); ++i) {
        CLSCONF c = results[i].first;
        dets[i].classID = c.cls;
        dets[i].confidence = c.conf;
    }
}

void DeepSortImpl::sort(cv::Mat& frame, DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        //result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

void DeepSortImpl::sort(cv::Mat& frame, DETECTIONSV2& detectionsv2) {
    std::vector<CLSCONF>& clsConf = detectionsv2.first;
    DETECTIONS& detections = detectionsv2.second;
    bool flag = featureExtractor->getRectsFeature(frame, detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detectionsv2);
        result.clear();
        results.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
            results.push_back(make_pair(CLSCONF(track.cls, track.conf) ,track.to_tlwh()));
        }
    }
}

void DeepSortImpl::sort(vector<DetectBox>& dets) {
    DETECTIONS detections;
    for (DetectBox i : dets) {
        DETECTBOX box(i.x1, i.y1, i.x2-i.x1, i.y2-i.y1);
        DETECTION_ROW d;
        d.tlwh = box;
        d.confidence = i.confidence;
        detections.push_back(d);
    }
    if (detections.size() > 0)
        sort(detections);
    dets.clear();
    for (auto r : result) {
        DETECTBOX i = r.second;
        DetectBox b(i(0), i(1), i(2), i(3), 1.);
        b.trackID = r.first;
        dets.push_back(b);
    }
}

void DeepSortImpl::sort(DETECTIONS& detections) {
    bool flag = featureExtractor->getRectsFeature(detections);
    if (flag) {
        objTracker->predict();
        objTracker->update(detections);
        result.clear();
        for (Track& track : objTracker->tracks) {
            if (!track.is_confirmed() || track.time_since_update > 1)
                continue;
            result.push_back(make_pair(track.track_id, track.to_tlwh()));
        }
    }
}

// build deepsort engine, save to workspace/deepsort.trtmodel
bool DeepSortImpl::build_model() {
    if(exists(this->engineFile)) {
        printf("%s has exists.\n", this->engineFile);
        return true;
    } else {
        printf("Building %s ...\n", this->engineFile);
    }

    // make_nvshared, destroy automatically
    auto builder = make_nvshared(nvinfer1::createInferBuilder(*gLogger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // parse network data from onnx file to `network`
    auto parser = make_nvshared(nvonnxparser::createParser(*network, *gLogger));
    if(!parser->parseFromFile(this->onnxFile, 1)) {
        printf("Failed to parse %s\n", this->onnxFile);
        return false;
    }

    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);

    const int IMG_HEIGHT = 128;
    const int IMG_WIDTH = 64;
    const int MAX_BATCH_SIZE = 128;
    nvinfer1::Dims dims = nvinfer1::Dims4{1, 3, IMG_HEIGHT, IMG_WIDTH};

    if (this->useInt8) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }

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
        printf("Build %s failed.\n", this->engineFile);
        return false;
    }

    // serialize model to engine file
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen(this->engineFile, "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    printf("Build done.\n");
    return true;
}

std::shared_ptr<DeepSort> create_deepsort(char* onnxFile,
                                          char* engineFile,
                                          bool useInt8,
                                          int batchSize,
                                          int featureDim,
                                          int gpuID,
                                          ILogger* logger) {
    
    shared_ptr<DeepSortImpl> instance(new DeepSortImpl(onnxFile, engineFile, useInt8, batchSize, featureDim, gpuID, logger));
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
