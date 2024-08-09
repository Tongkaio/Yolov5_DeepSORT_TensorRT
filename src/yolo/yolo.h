#ifndef YOLO_H
#define YOLO_H

#include <memory>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "datatype.h"

extern const char* cocolabels[];

class Yolo {
public:
    virtual void detect(cv::Mat& image, std::vector<DetectBox>& allDetections) = 0;
};

std::shared_ptr<Yolo> create_yolo(char* onnxFile,
                                  char* engineFile,
                                  bool useInt8,
                                  int input_batch,
                                  int input_channel,
                                  int input_height,
                                  int input_width,
                                  TRTLogger* logger);

#endif  // YOLO_H