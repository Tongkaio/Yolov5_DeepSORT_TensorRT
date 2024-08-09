#ifndef INFER_H
#define INFER_H

#include <memory>

class Infer {
public:
    virtual void forward(const std::string& file) = 0;
};


std::shared_ptr<Infer> create_infer(char* yolo_onnx,
                                    char* deepsort_onnx,
                                    char* yolo_engine,
                                    char* deepsort_engine,
                                    bool yolo_int8,
                                    bool deepsort_int8);

#endif  // INFER_H