#ifndef INFER_H
#define INFER_H

#include <memory>

class Infer {
public:
    virtual void forward(const std::string& file) = 0;
};


std::shared_ptr<Infer> create_infer(char* yolo_engine,
                                    char* deepsort_engine);

#endif  // INFER_H