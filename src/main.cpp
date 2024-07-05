#include <omp.h>
#include "infer.h"

int main(){
    // 0. Set the number of threads for OpenMP
    omp_set_num_threads(8);

    // 1. Create infer instance
    // The ONNX will be automatically parsed and an engine file will be created.
    auto infer = create_infer("yolov5s.trtmodel", "deepsort.trtmodel");
    if (infer == nullptr) {
        printf("Create Infer failed.\n");
        return -1;
    }

    // Inference
    infer->forward("test.mp4");

    return 0;
}