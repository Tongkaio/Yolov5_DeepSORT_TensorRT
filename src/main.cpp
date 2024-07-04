#include <omp.h>
#include "yolo.h"
#include "deepsort.h"
#include "infer.h"

int main(){
    omp_set_num_threads(8);
    // 1. build yolov5s engine, save to workspace/yolov5s.trtmodel
    if(!build_yolov5_model()) {
        return -1;
    }

    // 2. build deepsort engine, save to workspace/deepsort.trtmodel
    if(!build_deepsort_model()) {
        return -1;
    }

    // 3. inference
    Infer* infer = new Infer("test.mp4", "yolov5s.trtmodel", "deepsort.trtmodel");
    infer->forward();
    delete infer;

    return 0;
}