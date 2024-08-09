#include "infer.h"

#include <stdio.h>
#include <iostream>
#include <chrono>
#include <string>
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include "yolo.h"
#include "deepsort.h"
#include "datatype.h"
#include "utils.h"

class InferImpl : public Infer {
public:
    InferImpl(char* yolo_onnx,
              char* deepsort_onnx,
              char* yolo_engine,
              char* deepsort_engine,
              bool yolo_int8,
              bool deepsort_int8);
    bool create_model();
    void forward(const std::string& file) override;

private:
    void video_worker(const std::string& file);
    void yolo_worker();
    void deepsort_worker();
    void imshow_worker();

private:
    TRTLogger logger;
    char* yolo_onnx;
    char* deepsort_onnx;
    char* yolo_engine;
    char* deepsort_engine;
    bool yolo_int8;
    bool deepsort_int8;
    std::shared_ptr<Yolo> yolo;
    std::shared_ptr<DeepSort> DS;
    std::string video;

private:
    const int MAX_LENGTH = 5;
    int frameCount;
    int FRAME;
    std::atomic<bool> running{false};
    std::atomic<bool> done1{false};
    std::atomic<bool> done2{false};
    std::atomic<bool> done3{false};
    std::mutex mtx1;
    std::mutex mtx2;
    std::mutex mtx3;
    std::condition_variable cv1;
    std::condition_variable cv2;
    std::condition_variable cv3;
    std::queue<cv::Mat> q_pics;
    std::queue<std::pair<cv::Mat, std::vector<DetectBox>>> q_detects;
    std::queue<std::pair<cv::Mat, std::vector<DetectBox>>> q_sorts;
    std::thread video_thread;
    std::thread yolo_thread;
    std::thread deepsort_thread;
    std::thread imshow_thread;
};

InferImpl::InferImpl(char* yolo_onnx,
                     char* deepsort_onnx,
                     char* yolo_engine,
                     char* deepsort_engine,
                     bool yolo_int8,
                     bool deepsort_int8) {
    this->yolo_onnx = yolo_onnx;
    this->deepsort_onnx = deepsort_onnx;
    this->yolo_engine = yolo_engine;
    this->deepsort_engine = deepsort_engine;
    this->yolo_int8 = yolo_int8;
    this->deepsort_int8 = deepsort_int8;
}

bool InferImpl::create_model() {
    this->yolo = create_yolo(this->yolo_onnx, this->yolo_engine, this->yolo_int8, 1, 3, 640, 640, &this->logger);
    if (this->yolo == nullptr) {
        printf("Create Yolo failed.\n");
        return false;
    }
    this->DS = create_deepsort(this->deepsort_onnx, this->deepsort_engine, this->deepsort_int8, 128, 256, 0, &this->logger);
    if (this->DS == nullptr) {
        printf("Create Deepsort failed.\n");
        return false;
    }
    return true;
}

void InferImpl::forward(const std::string& file) {

    this->video = file;

    auto start = std::chrono::high_resolution_clock::now();

    this->running = true;
    this->frameCount = 0;

    this->video_thread = std::thread(&InferImpl::video_worker, this, this->video);
    this->yolo_thread = std::thread(&InferImpl::yolo_worker, this);
    this->deepsort_thread = std::thread(&InferImpl::deepsort_worker, this);
    this->imshow_thread = std::thread(&InferImpl::imshow_worker, this);

    this->video_thread.join();
    this->yolo_thread.join();
    this->deepsort_thread.join();
    this->imshow_thread.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    printf("Processd %d/%d frames, %f ms per frame.\n", frameCount, FRAME, total/(frameCount*1.0));
}

void InferImpl::video_worker(const std::string& file) {

    cv::VideoCapture cap(file);

    this->FRAME = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return;
    }

    cv::Mat image;
    while (running && cap.read(image)) {
        {
            std::unique_lock<std::mutex> lock1(this->mtx1);
            this->cv1.wait(lock1, [&](){ return this->q_pics.size() < this->MAX_LENGTH || !running; });
            if (!running) break;
            this->q_pics.push(image.clone());  // 创建副本并推入队列
        }
        cv1.notify_one();  // 通知 yolo_worker 有新的数据
        std::this_thread::yield();
    }

    done1 = true;
    cv1.notify_one();  // 通知 yolo_worker 已完成
    cap.release();
}

void InferImpl::yolo_worker() {
    while (running) {
        std::unique_lock<std::mutex> lock1(this->mtx1);
        this->cv1.wait(lock1, [&](){ return !this->q_pics.empty() || done1 || !running; });

        if ((q_pics.empty() && done1) || !running) break;

        cv::Mat image = this->q_pics.front();
        this->q_pics.pop();
        if (!done1) this->cv1.notify_one();  // 通知 video_worker 队列不满
        lock1.unlock();

        std::vector<DetectBox> allDetections;

        this->yolo->detect(image, allDetections);

        {
            std::unique_lock<std::mutex> lock2(this->mtx2);
            this->cv2.wait(lock2, [&](){ return q_detects.size() < this->MAX_LENGTH || !running;});
            if (!running) break;
            q_detects.push({image, allDetections});  // 创建副本并推入队列
        }
        cv2.notify_one();
        std::this_thread::yield();
    }

    done2 = true;
    cv2.notify_one();
}

void InferImpl::deepsort_worker() {
    while (running) {
        std::unique_lock<std::mutex> lock2(this->mtx2);
        this->cv2.wait(lock2, [&](){ return !this->q_detects.empty() || done2 || !running; });

        if ((q_detects.empty() && done2) || !running) break;

        cv::Mat image = this->q_detects.front().first;
        std::vector<DetectBox> allDetections = this->q_detects.front().second;
        this->q_detects.pop();
        if (!done2) this->cv2.notify_one();  // 通知 yolo_worker 队列不满
        lock2.unlock();

        DS->sort(image, allDetections);

        {
            std::unique_lock<std::mutex> lock3(this->mtx3);
            cv3.wait(lock3, [&](){ return this->q_sorts.size() < this->MAX_LENGTH || !running; });
            if (!running) break;
            q_sorts.push({image.clone(), allDetections});  // 创建副本并推入队列
        }
        cv3.notify_one();
        std::this_thread::yield();
    }  

    done3 = true;
    cv3.notify_one();
}

void InferImpl::imshow_worker() {
    while (running) {
        std::unique_lock<std::mutex> lock3(this->mtx3);

        cv3.wait(lock3, [&](){ return !this->q_sorts.empty() || done3 || !running; });

        if ((this->q_sorts.empty() && done3) || !running) break;

        cv::Mat image_result = this->q_sorts.front().first;
        std::vector<DetectBox> allDetections = this->q_sorts.front().second;
        this->q_sorts.pop();
        lock3.unlock();
        if (!done3) this->cv3.notify_one();  // 通知 deepsort_worker 队列不满
        
        draw_bboxs(image_result, allDetections);

        cv::imshow("Result", image_result);
        frameCount++;
        if (cv::waitKey(1) == 27) {
            std::cout << "Esc key is pressed by user. Stopping the video." << std::endl;
            running = false;
            cv1.notify_all();
            cv2.notify_all();
            cv3.notify_all();
            break;
        }
        std::this_thread::yield();
    }
}

std::shared_ptr<Infer> create_infer(char* yolo_onnx,
                                    char* deepsort_onnx,
                                    char* yolo_engine,
                                    char* deepsort_engine,
                                    bool yolo_int8,
                                    bool deepsort_int8) {
    std::shared_ptr<InferImpl> instance(new InferImpl(yolo_onnx,
                                                      deepsort_onnx,
                                                      yolo_engine,
                                                      deepsort_engine,
                                                      yolo_int8,
                                                      deepsort_int8));
    if (!instance->create_model()) {
        instance.reset();
        return instance;
    }
    return instance;
}