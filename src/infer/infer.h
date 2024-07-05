#ifndef INFER_H
#define INFER_H

#include <mutex>
#include <future>
#include <opencv2/opencv.hpp>
#include "yolo.h"
#include "deepsort.h"
#include "utils.h"

class Infer {
public:
    Infer(const std::string& video, const std::string& yolo_engine, const std::string& deepsort_engine);
    ~Infer();
    void forward();

private:
    void video_worker(const std::string& file);
    void yolo_worker();
    void deepsort_worker();
    void imshow_worker();

private:
    TRTLogger logger;
    Yolo* yolo;
    DeepSort* DS;
    string video;

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

#endif  // INFER_H