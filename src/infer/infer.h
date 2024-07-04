#ifndef INFER_H
#define INFER_H

#include <mutex>
#include <future>
#include <opencv2/opencv.hpp>
#include "yolo.h"

template<typename T>
struct Job {
    std::shared_ptr<std::promise<T>> pro;
    T input;
};

class Infer {
public:
    Infer(const std::string& video, const std::string& yolo_engine, const std::string& deepsort_engine);
    ~Infer();
    void forward();
    void video_capture(const std::string& file);
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
    std::atomic<bool> running_{false};
    std::mutex lock_;
    std::mutex sort_lock_;
    std::condition_variable pic_cv_;
    std::condition_variable bbox_cv_;
    std::condition_variable sort_cv_;
    std::queue<Job<cv::Mat>> pic_jobs_;
    std::queue<Job<std::vector<DetectBox>>> bbox_jobs_;
    std::queue<cv::Mat> q_pics;
    std::queue<std::pair<cv::Mat, std::vector<DetectBox>>> q_detects;
    std::queue<std::pair<cv::Mat, std::vector<DetectBox>>> q_sorts;
    std::thread video_thread;
    std::thread yolo_thread;
    std::thread deepsort_thread;
    std::thread imshow_thread;
};

#endif  // INFER_H