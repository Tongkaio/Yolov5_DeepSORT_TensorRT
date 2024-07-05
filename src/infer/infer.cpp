#include "infer.h"

#include <stdio.h>
#include <iostream>


Infer::Infer(const std::string& video, const std::string& yolo_engine, const std::string& deepsort_engine) {
    this->video = video;
    this->yolo = new Yolo(yolo_engine, 1, 3, 640, 640, this->logger);
    this->DS = new DeepSort(deepsort_engine, 128, 256, 0, &this->logger);
}

Infer::~Infer() {
    delete this->yolo;
    delete this->DS;
}

void Infer::forward() {

    auto start = std::chrono::high_resolution_clock::now();

    this->running = true;

    this->video_thread = std::thread(&Infer::video_worker, this, this->video);
    this->yolo_thread = std::thread(&Infer::yolo_worker, this);
    this->deepsort_thread = std::thread(&Infer::deepsort_worker, this);
    this->imshow_thread = std::thread(&Infer::imshow_worker, this);

    this->video_thread.join();
    this->yolo_thread.join();
    this->deepsort_thread.join();
    this->imshow_thread.join();

    auto end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration_cast<chrono::milliseconds>(end - start).count();
    printf("%f ms per frame\n", total/(this->frameCount*1.0));
}

void Infer::video_worker(const string& file) {

    cv::VideoCapture cap(file);

    this->frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Total number of frames: " << frameCount << std::endl;

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
        this_thread::yield();
    }

    done1 = true;
    cv1.notify_one();  // 通知 yolo_worker 已完成
    cap.release();
}

void Infer::yolo_worker() {
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
        this_thread::yield();
    }

    done2 = true;
    cv2.notify_one();
}

void Infer::deepsort_worker() {
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
        this_thread::yield();
    }  

    done3 = true;
    cv3.notify_one();
}

void Infer::imshow_worker() {
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
        if (cv::waitKey(1) == 27) {
            std::cout << "Esc key is pressed by user. Stopping the video." << std::endl;
            running = false;
            cv1.notify_all();
            cv2.notify_all();
            cv3.notify_all();
            break;
        }
        this_thread::yield();
    }
}
