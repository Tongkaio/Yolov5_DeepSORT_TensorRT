#include "infer.h"

#include <stdio.h>
#include <iostream>

// hsv to bgr
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v){
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f*s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0:r = v; g = t; b = p;break;
    case 1:r = q; g = v; b = p;break;
    case 2:r = p; g = v; b = t;break;
    case 3:r = p; g = q; b = v;break;
    case 4:r = t; g = p; b = v;break;
    case 5:r = v; g = p; b = q;break;
    default:r = 1; g = 1; b = 1;break;}
    return std::make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}

static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id){
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}

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
    while (cap.read(image)) {
        {
            std::unique_lock<std::mutex> lock1(this->mtx1);
            this->cv1.wait(lock1, [&](){ return this->q_pics.size() < this->MAX_LENGTH; });
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
    while (true) {
        std::unique_lock<std::mutex> lock1(this->mtx1);
        this->cv1.wait(lock1, [&](){ return !this->q_pics.empty() || done1; });

        if (q_pics.empty() && done1) break;

        cv::Mat image = this->q_pics.front();
        this->q_pics.pop();
        if (!done1) this->cv1.notify_one();  // 通知 video_worker 队列不满
        lock1.unlock();

        std::vector<DetectBox> allDetections;

        this->yolo->detect(image, allDetections);

        {
            std::unique_lock<std::mutex> lock2(this->mtx2);
            this->cv2.wait(lock2, [&](){ return q_detects.size() < this->MAX_LENGTH; });
            q_detects.push({image, allDetections});  // 创建副本并推入队列
        }
        cv2.notify_one();
        this_thread::yield();
    }

    done2 = true;
    cv2.notify_one();
}

void Infer::deepsort_worker() {
    while (true) {
        std::unique_lock<std::mutex> lock2(this->mtx2);
        this->cv2.wait(lock2, [&](){
            return !this->q_detects.empty() || done2;
        });

        if (q_detects.empty() && done2) break;

        cv::Mat image = this->q_detects.front().first;
        std::vector<DetectBox> allDetections = this->q_detects.front().second;
        this->q_detects.pop();
        if (!done2) this->cv2.notify_one();  // 通知 yolo_worker 队列不满
        lock2.unlock();

        DS->sort(image, allDetections);

        {
            std::unique_lock<std::mutex> lock3(this->mtx3);
            cv3.wait(lock3, [&](){ return this->q_sorts.size() < this->MAX_LENGTH; });
            q_sorts.push({image.clone(), allDetections});  // 创建副本并推入队列
        }
        cv3.notify_one();
        this_thread::yield();
    }  

    done3 = true;
    cv3.notify_one();
}

void Infer::imshow_worker() {
    while (true) {
        std::unique_lock<std::mutex> lock3(this->mtx3);

        cv3.wait(lock3, [&](){ return !this->q_sorts.empty() || done3; });

        if (this->q_sorts.empty() && done3) break;

        cv::Mat image_result = this->q_sorts.front().first;
        std::vector<DetectBox> allDetections = this->q_sorts.front().second;
        this->q_sorts.pop();
        lock3.unlock();
        if (!done3) this->cv3.notify_one();  // 通知 deepsort_worker 队列不满
        
        // draw bbox
        for (auto box : allDetections) {
            float left = box.x1;
            float top = box.y1;
            float right = box.x2;
            float bottom = box.y2;
            int class_label = box.classID;
            int track_label = box.trackID;
            float confidence = box.confidence;
            cv::Scalar color;
            tie(color[0], color[1], color[2]) = random_color(class_label);
            cv::rectangle(image_result, cv::Point(left, top), cv::Point(right, bottom), color, 3);

            auto name      = cocolabels[class_label];
            auto caption   = cv::format("%s ID: %d", name, track_label);
            int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image_result, cv::Point(left-3, top-33), cv::Point(left + text_width, top), color, -1);
            cv::putText(image_result, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        }

        cv::imshow("Result", image_result);
        if (cv::waitKey(1) == 27) {
            std::cout << "Esc key is pressed by user. Stopping the video." << std::endl;
            break;
        }
        this_thread::yield();
    }
}
