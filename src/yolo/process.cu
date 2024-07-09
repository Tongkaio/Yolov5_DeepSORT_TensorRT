#include "process.cuh"


__global__ void warp_affine_bilinear_kernel(uint8_t* src,
                                            int src_width,
                                            int src_height,
                                            float* dst,
                                            int dst_width,
                                            int dst_height,
                                            uint8_t fill_value,
                                            float* d2i) {
    // dst像素坐标，每个线程负责一个dst中的像素
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height) return;
    // 双线性插值
    // 1. 计算过当前像素坐标在src中的坐标，需要逆矩阵d2i
    float src_x = d2i[0] * dx + d2i[1] * dy + d2i[2];
    float src_y = d2i[3] * dx + d2i[4] * dy + d2i[5];

    // 2. 计算src坐标的左上和右下坐标
    int low_y = floorf(src_y);
    int low_x = floorf(src_x);
    int high_y = low_y + 1;
    int high_x = low_x + 1;
    
    // 3. 计算四个区域的宽高
    float height1 = src_y - low_y;
    float height2 = 1 - height1;
    float width1  = src_x - low_x;
    float width2  = 1 - width1;
    
    // 4. 每个像素在插值中的权重，等于其对角矩形区域的面积
    float w11 = height2 * width2;
    float w12 = height2 * width1;
    float w21 = height1 * width2;
    float w22 = height1 * width1;

    // 5. 三通道像素值默认为fill_value(114)灰色
    uint8_t fill_value_3_channel[] = {fill_value, fill_value, fill_value};
    uint8_t* v1 = fill_value_3_channel;
    uint8_t* v2 = fill_value_3_channel;
    uint8_t* v3 = fill_value_3_channel;
    uint8_t* v4 = fill_value_3_channel;

    // 6. 如果四个角像素在有效范围内，则将像素值更新为该像素(含三通道)
    if (low_y >= 0 && low_x >= 0 && low_y < src_height && low_x < src_width) {
        v1 = src + low_y * src_width * 3 + low_x * 3;  // 左上元素像素值(b,g,r)
    }
    if (low_y >= 0 && high_x >= 0 && low_y < src_height && high_x < src_width) {
        v2 = src + low_y * src_width * 3 + high_x * 3;  // 右上元素像素值(b,g,r)
    }
    if (high_y >= 0 && low_x >= 0 && high_y < src_height && low_x < src_width) {
        v3 = src + high_y * src_width * 3 + low_x * 3;  // 左下元素像素值(b,g,r)
    }
    if (high_y >= 0 && high_x >= 0 && high_y < src_height && high_x < src_width) {
        v4 = src + high_y * src_width * 3 + high_x * 3;  // 右下元素像素值(b,g,r)
    }

    // 7. 双线性插值，权重*像素值，bgr三通道都要算一遍
    float c0, c1, c2;  // 三通道像素值bgr
    c0 = w11 * v1[0] + w12 * v2[0] + w21 * v3[0] + w22 * v4[0];  // b
    c1 = w11 * v1[1] + w12 * v2[1] + w21 * v3[1] + w22 * v4[1];  // g
    c2 = w11 * v1[2] + w12 * v2[2] + w21 * v3[2] + w22 * v4[2];  // r

    // 8. 将像素值赋给dst
    // 8.1 bgr=>rgb: dst_c0是r，存储src_c2(r)；dst_c2是b，存储src_c0(b)；
    // 8.2 除以255进行归一化
    int area = dst_width * dst_height;
    float* dst_c0 = dst + dy * dst_width + dx;  // r层
    float* dst_c1 = dst_c0 + area;             // g层
    float* dst_c2 = dst_c1 + area;             // b层
    *dst_c0 = c2 / 255.0f;  // r
    *dst_c1 = c1 / 255.0f;  // g
    *dst_c2 = c0 / 255.0f;  // b
}

void warp_affine_bilinear(uint8_t* src,
                          int src_width,
                          int src_height,
                          float* dst,
                          int dst_width,
                          int dst_height,
                          uint8_t fill_value,
                          float* d2i,
                          cudaStream_t stream) {
    dim3 block_size(32, 32);
    dim3 grid_size((dst_width + 31)/32, (dst_height + 31)/32);
    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, stream>>>(
        src, src_width, src_height,
        dst, dst_width, dst_height,
        fill_value, d2i        
    );
}

__global__ void decode_kernel(float* predict, 
                              int num_bboxes,
                              int num_classes,
                              float confidence_threshold, 
                              float* invert_affine_matrix,
                              float* parray,
                              int max_objects,
                              int NUM_BOX_ELEMENT) {

    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem     = predict + (5 + num_classes) * position;
    float objectness = pitem[4];
    if(objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5;
    float confidence        = class_confidence[0];
    int label               = 0;
    for(int i = 1; i < num_classes; ++i){
        if(class_confidence[i] > confidence) {
            confidence = class_confidence[i];
            label      = i;
        }
    }

    if (label != 0) {  // only detect and track person(cocolabels[0])
        return;
    }

    confidence *= objectness;
    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);
    if(index >= max_objects)
        return;

    float cx         = pitem[0];
    float cy         = pitem[1];
    float width      = pitem[2];
    float height     = pitem[3];
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // left, top, right, bottom, confidence, class, keepflag
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    pout_item[0] = left;
    pout_item[1] = top;
    pout_item[2] = right;
    pout_item[3] = bottom;
    pout_item[4] = confidence;
    pout_item[5] = label;
    pout_item[6] = 1; // 1 = keep, 0 = ignore
}

__global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT) {

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);
    if (position >= count) 
        return;
    
    // left, top, right, bottom, confidence, class, keepflag
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]) continue;

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
}

void decode_kernel_invoker(float* predict,
                           int num_bboxes,
                           int num_classes,
                           float confidence_threshold,
                           float nms_threshold,
                           float* invert_affine_matrix,
                           float* parray,
                           int max_objects,
                           int NUM_BOX_ELEMENT,
                           cudaStream_t stream) {

    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;

    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold, 
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    );

    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}