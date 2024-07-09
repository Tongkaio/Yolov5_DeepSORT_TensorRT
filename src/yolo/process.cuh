#ifndef PROCESS_CUH
#define PROCESS_CUH

#include <stdint.h>
#include <cuda_runtime.h>

static __device__  inline void affine_project(float* matrix, float x, float y, float* ox, float* oy) {
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __device__ inline float box_iou(float aleft, float atop, float aright, float abottom, 
                                float bleft, float btop, float bright, float bbottom) {

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

__global__ void warp_affine_bilinear_kernel(uint8_t* src,
                                            int src_width,
                                            int src_height,
                                            float* dst,
                                            int dst_width,
                                            int dst_height,
                                            uint8_t fill_value,
                                            float* d2i);

void warp_affine_bilinear(uint8_t* src,
                          int src_width,
                          int src_height,
                          float* dst,
                          int dst_width,
                          int dst_height,
                          uint8_t fill_value,
                          float* d2i,
                          cudaStream_t stream);


__global__ void decode_kernel(float* predict, 
                              int num_bboxes,
                              int num_classes,
                              float confidence_threshold, 
                              float* invert_affine_matrix,
                              float* parray,
                              int max_objects,
                              int NUM_BOX_ELEMENT);

__global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT);

void decode_kernel_invoker(float* predict,
                           int num_bboxes,
                           int num_classes,
                           float confidence_threshold,
                           float nms_threshold,
                           float* invert_affine_matrix,
                           float* parray,
                           int max_objects,
                           int NUM_BOX_ELEMENT,
                           cudaStream_t stream);

#endif  // PROCESS_CUH