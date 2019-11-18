#pragma once

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "Utils.h"

static const int KEYPOINTS_CHANNEL = 3;
static const int NUM_KEYPOINTS = 7;
static const int KEYPOINT_SCALE = 1;

// #define KPT_PROFILE

class KeypointDetector
{
public:
    KeypointDetector(std::string onnxFile, std::string trtFile, int input_w, int input_h, int max_batch);
    ~KeypointDetector();
    std::vector<std::vector<cv::Point2f>> doInference(std::vector<cv::Mat>& imgs);
private:
    std::vector<cv::Point2f> interpretOutputTensor(float *tensor, int width, int height);
    Logger logger_;
#ifdef KPT_PROFILE
    nvinfer1::Profiler profiler_;
#endif
    nvinfer1::IExecutionContext* context_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IRuntime* runtime_;
    std::unique_ptr<float[]> inputData_;
    std::unique_ptr<float[]> outputData_;
    cudaStream_t stream_;
    std::unique_ptr<void*[]> buffers_;

    int inputW_;
    int inputH_;
    int maxBatch_;
};
