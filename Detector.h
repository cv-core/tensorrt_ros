#pragma once

#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include "Utils.h"

//#define PROFILE

static const int INPUT_CHANNEL = 3;
static const int CHECK_COUNT = 3;
static const int BATCH_SIZE = 2;

struct YoloKernel
{
    int width;
    int height;
    float anchors[CHECK_COUNT * 2];
};

struct Detection{
    float bbox[4];
    int classId;
    float prob;
};

class Detector
{
public:
    Detector(std::string onnxFile, std::string trtFile, int input_w, int input_h, int num_classes, float yolo_thresh, float nms_thresh);
    ~Detector();
    std::vector<std::vector<Detection>> doInference(std::vector<cv::Mat>& imgs);
private:
    void postProcessImg(cv::Mat& img, std::vector<Detection>& detections);
    void doNms(std::vector<Detection>& detections, float nmsThresh);
    std::vector<std::vector<Detection>> interpretOutputTensor(float *tensor, int batchSize);

    Logger logger_;
    Profiler profiler_;
    nvinfer1::IExecutionContext* context_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IRuntime* runtime_;
    std::unique_ptr<float[]> inputData_;
    std::unique_ptr<float[]> outputData_;
    cudaStream_t stream_;
    std::unique_ptr<void*[]> buffers_;

    std::vector<YoloKernel> yoloKernel_;

    int inputW_;
    int inputH_;
    int numClasses_;
    float yoloThresh_;
    float nmsThresh_;

    const YoloKernel yolo1_;
    const YoloKernel yolo2_;
    const YoloKernel yolo3_;
};
