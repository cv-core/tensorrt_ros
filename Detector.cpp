#include "Detector.h"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cublas_v2.h>
#include <cudnn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <time.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std;
using namespace nvinfer1;

Detector::Detector(string onnxFile, string trtFile, int input_w, int input_h, int num_classes, float yolo_thresh, float nms_thresh)
    : logger_(Logger::Severity::kINFO),
      profiler_("Layer times"),
      inputW_(input_w),
      inputH_(input_h),
      numClasses_(num_classes),
      yoloThresh_(yolo_thresh),
      nmsThresh_(nms_thresh),
      yolo1_({
              input_w / 32,
              input_h / 32,
              {116,90,  156,198,  373,326}
              }),

      yolo2_({
              input_w / 16,
              input_h / 16,
              {30,61,  62,45,  59,119}
              }),
      yolo3_({
              input_w / 8,
              input_h / 8,
              {10,13,  16,30,  33,23}
              })
{
    runtime_ = createInferRuntime(logger_);
    assert(runtime_ != nullptr);
    runtime_->setDLACore(0);

    engine_ = engineFromFiles(onnxFile, trtFile, runtime_, BATCH_SIZE, logger_, false);
    context_ = engine_->createExecutionContext();
#ifdef PROFILE
    context_->setProfiler(&profiler_);
#endif
    assert(context_ != nullptr);

    int64_t outputCount = 0;
    int nbBindings = engine_->getNbBindings();
    for (int i = 0; i < nbBindings; i++)
    {
        if (!engine_->bindingIsInput(i))
        {
            outputCount += volume(engine_->getBindingDimensions(i));
        }
    }
    outputData_.reset(new float[outputCount * BATCH_SIZE]);
    inputData_.reset(new float[inputW_ * inputH_ * INPUT_CHANNEL * BATCH_SIZE]);

    CUDA_CHECK(cudaStreamCreate(&stream_));

    buffers_.reset(new void*[nbBindings]);

    for (int b = 0; b < nbBindings; ++b)
    {
        int64_t size = volume(engine_->getBindingDimensions(b));
        CUDA_CHECK(cudaMalloc(&buffers_.get()[b], size * BATCH_SIZE * sizeof(float)));
    }

    yoloKernel_.push_back(yolo1_);
    yoloKernel_.push_back(yolo2_);
    yoloKernel_.push_back(yolo3_);
}

Detector::~Detector()
{
    // release the stream and the buffers
    cudaStreamDestroy(stream_);
    for (int b = 0; b < engine_->getNbBindings(); ++b)
    {
        CUDA_CHECK(cudaFree(buffers_.get()[b]));
    }
    // destroy the engine_
    context_->destroy();
    engine_->destroy();
    runtime_->destroy();
}

void Detector::postProcessImg(cv::Mat& img, vector<Detection>& detections)
{
    int h = inputH_;
    int w = inputW_;

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w) / width, float(h) / height);
    float scaleSize[] = {width * scale, height * scale};

    //correct box
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }

    if(nmsThresh_ > 0)
        doNms(detections, nmsThresh_);
}

void Detector::doNms(vector<Detection>& detections, float nmsThresh)
{
    vector<vector<Detection>> resClass;
    resClass.resize(numClasses_);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f, rbox[0] - rbox[2]/2.f),
            min(lbox[0] + lbox[2]/2.f, rbox[0] + rbox[2]/2.f),
            max(lbox[1] - lbox[3]/2.f, rbox[1] - rbox[3]/2.f),
            min(lbox[1] + lbox[3]/2.f, rbox[1] + rbox[3]/2.f),
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS = (interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS / (lbox[2]*lbox[3] + rbox[2]*rbox[3] - interBoxS);
    };

    auto fullOverlap = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f, rbox[0] - rbox[2]/2.f),
            min(lbox[0] + lbox[2]/2.f, rbox[0] + rbox[2]/2.f),
            max(lbox[1] - lbox[3]/2.f, rbox[1] - rbox[3]/2.f),
            min(lbox[1] + lbox[3]/2.f, rbox[1] + rbox[3]/2.f),
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return false;

        if (interBox[0] >= lbox[0] - lbox[2]/2.f && interBox[0] >= rbox[0] - rbox[2]/2.f
                && interBox[1] <= lbox[0] + lbox[2]/2.f && interBox[1] <= rbox[0] + rbox[2]/2.f
                && interBox[2] >= lbox[1] - lbox[3]/2.f && interBox[2] >= rbox[1] - rbox[3]/2.f
                && interBox[3] <= lbox[1] + lbox[3]/2.f && interBox[3] <= rbox[1] + rbox[3]/2.f)
            return true;
        return false;
    };

    vector<Detection> result;
    for (int i = 0; i < numClasses_; ++i)
    {
        auto& dets = resClass[i];
        if(dets.size() == 0)
            continue;

        sort(dets.begin(), dets.end(), [=](const Detection& left,const Detection& right){
                return left.prob > right.prob;
                });

        for (unsigned int m = 0; m < dets.size(); ++m)
        {
            auto& item = dets[m];
            for (unsigned int n = m + 1; n < dets.size(); ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }

        sort(dets.begin(), dets.end(), [=](const Detection& left,const Detection& right){
                return left.bbox[2] * left.bbox[3] > right.bbox[2] * right.bbox[3];
                });

        for (unsigned int m = 0; m < dets.size(); ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for (unsigned int n = m + 1; n < dets.size(); ++n)
            {
                if (fullOverlap(item.bbox, dets[n].bbox))
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    detections = move(result);
}

vector<vector<Detection>> Detector::interpretOutputTensor(float *tensor, int batchSize)
{
    auto Logist = [=](float data){
        return 1./(1. + exp(-data));
    };

    float *inputData = tensor;
    vector<vector<Detection>> results(batchSize);
    for (const auto& yolo : yoloKernel_)
    {
        for (int i = 0; i < BATCH_SIZE; i++)
        {
            int stride = yolo.width * yolo.height;
            if (i >= batchSize)
            {
                inputData += (5 + numClasses_) * stride * CHECK_COUNT;
                continue;
            }
            vector<Detection>& result = results[i];
            for (int j = 0; j < stride; ++j)
            {
                for (int k = 0; k < CHECK_COUNT; ++k)
                {
                    int beginIdx = (5 + numClasses_) * stride * k + j;
                    int objIndex = beginIdx + 4 * stride;

                    float objProb = Logist(inputData[objIndex]);
                    if(objProb <= yoloThresh_)
                        continue;

                    int classId = -1;
                    float maxProb = yoloThresh_;
                    for (int c = 0; c < numClasses_; ++c){
                        float cProb =  Logist(inputData[beginIdx + (5 + c) * stride]) * objProb;
                        if(cProb > maxProb){
                            maxProb = cProb;
                            classId = c;
                        }
                    }

                    if(classId >= 0) {
                        Detection det;
                        int row = j / yolo.width;
                        int cols = j % yolo.width;

                        det.bbox[0] = (cols + Logist(inputData[beginIdx])) / yolo.width;
                        det.bbox[1] = (row + Logist(inputData[beginIdx + stride])) / yolo.height;
                        det.bbox[2] = exp(inputData[beginIdx+2*stride]) * yolo.anchors[2*k];
                        det.bbox[3] = exp(inputData[beginIdx+3*stride]) * yolo.anchors[2*k + 1];
                        det.classId = classId;
                        det.prob = maxProb;

                        result.emplace_back(det);
                    }
                }
            }
            inputData += (5 + numClasses_) * stride * CHECK_COUNT;
        }
    }
    return results;
}

vector<vector<Detection>> Detector::doInference(vector<cv::Mat>& imgs)
{
    auto t_start = std::chrono::high_resolution_clock::now();
    int batchSize = imgs.size();
    float *input = inputData_.get();
    for(auto &img : imgs)
    {
        prepareImage(img, input, inputW_, inputH_, INPUT_CHANNEL);
        input += inputW_ * inputH_ * INPUT_CHANNEL;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    float total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    // std::cout << "Time taken for scaling is " << total << " ms." << std::endl;
    t_start = std::chrono::high_resolution_clock::now();

    int nbBindings = engine_->getNbBindings();

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CUDA_CHECK(cudaMemcpyAsync(buffers_.get()[0], inputData_.get(), batchSize * volume(engine_->getBindingDimensions(0)) * sizeof(float), cudaMemcpyHostToDevice, stream_));
#ifdef PROFILE
    context_->execute(batchSize, buffers_.get());
#else
    context_->enqueue(batchSize, buffers_.get(), stream_, nullptr);
#endif
    float *output = outputData_.get();
    for (int b = 0; b < nbBindings; ++b)
    {
        if (!engine_->bindingIsInput(b))
        {
            int64_t size = volume(engine_->getBindingDimensions(b));
            CUDA_CHECK(cudaMemcpyAsync(output, buffers_.get()[b], BATCH_SIZE * size * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            output += BATCH_SIZE * size;
        }
    }
    cudaStreamSynchronize(stream_);
#ifdef PROFILE
    std::cout << profiler_;
#endif

    t_end = std::chrono::high_resolution_clock::now();
    total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    // std::cout << "Time taken for inference is " << total << " ms." << std::endl;

    vector<vector<Detection>> results = interpretOutputTensor(outputData_.get(), batchSize);
    for (int i = 0; i < batchSize; i++)
    {
        postProcessImg(imgs[i], results[i]);
    }
    return results;
}

