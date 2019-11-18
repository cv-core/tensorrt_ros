#include "KeypointDetector.h"

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

using namespace nvinfer1;
using namespace std;

KeypointDetector::KeypointDetector(string onnxFile, string trtFile, int input_w, int input_h, int max_batch)
    : logger_(Logger::Severity::kINFO),
#ifdef KPT_PROFILE
      profiler_("Layer times"),
#endif
      inputW_(input_w),
      inputH_(input_h),
      maxBatch_(max_batch)
{
    runtime_ = createInferRuntime(logger_);
    assert(runtime_ != nullptr);
    runtime_->setDLACore(0);

    engine_ = engineFromFiles(onnxFile, trtFile, runtime_, maxBatch_, logger_, false);

    context_ = engine_->createExecutionContext();
#ifdef KPT_PROFILE
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
    outputData_.reset(new float[outputCount * maxBatch_]);
    inputData_.reset(new float[inputW_ * inputH_ * KEYPOINTS_CHANNEL * maxBatch_]);

    CUDA_CHECK(cudaStreamCreate(&stream_));

    buffers_.reset(new void*[nbBindings]);

    for (int b = 0; b < nbBindings; ++b)
    {
        int64_t size = volume(engine_->getBindingDimensions(b));
        CUDA_CHECK(cudaMalloc(&buffers_.get()[b], size * maxBatch_ * sizeof(float)));
    }
}

KeypointDetector::~KeypointDetector()
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

vector<cv::Point2f> KeypointDetector::interpretOutputTensor(float *tensor, int width, int height)
{
    const int size = inputW_ * inputH_ / (KEYPOINT_SCALE * KEYPOINT_SCALE);
    //const float scaleX = max(width / float(inputW_ / KEYPOINT_SCALE), height / float(inputH_ / KEYPOINT_SCALE));
    //const float scaleY = scaleX;
    const float scaleX = width / float(inputW_ / KEYPOINT_SCALE);
    const float scaleY = height / float(inputH_ / KEYPOINT_SCALE);
    vector<cv::Point2f> kpt(NUM_KEYPOINTS);
    for (int i = 0; i < NUM_KEYPOINTS; i++)
    {
        int largest = distance(tensor, max_element(tensor, tensor + size));
        //tensor[largest] = 0;
        //int second = distance(tensor, max_element(tensor, tensor + size));
        kpt[i] = cv::Point2f((largest % inputW_) * scaleX, (largest / inputW_) * scaleY);//cv::Point2f((3 * (largest % (inputW_ / KEYPOINT_SCALE)) + second % (inputW_ / KEYPOINT_SCALE)) / 4. * scale, (3 * (largest / (inputW_ / KEYPOINT_SCALE)) + second / (inputW_ / KEYPOINT_SCALE)) / 4. * scale);
        cv::Mat outImg(inputW_, inputH_, CV_32FC1);
        memcpy(outImg.data, tensor, sizeof(float) * inputW_ * inputH_);
        cv::Mat normImg(inputW_, inputH_, CV_8UC1);
	outImg *= 255 / tensor[largest];
        outImg.convertTo(normImg, CV_8UC1);
        //cv::imshow("kpt", normImg);
        //cv::waitKey(0);
	cv::threshold(normImg, normImg, 32, 0, cv::THRESH_TOZERO);
	cv::GaussianBlur(normImg, normImg, cv::Size(5, 5), 0, 0);
	cv::Mat mask;
	cv::dilate(normImg, mask, cv::Mat());
	cv::compare(normImg, mask, mask, cv::CMP_GE);
        cv::Mat non_plateau_mask;
        cv::erode(normImg, non_plateau_mask, cv::Mat());
        cv::compare(normImg, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
        cv::bitwise_and(mask, non_plateau_mask, mask);

        //cv::imshow("kpt", mask);
        //cv::waitKey(0);
        tensor += size;
    }
    return kpt;
}

vector<vector<cv::Point2f>> KeypointDetector::doInference(vector<cv::Mat>& imgs)
{
    int batchSize = imgs.size();
    float *input = inputData_.get();
    for(auto &img : imgs)
    {
        prepareImage(img, input, inputW_, inputH_, KEYPOINTS_CHANNEL, false, false, false);
	//for (int c = 0; c < 3; c++)
	//{
        //cv::Mat outImg(inputW_, inputH_, CV_32FC1);
        //memcpy(outImg.data, input + c * inputW_ * inputH_, sizeof(float) * inputW_ * inputH_);
        //cv::Mat normImg(inputW_, inputH_, CV_8UC1);
	//outImg *= 255;
        //outImg.convertTo(normImg, CV_8UC1);
        //cv::imshow("kpt", normImg);
        //cv::waitKey(0);
	//}
        input += inputW_ * inputH_ * KEYPOINTS_CHANNEL;
    }
    auto t_start = std::chrono::high_resolution_clock::now();

    int nbBindings = engine_->getNbBindings();

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CUDA_CHECK(cudaMemcpyAsync(buffers_.get()[0], inputData_.get(), batchSize * volume(engine_->getBindingDimensions(0)) * sizeof(float), cudaMemcpyHostToDevice, stream_));
#ifdef KPT_PROFILE
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
            CUDA_CHECK(cudaMemcpyAsync(output, buffers_.get()[b], batchSize * size * sizeof(float), cudaMemcpyDeviceToHost, stream_));
            output += maxBatch_ * size;
        }
    }
    cudaStreamSynchronize(stream_);
#ifdef KPT_PROFILE
    cout << profiler_;
#endif

    auto t_end = std::chrono::high_resolution_clock::now();
    auto total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
//    std::cout << "Time taken for keypoints is " << total << " ms." << std::endl;

    output = outputData_.get();
    vector<vector<cv::Point2f>> results(batchSize);
    for (int b = 0; b < batchSize; b++)
    {
        results[b] = interpretOutputTensor(output, imgs[b].cols, imgs[b].rows);
        output += inputW_ * inputH_ / (KEYPOINT_SCALE * KEYPOINT_SCALE) * NUM_KEYPOINTS;
    }
    return results;
}

