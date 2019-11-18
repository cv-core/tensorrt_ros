#include "Utils.h"

using namespace nvinfer1;
using namespace std;

//#define USE_GPU

void prepareImage(cv::Mat& img, float *data, int w, int h, int c, bool cvtColor, bool padCenter, bool pad, bool normalize)
{
    float scale = min(float(w) / img.cols, float(h) / img.rows);
    auto scaleSize = cv::Size(img.cols * scale, img.rows * scale);
    if (scaleSize.height < 1 || scaleSize.width < 1)
        pad = false;

#ifdef USE_GPU
    cv::cuda::GpuMat rgb_gpu;
    cv::cuda::GpuMat img_gpu(img);
    cv::cuda::resize(img_gpu, rgb_gpu, scaleSize, 0, 0, cv::INTER_NEAREST);
    if (cvtColor)
    {
        cv::cuda::cvtColor(rgb_gpu, rgb_gpu, CV_BGR2RGB);
    }
#else
    cv::Mat rgb;
    if (pad)
        cv::resize(img, rgb, scaleSize, 0, 0, cv::INTER_NEAREST);
    else
        cv::resize(img, rgb, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
    if (cvtColor)
    {
        cv::cvtColor(rgb, rgb, CV_BGR2RGB);
    }
#endif
//#ifndef USE_GPU
    //rgb_gpu.download(rgb);
//#endif

#ifdef USE_GPU
    cv::cuda::GpuMat cropped(h, w, CV_8UC3, 127);
#else
    cv::Mat cropped(h, w, CV_8UC3, cv::Scalar(127, 127, 127));
#endif
    if (pad)
    {
#ifdef USE_GPU
        if (padCenter)
        {
            cv::Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
            rgb_gpu.copyTo(cropped(rect));
        }
        else
        {
            cv::Rect rect(0, 0, scaleSize.width, scaleSize.height);
            rgb_gpu.copyTo(cropped(rect));
        }
#else
        if (padCenter)
        {
            cv::Rect rect((w - scaleSize.width) / 2, (h - scaleSize.height) / 2, scaleSize.width, scaleSize.height);
            rgb.copyTo(cropped(rect));
        }
        else
        {
            cv::Rect rect(0, 0, scaleSize.width, scaleSize.height);
            rgb.copyTo(cropped(rect));
        }
#endif
    }
    else
    {
        rgb.copyTo(cropped);
    }

    float factor = 1.0;
    if (normalize)
        factor = 1/255.0;
#ifdef USE_GPU
    cv::cuda::GpuMat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, factor);
    else
        cropped.convertTo(img_float, CV_32FC1, factor);
#else
    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, factor);
    else
        cropped.convertTo(img_float, CV_32FC1, factor);
#endif

    //HWC TO CHW
#ifdef USE_GPU
    cv::cuda::GpuMat input_channels_gpu[c];
    cv::cuda::split(img_float, input_channels_gpu);
    cv::Mat input_channels[c];
    for (int i = 0; i < c; ++i) 
    {
        input_channels_gpu[i].download(input_channels[i]);
    }
#else
    cv::Mat input_channels[c];
    cv::split(img_float, input_channels);
#endif

    int channelLength = h * w;
    for (int i = 0; i < c; ++i) 
    {
        memcpy(data, input_channels[i].data, channelLength * sizeof(float));
        data += channelLength;
    }
}

void setLayerPrecision(nvinfer1::INetworkDefinition*& network)
{
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto layer = network->getLayer(i);
        layer->setPrecision(nvinfer1::DataType::kINT8);
        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            layer->setOutputType(j, nvinfer1::DataType::kINT8);
        }
    }
}

void setDynamicRange(nvinfer1::INetworkDefinition*& network)
{
    string name = network->getLayer(0)->getInput(0)->getName();
    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        string name = network->getInput(i)->getName();
        //network->getInput(i)->setDynamicRange(-mPerTensorDynamicRangeMap.at(name), mPerTensorDynamicRangeMap.at(name));
        //for now, use a simplified version:
        network->getInput(i)->setDynamicRange(1e-12, 0.01);
    }
    for (int i = 0; i < network->getNbLayers(); ++i)
    {
        auto layer = network->getLayer(i);
        for (int j = 0; j < layer->getNbOutputs(); ++j)
        {
            std::string tensorName = layer->getOutput(j)->getName();
            layer->getOutput(j)->setDynamicRange(1e-12, 0.01);
        }
    }
}

void onnxToTRTModel(const std::string& modelFile,
                    unsigned int maxBatchSize,
                    IHostMemory*& trtModelStream, Logger &logger, bool useInt8, bool markOutput)
{
    IBuilder* builder = createInferBuilder(logger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    auto parser = nvonnxparser::createParser(*network, logger);

    std::ifstream onnx_file(modelFile.c_str(), std::ios::binary | std::ios::ate);
    std::streamsize file_size = onnx_file.tellg();
    onnx_file.seekg(0, std::ios::beg);
    std::vector<char> onnx_buf(file_size);
    if(!onnx_file.read(onnx_buf.data(), onnx_buf.size()) ) 
    {
        string msg("failed to open onnx file");
        logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    }

    if (!parser->parse(onnx_buf.data(), onnx_buf.size()))
    {
        string msg("failed to parse onnx file");
        logger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
    }

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);
    if (useInt8 && builder->platformHasFastInt8())
    {
      builder->setInt8Mode(true);
      builder->setInt8Calibrator(nullptr);
      setLayerPrecision(network);
      setDynamicRange(network);
    }
    else
    {
        builder->setFp16Mode(true);
    }
    builder->setStrictTypeConstraints(true);
    if (markOutput)
    {
        network->markOutput(*network->getLayer(network->getNbLayers()-1)->getOutput(0));
    }

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // serialize the engine, then close everything down
    parser->destroy();
    trtModelStream = engine->serialize();
    engine->destroy();
    network->destroy();
    builder->destroy();
}

ICudaEngine* engineFromFiles(string onnxFile, string trtFile, IRuntime *runtime, int batchSize, Logger &logger, bool useInt8, bool markOutput)
{
    ICudaEngine *engine;
    fstream file;
    file.open(trtFile, ios::binary | ios::in);
    if(!file.is_open())
    {
        IHostMemory* trtModelStream{nullptr};
        onnxToTRTModel(onnxFile, batchSize, trtModelStream, logger, useInt8, markOutput);
        assert(trtModelStream != nullptr);

        engine = runtime->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
        assert(engine != nullptr);
        trtModelStream->destroy();

        nvinfer1::IHostMemory* data = engine->serialize();
        std::ofstream save_file;
        save_file.open(trtFile, std::ios::binary | std::ios::out);

        save_file.write((const char*)data->data(), data->size());
        save_file.close();
    }
    else
    {
        file.seekg(0, ios::end);
        int length = file.tellg();
        file.seekg(0, ios::beg);
        std::unique_ptr<char[]> data(new char[length]);
        file.read(data.get(), length);

        file.close();

        engine = runtime->deserializeCudaEngine(data.get(), length, nullptr);
        assert(engine != nullptr);
    }
    return engine;
}
