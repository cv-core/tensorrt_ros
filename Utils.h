#pragma once

#include "NvInfer.h"
#include "EntropyCalibrator.h"
#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "NvOnnxParser.h"
#include <numeric>


#define CUDA_CHECK(callstr)                                                                    \
{                                                                                          \
    cudaError_t error_code = callstr;                                                      \
    if (error_code != cudaSuccess) {                                                       \
        std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; \
        assert(0);                                                                         \
    }                                                                                      \
}

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
            return;

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR: std::cerr << "INTERNAL_ERROR: "; break;
        case Severity::kERROR: std::cerr << "ERROR: "; break;
        case Severity::kWARNING: std::cerr << "WARNING: "; break;
        case Severity::kINFO: std::cout << "INFO: "; break;
        default: std::cout << "UNKNOWN: "; break;
        }
        std::cout << msg << std::endl;
    }

    Severity reportableSeverity;
};

struct Profiler : public nvinfer1::IProfiler
{
    struct Record
    {
        float time{0};
        int count{0};
    };

    virtual void reportLayerTime(const char* layerName, float ms)
    {
        mProfile[layerName].count++;
        mProfile[layerName].time += ms;
    }

    Profiler(
        const char* name,
        const std::vector<Profiler>& srcProfilers = std::vector<Profiler>())
        : mName(name)
    {
        for (const auto& srcProfiler : srcProfilers)
        {
            for (const auto& rec : srcProfiler.mProfile)
            {
                auto it = mProfile.find(rec.first);
                if (it == mProfile.end())
                {
                    mProfile.insert(rec);
                }
                else
                {
                    it->second.time += rec.second.time;
                    it->second.count += rec.second.count;
                }
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& out, const Profiler& value)
    {
        out << "========== " << value.mName << " profile ==========" << std::endl;
        float totalTime = 0;
        std::string layerNameStr = "TensorRT layer name";
        int maxLayerNameLength = std::max(static_cast<int>(layerNameStr.size()), 70);
        for (const auto& elem : value.mProfile)
        {
            totalTime += elem.second.time;
            maxLayerNameLength = std::max(maxLayerNameLength, static_cast<int>(elem.first.size()));
        }

        auto old_settings = out.flags();
        auto old_precision = out.precision();
        // Output header
        {
            out << std::setw(maxLayerNameLength) << layerNameStr << " ";
            out << std::setw(12) << "Runtime, "
                << "%"
                << " ";
            out << std::setw(12) << "Invocations"
                << " ";
            out << std::setw(12) << "Runtime, ms" << " ";
            out << std::setw(12) << "Single Runtime, ms" << std::endl;
        }
        for (const auto& elem : value.mProfile)
        {
            out << std::setw(maxLayerNameLength) << elem.first << " ";
            out << std::setw(12) << std::fixed << std::setprecision(1) << (elem.second.time * 100.0F / totalTime) << "%"
                << " ";
            out << std::setw(12) << elem.second.count << " ";
            out << std::setw(12) << std::fixed << std::setprecision(2) << elem.second.time << " ";
            out << std::setw(12) << std::fixed << std::setprecision(4) << elem.second.time / (float)elem.second.count << std::endl;
        }
        out.flags(old_settings);
        out.precision(old_precision);
        out << "========== " << value.mName << " total runtime = " << totalTime << " ms ==========" << std::endl;

        return out;
    }

private:
    std::string mName;
    std::map<std::string, Record> mProfile;
};

void prepareImage(cv::Mat& img, float *data, int w, int h, int c, bool cvtColor = true, bool padCenter = true, bool pad = true, bool normalize = true);

void setLayerPrecision(nvinfer1::INetworkDefinition*& network);

void setDynamicRange(nvinfer1::INetworkDefinition*& network);

void onnxToTRTModel(const std::string& modelFile,
                    unsigned int maxBatchSize,
                    nvinfer1::IHostMemory*& trtModelStream, Logger &logger, bool useInt8 = true, bool markOutput = false, nvinfer1::IInt8EntropyCalibrator* calibrator = nullptr);

nvinfer1::ICudaEngine* engineFromFiles(std::string onnxFile, std::string trtFile, nvinfer1::IRuntime *runtime,
                                       int batchSize, Logger &logger, bool useInt8 = true, bool markOutput = false, nvinfer1::IInt8EntropyCalibrator* calibrator = nullptr);

inline int64_t volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}
