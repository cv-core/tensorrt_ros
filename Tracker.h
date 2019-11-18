#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <vector>

#define USE_KLT


class Tracker
{
public:
    Tracker();
    void resetTracks(cv::Mat &img, std::vector<cv::Rect2d> &boxes);
    std::vector<cv::Rect2d> updateTracks(cv::Mat &img, std::vector<std::vector<cv::Point2f>> &keypoints);

private:
#ifndef USE_KLT
    cv::Ptr<cv::MultiTracker> tracker_;
#else
    cv::Ptr<cv::GFTTDetector> detector_;
    std::vector<cv::Point2f> points_[2];
    std::vector<int> num_points_;
    std::vector<cv::Rect2d> boxes_;
    std::vector<cv::Point2f> averages_;
    cv::Mat prev_img_;
#endif
};

