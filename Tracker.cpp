#include "Tracker.h"

using namespace std;

Tracker::Tracker()
{
    detector_ = cv::GFTTDetector::create(100, 0.1, 3);
}

void Tracker::resetTracks(cv::Mat &img, std::vector<cv::Rect2d> &boxes)
{
#ifndef USE_KLT
    tracker_ = cv::MultiTracker::create();
    for (cv::Rect2d box : boxes)
    {
        tracker_->add(cv::TrackerMOSSE::create(), img, box);
    }
#else
    points_[0].clear();
    points_[1].clear();
    num_points_.clear();
    averages_.clear();
    cv::cvtColor(img, prev_img_, cv::COLOR_BGR2GRAY);
    for (cv::Rect2d &b : boxes)
    {
        cv::Mat roi = prev_img_(b);
        //cv::Mat ycrcb;
        //cv::cvtColor(roi,ycrcb,CV_BGR2YCrCb);
        //vector<cv::Mat> channels;
        //cv::split(ycrcb,channels);
        //cv::equalizeHist(channels[0], channels[0]);
        //cv::merge(channels,ycrcb);
        //cv::cvtColor(ycrcb,roi,CV_YCrCb2BGR);
        std::vector<cv::KeyPoint> keypoints;
        detector_->detect(roi, keypoints);
        //cv::drawKeypoints(roi, keypoints, roi);
        //cv::imshow("roi", roi);
        //cv::waitKey(0);
        num_points_.push_back(keypoints.size());
        cv::Point2f avg(0, 0);
        for (cv::KeyPoint &k : keypoints)
        {
            cv::Point2f p = k.pt + static_cast<cv::Point2f>(b.tl());
            avg += p;
            points_[0].push_back(p);
        }
        avg *= 1.0 / keypoints.size();
        averages_.push_back(avg);
    }
    if (points_[0].size() > 0)
        cv::cornerSubPix(prev_img_, points_[0], cv::Size(10,10), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03));
    boxes_ = boxes;
#endif
}

std::vector<cv::Rect2d> Tracker::updateTracks(cv::Mat &img, std::vector<std::vector<cv::Point2f>> &keypoints)
{
#ifndef USE_KLT
    if (!tracker_)
    {
        return std::vector<cv::Rect2d>();
    }
    tracker_->update(img);
    return tracker_->getObjects();
#else
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    keypoints.clear();
    keypoints.resize(num_points_.size());
    if (!points_[0].empty())
    {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::calcOpticalFlowPyrLK(prev_img_, gray, points_[0], points_[1], status, err, cv::Size(50, 50), 3, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03), 0, 0.0001);
        int start_inx = 0;
        int inx = 0;
        for (unsigned int j = 0; j < num_points_.size(); j++)
        {
            cv::Point2f avg(0, 0);
            int old_size = num_points_[j];
            for (int i = 0; i < old_size; i++)
            {
                if (!status[start_inx + i])
                {
                    num_points_[j]--;
                    continue;
                }
                points_[1][inx++] = points_[1][start_inx + i];
                avg += points_[1][start_inx + i];
        keypoints[j].push_back(points_[1][start_inx + i]);
            }
            avg *= 1.0 / num_points_[j];
            cv::Point2f old_avg = averages_[j];
            boxes_[j].x += (avg - old_avg).x;
            boxes_[j].y += (avg - old_avg).y;
            averages_[j] = avg;
            start_inx += old_size;
        }
        points_[1].resize(inx);
        swap(points_[1], points_[0]);
        cv::swap(prev_img_, gray);
    }
#endif
    return boxes_;
}
