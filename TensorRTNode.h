#pragma once

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/Image.h>
#include <tensorrt_ros/BoundingBoxes.h>
#include <tensorrt_ros/BoundingBox.h>
#include <geometry_msgs/Point.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/image_encodings.h>

#include "Detector.h"
#include "ColorDetector.h"
#include "KeypointDetector.h"
#include "Tracker.h"

#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <boost/thread.hpp>
#include <mutex>
#include <condition_variable>

//#define TRACKING

namespace tensorrt_node
{

class TensorRTNode
{
public:
    TensorRTNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private);
    TensorRTNode() : TensorRTNode(ros::NodeHandle(), ros::NodeHandle("~")) {}
    ~TensorRTNode();
    void cameraCallback(const sensor_msgs::ImageConstPtr& msg1, const sensor_msgs::ImageConstPtr& msg2);
    void timerCallback(const ros::WallTimerEvent& event);
private:
    void detect();
    tensorrt_ros::BoundingBoxes processDetections(std::vector<Detection> &detections, cv::Mat &img, bool isCamera1);
    void drawDetections(cv::Mat &img, tensorrt_ros::BoundingBoxes &boxes);
    void rgbToHsv(float r, float g, float b, float *h, float *s, float *v);
    bool isInRange(float h, float s, float v, int color);
    int coneColors(std::vector<uint32_t> image_rgb);
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    image_transport::ImageTransport imageTransport_;
    image_transport::SubscriberFilter imageSubscriber1_;
    image_transport::SubscriberFilter imageSubscriber2_;
    std::unique_ptr<message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>> sync_;

    ros::Publisher boundingBoxesPublisher1_;
    image_transport::Publisher detectionImagePublisher1_;
    ros::Publisher boundingBoxesPublisher2_;
    image_transport::Publisher detectionImagePublisher2_;

    std::unique_ptr<Detector> detector_;
    std::unique_ptr<ColorDetector> colorDetector_;
    std::unique_ptr<KeypointDetector> keypointDetector_;
    std::unique_ptr<Tracker> tracker1_;
    std::unique_ptr<Tracker> tracker2_;

    std::shared_ptr<boost::thread> detectThread_;
    std::mutex detectLock_;
    std::condition_variable detectCond_;
    std::vector<cv::Mat> images_to_detect_;
    std::vector<tensorrt_ros::BoundingBoxes> detect_results_;
    std::vector<tensorrt_ros::BoundingBoxes> detections_to_track_;

    ros::WallTimer timer_;

    bool watchdog_;
    bool ranOnce_;
    bool cameraSwitch_;

    std::string camera1Topic;
    std::string camera2Topic;
    std::string carType_;

    double boxMinSizeRatio_;

    // 5/5/19 DUT18D
    const double x_b1{0.3175};
    const double x_b2{0.66};
    const double x_t1{0.434};
    const double x_t2{0.546};
    const double y_t{0.534};
};

}
