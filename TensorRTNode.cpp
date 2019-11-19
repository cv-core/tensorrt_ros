#include "TensorRTNode.h"
#include <chrono>

using namespace std;

namespace tensorrt_node
{

TensorRTNode::TensorRTNode(const ros::NodeHandle &nh, const ros::NodeHandle &nh_private)
    : nh_(nh), nh_private_(nh_private), imageTransport_(nh)
{
    string onnxFile;
    string trtFile;
    string calibFile;
    string keypointsOnnxFile;
    string keypointsTrtFile;
    string detectionsTopic1;
    string detectionImageTopic1;
    string detectionsTopic2;
    string detectionImageTopic2;
    bool useInt8;
    int yoloW;
    int yoloH;
    int yoloClasses;
    double yoloThresh;
    double yoloNms;
    int keypointsW;
    int keypointsH;
    int maxBatch;
    double timeout;

    nh_private_.param("onnx_path", onnxFile, string("yolov3.onnx"));
    nh_private_.param("trt_path", trtFile, string("yolov3.trt"));
    nh_private_.param("calib_path", calibFile, string("yolo.txt"));
    nh_private_.param("keypoints_onnx_path", keypointsOnnxFile, string("keypoints.onnx"));
    nh_private_.param("keypoints_trt_path", keypointsTrtFile, string("keypoints.trt"));
    nh_private_.param("camera1_topic", camera1Topic, string("/camera1/image_raw"));
    nh_private_.param("camera2_topic", camera2Topic, string("/camera2/image_raw"));
    nh_private_.param("detections_topic1", detectionsTopic1, string("/tensorrt/bounding_boxes1"));
    nh_private_.param("detection_image_topic1", detectionImageTopic1, string("/tensorrt/detection_image1"));
    nh_private_.param("detections_topic2", detectionsTopic2, string("/tensorrt/bounding_boxes2"));
    nh_private_.param("detection_image_topic2", detectionImageTopic2, string("/tensorrt/detection_image2"));
    nh_private_.param("timeout", timeout, 2.0);
    nh_private_.param("yolo_width", yoloW, 800);
    nh_private_.param("yolo_height", yoloH, 800);
    nh_private_.param("yolo_classes", yoloClasses, 80);
    nh_private_.param("yolo_detection_threshold", yoloThresh, 0.998);
    nh_private_.param("yolo_nms_threshold", yoloNms, 0.25);
    nh_private_.param("keypoints_width", keypointsW, 96);
    nh_private_.param("keypoints_height", keypointsH, 96);
    nh_private_.param("max_boxes", maxBatch, 100);
    nh_private_.param("box_min_size_ratio", boxMinSizeRatio_, 0.012);
    nh_private_.param("car_type", carType_, string("dut18d"));
    nh_private_.param("use_int8", useInt8, false);

    imageSubscriber1_.subscribe(imageTransport_, camera1Topic, 3);
    imageSubscriber2_.subscribe(imageTransport_, camera2Topic, 3);
    sync_.reset(new message_filters::Synchronizer<message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>>(message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>(5), imageSubscriber1_, imageSubscriber2_));
    sync_->registerCallback(boost::bind(&TensorRTNode::cameraCallback, this, _1, _2));

    boundingBoxesPublisher1_ = nh_.advertise<tensorrt_ros::BoundingBoxes>(detectionsTopic1, 2, false);
    detectionImagePublisher1_ = imageTransport_.advertise(detectionImageTopic1, 2);
    boundingBoxesPublisher2_ = nh_.advertise<tensorrt_ros::BoundingBoxes>(detectionsTopic2, 2, false);
    detectionImagePublisher2_ = imageTransport_.advertise(detectionImageTopic2, 2);
    detector_.reset(new Detector(ros::package::getPath("tensorrt_ros") + "/" +  onnxFile, ros::package::getPath("tensorrt_ros") + "/" + trtFile, ros::package::getPath("tensorrt_ros") + "/" + calibFile, yoloW, yoloH, yoloClasses, yoloThresh, yoloNms, useInt8));
    keypointDetector_.reset(new KeypointDetector(ros::package::getPath("tensorrt_ros") + "/" +  keypointsOnnxFile, ros::package::getPath("tensorrt_ros") + "/" + keypointsTrtFile, keypointsW, keypointsH, maxBatch));
#ifdef TRACKING
    tracker1_.reset(new Tracker());
    tracker2_.reset(new Tracker());
#endif
    timer_ = nh_.createWallTimer(ros::WallDuration(timeout), &TensorRTNode::timerCallback, this);
#ifdef TRACKING
    detectThread_.reset(new boost::thread(boost::bind(&TensorRTNode::detect, this)));
#endif
    watchdog_ = true;
    ranOnce_ = false;
    cameraSwitch_ = false;
}

TensorRTNode::~TensorRTNode()
{
    if (detectThread_)
    {
        detectThread_->interrupt();
        detectThread_->join();
    }
}

void TensorRTNode::cameraCallback(const sensor_msgs::ImageConstPtr& msg1, const sensor_msgs::ImageConstPtr& msg2)
{
    ranOnce_ = true;
    watchdog_ = true;
    cv_bridge::CvImagePtr cam_image1;
    cv_bridge::CvImagePtr cam_image2;

    try {
        cam_image1 = cv_bridge::toCvCopy(msg1, sensor_msgs::image_encodings::BGR8);
        cam_image2 = cv_bridge::toCvCopy(msg2, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
#ifndef TRACKING
    images_to_detect_.clear();
    cv::Mat cam_image1_half;
    cv::resize(cam_image1->image, cam_image1_half, cv::Size(), 0.5, 0.5);
    images_to_detect_.push_back(cam_image1_half);
    images_to_detect_.push_back(cam_image2->image.clone());
    vector<vector<Detection>> detect_results_ = detector_->doInference(images_to_detect_);
    tensorrt_ros::BoundingBoxes boxes2 = processDetections(detect_results_[1], images_to_detect_[1], false);
    boxes2.header = msg2->header;
    boundingBoxesPublisher2_.publish(boxes2);
    tensorrt_ros::BoundingBoxes boxes1 = processDetections(detect_results_[0], cam_image1->image, true);
    boxes1.header = msg1->header;
    boundingBoxesPublisher1_.publish(boxes1);
    drawDetections(cam_image1->image, boxes1);
    drawDetections(cam_image2->image, boxes2);
    detectionImagePublisher1_.publish(cam_image1->toImageMsg());
    detectionImagePublisher2_.publish(cam_image2->toImageMsg());
#else
    //vector<tensorrt_ros::BoundingBoxes> results;
    vector<vector<Detection>> results;
    cv::Mat old_image1, old_image2;
    if (detectLock_.try_lock())
    {
        results = detect_results_;
        if (!results.empty())
        {
            old_image1 = images_to_detect_[0];
            old_image2 = images_to_detect_[1];
        }
        images_to_detect_.clear();
        images_to_detect_.push_back(cam_image1->image.clone());
        images_to_detect_.push_back(cam_image2->image.clone());
        detectLock_.unlock();
        detectCond_.notify_one();
    }

    if (!results.empty())
    {
        tensorrt_ros::BoundingBoxes &boxes1 = results[0];
        tensorrt_ros::BoundingBoxes &boxes2 = results[1];
	boxes1.header = msg1->header;
	boxes2.header = msg2->header;
        vector<cv::Rect2d> rects;
        for (auto &box : boxes1.bounding_boxes)
        {
            rects.push_back(cv::Rect2d(cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax)));
        }
        tracker1_->resetTracks(old_image1, rects);
        rects.clear();
        for (auto &box : boxes2.bounding_boxes)
        {
            rects.push_back(cv::Rect2d(cv::Point(box.xmin, box.ymin), cv::Point(box.xmax, box.ymax)));
        }
        tracker2_->resetTracks(old_image2, rects);
        detections_to_track_ = {boxes1, boxes2};
    }

    if (!detections_to_track_.empty())
    {
        auto t_start = chrono::high_resolution_clock::now();
	vector<vector<cv::Point2f>> keypoints1;
	vector<vector<cv::Point2f>> keypoints2;
        vector<cv::Rect2d> trackedBoxes1 = tracker1_->updateTracks(cam_image1->image, keypoints1);
        vector<cv::Rect2d> trackedBoxes2 = tracker2_->updateTracks(cam_image2->image, keypoints2);
        for (int i = 0; i < trackedBoxes1.size(); i++)
        {
            detections_to_track_[0].bounding_boxes[i].xmin = trackedBoxes1[i].x;
            detections_to_track_[0].bounding_boxes[i].ymin = trackedBoxes1[i].y;
            detections_to_track_[0].bounding_boxes[i].xmax = trackedBoxes1[i].x + trackedBoxes1[i].width;
            detections_to_track_[0].bounding_boxes[i].ymax = trackedBoxes1[i].y + trackedBoxes1[i].height;
	    for (cv::Point2f &pt : keypoints1[i])
	    {
		geometry_msgs::Point point;
		point.x = pt.x;
		point.y = pt.y;
	        point.z = 0;
	        detections_to_track_[0].bounding_boxes[i].keypoints.clear();
	        detections_to_track_[0].bounding_boxes[i].keypoints.push_back(point);
	    }
        }
        for (int i = 0; i < trackedBoxes2.size(); i++)
        {
            detections_to_track_[1].bounding_boxes[i].xmin = trackedBoxes2[i].x;
            detections_to_track_[1].bounding_boxes[i].ymin = trackedBoxes2[i].y;
            detections_to_track_[1].bounding_boxes[i].xmax = trackedBoxes2[i].x + trackedBoxes2[i].width;
            detections_to_track_[1].bounding_boxes[i].ymax = trackedBoxes2[i].y + trackedBoxes2[i].height;
	    for (cv::Point2f &pt : keypoints2[i])
	    {
		geometry_msgs::Point point;
		point.x = pt.x;
		point.y = pt.y;
	        point.z = 0;
	        detections_to_track_[1].bounding_boxes[i].keypoints.clear();
	        detections_to_track_[1].bounding_boxes[i].keypoints.push_back(point);
	    }
        }
        drawDetections(cam_image1->image, detections_to_track_[0]);
        drawDetections(cam_image2->image, detections_to_track_[1]);
        boundingBoxesPublisher1_.publish(detections_to_track_[0]);
        detectionImagePublisher1_.publish(cam_image1->toImageMsg());
        boundingBoxesPublisher2_.publish(detections_to_track_[1]);
        detectionImagePublisher2_.publish(cam_image2->toImageMsg());
        auto t_end = chrono::high_resolution_clock::now();
        float total = chrono::duration<float, milli>(t_end - t_start).count();
        ROS_DEBUG("Time taken for tracking is  %f ms.\n", total);
    }
#endif
}

void TensorRTNode::timerCallback(const ros::WallTimerEvent& event)
{
    if (!ranOnce_)
        return;
    if (watchdog_)
    {
        watchdog_ = false;
	return;
    }
    ROS_WARN("TensorRT timed out");
    if (!cameraSwitch_)
    {
	imageSubscriber1_.unsubscribe();
	imageSubscriber2_.unsubscribe();
        imageSubscriber1_.subscribe(imageTransport_, camera1Topic, 3);
        imageSubscriber2_.subscribe(imageTransport_, camera1Topic, 3);
	cameraSwitch_ = true;
    }
    else
    {
	imageSubscriber1_.unsubscribe();
	imageSubscriber2_.unsubscribe();
        imageSubscriber1_.subscribe(imageTransport_, camera2Topic, 3);
        imageSubscriber2_.subscribe(imageTransport_, camera2Topic, 3);
	cameraSwitch_ = false;
    }
}

void TensorRTNode::detect()
{
    unique_lock<mutex> lock(detectLock_);
    while (!boost::this_thread::interruption_requested())
    {
        while(detectCond_.wait_for(lock, chrono::seconds(1)) == cv_status::timeout)
        {
            if (boost::this_thread::interruption_requested())
            {
                lock.unlock();
                return;
            }
        }
        detect_results_.clear();
        vector<vector<Detection>> results = detector_->doInference(images_to_detect_);
        detect_results_.push_back(processDetections(results[0], images_to_detect_[0], true));
        detect_results_.push_back(processDetections(results[1], images_to_detect_[1], false));
    }
    lock.unlock();
}

tensorrt_ros::BoundingBoxes TensorRTNode::processDetections(vector<Detection> &detections, cv::Mat &img, bool isCamera1)
{
    tensorrt_ros::BoundingBoxes boxes;
    vector<cv::Mat> rois;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        int left  = max((b[0]-b[2]/2.)*img.cols, 0.0);
        int right = min((b[0]+b[2]/2.)*img.cols, double(img.cols));
        int top   = max((b[1]-b[3]/2.)*img.rows, 0.0);
        int bot   = min((b[1]+b[3]/2.)*img.rows, double(img.rows));
	int h = bot - top;
	int w = right - left;
	double adder = 0.0;
	left = max((int)(left - adder * w), 0);
        top = max((int)(top - adder * h), 0);
        right = min((int)(right + adder * w), img.cols);
        bot = min((int)(bot + adder * h), img.rows);

        if (right - left <= img.cols * boxMinSizeRatio_ || bot - top <= img.rows * boxMinSizeRatio_)
            continue;
	double x = (right + left) / 2. / img.cols;
	double y = bot / (double)img.rows;
	if (carType_ == "dut18d" && isCamera1 && y > y_t && y - 1.0 > (1.0 - y_t) / (x_b2 - x_t2) * (x - x_b2) && y - 1.0 > (y_t - 1.0) / (x_t1 - x_b1) * (x - x_b1))
	    continue;

        tensorrt_ros::BoundingBox boundingBox;
        boundingBox.probability = item.prob;
        boundingBox.xmin = left;
        boundingBox.ymin = top;
        boundingBox.xmax = right;
        boundingBox.ymax = bot;

        cv::Rect box(cv::Point(left, top), cv::Point(right, bot));
        cv::Mat roi = img(box);
        rois.push_back(roi);
        //cv::imshow("roi", roi);
        //cv::imwrite(ros::package::getPath("tensorrt_ros") + "/cone0.jpg", roi);
        //cv::waitKey(0);
        //cv::Mat ycrcb;
        //cv::cvtColor(roi,ycrcb,CV_BGR2YCrCb);
        //vector<cv::Mat> channels;
        //cv::split(ycrcb,channels);
        //cv::equalizeHist(channels[0], channels[0]);
        //cv::merge(channels,ycrcb);
        //cv::cvtColor(ycrcb,roi,CV_YCrCb2BGR);
        //cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(100, 0.1, 3);
        //vector<cv::KeyPoint> keypoints;
        //detector->detect(roi, keypoints);
        //cv::drawKeypoints(roi, keypoints, roi);
        //cv::imshow("roi", roi);
	//cv::imwrite("roi.jpg", roi);
        //cv::waitKey(0);
        boundingBox.Class = "cone";
        boxes.bounding_boxes.push_back(boundingBox);
    }
    if (isCamera1)
    {
        vector<vector<cv::Point2f>> keypoints = keypointDetector_->doInference(rois);
        for (unsigned int i = 0; i < boxes.bounding_boxes.size(); i++)
        {
            //cv::Mat gray;
            //cv::cvtColor(rois[i], gray, cv::COLOR_BGR2GRAY);
            //cv::cornerSubPix(gray, keypoints[i], cv::Size(3, 3), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 40, 0.001));

            for (cv::Point2f pt : keypoints[i])
            {
                geometry_msgs::Point point;
                point.x = boxes.bounding_boxes[i].xmin + pt.x;
                point.y = boxes.bounding_boxes[i].ymin + pt.y;
                point.z = 0;
                boxes.bounding_boxes[i].keypoints.push_back(point);
            }
            //boxes.bounding_boxes[i].keypoints.erase(boxes.bounding_boxes[i].keypoints.begin() + 5, boxes.bounding_boxes[i].keypoints.end());
        }
    }
    return boxes;
}

void TensorRTNode::drawDetections(cv::Mat &img, tensorrt_ros::BoundingBoxes &boxes)
{
    for (auto &b : boxes.bounding_boxes)
    {
        cv::Rect box(cv::Point(b.xmin,b.ymin), cv::Point(b.xmax,b.ymax));
	for (auto &pt : b.keypoints)
	{
	    cv::circle(img, cv::Point2f(pt.x, pt.y), 1, cv::Scalar(0, 255, 0), -1, 8);
	}
	// Make bounding box color blue by default
        cv::Scalar boxColor(255, 0, 0);
	cv::rectangle(img, box, boxColor,2,8,0);
    }
}

}
