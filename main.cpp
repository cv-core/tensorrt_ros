#include "TensorRTNode.h"
#include <chrono>

int main(int argc, char** argv)
{
    ros::init(argc, argv, "tensorrt_ros");

    tensorrt_node::TensorRTNode node;

    ros::spin();
    return 0;

}
