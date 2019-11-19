mkdir -p ~/DUT18D_ws/src/cv/tensorrt_ros/launch/calib_imgs
cd ~/DUT18D_ws/src/cv/tensorrt_ros/launch/calib_imgs 
gsutil -m cp -p gs://mit-dut-driverless-external/tiling_dataset_non_square/* .
