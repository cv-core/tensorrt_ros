This reposity is built with Catkin on Ubuntu 18.04 and has additional
dependencies on CUDA10, TensorRT, and OpenCV 3.4. For best results, it is
recommended to use NVIDIA Driver 410 to build CUDA with TensorRT version
5.0.2.6, as these are the versions used on the MIT/DUT18D car.

## Installation Instructions:

### CUDA:
CUDA is an NVIDIA GPU programming language, with installation instructions
that can be found on NVIDIA's website at:
https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

### TensorRT:
TensorRT is a CUDA-based deep learning inference platform optimized for NVIDIA
GPU's, used to run cone detections and keypoint detections on the MIT/DUT18D
car. TensorRT installation instructions can also be found on NVIDIA's website
at:
https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html

### OpenCV:
OpenCV is an open-source computer vision library, used in this repository
for processing images and storing basic image-related data structures.
OpenCV 3.4 can be installed by running the following commands:

```bash
git clone git@github.com:opencv/opencv_contrib.git
sudo apt-get install -y qtbase5-dev qtdeclarative5-dev
cd opencv_contrib
git checkout 3.4
cd ..
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local -DWITH_CUDA=ON -DENABLE_FAST_MATH=1 -DCUDA_FAST_MATH=1 -DWITH_CUBLAS=1 -DBUILD_opencv_cudacodec=OFF -DWITH_QT=ON -DWITH_OPENGL=ON -DFORCE_VTK=OFF -DWITH_TBB=OFF -DWITH_MKL=ON -DMKL_USE_MULTITHREAD=ON -DMKL_WITH_TBB=ON -DWITH_IPP=ON -DWITH_GDAL=ON -DWITH_XINE=OFF -DBUILD_EXAMPLES=OFF -DCUDA_ARCH_PTX="" -DCUDA_ARCH_BIN="6.1" -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j"$(nproc)"
sudo make install
```
