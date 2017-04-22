# GPU-Tracking

## Installation

Read the following tutorials:

    http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html
    https://github.com/opencv/opencv_contrib

## Build

```
mkdir build
cd build
cmake -DOpenCV_DIR=/tmp/dmerigou/opencv-3.2.0/build2 -DCMAKE_BUILD_TYPE=RELEASE ..
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/dmerigou/ffmpeg-3.2.4/build/lib make
```

## Run

```
./main
```
