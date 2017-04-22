#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/dmerigou/ffmpeg-3.2.4/build/lib:/tmp/dmerigou/libv4l/usr/lib/libv4l
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/tmp/dmerigou/ffmpeg-3.2.4/build/lib/pkgconfig
export PKG_CONFIG_LIBDIR=$PKG_CONFIG_LIBDIR:/tmp/dmerigou/ffmpeg-3.2.4/build/lib

cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_CUDA=ON -D ENABLE_FAST_MATH=1 -D CUDA_FAST_MATH=1 \
    -DWITH_CUBLAS=1 -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.2.0/modules \
    -DWITH_NVCUVID=ON -DBUILD_PERF_TESTS=OFF -DBUILD_TESTS=OFF -DWITH_LIBV4L=ON -DWITH_V4L=ON \
    -DCUDA_HOST_COMPILER=/usr/bin/g++ -DFFMPEG_INCLUDE_DIRS=/tmp/dmerigou/ffmpeg-3.2.4/build/include \
    -DFFMPEG_LIBRARY_DIRS=/tmp/dmerigou/ffmpeg-3.2.4/build/lib  -DCMAKE_CXX_FLAGS="-D__STDC_CONSTANT_MACROS" \
    ..
