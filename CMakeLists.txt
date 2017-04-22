cmake_minimum_required(VERSION 2.8)
project( GPUTracking )


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )


find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)
list( APPEND CUDA_NVCC_FLAGS "-O3 --gpu-architecture compute_35 -std=c++11")

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -Wunused-function")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
cuda_add_executable( main src/main.cpp src/trackerKCFparallel.cu)


target_link_libraries( main ${OpenCV_LIBS})
