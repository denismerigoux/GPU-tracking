cmake_minimum_required(VERSION 2.8)
project( GPUTracking )

find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# Add OpenCV headers location to your include paths
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(AFTER ${OpenCV2_INCLUDE_PATH})

find_package(CUDA REQUIRED)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER /usr/bin/g++)
list( APPEND CUDA_NVCC_FLAGS "-O3 --gpu-architecture compute_61 -std=c++11")

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -Wunused-function")
SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}" )
cuda_add_executable( main src/main.cpp src/trackerKCFparallel.cu)

SET( CUFFT_LIB /usr/local/cuda/lib64/libcufft.so.8.0)


target_link_libraries( main ${OpenCV_LIBS} ${CUFFT_LIB})
