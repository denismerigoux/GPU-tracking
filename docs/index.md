# Parallel KCF Tracking
_A 15-618 Final Project by Ilaï Deutel and Denis Merigoux_

## Summary

We are implementing an optimized object tracker on NVIDIA GPUs using the [KCF algorithm](http://home.isr.uc.pt/~pedromartins/Publications/henriques_eccv2012.pdf). The goal is to perform real-time object tracking with a simple OpenCV interface (using the [OpenCV Tracking API](http://docs.opencv.org/trunk/d9/df8/group__tracking.html)).

## The challenge

The main dependency is temporal: you need to examine the frames in order, each one after the other. However, the algorithm present some points of synchronization that prevent total parallelization. The size of the working data is not that big (we only work one image at a time) but the complexity of the computations in the algorithm is relatively low, so the dominant factor between data access and computation is unclear.

The challenge is then to optimize the parallelization rate of the algorithm, find out the bottlenecks and try to overcome them The goal is to perform the computation in real time with the best framerate possible (with a decent resolution).

## Correctness and performance analysis

We have built a correctness and performance analysis engine, which shows performance per task (averaged over the number of frames) updated in real time, and stop if the parallel solution is not correct. We determine if the solution is correct by comparing the bounding boxes returned by the parallel implementation with those returned by the sequential implementation. They must be *exactly* the same (no approximation).

Example of output for an incorrect implementation:
```
=== Sequential ===
// Performance information for the 189 frames of the video, not shown here
=== Parallel ===
// Performance information for the first 4 frames of the video, not shown here
Correctness failed at frame 3
Bounding box mismatch:
* Sequential: [286 x 720 from (635, 219)]
* Parallel: [286 x 720 from (635, 221)]
```

## Analyzing performance of the sequential implementation

The table below gives the time (averaged over the number  taken by the baseline sequential algorithm for each subtask:

| Phase                    | Subtask                             | Time taken (ms) |
|--------------------------|-------------------------------------|-----------------|
| Detection                | Extract and pre-process the patches | 1.055           |
|                          | Non-compressed custom descriptors   | 0.193           |
|                          | Compressed descriptors              | 7.355           |
|                          | Compressed custom descriptors       | 2.774           |
|                          | Compress features and KRSL          | 9.372           |
|                          | Merge all features                  | 1.558           |
|                          | **Compute the gaussian kernel**     | **20.087**      |
|                          | Compute the FFT                     | 1.748           |
|                          | Calculate filter response           | 3.101           |
|                          | Extract maximum response            | 0.236           |
|                          | _Total_                             | _47.48_         |
| Extracting patches       | Update bounding box                 | 0.000           |
|                          | Non-compressed descriptors          | 1.02            |
|                          | Non-compressed custom descriptors   | 0.196           |
|                          | Compressed descriptors              | 7.301           |
|                          | Compressed custom descriptors       | 3.058           |
|                          | Update training data                | 3.027           |
|                          | _Total_                             | _14.602_        |
| Feature compression      | Update projection matrix            | 20.164          |
|                          | Compress                            | 4.249           |
|                          | Merge all features                  | 0.705           |
|                          | _Total_                             | _25.118_        |
| Least Squares Regression | Initialization                      | 0.000           |
|                          | **Calculate alphas**                | **18.57**       |
|                          | Compute FFT                         | 1.758           |
|                          | Add a small value                   | 0.378           |
|                          | New alphaf                          | 1.095           |
|                          | Update RLS Model                    | 0.837           |
|                          | _Total_                             | _22.638_        |
| **_Total time for a frame_** |                                 | **_111.353_**       |

## Preliminary results

Given the timing results on the sequential algorithm, we have decided to start improving the performance of [`denseGaussKernel`](https://github.com/denismerigoux/GPU-tracking/blob/master/src/trackerKCF.cpp), which is called in both the _Compute the gaussian kernel_ and _Calculate alphas_ phases.

This function uses calls to [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) and inverse FFT, therefore we have decided to use [the NVIDIA CUDA Fast Fourier Transform library (cuFFT)](https://developer.nvidia.com/cufft).

Our first parallel implementation of `denseGaussKernel` gives the following results:

| Phase                    | Subtask                             | Time taken (ms) |
|--------------------------|-------------------------------------|-----------------|
| Detection                | Extract and pre-process the patches | 1.037           |
|                          | Non-compressed custom descriptors   | 0.200           |
|                          | Compressed descriptors              | 6.613           |
|                          | Compressed custom descriptors       | 2.827           |
|                          | Compress features and KRSL          | 9.273           |
|                          | Merge all features                  | 1.570           |
|                          | **Compute the gaussian kernel**     | **10.809**      |
|                          | Compute the FFT                     | 1.881           |
|                          | Calculate filter response           | 3.167           |
|                          | Extract maximum response            | 0.240           |
|                          | _Total_                             | _37.617_        |
| Extracting patches       | Update bounding box                 | 0.000           |
|                          | Non-compressed descriptors          | 1.008           |
|                          | Non-compressed custom descriptors   | 0.197           |
|                          | Compressed descriptors              | 6.492           |
|                          | Compressed custom descriptors       | 3.487           |
|                          | Update training data                | 3.082           |
|                          | _Total_                             | _14.266_        |
| Feature compression      | Update projection matrix            | 20.618          |
|                          | Compress                            | 4.337           |
|                          | Merge all features                  | 0.708           |
|                          | _Total_                             | _25.663_        |
| Least Squares Regression | Initialization                      | 0.000           |
|                          | **Calculate alphas**                | **16.663**      |
|                          | Compute FFT                         | 1.886           |
|                          | Add a small value                   | 0.403           |
|                          | New alphaf                          | 1.147           |
|                          | Update RLS Model                    | 0.900           |
|                          | _Total_                             | _19.401_        |
| **_Total time for a frame_** |                                 | **_98.510_**    |

We have found ways to improve the performance again (by pipelining the operations on GPU instead of transfering to/from the GPU after calling each function). These preliminary results should be updated in the next few hours/days.

## Ressources

We have used the [sequential KCF OpenCV implementation](http://docs.opencv.org/trunk/d2/dff/classcv_1_1TrackerKCF.html) as a starting point. We are using the GPUs of the GHC machines to run our program.


## Deliverables
* [Project proposal](https://github.com/denismerigoux/GPU-tracking/raw/master/proposal/proposal.pdf)
* [Checkpoint Writeup](https://github.com/denismerigoux/GPU-tracking/raw/master/checkpoint/checkpoint.pdf)
