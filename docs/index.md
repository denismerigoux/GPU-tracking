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

The table below gives the time, averaged over the number of frames after the first frame (which serves as initialization), taken by the baseline sequential algorithm for each subtask:

| Phase                    | Subtask                             | Time taken (ms) |
|--------------------------|-------------------------------------|-----------------|
| Detection                | Extract and pre-process the patches | 1.083           |
|                          | Non-compressed custom descriptors   | 0.198           |
|                          | Compressed descriptors              | 7.408           |
|                          | Compressed custom descriptors       | 2.800           |
|                          | Compress features and KRSL          | 9.707           |
|                          | Merge all features                  | 1.569           |
|                          | **Compute the gaussian kernel**     | **20.637**      |
|                          | Compute the FFT                     | 1.775           |
|                          | Calculate filter response           | 3.109           |
|                          | Extract maximum response            | 0.238           |
|                          | _Total_                             | _48.524_        |
| Extracting patches       | Update bounding box                 | 0.000           |
|                          | Non-compressed descriptors          | 1.010           |
|                          | Non-compressed custom descriptors   | 0.198           |
|                          | Compressed descriptors              | 7.353           |
|                          | Compressed custom descriptors       | 3.147           |
|                          | Update training data                | 3.119           |
|                          | _Total_                             | _14.703_        |
| Feature compression      | **Update projection matrix**        | **20.746**      |
|                          | Compress                            | 4.315           |
|                          | Merge all features                  | 0.714           |
|                          | _Total_                             | _25.578_        |
| Least Squares Regression | Initialization                      | 0.000           |
|                          | **Calculate alphas**                | **18.884**      |
|                          | Compute FFT                         | 1.782           |
|                          | Add a small value                   | 0.384           |
|                          | New alphaf                          | 1.139           |
|                          | Update RLS Model                    | 0.926           |
|                          | _Total_                             | _22.927_        |
| **_Total time for a frame_** |                                 | **_113.279_**   |

## Preliminary results

Given the timing results on the sequential algorithm, we have decided to start improving the performance of [`denseGaussKernel`](https://github.com/denismerigoux/GPU-tracking/blob/master/src/trackerKCF.cpp), which is called in both the _Compute the gaussian kernel_ and _Calculate alphas_ phases.

This function uses calls to [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) and inverse FFT, therefore we have decided to use [the NVIDIA CUDA Fast Fourier Transform library (cuFFT)](https://developer.nvidia.com/cufft).

We have also implemented a parallel version of `updateProjectionMatrix`, which involves a very large matrix multiplication.

Our parallel implementation yield the following results:

| Phase | Subtask | Time taken |
|--------------------------|-------------------------------------|-----------------|
| Detection | Extract and pre-process the patch | 1.135 ms |
|  | Non-compressed custom descriptors | 0.204 ms |
|  | Compressed descriptors | 6.620 ms |
|  | Compressed custom descritors | 2.826 ms |
|  | Compress features and KRSL | 9.216 ms |
|  | Merge all features | 1.593 ms |
|  | **Compute the gaussian kernel** | **10.899** ms |
|  | Compute the FFT | 1.864 ms |
|  | Calculate filter response | 3.155 ms |
|  | Extract maximum response | 0.240 ms |
|  | *Total* | *37.751* ms |
| Extracting patches | Update bounding box | 0.000 ms |
|  | Non-compressed descriptors | 1.013 ms |
|  | Non-compressed custom descriptors | 0.196 ms |
|  | Compressed descriptors | 6.392 ms |
|  | Compressed custom descriptors | 3.468 ms |
|  | Update training data | 3.047 ms |
|  | *Total* | *14.017* ms |
| Feature compression | **Update projection matrix** | **14.028** ms |
|  | Compress | 4.345 ms |
|  | Merge all features | 0.717 ms |
|  | *Total* | *17.827* ms |
| Least Squares Regression | Initialization | 0.000 ms |
|  | **Calculate alphas** | **11.926** ms |
|  | Compute FFT | 1.872 ms |
|  | Add a small value | 0.412 ms |
|  | New Alphaf | 1.167 ms |
|  | Update RLS model | 0.922 ms |
|  | *Total* | *14.881* ms |
|  | ***Total time for a frame*** | ***86.017*** ms |

## Ressources

We have used the [sequential KCF OpenCV implementation](http://docs.opencv.org/trunk/d2/dff/classcv_1_1TrackerKCF.html) as a starting point. We are using the GPUs of the GHC machines to run our program.


## Deliverables
* [Project proposal](https://github.com/denismerigoux/GPU-tracking/raw/master/proposal/proposal.pdf)
* [Checkpoint Writeup](https://github.com/denismerigoux/GPU-tracking/raw/master/checkpoint/checkpoint.pdf)
