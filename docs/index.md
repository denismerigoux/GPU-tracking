# Parallel KCF Tracking
_A 15-618 Final Project by Ilaï Deutel and Denis Merigoux_

## Summary

We are implementing an optimized object tracker on NVIDIA GPUs using the [KCF algorithm](http://home.isr.uc.pt/~pedromartins/Publications/henriques_eccv2012.pdf). The goal is to perform real-time object tracking with a simple OpenCV interface (using the [OpenCV Tracking API](http://docs.opencv.org/trunk/d9/df8/group__tracking.html)).

## The challenge

The main dependency is temporal: you need to examine the frames in order, each one after the other. However, the algorithm present some points of synchronization that prevent total parallelization. The size of the working data is not that big (we only work one one image at a time) but the complexity of the computations in the algorithm is relatively low, so the dominant factor between data access and computation is unclear.

The challenge is then to optimize the parallelization rate of the algorithm, find out the bottlenecks and try to overcome them The goal is to perform the computation in real time with the best framerate possible (with a decent resolution).

## Ressources

We have used the [sequential KCF OpenCV implementation](http://docs.opencv.org/trunk/d2/dff/classcv_1_1TrackerKCF.html) as a starting point. We are using the GPUs of the GHC machines to run our program.


## Deliverables
* [Project proposal](https://github.com/denismerigoux/GPU-tracking/blob/master/proposal/proposal.pdf)
* [Checkpoint Writeup](https://github.com/denismerigoux/GPU-tracking/blob/master/checkpoint/checkpoint.pdf)
