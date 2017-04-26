# GPU-Tracking

## 15-618 Project information

### Proposal

See [the project proposal](proposal/proposal.pdf)

### Checkpoint

See [the project checkpoint writeup](checkpoint/checkpoint.pdf)

## Usage

### Installation

Read the following tutorials:

    http://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html
    https://github.com/opencv/opencv_contrib

Build scripts are available in the `build_scripts` folder.

### Build

The `Makefile` provided builds the project on `ghc39.andrew.cmu.edu`. For other machines and configurations, you should adjust the paths in the `Makefile` and in the `CMakeLists.txt`

### Run

Make sure the `tests` folder contains the video `chaplin.mp4`.

```
./main
```
