// Standard include files
#include "tracker.cpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

void track(bool parallel, VideoCapture &video, Rect2d bbox) {
    //TODO: Check correctness

    Ptr<Tracker> tracker;
    if (parallel) {
        tracker = new TackerKCFImplParallel(TackerKCFImplParallel::Params());
    }
    else {
        tracker = new TackerKCFImplSequential(TackerKCFImplSequential::Params());
    }

    // Read first frame.
    Mat frame;
    video.read(frame);

    // Initialize tracker with first frame and bounding box
    tracker->init(frame, bbox);

    while(video.read(frame)) {
        // Update tracking results
        tracker->update(frame, bbox);

        // Draw bounding box
        rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );

        // Display result
        //imshow("Tracking", frame);
        int k = waitKey(1);
        if(k == 27) break;

    }
}

int main(int argc, char **argv)
{
    // Read video
    VideoCapture video("tests/chaplin.mp4");

    // Check video is open
    if(!video.isOpened())
    {
        cout << "Could not read video file" << endl;
        return 1;
    }

    // Define an initial bounding box
    Rect2d bbox(587, 223, 286, 720);

    // bbox = selectROI(frame, false);

    // cout << "=== Sequential ===" << endl;
    // track(false, video, bbox);
    //
    // // Reset video
    // video.set(CV_CAP_PROP_POS_AVI_RATIO , 0);

    cout << "=== Parallel ===" << endl;
    track(true, video, bbox);

    return 0;

}
