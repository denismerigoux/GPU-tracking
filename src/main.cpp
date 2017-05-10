// Standard include files
#include "tracker.cpp"
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

inline bool compare_bounding_boxes(const Rect2d &bbox1, const Rect2d &bbox2) {
    return bbox1.tl().x == bbox2.tl().x
        && bbox1.tl().y == bbox2.tl().y
        && bbox1.br().x == bbox2.br().x
        && bbox1.br().y == bbox2.br().y;
}

void track(bool parallel, VideoCapture &video, Rect2d bbox, bool display, Rect2d *bounding_boxes, bool check_correctness, int num_frames) {
    if (parallel) {
        cout << "=== Parallel ===" << endl;
    }
    else {
        cout << "=== Sequential ===" << endl;
    }

    // Reset video
    video.set(CV_CAP_PROP_POS_AVI_RATIO , 0);


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

    int frame_id = 0;

    while (video.read(frame)) {
        if (frame_id == num_frames) {
          return;
        }
        // Update tracking results
        bool updated = tracker->update(frame, bbox);

        if (!updated) {
            return;
        }

        if (check_correctness) {
            if (!parallel) {
                bounding_boxes[frame_id] = bbox;
            }
            else {
                if (!compare_bounding_boxes(bbox, bounding_boxes[frame_id])) {
                    cerr << "Correctness failed at frame " << frame_id << endl
                        << "Bounding box mismatch:" << endl
                        << "* Sequential: "<< bbox << endl
                        << "* Parallel: "<< bounding_boxes[frame_id] << endl;
                    return;
                }
            }
        }

        frame_id++;

        if (display) {
            // Draw bounding box
            rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );

            // Display result
            imshow("Tracking", frame);
            int k = waitKey(1);
            if(k == 27) break;
        }
    }
}

void test_implementation(bool sequential, bool parallel, VideoCapture &video, Rect2d bbox, bool display, int num_frames) {
    if (sequential && parallel) {
        if (num_frames < 0) {
          num_frames = video.get(CV_CAP_PROP_FRAME_COUNT);
        }
        Rect2d bounding_boxes[num_frames];

        track(false, video, bbox, display, bounding_boxes, true, num_frames); // Sequential
        track(true, video, bbox, display, bounding_boxes, true, num_frames); // Parallel
    }
    else {
        track(parallel, video, bbox, display, NULL, false, num_frames); // Sequential
    }
}

int main(int argc, char **argv)
{
    // Read video
    VideoCapture video("tests/chaplin.mp4");

    // Check video is open
    if(!video.isOpened())
    {
        cerr << "Could not read video file" << endl;
        return 1;
    }

    // Define an initial bounding box
    Rect2d bbox(587, 223, 286, 720);

    // bbox = selectROI(frame, false);

    test_implementation(true, true, video, bbox, false, -1);

    return 0;

}
