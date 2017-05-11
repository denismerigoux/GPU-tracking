/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

#include "precomp.hpp"
#include <complex>
#include "cycleTimer.h"
#include <opencv2/core/cuda.hpp>

#if TIME
#include <iomanip>
#include <fstream>
#endif

/*---------------------------
|  TrackerKCF
|---------------------------*/
namespace cv {

  /*
 * Prototype
 */
  class TackerKCFImplParallel : public TrackerKCF {
  public:
    TackerKCFImplParallel( const TrackerKCF::Params &parameters = TrackerKCF::Params() );
    void read( const FileNode& /*fn*/ );
    void write( FileStorage& /*fs*/ ) const;
    void setFeatureExtractor(void (*f)(const Mat, const Rect, Mat&), bool pca_func = false);

    #if TIME == 2
    inline void write_line(std::ofstream &file, std::string mode, int frames,
      std::string step_label, std::string steps_details_label, double time) {
        file << "| " << step_label << " | " << mode << steps_details_label << \
          mode << " | " << mode << std::fixed << std::setprecision(3) << \
          (1000. * time) / (frame - 1) << " ms |\n" << mode;
    }
    #endif

    ~TackerKCFImplParallel() {
      #if TIME == 2
      std::string path = "results/parallel.md";
        std::ofstream file(path);
        file << "| Phase | Subtask | Time taken |\n";
        file << "|--------------------------|-------------------------------------|-----------------|\n";
        for (int i = 0; i < num_steps-1; i++) {
            for (int j = 0; j < num_steps_details[i]; j++) {
                std::string mode = "";
                if (i == 0 && j == 6 || // Compute the gaussian kernel
                    i == 2 && j == 0 || // Update projection matrix
                    i == 3 && j == 1) { // Calculate alphas
                    mode = "**";
                }
                write_line(file, mode, frame, (j==0) ? steps_labels[i]: "",
                  steps_details_labels[i][j], cumulated_details_times[i][j]);
            }
            write_line(file, "*", frame, "", "Total", cumulated_times[i]);
        }
        write_line(file, "***", frame, "Total time for a frame", "", cumulated_times[num_steps-1]);
        file.close();
      #endif
    }

  protected:
     /*
    * basic functions and vars
    */
    bool initImpl( const Mat& /*image*/, const Rect2d& boundingBox );
    bool updateImpl( const Mat& image, Rect2d& boundingBox );

    TrackerKCF::Params params;

    /*
    * KCF functions and vars
    */
    void createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const;
    void inline fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const;
    void inline fft2(const Mat src, Mat & dest) const;
    void inline ifft2(const Mat src, Mat & dest) const;
    void inline cudafft2(int num_channels, std::vector<cuda::GpuMat> & dest, std::vector<cuda::GpuMat> & layers_data);
    void inline cudafft2(const Mat src, Mat & dest);
    void inline cudaifft2(const cuda::GpuMat src, cuda::GpuMat & dest);
    void inline full2cce(const Mat src, Mat & dest);
    void inline cce2full(const Mat src, Mat & dest);
    void inline pixelWiseMult(const std::vector<cuda::GpuMat> src1, const std::vector<cuda::GpuMat>  src2, std::vector<cuda::GpuMat>  & dest, const int flags, const bool conjB) const;
    void inline sumChannels(std::vector<cuda::GpuMat> src, cuda::GpuMat & dest) const;
    void inline updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix,double pca_rate, int compressed_sz,
                                       std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat v);
    void inline compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, Mat& patch, TrackerKCF::MODE desc = GRAY) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )) const;
    void extractCN(Mat patch_data, Mat & cnFeatures) const;
    void denseGaussKernel(const double sigma, const Mat , const Mat y_data, Mat & k_data,
                          std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf );
    void calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data);
    void calcResponse(const Mat alphaf_data, const Mat alphaf_den_data, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data);

    void shiftRows(Mat& mat) const;
    void shiftRows(Mat& mat, int n) const;
    void shiftCols(Mat& mat, int n) const;

  private:
    double output_sigma;
    Rect2d roi;
    Mat hann; 	//hann window filter
    Mat hann_cn; //10 dimensional hann-window filter for CN features,

    Mat y,yf; 	// training response and its FFT
    Mat x; 	// observation and its FFT
    Mat k,kf;	// dense gaussian kernel and its FFT
    Mat kf_lambda; // kf+lambda
    Mat new_alphaf, alphaf;	// training coefficients
    Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
    Mat z; // model
    Mat response; // detection result
    Mat old_cov_mtx, proj_mtx; // for feature compression

    // pre-defined Mat variables for optimization of private functions
    Mat spec, spec2;
    std::vector<Mat> layers;
    std::vector<Mat> vxf,vyf,vxyf;
    Mat xy_data,xyf_data;
    Mat data_temp, compress_data;
    std::vector<Mat> layers_pca_data;
    std::vector<Scalar> average_data;
    Mat img_Patch;

    // storage for the extracted features, KRLS model, KRLS compressed model
    Mat X[2],Z[2],Zc[2];

    // storage of the extracted features
    std::vector<Mat> features_pca;
    std::vector<Mat> features_npca;
    std::vector<MODE> descriptors_pca;
    std::vector<MODE> descriptors_npca;

    // optimization variables for updateProjectionMatrix
    Mat data_pca, new_covar,w_data,u_data,vt_data;

    // custom feature extractor
    bool use_custom_extractor_pca;
    bool use_custom_extractor_npca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_pca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_npca;

    bool resizeImage; // resize the image whenever needed and the patch size is large

    int frame;

    // GpuMats for Guassian Kernel
    cuda::GpuMat xyf_c_gpu;
    cuda::GpuMat xyf_r_gpu;
    std::vector<cuda::GpuMat> xf_data_gpu;
    std::vector<cuda::GpuMat> yf_data_gpu;
    std::vector<cuda::GpuMat> layers_data_gpu;
    std::vector<cuda::GpuMat> xyf_v_gpu;

    //GpuMat for projection matrix
    cuda::GpuMat pca_data_gpu;


    #if TIME
    void static printTime(double time, const std::string prefix, const  std::string label) {
        static const int labelWidth = 50;
        static const int precision = 3;
        // Print the label
        std::cout << prefix << std::left << std::setw(labelWidth) << label
             << std::setfill(' ')
        // Print the time
             << std::fixed << std::setprecision(precision) << (1000. * time)
             << "ms" << std::endl;
    }

    static const int num_steps = 5;
    int total_lines;
    const std::string steps_labels[num_steps - 1] =
        {"Detection",
         "Extracting patches",
         "Feature compression",
         "Least Squares Regression"};

    double cumulated_times[num_steps];

    void printInitializationTime(double startTime) {
        double endTime = CycleTimer::currentSeconds();
        printTime(endTime - startTime, "", "Initialization");
    }

    void updateTime(double startTime, int step) {
        if (frame != 0 || step == num_steps-1) {
          double endTime = CycleTimer::currentSeconds();
          cumulated_times[step] += endTime - startTime;
        }
    }

    void printAverageTimes() {
        if (frame == 0) {
          printTime(cumulated_times[num_steps-1], "", "Time for the first frame");
          cumulated_times[num_steps-1] = 0;
          return;
        }
        if (frame > 1) {
            // Clear previous times
            for (int i = 0; i < total_lines; i++) {
                printf("\e[A");
            }
        }
        char buffer[45];
        sprintf(buffer, "Average time for the next %d frames", frame);
        printTime(cumulated_times[num_steps-1] / frame, "", buffer);
        for (int i = 0; i < num_steps-1; i++) {
            printTime(cumulated_times[i] / frame, "--> ",
                steps_labels[i]);
            #if TIME == 2
            for (int j = 0; j < num_steps_details[i]; j++) {
                printTime(cumulated_details_times[i][j] / frame, "-----> ",
                    steps_details_labels[i][j]);
            }
            #endif
        }
    }

    // TIME == 2: Detailed view
    #if TIME == 2
    static const int max_num_details = 11;
    const int num_steps_details[num_steps - 1] = {10, 6, 3, 6};
    const std::string steps_details_labels[num_steps - 1][max_num_details] =
        // Detection
        {{"Extract and pre-process the patch",
          "Non-compressed custom descriptors",
          "Compressed descriptors",
          "Compressed custom descritors",
          "Compress features and KRSL",
          "Merge all features",
          "Compute the gaussian kernel",
          "Compute the FFT",
          "Calculate filter response",
          "Extract maximum response"},
         // Extraction patches
         {"Update bounding box",
          "Non-compressed descriptors",
          "Non-compressed custom descriptors",
          "Compressed descriptors",
          "Compressed custom descriptors",
          "Update training data"},
         // Compression
         {"Update projection matrix",
          "Compress",
          "Merge all features"},
         // Least Squares
         {"Initialization",
          "Calculate alphas",
          "Compute FFT",
          "Add a small value",
          "New Alphaf",
          "Update RLS model"}};
    double cumulated_details_times[num_steps-1][max_num_details];

    void updateTimeDetail(double *startTime, int step, int step_detail) {
        double endTime = CycleTimer::currentSeconds();
        cumulated_details_times[step][step_detail] += endTime - *startTime;
        *startTime = endTime;
    }
    #endif
    #endif
  };



} /* namespace cv */
