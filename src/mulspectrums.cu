#include <opencv2/opencv_modules.hpp>

#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudev.hpp>
#define __OPENCV_BUILD 1
#include <opencv2/core/private.cuda.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;

//////////////////////////////////////////////////////////////////////////////
// mulSpectrums

namespace
{
    __device__ __forceinline__ float real(const double2& val)
    {
        return val.x;
    }

    __device__ __forceinline__ float imag(const double2& val)
    {
        return val.y;
    }

    __device__ __forceinline__ double2 cmul(const double2& a, const double2& b)
    {
        return make_double2((real(a) * real(b)) - (imag(a) * imag(b)),
                           (real(a) * imag(b)) + (imag(a) * real(b)));
    }

    __device__ __forceinline__ double2 conj(const double2& a)
    {
        return make_double2(real(a), -imag(a));
    }

    struct comlex_mul : binary_function<double2, double2, double2>
    {
        __device__ __forceinline__ double2 operator ()(const double2& a, const double2& b) const
        {
            return cmul(a, b);
        }
    };

    struct comlex_mul_conj : binary_function<double2, double2, double2>
    {
        __device__ __forceinline__ double2 operator ()(const double2& a, const double2& b) const
        {
            return cmul(a, conj(b));
        }
    };

    struct comlex_mul_scale : binary_function<double2, double2, double2>
    {
        float scale;

        __device__ __forceinline__ double2 operator ()(const double2& a, const double2& b) const
        {
            return scale * cmul(a, b);
        }
    };

    struct comlex_mul_conj_scale : binary_function<double2, double2, double2>
    {
        float scale;

        __device__ __forceinline__ double2 operator ()(const double2& a, const double2& b) const
        {
            return scale * cmul(a, conj(b));
        }
    };
}

void cv::cuda::mulSpectrums(InputArray _src1, InputArray _src2, OutputArray _dst, int flags, bool conjB, Stream& stream)
{
    (void) flags;

    GpuMat src1 = getInputMat(_src1, stream);
    GpuMat src2 = getInputMat(_src2, stream);

    CV_Assert( src1.type() == src2.type() && src1.type() == CV_64FC2 );
    CV_Assert( src1.size() == src2.size() );

    GpuMat dst = getOutputMat(_dst, src1.size(), CV_64FC2, stream);

    if (conjB)
        gridTransformBinary(globPtr<double2>(src1), globPtr<double2>(src2), globPtr<double2>(dst), comlex_mul_conj(), stream);
    else
        gridTransformBinary(globPtr<double2>(src1), globPtr<double2>(src2), globPtr<double2>(dst), comlex_mul(), stream);

    syncOutput(dst, _dst, stream);
}
