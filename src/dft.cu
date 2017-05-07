#include <limits>


#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/utility.hpp>

#define __OPENCV_BUILD 1
#include <opencv2/core/private.cuda.hpp>

#include <cublas.h>
#include <cufft.h>

using namespace cv;
using namespace cv::cuda;

#define error_entry(entry)  { entry, #entry }

struct ErrorEntry
{
    int code;
    const char* str;
};

const ErrorEntry cufft_errors[] =
{
    error_entry( CUFFT_INVALID_PLAN ),
    error_entry( CUFFT_ALLOC_FAILED ),
    error_entry( CUFFT_INVALID_TYPE ),
    error_entry( CUFFT_INVALID_VALUE ),
    error_entry( CUFFT_INTERNAL_ERROR ),
    error_entry( CUFFT_EXEC_FAILED ),
    error_entry( CUFFT_SETUP_FAILED ),
    error_entry( CUFFT_INVALID_SIZE ),
    error_entry( CUFFT_UNALIGNED_DATA )
};

struct ErrorEntryComparer
{
    int code;
    ErrorEntryComparer(int code_) : code(code_) {}
    bool operator()(const ErrorEntry& e) const { return e.code == code; }
};

cv::String getErrorString(int code, const ErrorEntry* errors, size_t n)
{
    size_t idx = std::find_if(errors, errors + n, ErrorEntryComparer(code)) - errors;

    const char* msg = (idx != n) ? errors[idx].str : "Unknown error code";
    cv::String str = cv::format("%s [Code = %d]", msg, code);

    return str;
}

const int cufft_error_num = sizeof(cufft_errors) / sizeof(cufft_errors[0]);

#define cufftSafeCall(expr)  ___cufftSafeCall(expr, __FILE__, __LINE__, CV_Func)

void ___cufftSafeCall(int err, const char* file, const int line, const char* func)
{
    if (CUFFT_SUCCESS != err)
    {
        String msg = getErrorString(err, cufft_errors, cufft_error_num);
        cv::error(cv::Error::GpuApiCallError, msg, func, file, line);
    }
}

enum DftFlags {
    DFT_COMPLEX_INPUT = 64,
    DFT_DOUBLE = 1024
};

class DFTImplCustom
{
    Size dft_size, dft_size_opt;
    bool is_1d_input, is_row_dft, is_scaled_dft, is_inverse, is_complex_input,
        is_complex_output, is_double_precision;

    cufftType dft_type;
    cufftHandle plan;

public:
    DFTImplCustom(Size dft_size, int flags)
        : dft_size(dft_size),
          dft_size_opt(dft_size),
          is_1d_input((dft_size.height == 1) || (dft_size.width == 1)),
          is_row_dft((flags & DFT_ROWS) != 0),
          is_scaled_dft((flags & DFT_SCALE) != 0),
          is_inverse((flags & DFT_INVERSE) != 0),
          is_complex_input((flags & DFT_COMPLEX_INPUT) != 0),
          is_complex_output(!(flags & DFT_REAL_OUTPUT)),
          is_double_precision((flags & DFT_DOUBLE) != 0),
          dft_type(!is_complex_input ? (is_double_precision ? CUFFT_D2Z : CUFFT_R2C)
           : (is_complex_output ? (is_double_precision ? CUFFT_Z2Z : CUFFT_C2C)
            : (is_double_precision? CUFFT_Z2D : CUFFT_C2R)))
    {
        // We don't support unpacked output (in the case of real input)
        CV_Assert( !(flags & DFT_COMPLEX_OUTPUT) );

        // We don't support real-to-real transform
        CV_Assert( is_complex_input || is_complex_output );

        if (is_1d_input && !is_row_dft)
        {
            // If the source matrix is single column handle it as single row
            dft_size_opt.width = std::max(dft_size.width, dft_size.height);
            dft_size_opt.height = std::min(dft_size.width, dft_size.height);
        }

        CV_Assert( dft_size_opt.width > 1 );

        if (is_1d_input || is_row_dft)
            cufftSafeCall( cufftPlan1d(&plan, dft_size_opt.width, dft_type, dft_size_opt.height) );
        else
            cufftSafeCall( cufftPlan2d(&plan, dft_size_opt.height, dft_size_opt.width, dft_type) );
    }

    ~DFTImplCustom()
    {
        cufftSafeCall( cufftDestroy(plan) );
    }

    void compute(InputArray _src, OutputArray _dst, Stream& stream)
    {
        GpuMat src = getInputMat(_src, stream);

        CV_Assert( src.type() == CV_32FC1 || src.type() == CV_32FC2
            || src.type() == CV_64FC2 || src.type() == CV_64FC1);
        CV_Assert( is_complex_input == (src.channels() == 2) );

        // Make sure here we work with the continuous input,
        // as CUFFT can't handle gaps
        GpuMat src_cont;
        if (src.isContinuous())
        {
            src_cont = src;
        }
        else
        {
            BufferPool pool(stream);
            src_cont.allocator = pool.getAllocator();
            createContinuous(src.rows, src.cols, src.type(), src_cont);
            src.copyTo(src_cont, stream);
        }

        cufftSafeCall( cufftSetStream(plan, StreamAccessor::getStream(stream)) );

        if (is_complex_input)
        {
            if (is_complex_output)
            {
                if (is_double_precision)
                {
                    createContinuous(dft_size, CV_64FC2, _dst);
                    GpuMat dst = _dst.getGpuMat();

                    cufftSafeCall(cufftExecZ2Z(
                            plan, src_cont.ptr<cufftDoubleComplex>(), dst.ptr<cufftDoubleComplex>(),
                            is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
                } else
                {
                    createContinuous(dft_size, CV_32FC2, _dst);
                    GpuMat dst = _dst.getGpuMat();

                    cufftSafeCall(cufftExecC2C(
                            plan, src_cont.ptr<cufftComplex>(), dst.ptr<cufftComplex>(),
                            is_inverse ? CUFFT_INVERSE : CUFFT_FORWARD));
                }
            }
            else
            {
                if (is_double_precision)
                {
                    createContinuous(dft_size, CV_64F, _dst);
                    GpuMat dst = _dst.getGpuMat();

                    cufftSafeCall(cufftExecZ2D(
                            plan, src_cont.ptr<cufftDoubleComplex>(), dst.ptr<cufftDoubleReal>()));
                } else
                {
                    createContinuous(dft_size, CV_32F, _dst);
                    GpuMat dst = _dst.getGpuMat();

                    cufftSafeCall(cufftExecC2R(
                            plan, src_cont.ptr<cufftComplex>(), dst.ptr<cufftReal>()));
                }
            }
        }
        else
        {
            if (is_double_precision)
            {
                // We could swap dft_size for efficiency. Here we must reflect it
                if (dft_size == dft_size_opt)
                    createContinuous(Size(dft_size.width / 2 + 1, dft_size.height), CV_64FC2, _dst);
                else
                    createContinuous(Size(dft_size.width, dft_size.height / 2 + 1), CV_64FC2, _dst);

                GpuMat dst = _dst.getGpuMat();

                cufftSafeCall(cufftExecD2Z(
                                  plan, src_cont.ptr<cufftDoubleReal>(), dst.ptr<cufftDoubleComplex>()));
            } else
            {
                // We could swap dft_size for efficiency. Here we must reflect it
                if (dft_size == dft_size_opt)
                    createContinuous(Size(dft_size.width / 2 + 1, dft_size.height), CV_32FC2, _dst);
                else
                    createContinuous(Size(dft_size.width, dft_size.height / 2 + 1), CV_32FC2, _dst);

                GpuMat dst = _dst.getGpuMat();

                cufftSafeCall(cufftExecR2C(
                                  plan, src_cont.ptr<cufftReal>(), dst.ptr<cufftComplex>()));
            }
        }

        if (is_scaled_dft)
            cuda::multiply(_dst, Scalar::all(1. / dft_size.area()), _dst, 1, -1, stream);
    }
};



Ptr<DFTImplCustom> createDFTCustom(Size dft_size, int flags)
{
    return makePtr<DFTImplCustom>(dft_size, flags);
}

void cv::cuda::dft(InputArray _src, OutputArray _dst, Size dft_size, int flags, Stream& stream)
{
    if (getInputMat(_src, stream).channels() == 2)
        flags |= DFT_COMPLEX_INPUT;

    Ptr<DFTImplCustom> dft = createDFTCustom(dft_size, flags);
    dft->compute(_src, _dst, stream);
}
