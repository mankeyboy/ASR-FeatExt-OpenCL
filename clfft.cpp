#include <iostream>
#include "clfft.h"

extern bool verbose, use_apple_oclfft;

bool clFFT::init(cl_context context, cl_command_queue cmd_queue, size_t len, size_t maxbatch)
{
	this->cmd_queue = cmd_queue;
	this->len = len;
	using_apple = use_apple_oclfft; 
    if (verbose)
        std::cout << "Using " << (using_apple ? "Apple" : "AMD") << " OpenCL FFT\n";
	if (using_apple)
	{
		cl_int err;
		clFFT_Dim3 dim = {len, 1, 1};
		if (!(fft_plan.apple = clFFT_CreatePlan(context, dim, clFFT_1D, clFFT_InterleavedComplexFormat, &err)))
			return false;
	}
	else
	{
		clAmdFftSetupData fft_setup;
		clAmdFftInitSetupData(&fft_setup);
		clAmdFftSetup(&fft_setup);
		clAmdFftCreateDefaultPlan(&fft_plan.amd, context, CLFFT_1D, &len);
		clAmdFftSetPlanPrecision(fft_plan.amd, CLFFT_SINGLE);
		clAmdFftSetResultLocation(fft_plan.amd, CLFFT_OUTOFPLACE);
		clAmdFftSetLayout(fft_plan.amd, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
		size_t fft_stride = 1;
		clAmdFftSetPlanInStride(fft_plan.amd, CLFFT_1D, &fft_stride);
		clAmdFftSetPlanOutStride(fft_plan.amd, CLFFT_1D, &fft_stride);
		clAmdFftSetPlanBatchSize(fft_plan.amd, maxbatch);
		clAmdFftSetPlanDistance(fft_plan.amd, len, len / 2 + 1);
		clAmdFftStatus fftStatus = clAmdFftBakePlan(fft_plan.amd, 1, &cmd_queue, NULL, NULL);
		if (fftStatus != CLFFT_SUCCESS)
			return false;
		size_t tmp_size;
		clAmdFftGetTmpBufSize(fft_plan.amd, &tmp_size);

		if (tmp_size > 0)
			d_tmp = clCreateBuffer(context, CL_MEM_READ_WRITE, tmp_size, NULL, NULL);
	}
	return true;
}

void clFFT::cleanup()
{
	if (using_apple)
		clFFT_DestroyPlan(fft_plan.apple);
	else
	{
		clAmdFftDestroyPlan(&fft_plan.amd);
		clAmdFftTeardown();
	}
}

bool clFFT::apply(cl_mem in, cl_mem out, size_t batch, cl_event * evt)
{
	if (using_apple)
	{
		cl_int ret_fft = clFFT_ExecuteInterleaved(cmd_queue, fft_plan.apple, batch, clFFT_Forward, in, out, 0, NULL, evt);
		if (ret_fft != CL_SUCCESS)
			return false;
	}
	else
	{
		clAmdFftStatus ret_fft = clAmdFftEnqueueTransform(fft_plan.amd, CLFFT_FORWARD, 1, &cmd_queue, 0, NULL, evt, &in, &out, d_tmp);
		if (ret_fft != CLFFT_SUCCESS)
			return false;
	}
	return true;
}
