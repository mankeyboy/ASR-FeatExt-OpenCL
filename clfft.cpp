#include <iostream>
#include "clfft.h"

extern bool verbose;

bool clFFT::init(cl_context context, cl_command_queue cmd_queue, size_t len, size_t maxbatch)
{
	this->cmd_queue = cmd_queue;
	this->len = len;
    if (verbose)
        std::cout << "Using " << "Apple" << " OpenCL FFT\n";
		cl_int err;
		clFFT_Dim3 dim = {len, 1, 1};
		if (!(fft_plan.apple = clFFT_CreatePlan(context, dim, clFFT_1D, clFFT_InterleavedComplexFormat, &err)))
			return false;
	return true;
}

void clFFT::cleanup()
{
	clFFT_DestroyPlan(fft_plan.apple);
}

bool clFFT::apply(cl_mem in, cl_mem out, size_t batch, cl_event * evt)
{
	cl_int ret_fft = clFFT_ExecuteInterleaved(cmd_queue, fft_plan.apple, batch, clFFT_Forward, in, out, 0, NULL, evt);
	if (ret_fft != CL_SUCCESS)
		return false;
	return true;
}
