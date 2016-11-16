#ifndef _CLFFT_H_
#define _CLFFT_H_

#include <CL/cl.h>
#include "AppleFFT/clFFT.h"

class clFFT
{
private:
	union {
		clFFT_Plan apple;
	} fft_plan;
	cl_command_queue cmd_queue;
	cl_mem d_tmp;
	size_t len;
public:
	clFFT() : cmd_queue(0), d_tmp(0), len(0) {}
	bool init(cl_context context, cl_command_queue cmd_queue, size_t len, size_t maxbatch);
	void cleanup();
	bool apply(cl_mem in, cl_mem out, size_t batch, cl_event * evt = NULL);

	size_t get_element_size() const {return 2;}
	size_t get_output_vector_size() const {return len;}
};

#endif
