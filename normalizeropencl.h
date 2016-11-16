#pragma once
#include <CL\cl.h>
#include "normalizer.h"

class NormalizerOpenCL
{
private:
	cl_mem d_mean,
		d_var,
		d_minmax;
	float * m_var,
		*m_minmax;
	Normalizer::norm_t m_norm_type;
	int m_dim, m_dim2;
	bool m_using_doubles;
	cl_context context;
	cl_command_queue cmd_queue;
	cl_program program;
	cl_kernel kernel_sum,
		kernel_finalizeSum,
		kernel_normalize;
public:
	NormalizerOpenCL() : d_mean(nullptr), d_var(nullptr), m_var(nullptr), d_minmax(nullptr), m_minmax(nullptr), m_norm_type(Normalizer::NORM_NONE), m_dim(0), m_using_doubles(false),
		context(0), cmd_queue(0), program(0), kernel_sum(0), kernel_finalizeSum(0), kernel_normalize(0) {}
	void init(Normalizer::norm_t norm_type, int dim, int dim2, bool using_doubles, cl_context context, cl_command_queue cmd_queue, cl_device_id opencl_device);
	void cleanup();

	void normalize(cl_mem data, int offset, int window_count, bool use_last_stats = false);
};