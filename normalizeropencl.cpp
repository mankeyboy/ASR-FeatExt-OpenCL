#include <string>
#include <algorithm>
#include <sstream>
#include "normalizeropencl.h"
#include "openclkernelloader.h"

static const int SUM_GRID_SIZE = 256;
static int MAX_BLOCK_SIZE = 256;

void NormalizerOpenCL::init(Normalizer::norm_t norm_type, int dim, int dim2, bool using_doubles, cl_context context, cl_command_queue cmd_queue, cl_device_id opencl_device)
{
	this->context = context;
	this->cmd_queue = cmd_queue;
	m_norm_type = norm_type;
	m_dim = dim;
	m_dim2 = dim2;
	m_using_doubles = using_doubles;
	//size_t element_size = m_using_doubles ? sizeof(double) : sizeof(float);

	if (m_norm_type != Normalizer::NORM_NONE)
		d_mean = clCreateBuffer(context, CL_MEM_READ_WRITE, SUM_GRID_SIZE * m_dim2 * sizeof(float), NULL, NULL);
	else
		d_mean = nullptr;
	if (m_norm_type == Normalizer::NORM_CVN)
	{
		d_var = clCreateBuffer(context, CL_MEM_READ_WRITE, SUM_GRID_SIZE * m_dim2 * sizeof(float), NULL, NULL);
		m_var = new float[m_dim];
	}
	else
	{
		d_var = nullptr;
		m_var = nullptr;
	}
	if (m_norm_type == Normalizer::NORM_MINMAX)
	{
		d_minmax = clCreateBuffer(context, CL_MEM_READ_WRITE, SUM_GRID_SIZE * m_dim2 * 2 * sizeof(float), NULL, NULL);
		m_minmax = new float[2 * m_dim];
	}
	else
	{
		d_minmax = nullptr;
		m_minmax = nullptr;
	}

	if (m_norm_type == Normalizer::NORM_NONE)
		return;

	std::string program_source;
	if (!LoadOpenCLKernel("norm", program_source))
		throw std::runtime_error("Error while loading normalization OpenCL kernel\n");
	const char * program_source_ptr = program_source.c_str();
	size_t program_source_size = program_source.size();
	program = clCreateProgramWithSource(context, 1, &program_source_ptr, &program_source_size, NULL);
	if (!program)
		throw std::runtime_error("Can't create OpenCL program");
	std::stringstream buildOptions;
	switch (m_norm_type)
	{
	case Normalizer::NORM_CVN:
		buildOptions << " -D WANTVAR";
		break;
	case Normalizer::NORM_MINMAX:
		buildOptions << " -D WANTMINMAX";
		break;
	}
	cl_int ret = clBuildProgram(program, 0, NULL, buildOptions.str().c_str(), NULL, NULL);
	if (ret != CL_SUCCESS)
	{
		size_t len;
		clGetProgramBuildInfo(program, opencl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char * buffer = new char[len];
		clGetProgramBuildInfo(program, opencl_device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		std::string msg = "Can't build OpenCL program:\n";
		msg += buffer;
		std::runtime_error e(msg.c_str());
		delete[] buffer;
		throw e;
	}

	cl_int param_dim = m_dim2;
	kernel_sum = clCreateKernel(program, "kernelSum", NULL);
	kernel_finalizeSum = clCreateKernel(program, "kernelFinalizeSum", NULL);
	kernel_normalize = clCreateKernel(program, "kernelNormalize", NULL);
	clSetKernelArg(kernel_sum, 2, sizeof(cl_int), &param_dim);
	clSetKernelArg(kernel_finalizeSum, 1, sizeof(cl_int), &param_dim);
	clSetKernelArg(kernel_normalize, 2, sizeof(cl_int), &param_dim);
	clSetKernelArg(kernel_normalize, 5, sizeof(cl_float) * param_dim, NULL);
	if (m_norm_type == Normalizer::NORM_CVN)
		clSetKernelArg(kernel_normalize, 7, sizeof(cl_float) * param_dim, NULL);
	else if (m_norm_type == Normalizer::NORM_MINMAX)
		clSetKernelArg(kernel_normalize, 7, sizeof(cl_float) * param_dim, NULL);

	clSetKernelArg(kernel_sum, 1, sizeof(cl_mem), &d_mean);
	clSetKernelArg(kernel_finalizeSum, 0, sizeof(cl_mem), &d_mean);
	clSetKernelArg(kernel_normalize, 1, sizeof(cl_mem), &d_mean);
	switch (m_norm_type)
	{
	case Normalizer::NORM_CVN:
		clSetKernelArg(kernel_sum, 5, sizeof(cl_mem), &d_var);
		clSetKernelArg(kernel_finalizeSum, 4, sizeof(cl_mem), &d_var);
		clSetKernelArg(kernel_normalize, 6, sizeof(cl_mem), &d_var);
		break;
	case Normalizer::NORM_MINMAX:
		clSetKernelArg(kernel_sum, 5, sizeof(cl_mem), &d_minmax);
		clSetKernelArg(kernel_finalizeSum, 4, sizeof(cl_mem), &d_minmax);
		clSetKernelArg(kernel_normalize, 6, sizeof(cl_mem), &d_minmax);
		break;
	}
}

void NormalizerOpenCL::cleanup()
{
	clReleaseKernel(kernel_sum);
	clReleaseKernel(kernel_finalizeSum);
	clReleaseKernel(kernel_normalize);
	clReleaseMemObject(d_mean);
	clReleaseMemObject(d_var);
	clReleaseMemObject(d_minmax);
	delete[] m_var;
	delete[] m_minmax;
}

void NormalizerOpenCL::normalize(cl_mem data, int offset, int window_count, bool use_last_stats)
{
	int sumGridSize = std::min(SUM_GRID_SIZE, window_count);

	size_t global_work_size[2],
		local_work_size[2];
	local_work_size[0] = std::min(16, m_dim2);
	local_work_size[1] = 1;
	global_work_size[0] = ((m_dim2 + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
	global_work_size[1] = sumGridSize;

	if (!use_last_stats)
	{
		clSetKernelArg(kernel_sum, 0, sizeof(cl_mem), &data);
		cl_int param_int = window_count;
		clSetKernelArg(kernel_sum, 3, sizeof(cl_int), &param_int);
		clSetKernelArg(kernel_finalizeSum, 3, sizeof(cl_int), &param_int);
		param_int = offset;
		clSetKernelArg(kernel_sum, 4, sizeof(cl_int), &param_int);
		clEnqueueNDRangeKernel(cmd_queue, kernel_sum, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);

		param_int = sumGridSize;
		clSetKernelArg(kernel_finalizeSum, 2, sizeof(cl_int), &param_int);
		clEnqueueNDRangeKernel(cmd_queue, kernel_finalizeSum, 1, NULL, global_work_size, local_work_size, 0, NULL, NULL);
		//check variance & minmax here
	}
	local_work_size[1] = MAX_BLOCK_SIZE / local_work_size[0];
	global_work_size[1] = 256 * local_work_size[1];

	clSetKernelArg(kernel_normalize, 0, sizeof(cl_mem), &data);
	cl_int param_int = window_count;
	clSetKernelArg(kernel_normalize, 3, sizeof(cl_int), &param_int);
	param_int = offset;
	clSetKernelArg(kernel_normalize, 4, sizeof(cl_int), &param_int);
	clEnqueueNDRangeKernel(cmd_queue, kernel_normalize, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
}