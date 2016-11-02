#include <sstream>
#include "deltaopencl.h"
#include "openclkernelloader.h"

static const int DELTA_TILE_SIZE = 16;

void DeltaOpenCL::init(int dim, int dim2, int window_limit, int delta_size, cl_context context, cl_command_queue cmd_queue, cl_device_id opencl_device)
{
    this->context = context;
    this->cmd_queue = cmd_queue;
    m_dim = dim;
    m_dim2 = dim2;
    m_window_limit = window_limit;
    m_delta_size = delta_size;
    d_output = clCreateBuffer(context, CL_MEM_READ_WRITE, m_dim2 * m_window_limit * sizeof(float), NULL, NULL);

    std::string program_source;
    if (!LoadOpenCLKernel("delta", program_source))
        throw std::runtime_error("Error while loading segmenter OpenCL kernel\n");
    const char * program_source_ptr = program_source.c_str();
    size_t program_source_size = program_source.size();
    program = clCreateProgramWithSource(context, 1, &program_source_ptr, &program_source_size, NULL);
    if (!program)
        throw std::runtime_error("Can't create OpenCL program");
    std::stringstream buildOptions;
	buildOptions << "-D DELTA_TILE_SIZE=" << DELTA_TILE_SIZE;
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

    kernel_delta = clCreateKernel(program, "kernelDelta", NULL);
    cl_int param_int = m_dim2;
    clSetKernelArg(kernel_delta, 1, sizeof(cl_mem), &d_output);
	clSetKernelArg(kernel_delta, 2, sizeof(cl_int), &param_int);
    param_int = m_delta_size;
    clSetKernelArg(kernel_delta, 4, sizeof(cl_int), &param_int);
	clSetKernelArg(kernel_delta, 5, sizeof(cl_float) * DELTA_TILE_SIZE * (DELTA_TILE_SIZE + 2 * m_delta_size), NULL);
}

void DeltaOpenCL::cleanup()
{
    clReleaseMemObject(d_output);
    clReleaseKernel(kernel_delta);
    clReleaseProgram(program);
}

void DeltaOpenCL::apply(cl_mem d_data, int window_count)
{
    size_t global_work_size[2],
		local_work_size[2];
	local_work_size[0] = DELTA_TILE_SIZE;
	local_work_size[1] = DELTA_TILE_SIZE;
	global_work_size[0] = ((m_dim2 + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
	global_work_size[1] = 256 * local_work_size[1];
	
	clSetKernelArg(kernel_delta, 0, sizeof(cl_mem), &d_data);
    cl_int param_int = window_count;
	clSetKernelArg(kernel_delta, 3, sizeof(cl_int), &param_int);
	cl_int ret = clEnqueueNDRangeKernel(cmd_queue, kernel_delta, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS)
		throw std::runtime_error("Error in kernelDelta");
}