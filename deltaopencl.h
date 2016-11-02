#pragma once
#include <CL\cl.h>

class DeltaOpenCL
{
private:
    int m_dim,
        m_dim2,
        m_window_limit,
        m_delta_size;
    cl_mem d_output;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_program program;
    cl_kernel kernel_delta;
public:
    DeltaOpenCL() : m_dim(0), m_window_limit(0), d_output(nullptr), context(0), cmd_queue(0), program(0), kernel_delta(0) {}
    void init(int dim, int dim2, int window_limit, int delta_size, cl_context context, cl_command_queue cmd_queue, cl_device_id opencl_device);
    void cleanup();

    void apply(cl_mem d_data, int window_count);
    cl_mem get_output_buffer() { return d_output; }
};