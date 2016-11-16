#pragma once
#include <cmath>
#include <CL/cl.h>
#include "stopwatchopencl.h"

class SegmenterOpenCL
{
private:
    size_t m_buffer_size;
    cl_mem d_tmpbuffer,
        d_window;
    int m_window_size,
        m_window_size2,
        m_shift,
        m_deltasize,
        m_remaining_samples,
        m_samples;
    bool m_flushed,
        m_last_calc_flushed;
    cl_context context;
    cl_command_queue cmd_queue;
    cl_program program;
    cl_kernel kernel_segment;

    void segment_data(cl_mem d_data_out, int window_count);
public:
    void init(int window_size, int shift, int window_limit, int deltasize, cl_context context, cl_command_queue cmd_queue, cl_device_id opencl_device);
    void cleanup();

    void set_window(const float * window);
    void set_input(const short * data_in, cl_mem d_data_out, int samples, int & window_count, int & window_count_no_delta, StopWatchOpenCL * swmem = NULL, float * timemem = NULL);
    void flush(cl_mem d_data_out, int & window_count, int & window_count_no_delta);

    int get_remaining_samples() const { return m_remaining_samples; }
    int get_samples() const { return m_samples; }
    bool is_flushed() const { return m_flushed; }
    bool was_flushed() const { return m_last_calc_flushed; }

    int estimated_window_count(int samples) const
    {
        return floor(float(samples - (m_window_size - m_shift)) / m_shift);
    }
};