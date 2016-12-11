#include <cstring>
#include <stdexcept>
#include "openclkernelloader.h"
#include "segmenteropencl.h"
#include "clmemset.h"
//#include "debug.h"

static int MAX_BLOCK_SIZE = 256;

static inline unsigned int ceil2(unsigned int v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void SegmenterOpenCL::segment_data(cl_mem d_data_out, int window_count)
{
    size_t global_work_size[2],
        local_work_size[2];
    local_work_size[0] = std::min(MAX_BLOCK_SIZE, m_window_size2);
    local_work_size[1] = 1;
    global_work_size[0] = ((m_window_size2 + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE) * local_work_size[0];
    global_work_size[1] = 256 * local_work_size[1];

    clSetKernelArg(kernel_segment, 1, sizeof(cl_mem), &d_data_out);
    cl_int clwindow_count = window_count;
    clSetKernelArg(kernel_segment, 3, sizeof(cl_int), &clwindow_count);
    //if (benchmark)
    //{
    //    swcpu.start();
    //    sw.start();
    //}
    clMemset(cmd_queue, d_data_out, 0, 0, m_window_size2 * window_count * sizeof(float), 0, NULL, NULL);
    cl_int ret = clEnqueueNDRangeKernel(cmd_queue, kernel_segment, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (ret != CL_SUCCESS)
        throw std::runtime_error("Error in kernelSegmentWindow");
    //if (benchmark)
    //{
    //    benchmark_win += sw.stop();
    //    benchmark_cpu_win += swcpu.stop();
    //}
    ///*if (debug_mode & AFET_DEBUG_SAVE_BUFFERS)
    //*/    export_cl_buffer(cmd_queue, d_data_out, m_window_size2, window_count, sizeof(float), "seg_wave.dat");
}

void SegmenterOpenCL::init(int window_size, int shift, int window_limit, int deltasize, cl_context context, cl_command_queue cmd_queue, cl_device_id opencl_device)
{
    this->context = context;
    this->cmd_queue = cmd_queue;
    m_window_size = window_size;
    m_shift = shift;
    m_deltasize = deltasize;
    m_remaining_samples = 0;
    m_samples = 0;
    m_flushed = true;
    m_last_calc_flushed = false;
    m_window_size2 = ceil2((float)m_window_size);
    m_buffer_size = window_limit * shift + window_size - shift;
    d_tmpbuffer = clCreateBuffer(context, CL_MEM_READ_ONLY, m_buffer_size * sizeof(short), NULL, NULL);
    d_window = clCreateBuffer(context, CL_MEM_READ_ONLY, m_window_size2 * sizeof(float), NULL, NULL);

    std::string program_source;
    if (!LoadOpenCLKernel("segmenter", program_source))
        throw std::runtime_error("Error while loading segmenter OpenCL kernel\n");
    const char * program_source_ptr = program_source.c_str();
    size_t program_source_size = program_source.size();
    program = clCreateProgramWithSource(context, 1, &program_source_ptr, &program_source_size, NULL);
    if (!program)
        throw std::runtime_error("Can't create OpenCL program");
    cl_int ret = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
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

    kernel_segment = clCreateKernel(program, "kernelSegmentWindow", NULL);
    clSetKernelArg(kernel_segment, 0, sizeof(cl_mem), &d_tmpbuffer);
    clSetKernelArg(kernel_segment, 2, sizeof(cl_mem), &d_window);
    cl_int param_int = m_window_size;
    clSetKernelArg(kernel_segment, 4, sizeof(cl_int), &param_int);
    param_int = m_window_size2;
    clSetKernelArg(kernel_segment, 5, sizeof(cl_int), &param_int);
    param_int = m_shift;
    clSetKernelArg(kernel_segment, 6, sizeof(cl_int), &param_int);
    clSetKernelArg(kernel_segment, 7, sizeof(cl_float) * std::min(MAX_BLOCK_SIZE, m_window_size2), NULL);
}

void SegmenterOpenCL::cleanup()
{
    clReleaseMemObject(d_tmpbuffer);
    clReleaseMemObject(d_window);
    clReleaseKernel(kernel_segment);
    clReleaseProgram(program);
}

void SegmenterOpenCL::set_window(const float * window)
{
    float * tmp = new float[m_window_size2];
    memcpy(tmp, window, m_window_size * sizeof(float));
    if (m_window_size2 > m_window_size)
        memset(tmp + m_window_size, 0, (m_window_size2 - m_window_size) * sizeof(float));
    clEnqueueWriteBuffer(cmd_queue, d_window, CL_TRUE, 0, m_window_size2 * sizeof(float), tmp, 0, NULL, NULL);
    delete[] tmp;
}

void SegmenterOpenCL::set_input(const short * data_in, cl_mem d_data_out, int samples, int & window_count, int & window_count_no_delta, float * timemem)
{
    m_last_calc_flushed = m_flushed;
    if (m_last_calc_flushed)
    {
        clEnqueueWriteBuffer(cmd_queue, d_tmpbuffer, CL_FALSE, 0, samples * sizeof(short), data_in, 0, NULL, NULL);
        
        window_count_no_delta = estimated_window_count(samples);
        window_count = window_count_no_delta - m_deltasize;
        if (window_count <= 0)
            throw std::runtime_error("Can't process data, window count is too small");

        segment_data(d_data_out, window_count_no_delta);

        int processed_samples = (window_count - m_deltasize) * m_shift + m_window_size - m_shift;
        if (processed_samples <= 0)
            throw std::runtime_error("Processed samples <= 0, this should never happen");
        m_remaining_samples = samples - processed_samples + m_window_size - m_shift;

        clEnqueueCopyBuffer(cmd_queue, d_tmpbuffer, d_tmpbuffer, (samples - m_remaining_samples) * sizeof(short), 0, m_remaining_samples * sizeof(short), 0, NULL, NULL);
        
        m_flushed = false;
    }
    else
    {
        clEnqueueWriteBuffer(cmd_queue, d_tmpbuffer, CL_FALSE, m_remaining_samples * sizeof(short), samples * sizeof(short), data_in, 0, NULL,NULL);
        
        samples += m_remaining_samples;
        window_count_no_delta = estimated_window_count(samples);
        window_count = window_count_no_delta - 2 * m_deltasize;
        if (window_count > 0)
        {
            segment_data(d_data_out, window_count_no_delta);
        }
        else
            window_count = 0;

        int processed_samples = window_count * m_shift + m_window_size - m_shift;
        m_remaining_samples = samples - processed_samples + m_window_size - m_shift;

        clEnqueueCopyBuffer(cmd_queue, d_tmpbuffer, d_tmpbuffer, (samples - m_remaining_samples) * sizeof(short), 0, m_remaining_samples * sizeof(short), 0, NULL, NULL);
        
    }
    m_samples = samples;
}

void SegmenterOpenCL::flush(cl_mem d_data_out, int & window_count, int & window_count_no_delta)
{
    m_flushed = true;
    window_count_no_delta = estimated_window_count(m_remaining_samples);
    window_count = window_count_no_delta - m_deltasize;
    if (window_count <= 0)
        return;

    segment_data(d_data_out, window_count_no_delta);
}