#define _USE_MATH_DEFINES
#include <cfloat>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include "mfccopencl.h"
#include "openclkernelloader.h"
#include "clmemset.h"
#include <oclUtils.h>
#include <shrQATest.h>
#include <oclDCT8x8_common.h>

static int MAX_BLOCK_SIZE = 256,
	SUM_GRID_SIZE = 256,
	TRANSPOSE_TILE_SIZE = 16,
	DELTA_TILE_SIZE = 16;

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
static inline float hz2mel(float f) {return 1127 * log(f / 700 + 1);}
static inline float mel2hz(float f) {return 700 * (exp(f / 1127) - 1);}

void MfccOpenCL::refresh_filters()
{
	float * centers = new float [m_num_banks + 2];
	float * filters = new float [2 * m_window_size2];
	int * filter_beg = new int [m_num_banks + 2];
	memset(filters, 0, 2 * m_window_size2 * sizeof(float));

	float minmel = hz2mel(m_low_freq),
		maxmel = hz2mel(m_high_freq);
	for (int i = 0; i < m_num_banks + 2; i++)
	{
		float f = mel2hz(i / float(m_num_banks + 1) * (maxmel - minmel) + minmel);
		float o = 2 * (float)M_PI * f / m_sample_rate;
		o = o + 2 * atan(((1 - m_alpha) * sin(o)) / (1 - (1 - m_alpha) * cos(o)));
		centers[i] = m_sample_rate * o / (2 * (float)M_PI);

		filter_beg[i] = floor(centers[i] * m_window_size2 / m_sample_rate + 0.5);
	}

	for (int i = 0; i < m_num_banks; i++)
	{
		float cl = centers[i],
			cc = centers[i + 1],
			cr = centers[i + 2];
		int il = floor(m_window_size2 * cl / m_sample_rate + 0.5),
			ic = floor(m_window_size2 * cc / m_sample_rate + 0.5),
			ir = floor(m_window_size2 * cr / m_sample_rate + 0.5);
		for (int j = il; j < ir; j++)
		{
			float lowslope = (j * m_sample_rate / (m_window_size2) - cl) / (cc - cl);
			float highslope = (j * m_sample_rate / (m_window_size2) - cr) / (cc - cr);
			filters[(i % 2) * m_window_size2 + j] = std::max(0.0f, std::min(lowslope, highslope));
		}
	}
	clEnqueueWriteBuffer(cmd_queue, d_filters, CL_FALSE, 0, 2 * m_window_size2 * sizeof(float), filters, 0, NULL, NULL);
	clEnqueueWriteBuffer(cmd_queue, d_filter_beg, CL_FALSE, 0, (m_num_banks + 2) * sizeof(float), filter_beg, 0, NULL, NULL);

	delete [] centers;
	delete [] filters;
	delete [] filter_beg;
}

void MfccOpenCL::get_output(float * data_out, cl_mem d_buff, int width, int height, int spitch, int dpitch, int buff_offset)
{
	if (width == spitch && spitch == dpitch)
		clEnqueueReadBuffer(cmd_queue, d_buff, CL_TRUE, buff_offset * sizeof(float), height * spitch * sizeof(float), data_out, 0, NULL, NULL);
	else
	{
		float * tmp = new float [height * spitch];
		clEnqueueReadBuffer(cmd_queue, d_buff, CL_TRUE, buff_offset * sizeof(float), height * spitch * sizeof(float), tmp, 0, NULL, NULL);
		for (int i = 0; i < height; i++)
			memcpy(data_out + i * dpitch, tmp + i * spitch, width * sizeof(float));
		delete [] tmp;
	}
}

MfccOpenCL::MfccOpenCL(int input_buffer_size,
    int window_size,
    int shift,
    int num_banks,
    float sample_rate,
    float low_freq,
    float high_freq,
    int ceps_len,
    bool want_c0,
    float lift_coef,
    Normalizer::norm_t norm,
    dyn_t dyn,
    int delta_l1,
    int delta_l2,
    bool norm_after_dyn,
    cl_device_id opencl_device)
    : MfccBase(input_buffer_size, window_size, shift, num_banks, sample_rate, low_freq, high_freq, ceps_len,
    want_c0, lift_coef, norm, dyn, delta_l1, delta_l2, norm_after_dyn),
    d_data(NULL),
    d_fft(NULL),
    d_mel_energies(NULL),
    d_filters(NULL),
    d_filter_beg(NULL),
    d_mfcc(NULL),
    d_dct_matrix(NULL),
    d_delta_in(NULL),
    context(0),
    cmd_queue(0),
    program(0),
    kernel_transpose(0),
    kernel_filter(0)
{
    bool using_doubles = false;

	m_window_size2 = ceil2((float)m_window_size);
	m_num_banks2 = ceil2((float)m_num_banks);
	m_dct_len2 = ceil2((float)m_dct_len);
	if (m_dyn != DYN_NONE)
	{
		m_window_limit = m_input_window_limit + 2 + 3 * (m_delta_l1 + m_delta_l2);
	}
	else
	{
		m_delta_l1 = m_delta_l2 = 0;
		m_window_limit = m_input_window_limit + 2;
	}
	cl_uint memoryAlign;
	clGetDeviceInfo(opencl_device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(memoryAlign), &memoryAlign, NULL);
	memoryAlign /= 8 * sizeof(float);
	m_window_limit = memoryAlign * (int)((m_window_limit + memoryAlign - 1) / memoryAlign);
	m_buffer_size = m_window_limit * m_shift + m_window_size - m_shift;
	m_data_length = m_window_limit * m_window_size2;

	cl_int clerr;
	context = clCreateContext(NULL, 1, &opencl_device, NULL, NULL, NULL);
	if (!context)
		throw std::runtime_error("Can't create OpenCL context");
	cmd_queue = clCreateCommandQueue(context, opencl_device, 0, NULL);
	
	if (!clfft.init(context, cmd_queue, m_window_size2, m_window_limit))
		throw std::runtime_error("Can't create FFT plan");

	
    segmenter.init(m_window_size, m_shift, m_window_limit, m_delta_l1 + m_delta_l2, context, cmd_queue, opencl_device);

	d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, clfft.get_element_size() * m_data_length * sizeof(float), NULL, NULL);
	d_fft = clCreateBuffer(context, CL_MEM_READ_WRITE, clfft.get_output_vector_size() * m_window_limit * 2 * sizeof(float), NULL, NULL);
	d_mel_energies = clCreateBuffer(context, CL_MEM_READ_WRITE, m_num_banks2 * m_window_limit * sizeof(float), NULL, NULL);
	d_filters = clCreateBuffer(context, CL_MEM_READ_ONLY, 2 * m_window_size2 * sizeof(float), NULL, NULL);
	d_filter_beg = clCreateBuffer(context, CL_MEM_READ_ONLY, (m_num_banks + 2) * sizeof(float), NULL, NULL);
	clMemset(cmd_queue, d_mel_energies, 0, 0, m_num_banks2 * m_window_limit * sizeof(float), 0, NULL, NULL);
	if (m_ceps_len > 0)
	{
		d_mfcc = clCreateBuffer(context, CL_MEM_READ_WRITE, m_dct_len2 * m_window_limit * sizeof(float), NULL, NULL);
		d_dct_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, m_dct_len2 * m_num_banks2 * sizeof(float), NULL, NULL);

		float * dct_matrix = new float [m_dct_len2 * m_num_banks2];
		memset(dct_matrix, 0, m_num_banks2 * m_dct_len2 * sizeof(float));
		float normfact = sqrt(2.0 / m_num_banks);
		for (int iy = 0; iy < m_num_banks; iy++)
		{
			for (int ix = 1; ix <= m_ceps_len; ix++)
			{
				float lifter = (1 + lift_coef / 2 * sinf((float)M_PI * (float)ix / lift_coef));
				dct_matrix[m_dct_len2 * iy + ix - 1] = lifter * normfact * cosf((float)M_PI * ix * (iy + 0.5f) / m_num_banks);
			}
		}
		if (m_want_c0)
			for (int iy = 0; iy < m_num_banks; iy++)
				dct_matrix[m_dct_len2 * iy + m_ceps_len] = normfact;
		clEnqueueWriteBuffer(cmd_queue, d_dct_matrix, CL_TRUE, 0, m_dct_len2 * m_num_banks2 * sizeof(float), dct_matrix, 0, NULL, NULL);
		delete [] dct_matrix;
	}
	if (m_norm != Normalizer::NORM_NONE)
	{
        int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks,
            cols2 = m_ceps_len > 0 ? m_dct_len2 : m_num_banks2;
        normalizer.init(m_norm, cols, cols2, using_doubles, context, cmd_queue, opencl_device);
        if (m_dyn == DYN_DELTA || m_dyn == DYN_ACC)
            normalizer_delta.init(m_norm, cols, cols2, using_doubles, context, cmd_queue, opencl_device);
        if (m_dyn == DYN_ACC)
            normalizer_acc.init(m_norm, cols, cols2, using_doubles, context, cmd_queue, opencl_device);
	}
	if (m_dyn != DYN_NONE)
	{
    	int cols2 = m_ceps_len > 0 ? m_dct_len2 : m_num_banks2,
			rows = m_window_limit + 2 * (m_delta_l1 + m_delta_l2);
	
		//d_delta = clCreateBuffer(context, CL_MEM_READ_WRITE, cols2 * (m_window_limit + 2 * m_delta_l2) * sizeof(float), NULL, NULL);
        delta.init(cols2, cols2, m_window_limit + 2 * m_delta_l2, m_delta_l1, context, cmd_queue, opencl_device);
		if (m_dyn == DYN_ACC)
            delta_acc.init(cols2, cols2, m_window_limit, m_delta_l2, context, cmd_queue, opencl_device);
		    //d_acc = clCreateBuffer(context, CL_MEM_READ_WRITE, cols2 * m_window_limit * sizeof(float), NULL, NULL);
		d_delta_in = clCreateBuffer(context, CL_MEM_READ_WRITE, cols2 * rows * sizeof(float), NULL, NULL);
	}

	std::string program_source;
	if (!LoadOpenCLKernel("mfcc", program_source))
		throw std::runtime_error("Error while loading OpenCL kernel files\n");
	const char * program_source_ptr = program_source.c_str();
	size_t program_source_size = program_source.size();
	program = clCreateProgramWithSource(context, 1, &program_source_ptr, &program_source_size, NULL);
	if (!program)
		throw std::runtime_error("Can't create OpenCL program");
	std::stringstream buildOptions;
    buildOptions << "-D TRANSPOSE_TILE_SIZE=" << TRANSPOSE_TILE_SIZE;
	buildOptions << " -D USE_APPLE_FFT";
	cl_int ret = clBuildProgram(program, 0, NULL, buildOptions.str().c_str(), NULL, NULL);
	if (ret != CL_SUCCESS)
	{
		size_t len;
		clGetProgramBuildInfo(program, opencl_device, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
		char * buffer = new char [len];
		clGetProgramBuildInfo(program, opencl_device, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
		std::string msg = "Can't build OpenCL program:\n";
		msg += buffer;
		std::runtime_error e(msg.c_str());
		delete [] buffer;
		throw e;
	}

	kernel_transpose = clCreateKernel(program, "kernelTranspose", NULL);
	clSetKernelArg(kernel_transpose, 0, sizeof(cl_mem), &d_fft);
	clSetKernelArg(kernel_transpose, 1, sizeof(cl_mem), &d_data);
	cl_int param_int = clfft.get_output_vector_size();
	clSetKernelArg(kernel_transpose, 2, sizeof(cl_int), &param_int);
	param_int = m_window_limit;
	clSetKernelArg(kernel_transpose, 4, sizeof(cl_int), &param_int);
	cl_float param_float = 1.f / m_window_size2;
	clSetKernelArg(kernel_transpose, 5, sizeof(cl_float), &param_float);

	kernel_filter = clCreateKernel(program, "kernelFilter", NULL);
	clSetKernelArg(kernel_filter, 0, sizeof(cl_mem), &d_data);
	clSetKernelArg(kernel_filter, 1, sizeof(cl_mem), &d_mel_energies);
	clSetKernelArg(kernel_filter, 2, sizeof(cl_mem), &d_filter_beg);
	clSetKernelArg(kernel_filter, 3, sizeof(cl_mem), &d_filters);
	param_int = m_window_limit;
	clSetKernelArg(kernel_filter, 4, sizeof(cl_int), &param_int);
	param_int = m_window_size2;
	clSetKernelArg(kernel_filter, 6, sizeof(cl_int), &param_int);
	param_int = m_num_banks;
	clSetKernelArg(kernel_filter, 7, sizeof(cl_int), &param_int);
	param_int = m_num_banks2;
	clSetKernelArg(kernel_filter, 8, sizeof(cl_int), &param_int);
	param_float = 1e-30f;
	clSetKernelArg(kernel_filter, 9, sizeof(cl_float), &param_float);

	refresh_filters();

	initDCT8x8(context, cmd_queue);

}

MfccOpenCL::~MfccOpenCL()
{
    segmenter.cleanup();
    normalizer.cleanup();
    normalizer_delta.cleanup();
    normalizer_acc.cleanup();
    delta.cleanup();
    delta_acc.cleanup();
	clReleaseMemObject(d_data);
	clReleaseMemObject(d_fft);
	clReleaseMemObject(d_mel_energies);
	clReleaseMemObject(d_filters);
	clReleaseMemObject(d_filter_beg);
	clReleaseMemObject(d_mfcc);
	clReleaseMemObject(d_dct_matrix);
	clReleaseMemObject(d_delta_in);
	clReleaseKernel(kernel_transpose);
	clReleaseKernel(kernel_filter);
	clReleaseProgram(program);
	clfft.cleanup();
	clReleaseCommandQueue(cmd_queue);
	clReleaseContext(context);
}

void MfccOpenCL::set_window(const float * window)
{
    segmenter.set_window(window);
}

void MfccOpenCL::fft(int window_count)
{
    	if (!clfft.apply(d_data, d_fft, window_count, NULL))
		throw std::runtime_error("Error while computing FFT");
	
	size_t global_work_size[2],
		local_work_size[2];
	local_work_size[0] = TRANSPOSE_TILE_SIZE;
	local_work_size[1] = TRANSPOSE_TILE_SIZE;
	global_work_size[0] = (((m_window_size2 / 2 + 1) + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];  //TODO: should be the same as in plp code?
	global_work_size[1] = 16 * local_work_size[1];
	
	int window_count2 = int((window_count + TRANSPOSE_TILE_SIZE - 1) / TRANSPOSE_TILE_SIZE) * TRANSPOSE_TILE_SIZE;
	cl_int clwindow_count = window_count2;
	clSetKernelArg(kernel_transpose, 3, sizeof(cl_int), &clwindow_count);
	
	cl_int ret = clEnqueueNDRangeKernel(cmd_queue, kernel_transpose, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
	if (ret != CL_SUCCESS)
		throw std::runtime_error("Error in kernelTranspose");
}

void MfccOpenCL::filter(int window_count)
{
	refresh_filters();
	size_t global_work_size[1],
		local_work_size[2];
	local_work_size[0] = 128;
	global_work_size[0] = 128 * local_work_size[0];

	int window_count2 = int((window_count + TRANSPOSE_TILE_SIZE - 1) / TRANSPOSE_TILE_SIZE) * TRANSPOSE_TILE_SIZE;
	cl_int clwindow_count = window_count2;
	clSetKernelArg(kernel_filter, 5, sizeof(cl_int), &clwindow_count);

	cl_int ret = clEnqueueNDRangeKernel(cmd_queue, kernel_filter, 1, NULL, global_work_size, local_work_size, 0, NULL,  NULL);
	if (ret != CL_SUCCESS)
		throw std::runtime_error("Error in kernelFilter");
}

void MfccOpenCL::dct(int window_count)
{
    	/*clAmdBlasStatus status = clAmdBlasSgemmEx(clAmdBlasRowMajor, clAmdBlasNoTrans, clAmdBlasNoTrans, window_count, m_dct_len2, m_num_banks2, 1,
		d_mel_energies, 0, m_num_banks2,
		d_dct_matrix, 0, m_dct_len2, 0,
		d_mfcc, 0, m_dct_len2,
		1, &cmd_queue, 0, NULL, NULL);
	if (status != clAmdBlasSuccess)
		throw std::runtime_error("Error while computing DCT");*/
	DCT8x8(
		cmd_queue,
		d_dct_matrix,
		d_mfcc,
		m_num_banks2,
		m_dct_len2,
		window_count,
		DCT_FORWARD
	);
	/*for (int i = 0; i < window_count; i++)
	for (int j = 0; j < m_dct_len2; j++)
	{
		float sum = 0;
		for (int k = 0; k < m_num_banks2; k++)
			sum += m_mel_energies[m_num_banks2 * i + k] * m_dct_matrix[m_dct_len * k + j];
		m_mfcc[m_dct_len2 * i + j] = sum;
	}*/
}

void MfccOpenCL::do_delta(int window_count, bool first_call, bool last_call)
{
	if (m_dyn == DYN_NONE)
		return;
	if (first_call && last_call)
		throw std::runtime_error("Invalid arguments");
	int cols2 = m_ceps_len > 0 ? m_dct_len2 : m_num_banks2;
	cl_mem src = m_ceps_len > 0 ? d_mfcc : d_mel_energies;

    if (first_call)
	{
		clEnqueueCopyBuffer(cmd_queue, src, d_delta_in, 0, cols2 * (m_delta_l1 + m_delta_l2) * sizeof(float), cols2 * (window_count + m_delta_l1 + m_delta_l2) * sizeof(float), 0, NULL, NULL);
		for (int i = 0; i < m_delta_l1 + m_delta_l2; i++)
			clEnqueueCopyBuffer(cmd_queue, src, d_delta_in, 0, cols2 * i * sizeof(float), cols2 * sizeof(float), 0, NULL, NULL);
	}
	else if (last_call)
	{
	    clEnqueueCopyBuffer(cmd_queue, src, d_delta_in, 0, 0, cols2 * (window_count + m_delta_l1 + m_delta_l2) * sizeof(float), 0, NULL, NULL);
		for (int i = 0; i < m_delta_l1 + m_delta_l2; i++)
			clEnqueueCopyBuffer(cmd_queue, src, d_delta_in, cols2 * (window_count + m_delta_l1 + m_delta_l2 - 1) * sizeof(float), cols2 * (i + window_count + m_delta_l1 + m_delta_l2) * sizeof(float), cols2 * sizeof(float), 0, NULL, NULL);
	}
	else
		clEnqueueCopyBuffer(cmd_queue, src, d_delta_in, 0, 0, cols2 * (window_count + 2 * (m_delta_l1 + m_delta_l2)) * sizeof(float), 0, NULL, NULL);
    
    delta.apply(d_delta_in, window_count + 2 * m_delta_l2);
    if (m_dyn == DYN_ACC)
        delta_acc.apply(delta.get_output_buffer(), window_count);
}

void MfccOpenCL::normalize(int window_count, bool use_last_stats)
{
	if (m_norm == Normalizer::NORM_NONE)
		return;
	int cols2 = m_ceps_len > 0 ? m_dct_len2 : m_num_banks2;
	cl_mem src = m_ceps_len > 0 ? d_mfcc : d_mel_energies;

	if (m_norm_after_dyn)
    {
        normalizer.normalize(src, segmenter.was_flushed() ? 0 : (m_delta_l1 + m_delta_l2) * cols2, window_count, use_last_stats);
        if (m_dyn == DYN_DELTA || m_dyn == DYN_ACC)
            normalizer_delta.normalize(delta.get_output_buffer(), m_delta_l2 * cols2, window_count, use_last_stats);
        if (m_dyn == DYN_ACC)
            normalizer_acc.normalize(delta_acc.get_output_buffer(), 0, window_count, use_last_stats);
    }
    else
        normalizer.normalize(src, 0, window_count, use_last_stats);
}

int MfccOpenCL::set_input(const short * data, int samples)
{
	/*if (samples > m_input_buffer_size)
		throw std::runtime_error("Can't process data, buffer is too small");
	int window_count;
	m_last_calc_flushed = m_flushed;
	m_last_block = false;

	if (m_last_calc_flushed)
	{
        if (benchmark)
            swmem.start();
		clEnqueueWriteBuffer(cmd_queue, d_populate, CL_FALSE, 0, samples * sizeof(short), data, 0, NULL, benchmark ? &swmem.event : NULL);
        if (benchmark)
            benchmark_mem += swmem.stop();

		int window_count_no_delta = estimated_window_count(samples);
		window_count = window_count_no_delta - (m_delta_l1 + m_delta_l2);
		if (window_count <= 0)
			throw std::runtime_error("Can't process data, window count is too small");

		segment_data(window_count_no_delta);
		fft(window_count_no_delta);

		int processed_samples = (window_count - (m_delta_l1 + m_delta_l2)) * m_shift + m_window_size - m_shift;
		if (processed_samples <= 0)
			throw std::runtime_error("Processed samples <= 0, this should never happen");
		m_remaining_samples = samples - processed_samples + m_window_size - m_shift;
        if (benchmark)
            swmem.start();
		clEnqueueCopyBuffer(cmd_queue, d_populate, d_populate, (samples - m_remaining_samples) * sizeof(short), 0, m_remaining_samples * sizeof(short), 0, NULL, benchmark ? &swmem.event : NULL);
        if (benchmark)
            benchmark_mem += swmem.stop();
		m_flushed = false;
	}
	else
	{
        if (benchmark)
            swmem.start();
		clEnqueueWriteBuffer(cmd_queue, d_populate, CL_FALSE, m_remaining_samples * sizeof(short), samples * sizeof(short), data, 0, NULL, benchmark ? &swmem.event : NULL);
        if (benchmark)
            benchmark_mem += swmem.stop();

		samples += m_remaining_samples;
		int window_count_no_delta = estimated_window_count(samples);
		window_count = window_count_no_delta - 2 * (m_delta_l1 + m_delta_l2);
		if (window_count > 0)
		{
			segment_data(window_count_no_delta);
			fft(window_count_no_delta);
		}
		else
			window_count = 0;

		int processed_samples = window_count * m_shift + m_window_size - m_shift;
		m_remaining_samples = samples - processed_samples + m_window_size - m_shift;
        if (benchmark)
            swmem.start();
		clEnqueueCopyBuffer(cmd_queue, d_populate, d_populate, (samples - m_remaining_samples) * sizeof(short), 0, m_remaining_samples * sizeof(short), 0, NULL, benchmark ? &swmem.event : NULL);
        if (benchmark)
            benchmark_mem += swmem.stop();
	}
	m_samples = samples;
	return window_count;*/
    if (samples > m_input_buffer_size)
        throw std::runtime_error("Can't process data, buffer is too small");
    int window_count, window_count_no_delta;
    segmenter.set_input(data, d_data, samples, window_count, window_count_no_delta, NULL);
    if (window_count <= 0)
        return 0;
    fft(window_count_no_delta);
    return window_count;
}

int MfccOpenCL::flush()
{
	if (m_last_block) //nothing to flush
		return 0;
	m_last_block = true;
    int window_count, window_count_no_delta;
    segmenter.flush(d_data, window_count, window_count_no_delta);
    if (window_count <= 0)
        return 0;
    fft(window_count_no_delta);
    return window_count;
}

void MfccOpenCL::apply()
{
	if (m_last_block)
	{
	    int window_count_no_delta = estimated_window_count(segmenter.get_remaining_samples());
	    int window_count = window_count_no_delta - (m_delta_l1 + m_delta_l2);

	    if (window_count > 0)
	    {
		filter(window_count_no_delta);
		if (m_ceps_len > 0)
			dct(window_count_no_delta);
		if (!m_norm_after_dyn && m_norm != Normalizer::NORM_NONE)
			normalize(window_count_no_delta, true);
		if (m_dyn != DYN_NONE)
			do_delta(window_count, false, true);
		if (m_norm_after_dyn && m_norm != Normalizer::NORM_NONE)
			normalize(window_count, true);
	    }
	}
	else if (segmenter.was_flushed())
	{
		int window_count_no_delta = estimated_window_count(segmenter.get_samples());
		int window_count = window_count_no_delta - (m_delta_l1 + m_delta_l2);
		if (window_count <= 0)
			throw std::runtime_error("Can't process data, window count is too small");

		filter(window_count_no_delta);
		if (m_ceps_len > 0)
			dct(window_count_no_delta);
		if (!m_norm_after_dyn && m_norm != Normalizer::NORM_NONE)
			normalize(window_count_no_delta, false);
		if (m_dyn != DYN_NONE)
			do_delta(window_count, true, false);
		if (m_norm_after_dyn && m_norm != Normalizer::NORM_NONE)
			normalize(window_count, false);
	}
	else
	{
		int window_count_no_delta = estimated_window_count(segmenter.get_samples());
		int window_count = window_count_no_delta - 2 * (m_delta_l1 + m_delta_l2);
		if (window_count > 0)
		{
			filter(window_count_no_delta);
			if (m_ceps_len > 0)
				dct(window_count_no_delta);
			if (!m_norm_after_dyn && m_norm != Normalizer::NORM_NONE)
				normalize(window_count_no_delta, false);
			if (m_dyn != DYN_NONE)
				do_delta(window_count, false, false);
			if (m_norm_after_dyn && m_norm != Normalizer::NORM_NONE)
				normalize(window_count, false);
		}
	}
}

void MfccOpenCL::get_output_data(float * data_out, int window_count)
{
	if (window_count > m_window_limit)
		throw std::runtime_error("Window count too high");
	int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks,
		cols2 = m_ceps_len > 0 ? m_dct_len2 : m_num_banks2,
		pitch = cols;
	switch (m_dyn)
	{
		case DYN_DELTA: pitch *= 2; break;
		case DYN_ACC:   pitch *= 3; break;
	}
	cl_mem src = m_ceps_len > 0 ? d_mfcc : d_mel_energies;
    get_output(data_out, src, cols, window_count, cols2, pitch, segmenter.was_flushed() ? 0 : ((m_delta_l1 + m_delta_l2) * cols2));
	if (m_dyn == DYN_DELTA || m_dyn == DYN_ACC)
		get_output(data_out + cols, delta.get_output_buffer(), cols, window_count, cols2, pitch, m_delta_l2 * cols2);
	if (m_dyn == DYN_ACC)
		get_output(data_out + 2 * cols, delta_acc.get_output_buffer(), cols, window_count, cols2, pitch);
}
