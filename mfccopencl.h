#ifndef _MFCCOPENCL_H_
#define _MFCCOPENCL_H_

#include "mfccbase.h"
#include <cstdio>
#include <CL/cl.h>
#include "clfft.h"
#include "segmenteropencl.h"
#include "normalizeropencl.h"
#include "deltaopencl.h"

class MfccOpenCL : public MfccBase
{
private:
	int m_buffer_size,
		m_window_limit,
		m_data_length,
		m_window_size2,
		m_num_banks2,
		m_dct_len2;
    cl_mem d_data,
        d_fft,
        d_mel_energies,
        d_filters,
        d_filter_beg,
        d_mfcc,
        d_dct_matrix,
        d_delta_in;
	float * m_data,
		*m_mel_energies,
		*m_mfcc,
		*m_dct_matrix,
		*m_filters,
		*m_delta_in;
	cl_context context;
	cl_command_queue cmd_queue;
	cl_program program;
    cl_kernel kernel_transpose,
        kernel_filter;
	clFFT clfft;
    SegmenterOpenCL segmenter;
    NormalizerOpenCL normalizer,
        normalizer_delta,
        normalizer_acc;
    DeltaOpenCL delta,
        delta_acc;

	void refresh_filters();
	void get_output(float * data_out, cl_mem d_buff, int width, int height, int spitch, int dpitch, int buff_offset = 0);
public:
	MfccOpenCL(int input_buffer_size,
		int window_size,
		int shift,
		int num_banks,
		float sample_rate,
		float low_freq,
		float high_freq,
		int ceps_len,
		bool want_c0,
		float lift_coef,
		Normalizer::norm_t norm = Normalizer::NORM_NONE,
		dyn_t dyn = DYN_NONE,
		int delta_l1 = 1,
		int delta_l2 = 1,
		bool norm_after_dyn = true,
		cl_device_id opencl_device = 0);
	~MfccOpenCL();

	void set_window(const float * window);
	void fft(int window_count);
	void filter(int window_count);
	void dct(int window_count);
	void do_delta(int window_count, bool first_call, bool last_call);
	void normalize(int window_count, bool use_last_stats);
	int set_input(const short * data, int samples);
	int flush();
	void apply();

	void get_output_data(float * data_out, int window_count);
};

#endif
