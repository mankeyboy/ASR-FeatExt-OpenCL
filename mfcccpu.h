#ifndef _MFCCCPU_H_
#define _MFCCCPU_H_

#include "mfccbase.h"
#include "segmentercpu.h"
#include "normalizercpu.h"
#include "deltacpu.h"
#include <fftw3.h>

class MfccCpu : public MfccBase
{
private:
	int m_buffer_size,
		m_window_limit,
		m_data_length,
		m_window_size2;
	float * m_data,
		*m_mel_energies,
		*m_mfcc,
		*m_dct_matrix,
		*m_filters,
		*m_delta_in;
	int * m_filter_beg;
	fftwf_complex * m_fft;
	fftwf_plan m_fft_plan;
	SegmenterCPU segmenter;
	NormalizerCPU normalizer,
		normalizer_delta,
		normalizer_acc;
	DeltaCPU delta, delta_acc;

	void refresh_filters();
	void get_output(float * data_out, float * buff, int width, int height, int spitch, int dpitch);
public:
	MfccCpu(int input_buffer_size,
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
		bool norm_after_dyn = true);
	~MfccCpu();

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
