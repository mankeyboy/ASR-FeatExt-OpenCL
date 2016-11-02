#define _USE_MATH_DEFINES
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cfloat>
#include <stdexcept>
#include "mfcccpu.h"
#include "debug.h"

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
static inline float hz2mel(float f) { return 1127 * log(f / 700 + 1); }
static inline float mel2hz(float f) { return 700 * (exp(f / 1127) - 1); }

void MfccCpu::refresh_filters()
{
	float * centers = new float[m_num_banks + 2];
	m_filters = new float[2 * m_window_size2];
	m_filter_beg = new int[m_num_banks + 2];
	memset(m_filters, 0, 2 * m_window_size2 * sizeof(float));

	float minmel = hz2mel(m_low_freq),
		maxmel = hz2mel(m_high_freq);
	for (int i = 0; i < m_num_banks + 2; i++)
	{
		float f = mel2hz(i / float(m_num_banks + 1) * (maxmel - minmel) + minmel);
		float o = 2 * (float)M_PI * f / m_sample_rate;
		o = o + 2 * atan(((1 - m_alpha) * sin(o)) / (1 - (1 - m_alpha) * cos(o)));
		centers[i] = m_sample_rate * o / (2 * (float)M_PI);

		m_filter_beg[i] = floor(centers[i] * m_window_size2 / m_sample_rate + 0.5);
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
			float lowslope = (j * m_sample_rate / (m_window_size2)-cl) / (cc - cl);
			float highslope = (j * m_sample_rate / (m_window_size2)-cr) / (cc - cr);
			m_filters[(i % 2) * m_window_size2 + j] = std::max(0.0f, std::min(lowslope, highslope));
		}
	}

	delete[] centers;
}

void MfccCpu::get_output(float * data_out, float * buff, int width, int height, int spitch, int dpitch)
{
	if (width == spitch == dpitch)
		memcpy(data_out, buff, height * spitch * sizeof(float));
	else
	{
		for (int i = 0; i < height; i++)
			memcpy(data_out + i * dpitch, buff + i * spitch, width * sizeof(float));
	}
}

MfccCpu::MfccCpu(int input_buffer_size,
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
	bool norm_after_dyn)
	: MfccBase(input_buffer_size, window_size, shift, num_banks, sample_rate, low_freq, high_freq, ceps_len,
		want_c0, lift_coef, norm, dyn, delta_l1, delta_l2, norm_after_dyn),
	m_dct_matrix(NULL),
	m_mfcc(NULL),
	m_delta_in(NULL)
{
	m_window_size2 = ceil2((float)m_window_size);
	if (m_dyn != DYN_NONE)
	{
		m_window_limit = m_input_window_limit + 2 + 3 * (m_delta_l1 + m_delta_l2);
	}
	else
	{
		m_delta_l1 = m_delta_l2 = 0;
		m_window_limit = m_input_window_limit + 2;
	}
	m_buffer_size = m_window_limit * m_shift + m_window_size - m_shift;
	m_data_length = m_window_limit * m_window_size2;

	segmenter.init(m_window_size, m_shift, m_window_limit, m_delta_l1 + m_delta_l2);

	m_data = fftwf_alloc_real(m_data_length);
	memset(m_data, 0, m_data_length * sizeof(float));
	m_fft = fftwf_alloc_complex(m_data_length);
	m_mel_energies = new float[m_num_banks * m_window_limit];

	m_fft_plan = fftwf_plan_many_dft_r2c(1, &m_window_size2, m_window_limit, m_data, NULL, 1, m_window_size2, m_fft, NULL, 1, m_window_size2, 0);
	if (!m_fft_plan)
		throw std::runtime_error("Can't create FFTW plan.");

	if (m_ceps_len > 0)
	{
		m_mfcc = new float[m_dct_len * m_window_limit];

		m_dct_matrix = new float[m_num_banks * m_dct_len];
		memset(m_dct_matrix, 0, m_num_banks * m_dct_len * sizeof(float));
		float normfact = sqrt(2.0 / m_num_banks);
		for (int iy = 0; iy < m_num_banks; iy++)
		{
			for (int ix = 1; ix <= m_ceps_len; ix++)
			{
				float lifter = (1 + lift_coef / 2 * sinf((float)M_PI * (float)ix / lift_coef));
				m_dct_matrix[m_dct_len * iy + ix - 1] = lifter * normfact * cosf((float)M_PI * ix * (iy + 0.5f) / m_num_banks);
			}
		}
		if (m_want_c0)
			for (int iy = 0; iy < m_num_banks; iy++)
				m_dct_matrix[m_dct_len * iy + m_ceps_len] = normfact;
	}
	if (m_norm != Normalizer::NORM_NONE)
	{
		int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks;
		normalizer.init(m_norm, cols);
		if (m_dyn == DYN_DELTA || m_dyn == DYN_ACC)
			normalizer_delta.init(m_norm, cols);
		if (m_dyn == DYN_ACC)
			normalizer_acc.init(m_norm, cols);
	}
	if (m_dyn != DYN_NONE)
	{
		int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks,
			rows = m_window_limit + 2 * (m_delta_l1 + m_delta_l2);

		delta.init(cols, m_window_limit + 2 * m_delta_l2, m_delta_l1);
		//m_delta = new float [cols * (m_window_limit + 2 * m_delta_l2)];
		if (m_dyn == DYN_ACC)
			//m_acc = new float [cols * m_window_limit];
			delta_acc.init(cols, m_window_limit, m_delta_l2);
		m_delta_in = new float[cols * rows];
	}

	refresh_filters();
}

MfccCpu::~MfccCpu()
{
	segmenter.cleanup();
	normalizer.cleanup();
	normalizer_delta.cleanup();
	normalizer_acc.cleanup();
	delta.cleanup();
	delta_acc.cleanup();
	fftwf_free(m_data);
	fftwf_free(m_fft);
	fftwf_destroy_plan(m_fft_plan);
	fftwf_cleanup();
	delete[] m_filters;
	delete[] m_filter_beg;
	delete[] m_mel_energies;
	delete[] m_dct_matrix;
	delete[] m_mfcc;
	delete[] m_delta_in;
}

void MfccCpu::set_window(const float * window)
{
	segmenter.set_window(window);
}

void MfccCpu::fft(int window_count)
{
	fftwf_execute(m_fft_plan);
	if (debug_mode & AFET_DEBUG_SAVE_BUFFERS)
		export_c_buffer(m_fft, m_window_size2 / 2 + 1, window_count, sizeof(fftwf_complex), "fft.dat");
}

void MfccCpu::filter(int window_count)
{
	refresh_filters();
	for (int i = 0; i < window_count; i++)
	{
		float sum[2] = { 0,0 };
		int curf = 0;
		int lastf = m_filter_beg[m_num_banks + 1];
		for (int j = m_filter_beg[0]; j <= lastf; j++)
		{
			fftwf_complex & c = m_fft[m_window_size2 * i + j];
			float v = sqrt(c[0] * c[0] + c[1] * c[1]) / m_window_size2;
			//float v = sqrt(c[0]*c[0] + c[1]*c[1]);

			while (j == m_filter_beg[curf + 1])
			{
				curf++;
				if (curf >= 2)
				{
					int sumidx = curf % 2;
					m_mel_energies[m_num_banks * i + curf - 2] = log(std::max(sum[sumidx], 1e-30f));
					sum[sumidx] = 0;
				}
			}
			sum[0] += m_filters[j] * v;
			sum[1] += m_filters[m_window_size2 + j] * v;
		}
	}
}

void MfccCpu::dct(int window_count)
{
	for (int i = 0; i < window_count; i++)
		for (int j = 0; j < m_dct_len; j++)
		{
			float sum = 0;
			for (int k = 0; k < m_num_banks; k++)
				sum += m_mel_energies[m_num_banks * i + k] * m_dct_matrix[m_dct_len * k + j];
			m_mfcc[m_dct_len * i + j] = sum;
		}
}

void MfccCpu::do_delta(int window_count, bool first_call, bool last_call)
{
	if (m_dyn == DYN_NONE)
		return;
	if (first_call && last_call)
		return;
	int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks;
	float * src = m_ceps_len > 0 ? m_mfcc : m_mel_energies;

	if (first_call)
	{
		memcpy(m_delta_in + cols * (m_delta_l1 + m_delta_l2), src, cols * (window_count + m_delta_l1 + m_delta_l2) * sizeof(float));
		for (int i = 0; i < m_delta_l1 + m_delta_l2; i++)
			memcpy(m_delta_in + cols * i, src, cols * sizeof(float));
	}
	else if (last_call)
	{
		memcpy(m_delta_in, src, cols * (window_count + m_delta_l1 + m_delta_l2) * sizeof(float));
		for (int i = 0; i < m_delta_l1 + m_delta_l2; i++)
			memcpy(m_delta_in + cols * (i + window_count + m_delta_l1 + m_delta_l2), src + cols * (window_count + m_delta_l1 + m_delta_l2 - 1), cols * sizeof(float));
	}
	else
		memcpy(m_delta_in, src, cols * (window_count + 2 * (m_delta_l1 + m_delta_l2)) * sizeof(float));

	//delta(m_delta_in, m_delta, cols, window_count + 2 * m_delta_l2, m_delta_l1);
	delta.apply(m_delta_in, window_count + 2 * m_delta_l2);
	if (m_dyn == DYN_ACC)
		//delta(m_delta, m_acc, cols, window_count, m_delta_l2);
		delta_acc.apply(delta.get_output_buffer(), window_count);
}

void MfccCpu::normalize(int window_count, bool use_last_stats)
{
	if (m_norm == Normalizer::NORM_NONE)
		return;
	int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks;
	float * src = m_ceps_len > 0 ? m_mfcc : m_mel_energies;

	if (m_norm_after_dyn)
	{
		normalizer.normalize(segmenter.was_flushed() ? src : src + (m_delta_l1 + m_delta_l2) * cols, window_count, use_last_stats);
		if (m_dyn == DYN_DELTA || m_dyn == DYN_ACC)
			normalizer_delta.normalize(delta.get_output_buffer() + m_delta_l2 * cols, window_count, use_last_stats);
		if (m_dyn == DYN_ACC)
			normalizer_acc.normalize(delta_acc.get_output_buffer(), window_count, use_last_stats);
	}
	else
		normalizer.normalize(src, window_count, use_last_stats);
}

int MfccCpu::set_input(const short * data, int samples)
{
	/*
	if (samples > m_input_buffer_size)
	throw std::runtime_error("Can't process data, buffer is too small");
	int window_count;
	m_last_calc_flushed = m_flushed;
	m_last_block = false;

	if (m_last_calc_flushed)
	{
	memcpy(m_populate, data, samples * sizeof(short));
	//for (int i = 0; i < samples; i++)
	//m_populate[i] = data[i];

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
	memcpy(m_populate, m_populate + samples - m_remaining_samples, m_remaining_samples * sizeof(short));
	m_flushed = false;
	}
	else
	{
	memcpy(m_populate + m_remaining_samples, data, samples * sizeof(short));
	//for (int i = 0; i < samples; i++)
	//m_populate[i + m_remaining_samples] = data[i];

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
	memcpy(m_populate, m_populate + samples - m_remaining_samples, m_remaining_samples * sizeof(short));
	}
	m_samples = samples;
	return window_count;
	*/
	if (samples > m_input_buffer_size)
		throw std::runtime_error("Can't process data, buffer is too small");
	int window_count, window_count_no_delta;
	segmenter.set_input(data, m_data, samples, window_count, window_count_no_delta);
	if (window_count <= 0)
		return 0;
	fft(window_count_no_delta);
	return window_count;
}

int MfccCpu::flush()
{
	if (m_last_block) //nothing to flush
		return 0;
	m_last_block = true;
	/*m_flushed = true;
	int window_count_no_delta = estimated_window_count(m_remaining_samples);
	int window_count = window_count_no_delta - (m_delta_l1 + m_delta_l2);
	if (window_count <= 0)
	return 0;

	segment_data(window_count_no_delta);
	fft(window_count_no_delta);

	return window_count;*/
	int window_count, window_count_no_delta;
	segmenter.flush(m_data, window_count, window_count_no_delta);
	if (window_count <= 0)
		return 0;
	fft(window_count_no_delta);
	return window_count;
}

void MfccCpu::apply()
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
	else if (segmenter.was_flushed())//m_last_calc_flushed)
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

void MfccCpu::get_output_data(float * data_out, int window_count)
{
	if (window_count > m_window_limit)
		throw std::runtime_error("Window count too high");
	int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks,
		pitch = cols;
	switch (m_dyn)
	{
	case DYN_DELTA: pitch *= 2; break;
	case DYN_ACC:   pitch *= 3; break;
	}
	float * src = m_ceps_len > 0 ? m_mfcc : m_mel_energies;
	get_output(data_out, segmenter.was_flushed() ? src : src + (m_delta_l1 + m_delta_l2) * cols, cols, window_count, cols, pitch);
	if (m_dyn == DYN_DELTA || m_dyn == DYN_ACC)
		get_output(data_out + cols, delta.get_output_buffer() + m_delta_l2 * cols, cols, window_count, cols, pitch);
	if (m_dyn == DYN_ACC)
		get_output(data_out + 2 * cols, delta_acc.get_output_buffer(), cols, window_count, cols, pitch);
}
