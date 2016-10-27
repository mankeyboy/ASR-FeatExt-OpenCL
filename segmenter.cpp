#include <cstring>
#include <stdexcept>
#include "Segmenter.h"

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

void Segmenter::segment_data(float * data_out, int window_count)
{
	for (int i = 0; i < window_count; i++)
	{
		for (int j = 0; j < m_window_size; j++)
		{
			data_out[m_window_size2 * i + j] = m_window[j] * m_tmpbuffer[i * m_shift + j];
		}
	}
}

void Segmenter::init(int window_size, int shift, int window_limit, int deltasize)
{
	m_window_size = window_size;
	m_shift = shift;
	m_deltasize = deltasize;
	m_remaining_samples = 0;
	m_samples = 0;
	m_flushed = true;
	m_last_calc_flushed = false;
	m_window_size2 = ceil2((float)m_window_size);
	m_buffer_size = window_limit * shift + window_size - shift;
	m_tmpbuffer = new short[m_buffer_size];
	m_window = new float[m_window_size];
}

void Segmenter::cleanup()
{
	delete[] m_tmpbuffer;
	delete[] m_window;
}

void Segmenter::set_window(const float * window)
{
	memcpy(m_window, window, m_window_size * sizeof(float));
}

void Segmenter::set_input(const short * data_in, float * data_out, int samples, int & window_count, int & window_count_no_delta)
{
	m_last_calc_flushed = m_flushed;
	if (m_last_calc_flushed)
	{
		memcpy(m_tmpbuffer, data_in, samples * sizeof(short));
		window_count_no_delta = estimated_window_count(samples);
		window_count = window_count_no_delta - m_deltasize;
		if (window_count <= 0)
			throw std::runtime_error("Can't process data, window count is too small");

		segment_data(data_out, window_count_no_delta);

		int processed_samples = (window_count - m_deltasize) * m_shift + m_window_size - m_shift;
		if (processed_samples <= 0)
			throw std::runtime_error("Processed samples <= 0, this should never happen");
		m_remaining_samples = samples - processed_samples + m_window_size - m_shift;
		memcpy(m_tmpbuffer, m_tmpbuffer + samples - m_remaining_samples, m_remaining_samples * sizeof(short));
		m_flushed = false;
	}
	else
	{
		memcpy(m_tmpbuffer + m_remaining_samples, data_in, samples * sizeof(short));

		samples += m_remaining_samples;
		window_count_no_delta = estimated_window_count(samples);
		window_count = window_count_no_delta - 2 * m_deltasize;
		if (window_count > 0)
		{
			segment_data(data_out, window_count_no_delta);
		}
		else
			window_count = 0;

		int processed_samples = window_count * m_shift + m_window_size - m_shift;
		m_remaining_samples = samples - processed_samples + m_window_size - m_shift;
		memcpy(m_tmpbuffer, m_tmpbuffer + samples - m_remaining_samples, m_remaining_samples * sizeof(short));
	}
	m_samples = samples;
}

void Segmenter::flush(float * data_out, int & window_count, int & window_count_no_delta)
{
	m_flushed = true;
	window_count_no_delta = estimated_window_count(m_remaining_samples);
	window_count = window_count_no_delta - m_deltasize;
	if (window_count <= 0)
		return;

	segment_data(data_out, window_count_no_delta);
}