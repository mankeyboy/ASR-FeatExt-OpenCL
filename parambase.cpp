#include <cmath>
#include "parambase.h"

ParamBase::ParamBase(int input_buffer_size, int window_size, int shift, Normalizer::norm_t norm, dyn_t dyn)
	: m_window_size(window_size),
	m_shift(shift),
	m_alpha(1),
	m_norm(norm),
	m_dyn(dyn),
	m_last_block(false)
{
	m_input_window_limit = estimated_window_count(input_buffer_size);
	m_input_buffer_size = m_input_window_limit * m_shift + m_window_size - m_shift;
}

int ParamBase::estimated_window_count(int samples) const
{
	return floor(float(samples - (m_window_size - m_shift)) / m_shift);
}
