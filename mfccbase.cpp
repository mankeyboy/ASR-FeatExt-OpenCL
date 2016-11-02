#include "mfccbase.h"

MfccBase::MfccBase(int input_buffer_size,
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
	: ParamBase(input_buffer_size, window_size, shift, norm, dyn),
	m_num_banks(num_banks),
	m_sample_rate(sample_rate),
	m_low_freq(low_freq),
	m_high_freq(high_freq),
	m_ceps_len(ceps_len),
	m_want_c0(want_c0),
	m_lift_coef(lift_coef),
	m_delta_l1(dyn != DYN_NONE ? delta_l1 : 0),
	m_delta_l2(dyn == DYN_ACC ? delta_l2 : 0),
	m_dct_len(want_c0 ? (m_ceps_len + 1) : m_ceps_len),
	m_norm_after_dyn(norm_after_dyn)
{
}

int MfccBase::get_output_data_width() const
{
	int cols = m_ceps_len > 0 ? m_dct_len : m_num_banks,
		pitch = cols;
	switch (m_dyn)
	{
	case DYN_DELTA: pitch *= 2; break;
	case DYN_ACC:   pitch *= 3; break;
	}
	return pitch;
}
