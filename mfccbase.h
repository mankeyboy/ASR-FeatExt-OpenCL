#ifndef _MFCCBASE_H_
#define _MFCCBASE_H_

#include "parambase.h"

class MfccBase : public ParamBase
{
protected:
	int m_num_banks,
		m_ceps_len,
		m_dct_len,
		m_delta_l1,
		m_delta_l2;
	float m_sample_rate,
		m_low_freq,
		m_high_freq,
		m_lift_coef;
	bool m_want_c0,
		m_norm_after_dyn;
public:
	MfccBase(int input_buffer_size,
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
	virtual ~MfccBase() {}

	int get_output_data_width() const;

	/*int get_num_banks() const {return m_num_banks;}
	int get_ceps_len() const {return m_ceps_len;}
	int get_dct_len() const {return m_dct_len;}
	int get_delta_l1() const {return m_delta_l1;}
	int get_delta_l2() const {return m_delta_l2;}
	float get_sample_rate() const {return m_sample_rate;}
	float get_low_freq() const {return m_low_freq;}
	float get_high_freq() const {return m_high_freq;}
	float get_alpha() const {return m_alpha;}
	float get_lift_coef() const {return m_lift_coef;}
	bool get_want_c0() const {return m_want_c0;}
	bool get_norm_after_dyn() const {return m_norm_after_dyn;}*/
};

#endif
