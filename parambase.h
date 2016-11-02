#ifndef _PARAMBASE_H_
#define _PARAMBASE_H_

#include "normalizer.h"

class ParamBase
{
public:
	enum dyn_t { DYN_NONE, DYN_DELTA, DYN_ACC };
protected:
	int m_input_buffer_size,
		m_input_window_limit,
		m_window_size,
		m_shift;
	float m_alpha;
	Normalizer::norm_t m_norm;
	dyn_t m_dyn;
	bool m_last_block;
public:
	ParamBase(int input_buffer_size, int window_size, int shift, Normalizer::norm_t norm, dyn_t dyn);
	virtual ~ParamBase() {}

	int get_input_buffer_size() const { return m_input_buffer_size; }
	int estimated_window_count(int samples) const;
	void set_alpha(float alpha) { m_alpha = alpha; }

	virtual void set_window(const float * window) = 0;
	virtual int set_input(const short * data, int samples) = 0;
	virtual int flush() = 0;
	virtual void apply() = 0;
	virtual int get_output_data_width() const = 0;
	virtual void get_output_data(float * data_out, int window_count) = 0;
};

#endif