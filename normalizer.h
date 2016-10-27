#pragma once

namespace Normalizer
{
	enum norm_t { NORM_NONE, NORM_CMN, NORM_CVN, NORM_MINMAX };
}

class NormalizerC
{
private:
	float * m_mean,
		*m_var,
		*m_minmax;
	Normalizer::norm_t m_norm_type;
	int m_dim;
public:
	NormalizerC() : m_mean(nullptr), m_var(nullptr), m_minmax(nullptr), m_norm_type(Normalizer::NORM_NONE), m_dim(0) {}
	void init(Normalizer::norm_t norm_type, int dim);
	void cleanup();

	void normalize(float * data, int window_count, bool use_last_stats = false);
};