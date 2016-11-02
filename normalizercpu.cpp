#include <cmath>
#include <algorithm>
#include <iostream>
#include "normalizercpu.h"

void NormalizerCPU::init(Normalizer::norm_t norm_type, int dim)
{
	m_norm_type = norm_type;
	m_dim = dim;
	m_mean = m_norm_type != Normalizer::NORM_NONE ? new float[m_dim] : nullptr;
	m_var = m_norm_type == Normalizer::NORM_CVN ? new float[m_dim] : nullptr;
	m_minmax = m_norm_type == Normalizer::NORM_MINMAX ? new float[m_dim] : nullptr;
}

void NormalizerCPU::cleanup()
{
	delete[] m_mean;
	delete[] m_var;
	delete[] m_minmax;
}

void NormalizerCPU::normalize(float * data, int window_count, bool use_last_stats)
{
	if (!use_last_stats)
	{
		switch (m_norm_type)
		{
		case Normalizer::NORM_CMN:
			for (int i = 0; i < m_dim; i++)
			{
				double sum = 0;
				for (int j = 0; j < window_count; j++)
					sum += data[m_dim * j + i];
				m_mean[i] = sum / window_count;
			}
			break;
		case Normalizer::NORM_CVN:
			for (int i = 0; i < m_dim; i++)
			{
				double sum = 0,
					sum2 = 0;
				for (int j = 0; j < window_count; j++)
				{
					float v = data[m_dim * j + i];
					sum += v;
					sum2 += v * v;
				}
				m_mean[i] = sum / window_count;
				m_var[i] = sqrt((window_count - 1) / (sum2 - sum * (sum / window_count)));
			}
			break;
		case Normalizer::NORM_MINMAX:
			for (int i = 0; i < m_dim; i++)
			{
				double sum = 0;
				float minv = FLT_MAX,
					maxv = -FLT_MAX;
				for (int j = 0; j < window_count; j++)
				{
					float v = data[m_dim * j + i];
					sum += v;
					minv = std::min(minv, v);
					maxv = std::max(maxv, v);
				}
				m_mean[i] = sum / window_count;
				m_minmax[i] = 1.f / std::max(abs(minv - m_mean[i]), abs(maxv - m_mean[i]));
			}
			break;
		}
	}
	switch (m_norm_type)
	{
	case Normalizer::NORM_CMN:
		for (int i = 0; i < window_count; i++)
			for (int j = 0; j < m_dim; j++)
				data[m_dim * i + j] -= m_mean[j];
		break;
	case Normalizer::NORM_CVN:
		for (int i = 0; i < window_count; i++)
			for (int j = 0; j < m_dim; j++)
				data[m_dim * i + j] = (data[m_dim * i + j] - m_mean[j]) * m_var[j];
		break;
	case Normalizer::NORM_MINMAX:
		for (int i = 0; i < window_count; i++)
			for (int j = 0; j < m_dim; j++)
				data[m_dim * i + j] = (data[m_dim * i + j] - m_mean[j]) * m_minmax[j];
		break;
	}
}