#include "delta.h"

void Delta::init(int dim, int window_limit, int delta_size)
{
	m_dim = dim;
	m_window_limit = window_limit;
	m_delta_size = delta_size;
	m_output = new float[dim * window_limit];
}

void Delta::cleanup()
{
	delete[] m_output;
}

void Delta::apply(const float * data, int window_count)
{
	for (int i = 0; i < window_count; i++)
		for (int j = 0; j < m_dim; j++)
		{
			float num = 0,
				den = 0;
			for (int l = 1; l <= m_delta_size; l++)
			{
				num += l * (data[m_dim * (i + m_delta_size + l) + j] - data[m_dim * (i + m_delta_size - l) + j]);
				den += l * l;
			}
			m_output[m_dim * i + j] = num / (2 * den);
		}
}