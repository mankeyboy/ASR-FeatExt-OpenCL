#pragma once

class DeltaCPU
{
private:
    int m_dim,
        m_window_limit,
        m_delta_size;
    float * m_output;
public:
    DeltaCPU() : m_dim(0), m_window_limit(0), m_output(nullptr) {}
    void init(int dim, int window_limit, int delta_size);
    void cleanup();

    void apply(const float * data, int window_count);
    float * get_output_buffer() { return m_output; }
};