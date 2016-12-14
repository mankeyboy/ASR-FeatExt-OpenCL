#include <iostream>
#include <fstream>
#include <cstdint>
#ifdef AFET_CUDA
#include <cuda_runtime.h>
#endif
#define EXTERN
#include "debug.h"

void export_c_buffer(void * data, size_t width, size_t height, size_t elemsize, std::string filename)
{
	std::cout << "[Debug] Exporting C buffer of size " << width << " x " << height << " x " << elemsize << " B to file '" << filename.c_str() << "'\n";
	size_t datasize = width * height * elemsize;

    uint32_t w = width,
        h = height,
        e = elemsize;
	std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
    fout.write((const char *)&w, sizeof(w));
    fout.write((const char *)&h, sizeof(h));
    fout.write((const char *)&e, sizeof(e));
	fout.write((const char *)data, datasize);
	fout.close();
}

#ifdef AFET_CUDA
void export_cuda_buffer(void * d_data, size_t width, size_t height, size_t elemsize, std::string filename)
{
	std::cout << "[Debug] Exporting CUDA buffer of size " << width << " x " << height << " x " << elemsize << " B to file '" << filename.c_str() << "'\n";

	size_t datasize = width * height * elemsize;
	char * data = new char [datasize];
	cudaMemcpy(data, d_data, datasize, cudaMemcpyDeviceToHost);

    uint32_t w = width,
        h = height,
        e = elemsize;
	std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
    fout.write((const char *)&w, sizeof(w));
    fout.write((const char *)&h, sizeof(h));
    fout.write((const char *)&e, sizeof(e));
	fout.write((const char *)data, datasize);
	fout.close();

	delete [] data;
}
#endif

#ifdef AFET_OPENCL
void export_cl_buffer(cl_command_queue cmd_queue, cl_mem d_data, size_t width, size_t height, size_t elemsize, std::string filename)
{
	std::cout << "[Debug] Exporting OpenCL buffer of size " << width << " x " << height << " x " << elemsize << " B to file '" << filename.c_str() << "'\n";

	size_t datasize = width * height * elemsize;
	char * data = new char [datasize];
	clEnqueueReadBuffer(cmd_queue, d_data, CL_TRUE, 0, datasize, data, 0, NULL, NULL);

    uint32_t w = width,
        h = height,
        e = elemsize;
	std::ofstream fout(filename.c_str(), std::ios::out | std::ios::binary);
    fout.write((const char *)&w, sizeof(w));
    fout.write((const char *)&h, sizeof(h));
    fout.write((const char *)&e, sizeof(e));
	fout.write((const char *)data, datasize);
	fout.close();

	delete [] data;
}
#endif
