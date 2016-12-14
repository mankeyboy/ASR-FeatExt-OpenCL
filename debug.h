#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <string>
#ifdef AFET_OPENCL
#include <CL/cl.h>
#endif

#ifndef EXTERN
#define EXTERN extern
#endif

#define AFET_DEBUG_SAVE_BUFFERS 0x01
#define AFET_DEBUG_FLUSH        0x02
#define AFET_DEBUG_NOFADVISE    0x04
#define AFET_DEBUG_NOINPUT      0x08
#define AFET_DEBUG_NOOUTPUT     0x10

EXTERN int debug_mode;

void export_c_buffer(void * data, size_t width, size_t height, size_t elemsize, std::string filename);
#ifdef AFET_CUDA
void export_cuda_buffer(void * d_data, size_t width, size_t height, size_t elemsize, std::string filename);
#endif
#ifdef AFET_OPENCL
void export_cl_buffer(cl_command_queue cmd_queue, cl_mem d_data, size_t width, size_t height, size_t elemsize, std::string filename);
#endif

#endif
