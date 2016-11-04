#ifndef _CLMEMSET_H_
#define _CLMEMSET_H_

#include <CL/cl.h>

cl_int clMemset(cl_command_queue command_queue, cl_mem buffer, int value, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event);

#endif