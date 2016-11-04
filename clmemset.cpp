#include <cstring>
#include "clmemset.h"

cl_int clMemset(cl_command_queue command_queue, cl_mem buffer, int value, size_t offset, size_t size, cl_uint num_events_in_wait_list, const cl_event * event_wait_list, cl_event * event)
{
	char * buf = new char [size];
	memset(buf, value, size);
	cl_int ret = clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE, offset, size, buf, num_events_in_wait_list, event_wait_list, event);
	delete [] buf;
	return ret;
}