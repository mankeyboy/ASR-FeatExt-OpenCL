__kernel void kernelSegmentWindow(__global short * tmp, __global float * data, __global float * window, int window_count, int window_size, int window_size2, int shift, __local float * win)
{
	int win_index = get_global_id(0),
		frame_index = get_local_size(1) * get_group_id(1),
		idx_shift = get_global_size(1);
	
	win[get_local_id(0)] = window[win_index];
	barrier(CLK_LOCAL_MEM_FENCE);

	while (frame_index < window_count)
	{
		if (win_index < window_size)
		{
			int indexIn = frame_index * shift + win_index;
			data[window_size2 * frame_index + win_index] = tmp[indexIn] * win[get_local_id(0)];
		}
		else
			data[window_size2 * frame_index + win_index] = 0;

		frame_index += idx_shift;
	}
}