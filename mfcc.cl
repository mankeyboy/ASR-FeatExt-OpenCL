__kernel void kernelTranspose(__global float2 * data_in, __global float * data_out, int width, int height, int opitch, float norm_factor)
{
	__constant int TRANSPOSE_TILE_SIZE = 16;
	//__local float tile[TRANSPOSE_TILE_SIZE][TRANSPOSE_TILE_SIZE + 1];
	__local float tile[16][16 + 1];
	int xIndex = get_global_id(0),
		yIndex = get_global_id(1),
		xIndexO = get_local_size(1) * get_group_id(1) + get_local_id(0),
		yIndexO = get_local_size(0) * get_group_id(0) + get_local_id(1),
		gridHeight = get_global_size(1);
	int rep = (height + gridHeight - 1) / gridHeight;
	for (int i = 0; i < rep; i++)
	{
		int shift = gridHeight * i;
		if (xIndex < width && yIndex + shift < height)
		{
			float2 v = data_in[width * (yIndex + shift) + xIndex];
			tile[get_local_id(1)][get_local_id(0)] = sqrt(v.x * v.x + v.y * v.y) * norm_factor;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		if (xIndexO + shift < height && yIndexO < width)
			data_out[opitch * yIndexO + xIndexO + shift] = tile[get_local_id(0)][get_local_id(1)];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void kernelFilter(__global float * data_in, __global float * data_out, __constant int * filter_beg, __constant float * filters, int pitch, int window_count, int window_size2, int num_banks, int num_banks2, float log_threshold)
{
	int frame_index = get_global_id(0),
		idx_shift = get_global_size(0);
	
	while (frame_index < window_count)
	{
		float sum[2] = {0,0};
		int curf = 0;
		int lastf = filter_beg[num_banks + 1];
		for (int i = filter_beg[0]; i <= lastf; i++)
		{
			float v = data_in[pitch * i + frame_index];

			if (i == filter_beg[curf + 1])
			{
				curf++;
				if (curf >= 2)
				{
					int sumidx = curf % 2;
					data_out[num_banks2 * frame_index + curf - 2] = log(max(sum[sumidx], log_threshold));
					sum[sumidx] = 0;
				}
			}
			sum[0] += filters[i] * v;
			sum[1] += filters[window_size2 + i] * v;
		}
		frame_index += idx_shift;
	}
}