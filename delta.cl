float safe_read(__global float * data, int frame, int col, int pitch, int window_count)
{
	return (frame < window_count) ? data[pitch * frame + col] : 0;
}

__kernel void kernelDelta(__global float * data_in, __global float * data_out, int cols, int window_count, int l1, __local float * tile)
{
	int frame_index = get_global_id(1),
		col = get_global_id(0),
		idx_shift = get_global_size(1);

	while (true)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		tile[DELTA_TILE_SIZE * get_local_id(1) + get_local_id(0)] = safe_read(data_in, frame_index, col, cols, window_count + 2 * l1);
		if (get_local_id(1) < 2 * l1)
			tile[DELTA_TILE_SIZE * (DELTA_TILE_SIZE + get_local_id(1)) + get_local_id(0)] = safe_read(data_in, frame_index + DELTA_TILE_SIZE, col, cols, window_count + 2 * l1);
		barrier(CLK_LOCAL_MEM_FENCE);

		if (frame_index >= window_count)
			return;

		float num = 0,
			den = 0;
		for (int l = 1; l <= l1; l++)
		{
			num += l * (tile[DELTA_TILE_SIZE * (get_local_id(1) + l1 + l) + get_local_id(0)] - tile[DELTA_TILE_SIZE * (get_local_id(1) + l1 - l) + get_local_id(0)]);
			den += l * l;
		}
		data_out[cols * frame_index + col] = num / (2 * den);

		frame_index += idx_shift;
	}
}