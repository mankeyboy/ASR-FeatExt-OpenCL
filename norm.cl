__kernel void kernelSum(__global float * data, __global float * mean,
	int cols, int window_count, int data_offset
#if defined(WANTVAR)
	,__global float * var
#endif
#if defined(WANTMINMAX)
	,__global float2 * minmax
#endif
)
{
	int rep = ceil((float)window_count / get_global_size(1)),
		group_id_y = get_group_id(1),
		frame_index = group_id_y * rep;
	int xIndex = get_global_id(0);
	if (xIndex >= cols)
        return;

	float sum = 0,
		sum2 = 0;
	float2 minmaxv = (float2)(FLT_MAX, -FLT_MAX);
	for (int r = 0; r < rep && frame_index < window_count; r++, frame_index++)
	{
		float v = data[data_offset + cols * frame_index + xIndex];
		sum += v;
#if defined(WANTVAR)
		sum2 += v * v;
#endif
#if defined(WANTMINMAX)
		minmaxv.x = min(minmaxv.x, v);
		minmaxv.y = max(minmaxv.y, v);
#endif
	}
	mean[cols * group_id_y + xIndex] = sum;
#if defined(WANTVAR)
	var[cols * group_id_y + xIndex] = sum2;
#endif
#if defined(WANTMINMAX)
	minmax[cols * group_id_y + xIndex] = minmaxv;
#endif
}

__kernel void kernelFinalizeSum(__global float * mean,
	int cols, int rows, int window_count
#if defined(WANTVAR)
	,__global float * var
#endif
#if defined(WANTMINMAX)
	,__global float2 * minmax
#endif
)
{
	int xIndex = get_global_id(0);
	if (xIndex >= cols)
        return;

	float sum = 0,
		sum2 = 0;
	float2 minmaxv = (float2)(FLT_MAX, -FLT_MAX);
	for (int i = 0; i < rows; i++)
	{
		sum += mean[cols * i + xIndex];
#if defined(WANTVAR)
		sum2 += var[cols * i + xIndex];
#endif
#if defined(WANTMINMAX)
		minmaxv.x = min(minmaxv.x, minmax[cols * i + xIndex].x);
		minmaxv.y = max(minmaxv.y, minmax[cols * i + xIndex].y);
#endif
	}
	mean[xIndex] = sum / window_count;
#if defined(WANTVAR)
	float v = rsqrt((sum2 - sum * (sum / window_count)) / (window_count - 1));
	var[xIndex] = v;
#endif
#if defined(WANTMINMAX)
	minmax[xIndex] = minmaxv;
#endif
}

__kernel void kernelNormalize(__global float * data, __global float * mean,
	int cols, int window_count, int data_offset,
	__local float * smean
#if defined(WANTVAR)
	,__global float * var
	,__local float * svar
#endif
#if defined(WANTMINMAX)
	,__global float2 * minmax
	,__local float * sminmax
#endif
)
{
	int xIndex = get_global_id(0),
		frame_index = get_global_id(1),
		idx_shift = get_global_size(1);
	if (xIndex >= cols)
        return;

	if (get_local_id(1) == 0)
	{
		float m = mean[xIndex];
		smean[get_local_id(0)] = m;
#if defined(WANTVAR)
		svar[get_local_id(0)] = var[xIndex];
#endif
#if defined(WANTMINMAX)
		sminmax[get_local_id(0)] = 1.f / max(fabs(minmax[xIndex].x - m), fabs(minmax[xIndex].y - m));
#endif
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	while (frame_index < window_count)
	{
		int idx = data_offset + cols * frame_index + xIndex;
		float v = data[idx];
#if defined(WANTVAR)
		data[idx] = (v - smean[get_local_id(0)]) * svar[get_local_id(0)];
#elif defined(WANTMINMAX)
		data[idx] = (v - smean[get_local_id(0)]) * sminmax[get_local_id(0)];
#else
		data[idx] = v - smean[get_local_id(0)];
#endif

		frame_index += idx_shift;
	}
}