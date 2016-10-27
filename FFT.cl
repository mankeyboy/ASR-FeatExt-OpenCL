// FFT Kernel Implementation

__constant float COS_8 = 0.923879532511f; // cos(Pi/8)
__constant float SIN_8 = 0.382683432365f; // sin(Pi/8)

#define mul_p0q8 mul_p0q4
float2  mul_p1q8(float2 a) { return mul_1((float2)(COS_8,-SIN_8),a); }

#define mul_p2q8 mul_p1q4
float2  mul_p3q8(float2 a) { return mul_1((float2)(SIN_8,-COS_8),a); }

#define mul_p4q8 mul_p2q4
float2  mul_p5q8(float2 a) { return mul_1((float2)(-SIN_8,-COS_8),a); }

#define mul_p6q8 mul_p3q4
float2  mul_p7q8(float2 a) { return mul_1((float2)(-COS_8,-SIN_8),a); }

// Compute in-place DFT2 and twiddle
#define DFT2_TWIDDLE(a,b,t) { float2 tmp = t(a-b); a += b; b = tmp; }

// T = N/16 = number of threads.
// P is the length of input sub-sequences, 1,16,256,...,N/16.

__kernel void fft_radix16(__global const float2 * x,__global float2 * y,int p)
{
  int t = get_global_size(0); // number of threads
  int i = get_global_id(0); // current thread
  int k = i & (p-1); // index in input sequence, in 0..P-1
  // Inputs indices are I+{0,..,15}*T

  x += i;
  // Output indices are J+{0,..,15}*P, where
  // J is I with four 0 bits inserted at bit log2(P)
  y += ((i-k)<<4) + k;

  // Load
  float2 u[16];
  for (int m=0;m<16;m++) u[m] = x[m*t];

  // Twiddle, twiddling factors are exp(_I*PI*{0,..,15}*K/4P)
  float alpha = -M_PI*(float)k/(float)(8*p);
  for (int m=1;m<16;m++) u[m] = mul_1(exp_alpha_1(m*alpha),u[m]);

  // 8x in-place DFT2 and twiddle (1)
  DFT2_TWIDDLE(u[0],u[8],mul_p0q8);
  DFT2_TWIDDLE(u[1],u[9],mul_p1q8);
  DFT2_TWIDDLE(u[2],u[10],mul_p2q8);
  DFT2_TWIDDLE(u[3],u[11],mul_p3q8);
  DFT2_TWIDDLE(u[4],u[12],mul_p4q8);
  DFT2_TWIDDLE(u[5],u[13],mul_p5q8);
  DFT2_TWIDDLE(u[6],u[14],mul_p6q8);
  DFT2_TWIDDLE(u[7],u[15],mul_p7q8);

  // 8x in-place DFT2 and twiddle (2)
  DFT2_TWIDDLE(u[0],u[4],mul_p0q4);
  DFT2_TWIDDLE(u[1],u[5],mul_p1q4);
  DFT2_TWIDDLE(u[2],u[6],mul_p2q4);
  DFT2_TWIDDLE(u[3],u[7],mul_p3q4);
  DFT2_TWIDDLE(u[8],u[12],mul_p0q4);
  DFT2_TWIDDLE(u[9],u[13],mul_p1q4);
  DFT2_TWIDDLE(u[10],u[14],mul_p2q4);
  DFT2_TWIDDLE(u[11],u[15],mul_p3q4);

  // 8x in-place DFT2 and twiddle (3)
  DFT2_TWIDDLE(u[0],u[2],mul_p0q2);
  DFT2_TWIDDLE(u[1],u[3],mul_p1q2);
  DFT2_TWIDDLE(u[4],u[6],mul_p0q2);
  DFT2_TWIDDLE(u[5],u[7],mul_p1q2);
  DFT2_TWIDDLE(u[8],u[10],mul_p0q2);
  DFT2_TWIDDLE(u[9],u[11],mul_p1q2);
  DFT2_TWIDDLE(u[12],u[14],mul_p0q2);
  DFT2_TWIDDLE(u[13],u[15],mul_p1q2);

  // 8x DFT2 and store (reverse binary permutation)
  y[0]    = u[0]  + u[1];
  y[p]    = u[8]  + u[9];
  y[2*p]  = u[4]  + u[5];
  y[3*p]  = u[12] + u[13];
  y[4*p]  = u[2]  + u[3];
  y[5*p]  = u[10] + u[11];
  y[6*p]  = u[6]  + u[7];
  y[7*p]  = u[14] + u[15];
  y[8*p]  = u[0]  - u[1];
  y[9*p]  = u[8]  - u[9];
  y[10*p] = u[4]  - u[5];
  y[11*p] = u[12] - u[13];
  y[12*p] = u[2]  - u[3];
  y[13*p] = u[10] - u[11];
  y[14*p] = u[6]  - u[7];
  y[15*p] = u[14] - u[15];
}