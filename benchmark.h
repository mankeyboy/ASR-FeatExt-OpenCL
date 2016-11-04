#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#ifndef EXTERN
#define EXTERN extern
#endif

EXTERN float benchmark_win,
       benchmark_fft,
       benchmark_trn,
       benchmark_filt,
       benchmark_dct,
       benchmark_traps,
       benchmark_lpc,
       benchmark_cep,
       benchmark_dlt,benchmark_nrm;
EXTERN float benchmark_cpu_win,
       benchmark_cpu_fft,
       benchmark_cpu_trn,
       benchmark_cpu_filt,
       benchmark_cpu_dct,
       benchmark_cpu_traps,
       benchmark_cpu_lpc,
       benchmark_cpu_cep,
       benchmark_cpu_dlt,
       benchmark_cpu_nrm;
EXTERN float benchmark_mem;
EXTERN bool benchmark;

#endif
