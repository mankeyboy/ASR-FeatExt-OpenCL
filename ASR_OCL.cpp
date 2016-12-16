#define _USE_MATH_DEFINES
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <tchar.h>
#include <memory.h>
#include <fstream>
#include <ctime>
#include <queue>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <string>

#include "CL\cl.h"
#include "utils.h"
#include "segmentercpu.h"
#include "mfcccpu.h"
#include "mfccopencl.h"

//for perf. counters
#include <Windows.h>
//for wav file read
#include <sndfile.h>

#include "debug.h"
#include "benchmark.h"
#include "normalizer.h"

#include "mfcccpu.h"
#include "mfccopencl.h"

typedef int cl_device_id;

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace std;

bool verbose = false, use_apple_oclfft = false, use_trans_filt_combined = false;

boost::mutex global_stream_lock, file_list_lock;

enum Method_t
{
	Method_MFCC
};

enum Platform_t
{
	Platform_Auto,
	Platform_CPU,
	Platform_OpenCL
};

struct SDevice
{
	int platform_id,
		device_id,
		cl_device_type;
	SDevice() : platform_id(0), device_id(0), cl_device_type(CL_DEVICE_TYPE_ALL) {}
};

struct SVTLNAlpha
{
	float min, max, step;
	SVTLNAlpha() : min(1), max(1), step(1) {}
	SVTLNAlpha(float alpha) : min(alpha), max(alpha), step(1) {}
};

struct SConfig
{
	Method_t method;
	Platform_t platform;
	SVTLNAlpha alpha;
	float window_size,
		shift;
	int num_banks,
		ceps_len,
		norm_type,
		dyn_type,
		delta_l1,
		delta_l2,
		traps_len,
		traps_dct_len,
		model_order;
	float sample_rate,
		low_freq,
		high_freq,
		lift_coef;
	bool want_c0,
		norm_after_dyn,
		text_output;
	//OpenCL config
	cl_device_id opencl_device;
};

struct SProcessedFile
{
	string input, output;
	SProcessedFile() {}
	SProcessedFile(const string & input, const string & output) : input(input), output(output) {}
};

void process_files_worker(std::list<SProcessedFile> * files, SConfig & cfg, int sample_limit, int thread_id, int num_threads)
{
	ParamBase * param = NULL;
	void * data = NULL;
	try
	{
		int window_size = cfg.sample_rate * cfg.window_size * 1e-3,
			shift = cfg.sample_rate * cfg.shift * 1e-3;

		Normalizer::norm_t norm_type = Normalizer::NORM_NONE;
		ParamBase::dyn_t dyn_type = ParamBase::DYN_NONE;
		switch (cfg.norm_type)
		{
		case 1: norm_type = Normalizer::NORM_CMN; break;
		case 2: norm_type = Normalizer::NORM_CVN; break;
		case 3: norm_type = Normalizer::NORM_MINMAX; break;
		}
		switch (cfg.dyn_type)
		{
		case 1: dyn_type = ParamBase::DYN_DELTA; break;
		case 2: dyn_type = ParamBase::DYN_ACC; break;
		}

		int htk_param_kind;
		switch (cfg.method)
		{
		case Method_MFCC: htk_param_kind = 6; break;
		case Method_PLP: htk_param_kind = 11; break;
		default: htk_param_kind = 9;
		}
		if (dyn_type == ParamBase::DYN_DELTA)
			htk_param_kind |= 000400;
		else if (dyn_type == ParamBase::DYN_ACC)
			htk_param_kind |= 000400 | 001000;
		if (cfg.method == Method_PLP || cfg.want_c0)
			htk_param_kind |= 020000;
		if (norm_type != Normalizer::NORM_NONE)
			htk_param_kind |= 004000;

		switch (cfg.platform)
		{
		case Platform_CPU:
			switch (cfg.method)
			{
			case Method_MFCC:
				param = new MfccCpu(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.ceps_len, cfg.want_c0, cfg.lift_coef, norm_type, dyn_type, cfg.delta_l1, cfg.delta_l2, cfg.norm_after_dyn);
				break;
			case Method_TRAPS:
				param = new TrapsCpu(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.traps_len, cfg.traps_dct_len, cfg.want_c0, norm_type);
				break;
			case Method_PLP:
				param = new PlpCpu(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.model_order, norm_type, dyn_type, cfg.delta_l1, cfg.delta_l2, cfg.norm_after_dyn);
				break;
			}
			break;
		case Platform_CUDA:
			break;
		case Platform_OpenCL:
			switch (cfg.method)
			{
			case Method_MFCC:
				param = new MfccOpenCL(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.ceps_len, cfg.want_c0, cfg.lift_coef, norm_type, dyn_type, cfg.delta_l1, cfg.delta_l2, cfg.norm_after_dyn, cfg.opencl_device);
				break;
			case Method_TRAPS:
				param = new TrapsOpenCL(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.traps_len, cfg.traps_dct_len, cfg.want_c0, norm_type, cfg.opencl_device);
				break;
			case Method_PLP:
				param = new PlpOpenCL(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.model_order, norm_type, dyn_type, cfg.delta_l1, cfg.delta_l2, cfg.norm_after_dyn, cfg.opencl_device);
				break;
			}
			break;
		default:
			return;
		}
		if (param == NULL)
			throw std::runtime_error("Unsupported platform");
		float * window = new float[window_size];
		for (int i = 0; i < window_size; i++)
			window[i] = (0.56f - 0.46f * cos((2.0f * M_PI * i) / window_size)) / 32768.f;
		param->set_window(window);
		delete[] window;
		sw.stop();
		if (benchmark)
		{
			global_stream_lock.lock();
			std::cout << "[Benchmark] Parameterizer init time: " << sw.getTime() << " s\n";
			global_stream_lock.unlock();
		}
		sw.reset();

		int samples_in_limit = param->get_input_buffer_size();
		int windows_out_limit = param->estimated_window_count(samples_in_limit);
		int data_out_width = param->get_output_data_width();

		int buffer_size = std::max(samples_in_limit * sizeof(short), data_out_width * windows_out_limit * sizeof(float));
		data = new char[buffer_size];

		while (true)
		{
			file_list_lock.lock();
			if (files->empty())
			{
				file_list_lock.unlock();
				break;
			}
			SProcessedFile file = files->front();
			files->pop_front();
			file_list_lock.unlock();

			const fs::path & file_in = file.input,
				&file_out = file.output;
			std::string input_file_name = file_in.generic_string();
			global_stream_lock.lock();
			std::cout << "Processing \"" << input_file_name << "\"\n";
			global_stream_lock.unlock();
			SF_INFO info;
			info.format = 0;
			SNDFILE * f = NULL;
#ifdef _WIN32
			f = sf_open(input_file_name.c_str(), SFM_READ, &info);
#else
			int fd = open(input_file_name.c_str(), O_RDONLY);
			if (fd >= 0)
			{
				if (!(debug_mode & AFET_DEBUG_NOFADVISE))
				{
					int ret = posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
					ret |= posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE);
					if (ret)
					{
						global_stream_lock.lock();
						std::cout << "Warning: posix_fadvise failed";
						global_stream_lock.unlock();
					}
				}
				f = sf_open_fd(fd, SFM_READ, &info, 1);
			}
#endif
			if (f == NULL)
			{
				global_stream_lock.lock();
				std::cerr << "Can't open \"" << input_file_name << "\"\n";
				global_stream_lock.unlock();
				continue;
			}
			sf_command(f, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
			if (info.frames < 1)
			{
				sf_close(f);
				global_stream_lock.lock();
				std::cerr << "Error while loading \"" << input_file_name << "\"\n";
				global_stream_lock.unlock();
				continue;
			}
			if (info.samplerate != cfg.sample_rate)
			{
				sf_close(f);
				global_stream_lock.lock();
				std::cerr << "File \"" << input_file_name << "\" has incorrect sample rate (" << info.samplerate << "), aborting program.\n";
				global_stream_lock.unlock();
				break;
			}

			std::vector<FILE *> vecfout;
			std::vector<int> vecframes;
			int fidx = 0;
			for (float alpha = cfg.alpha.min; alpha <= cfg.alpha.max; alpha = cfg.alpha.min + fidx * cfg.alpha.step)
			{
				fs::path file_out_name;
				if (cfg.alpha.max - cfg.alpha.min < cfg.alpha.step)
					file_out_name = file_out;  //only 1 alpha
				else
					file_out_name = file_out.parent_path() / fs::path(file_out.stem().string() + "-" + std::to_string(alpha) + file_out.extension().string());
				if (verbose)
				{
					global_stream_lock.lock();
					std::cout << "Creating output file: " << file_out_name.string() << std::endl;
					global_stream_lock.unlock();
				}
				FILE * fout = NULL;
				if (!(debug_mode & AFET_DEBUG_NOOUTPUT))
				{
					fout = fopen(file_out_name.string().c_str(), cfg.text_output ? "w" : "wb");
					if (fout == NULL)
						throw std::runtime_error("Can't create output file: " + file_out_name.string());
#ifndef _WIN32
					if (!(debug_mode & AFET_DEBUG_NOFADVISE))
					{
						int fd = fileno(fout);
						if (fd > 0)
						{
							int ret = posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
							ret |= posix_fadvise(fd, 0, 0, POSIX_FADV_NOREUSE);
							if (ret)
							{
								global_stream_lock.lock();
								std::cout << "Warning: posix_fadvise failed";
								global_stream_lock.unlock();
							}
						}
					}
#endif
					if (!cfg.text_output)
						write_htk_header(fout, 0, 0, 0, 0); //Dummy header for correct data offset
				}
				vecfout.push_back(fout);
				fidx++;
			}

			int samples = info.frames;
			int windows_total_in = 0,
				windows_total_out = 0,
				windows_out,
				block = 0;
			long double frame_time_step = cfg.shift / cfg.sample_rate,
				frame_time = 0.5f * cfg.window_size / cfg.sample_rate;
			while (samples > 0)
			{
				int samples_in = std::min<size_t>(samples, samples_in_limit);
				if (benchmark)
					sw.start();
				if (!(debug_mode & AFET_DEBUG_NOINPUT))
					sf_readf_short(f, (short *)data, samples_in);
				//sf_readf_float(f, data, samples_in);
				if (benchmark)
					sw.stop();

				windows_out = param->set_input((const short *)data, samples_in);
				windows_total_in += windows_out;
				for (fidx = 0; fidx < vecfout.size(); fidx++)
				{
					windows_total_out += windows_out;
					float alpha = cfg.alpha.min + fidx * cfg.alpha.step;

					param->set_alpha(alpha);
					param->apply();
					param->get_output_data((float *)data, windows_out);
					if (verbose)
					{
						global_stream_lock.lock();
						std::cout << "\tBlock " << block++ << ": " << samples_in << " samples, " << windows_out << " frames, alpha = " << alpha << std::endl;
						global_stream_lock.unlock();
					}

					if (!(debug_mode & AFET_DEBUG_NOOUTPUT))
					{
						FILE * fout = vecfout[fidx];
						if (benchmark)
							sw.start();
						if (cfg.text_output)
						{
							for (int f = 0; f < windows_out; f++)
							{
								fprintf(fout, "| %f |", frame_time + (windows_total_in - windows_out + f) * frame_time_step);
								for (int i = 0; i < data_out_width; i++)
									fprintf(fout, " %f |", ((float *)data)[data_out_width * f + i]);
								fprintf(fout, "\n");
							}
						}
						else
							write_htk_d(fout, (float *)data, data_out_width * windows_out);
						if (debug_mode & AFET_DEBUG_FLUSH)
							fflush(fout);
						if (benchmark)
							sw.stop();
					}
				}
				samples -= samples_in;
			}
			windows_out = param->flush();
			if (windows_out > 0)
			{
				windows_total_in += windows_out;
				for (fidx = 0; fidx < vecfout.size(); fidx++)
				{
					windows_total_out += windows_out;
					float alpha = cfg.alpha.min + fidx * cfg.alpha.step;
					param->set_alpha(alpha);
					param->apply();
					param->get_output_data((float *)data, windows_out);
					if (verbose)
					{
						global_stream_lock.lock();
						std::cout << "\tFlushing: " << windows_out << " frames, alpha = " << alpha << std::endl;
						global_stream_lock.unlock();
					}

					if (!(debug_mode & AFET_DEBUG_NOOUTPUT))
					{
						FILE * fout = vecfout[fidx];
						if (benchmark)
							sw.start();
						if (cfg.text_output)
						{
							for (int f = 0; f < windows_out; f++)
							{
								fprintf(fout, "| %f |", frame_time + (windows_total_in - windows_out + f) * frame_time_step);
								for (int i = 0; i < data_out_width; i++)
									fprintf(fout, " %f |", ((float *)data)[data_out_width * f + i]);
								fprintf(fout, "\n");
							}
						}
						else
							write_htk_d(fout, (float *)data, data_out_width * windows_out);
						if (debug_mode & AFET_DEBUG_FLUSH)
							fflush(fout);
						if (benchmark)
							sw.stop();
					}
				}
			}
			else if (verbose)
			{
				global_stream_lock.lock();
				std::cout << "\tNothing to flush\n";
				global_stream_lock.unlock();
			}
			if (verbose)
			{
				global_stream_lock.lock();
				std::cout << "\tTotal frames processed per VTLN alpha: " << windows_total_in << "\n\tTotal frames processed: " << windows_total_out << std::endl;
				global_stream_lock.unlock();
			}

			sf_close(f);
			sw.start();
			for (float alpha = cfg.alpha.min, fidx = 0; alpha <= cfg.alpha.max; alpha += cfg.alpha.step, fidx++)
			{
				FILE * fout = vecfout[fidx];
				if (!(debug_mode & AFET_DEBUG_NOOUTPUT) && !cfg.text_output)
				{
					fseek(fout, 0, SEEK_SET);
					write_htk_header(fout, data_out_width, windows_total_in, cfg.shift * 10000, htk_param_kind);
				}
				fclose(fout);
			}
			sw.stop();
		}
		if (benchmark)
		{
			global_stream_lock.lock();
			std::cout << "[Benchmark] File IO time:   " << sw.getTime() << " s\n";
			global_stream_lock.unlock();
		}

		delete[](char *)data;
		delete param;
	}
	catch (const std::exception & e)
	{
		delete[](char *)data;
		delete param;
		std::cerr << "Exception caught in thread " << thread_id << ": " << e.what() << "\nTerminating the thread " << thread_id << std::endl;
	}
	catch (...)
	{
		delete[](char *)data;
		delete param;
		std::cerr << "Unknown exception caught in thread " << thread_id << "\nTerminating the thread " << thread_id << std::endl;
	}
}

void process_files_mt(const std::vector<fs::path> & input, const std::vector<fs::path> & output, SConfig & cfg, int sample_limit, int num_threads)
{
	if (cfg.sample_rate <= 0)
	{
		if (verbose)
			std::cout << "Sample rate not set, opening the first file: " << input.front().string() << "\n";
		SF_INFO info;
		info.format = 0;
		SNDFILE * f = sf_open(input.front().string().c_str(), SFM_READ, &info);
		sf_close(f);
		if (f == NULL)
		{
			std::cerr << "Error while opening the file, processing failed\n";
			return;
		}
		cfg.sample_rate = info.samplerate;
		if (verbose)
			std::cout << "Sample rate set to " << cfg.sample_rate << " Hz\n";
	}
	if (cfg.high_freq <= 0)
		cfg.high_freq = 0.5 * cfg.sample_rate;
	std::list<SProcessedFile> files;
	for (int fidx = 0; fidx < input.size(); fidx++)
		files.push_back(SProcessedFile(input[fidx], output[fidx]));


	boost::thread_group threads;
	for (int i = 0; i < num_threads; i++)
		threads.add_thread(new boost::thread(process_files_worker, &files, cfg, sample_limit, i, num_threads));
	threads.join_all();

}

void printOpenCLDevices(std::ostream & os = std::cout)
{
	const size_t BUFFSIZE = 128;

	cl_platform_id * platform_ids = NULL;
	cl_device_id * device_ids = NULL;
	cl_uint nplatforms, ndevices;
	cl_int err;

	err = clGetPlatformIDs(0, NULL, &nplatforms);
	if (err != CL_SUCCESS)
		goto printOpenCLDevices_error;
	platform_ids = (cl_platform_id *)malloc(nplatforms * sizeof(cl_platform_id));
	err = clGetPlatformIDs(nplatforms, platform_ids, NULL);
	if (err != CL_SUCCESS)
		goto printOpenCLDevices_error;
	os << "Available OpenCL platforms and devices:\n";
	for (int i = 0; i < nplatforms; i++)
	{
		char vendor[BUFFSIZE], name[BUFFSIZE], version[BUFFSIZE];
		err |= clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR, BUFFSIZE, vendor, NULL);
		err |= clGetPlatformInfo(platform_ids[i], CL_PLATFORM_NAME, BUFFSIZE, name, NULL);
		err |= clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VERSION, BUFFSIZE, version, NULL);
		if (err != CL_SUCCESS)
			goto printOpenCLDevices_error;
		os << "  Platform " << i << ": " << vendor << ", " << name << ", " << version << std::endl;

		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, 0, NULL, &ndevices);
		if (err != CL_SUCCESS)
			goto printOpenCLDevices_error;
		device_ids = (cl_device_id *)malloc(ndevices * sizeof(cl_device_id));
		err = clGetDeviceIDs(platform_ids[i], CL_DEVICE_TYPE_ALL, ndevices, device_ids, NULL);
		if (err != CL_SUCCESS)
			goto printOpenCLDevices_error;
		for (int j = 0; j < ndevices; j++)
		{
			err |= clGetDeviceInfo(device_ids[j], CL_DEVICE_NAME, BUFFSIZE, name, NULL);
			err |= clGetDeviceInfo(device_ids[j], CL_DEVICE_VERSION, BUFFSIZE, version, NULL);
			os << "    Device " << j << ": " << name << "\t [" << version << "]\n";
		}
		free(device_ids);
		device_ids = NULL;
	}
	free(platform_ids);
	platform_ids = NULL;

printOpenCLDevices_error:
	if (platform_ids)
		free(platform_ids);
	if (device_ids)
		free(device_ids);
}

void help(const po::options_description & desc, std::ostream & os = std::cout)
{
	os << "Usage:\n  afet.exe [options] ([--wav-file] | --scp) file\n" << desc << std::endl;
	os << "Afet was compiled with support for: CPU"
#ifdef AFET_CUDA
		", CUDA"
#endif
#ifdef AFET_OPENCL
		", OpenCL"
#endif
		<< "\n\n";
#ifdef AFET_CUDA
	printCudaDevices(os);
	os << std::endl;
#endif
#ifdef AFET_OPENCL
	printOpenCLDevices(os);
#endif
}

std::istream & operator >> (std::istream & is, Method_t & method)
{
	std::string token;
	is >> token;
	if (boost::iequals(token, "mfcc"))
		method = Method_MFCC;
	else if (boost::iequals(token, "traps"))
		method = Method_TRAPS;
	else if (boost::iequals(token, "plp"))
		method = Method_PLP;
	return is;
}

std::istream & operator >> (std::istream & is, Platform_t & platform)
{
	std::string token;
	is >> token;
	if (boost::iequals(token, "auto"))
		platform = Platform_Auto;
	else if (boost::iequals(token, "cpu"))
		platform = Platform_CPU;
	else if (boost::iequals(token, "cuda"))
		platform = Platform_CUDA;
	else if (boost::iequals(token, "opencl"))
		platform = Platform_OpenCL;
	else
		throw std::runtime_error("Unknown platform");
	return is;
}

int str2clDeviceType(const std::string & t)
{
#ifdef AFET_OPENCL
	if (boost::iequals(t, "default"))
		return CL_DEVICE_TYPE_DEFAULT;
	else if (boost::iequals(t, "cpu"))
		return CL_DEVICE_TYPE_CPU;
	else if (boost::iequals(t, "gpu"))
		return CL_DEVICE_TYPE_GPU;
	else if (boost::iequals(t, "accelerator"))
		return CL_DEVICE_TYPE_ACCELERATOR;
#ifdef CL_DEVICE_TYPE_CUSTOM
	else if (boost::iequals(t, "custom"))
		return CL_DEVICE_TYPE_CUSTOM;
#endif
	else if (boost::iequals(t, "all"))
		return CL_DEVICE_TYPE_ALL;
	else
		throw std::runtime_error("Unknown device type");
#else
	return 0;
#endif
}

std::istream & operator >> (std::istream & is, SDevice & dev)
{
	std::string token;
	is >> token;
	size_t colonpos = token.rfind(':');
	if (colonpos == std::string::npos)
	{
		dev.platform_id = 0;
		if (std::stringstream(token) >> dev.device_id)
			dev.cl_device_type = CL_DEVICE_TYPE_ALL;
		else
		{
			dev.device_id = 0;
			dev.cl_device_type = str2clDeviceType(token);
		}
	}
	else
	{
		dev.device_id = atoi(token.substr(colonpos + 1).c_str());
		token = token.substr(0, colonpos);
		colonpos = token.find(':');
		if (colonpos == std::string::npos)
		{
			if (std::stringstream(token) >> dev.platform_id)
				dev.cl_device_type = CL_DEVICE_TYPE_ALL;
			else
			{
				dev.platform_id = 0;
				dev.cl_device_type = str2clDeviceType(token);
			}
		}
		else
		{
			dev.platform_id = atoi(token.substr(colonpos + 1).c_str());
			dev.cl_device_type = str2clDeviceType(token.substr(0, colonpos));
		}
	}
	return is;
}

std::istream & operator >> (std::istream & is, SVTLNAlpha & alpha)
{
	std::string token;
	is >> token;
	boost::smatch sm;
	boost::regex e("([^:]+):([^:]+):([^:]+)");
	if (boost::regex_match(token, sm, e))
	{
		alpha.min = atof(sm.str(1).c_str());
		alpha.step = atof(sm.str(2).c_str());
		alpha.max = atof(sm.str(3).c_str());
	}
	else
		alpha = SVTLNAlpha(atof(token.c_str()));
	return is;
}

int main(int argc, char * argv[])
{
	SConfig cfg = { Method_MFCC, Platform_Auto, SVTLNAlpha(1), 25, 10, 15, 12, 2, 0, 3, 3, 31, 10, 8, 0, 64, 0, 22, true, true, false, 0 };
	SDevice device_spec;
	int sample_limit = 10000000,
		num_threads = 1;
	std::string input_dir_arg, output_dir_arg, wav_file, scp_file, output_ext("txt"), config_file, htk_config_file;
	benchmark = false;
	debug_mode = 0;

	po::options_description prog_desc("Program options");
	prog_desc.add_options()
		("help,h", "Print help and exit")
		("dev,d", po::value<SDevice>(&device_spec), "Cuda/OpenCL device to use (for OpenCL use format platform_id:device_id)")
		("ext", po::value<std::string>(&output_ext)->default_value(output_ext), "Output file extension")
		("sample-limit", po::value<int>(&sample_limit)->default_value(sample_limit), "Suggested size of GPU buffer in samples")
		("input-dir", po::value<std::string>(&input_dir_arg), "Input file directory")
		("output-dir", po::value<std::string>(&output_dir_arg), "Output file directory")
		("wav-file", po::value<std::string>(&wav_file), "Input file name")
		("scp", po::value<std::string>(&scp_file), "SCP file containing list of files to process")
		("config,c", po::value<std::string>(&config_file), "Afet configuration file")
		("htk-config", po::value<std::string>(&htk_config_file), "HTK configuration file")
		("platform,p", po::value<Platform_t>(&cfg.platform), "Computation platform, valid options are:\n\"Auto\" (default), \"CPU\", \"CUDA\" or \"OpenCL\"")
		("text-output", po::value<bool>(&cfg.text_output), "Specifies if output should be text file")
		("benchmark,b", "Prints the duration of various parts of the processing")
		("verbose,v", "Verbose mode");
	po::options_description param_desc("Parameterization options");
	param_desc.add_options()
		("method,m", po::value<Method_t>(&cfg.method)->default_value(cfg.method), "Parameterization method, valid options are:\n\"MFCC\" (default), \"TRAPS\" or \"PLP\"")
		("window-size", po::value<float>(&cfg.window_size)->default_value(cfg.window_size), "Window size in [ms]")
		("shift", po::value<float>(&cfg.shift)->default_value(cfg.shift), "Window shift in [ms]")
		("banks", po::value<int>(&cfg.num_banks)->default_value(cfg.num_banks), "Number of mel banks")
		("ceps", po::value<int>(&cfg.ceps_len)->default_value(cfg.ceps_len), "Number of cepstral coefficients (0 means no dct)")
		("sample-rate", po::value<float>(&cfg.sample_rate)->default_value(cfg.sample_rate), "Input data sample rate")
		("low-freq", po::value<float>(&cfg.low_freq)->default_value(cfg.low_freq), "Low frequency cutoff")
		("high-freq", po::value<float>(&cfg.high_freq)->default_value(cfg.high_freq), "High frequency cutoff")
		("lift-coef", po::value<float>(&cfg.lift_coef)->default_value(cfg.lift_coef), "Liftering coefficient")
		("c0", po::value<bool>(&cfg.want_c0)->default_value(cfg.want_c0), "Keep the 0th coefficient")
		("alpha", po::value<SVTLNAlpha>(&cfg.alpha), "VTLN parameter alpha")
		("norm", po::value<int>(&cfg.norm_type)->default_value(cfg.norm_type), "Feature normalization (0 - none, 1 - CMN, 2 - CVN, 3 - minmax)")
		("dyn", po::value<int>(&cfg.dyn_type)->default_value(cfg.dyn_type), "Dynamic coefficients (0 - none, 1 - delta, 2 - delta & acceleration)")
		("l1", po::value<int>(&cfg.delta_l1)->default_value(cfg.delta_l1), "Width of delta coefficients calculation")
		("l2", po::value<int>(&cfg.delta_l2)->default_value(cfg.delta_l2), "Width of acceleration coefficients calculation")
		("norm-after-dyn", po::value<bool>(&cfg.norm_after_dyn)->default_value(cfg.norm_after_dyn), "Normalize features after dynamic coefficients calculation")
		("traps-length", po::value<int>(&cfg.traps_len)->default_value(cfg.traps_len), "TRAPS length")
		("traps-dct", po::value<int>(&cfg.traps_dct_len)->default_value(cfg.traps_dct_len), "Number of output TRAPS coefficients")
		("model-order", po::value<int>(&cfg.model_order)->default_value(cfg.model_order), "PLP model order");
	po::options_description hidden_desc("Hidden options");
	hidden_desc.add_options()
		("debug", po::value<int>(&debug_mode), "Debug mode")
		("threads", po::value<int>(&num_threads), "Number of processing threads")
		("applefft", "Use Apple OpenCL FFT")
		("transfilt", "Use one kernel to transpose and filter FFT output");
	po::positional_options_description pos_desc;
	pos_desc.add("wav-file", 1);

	po::options_description desc, visible_desc;
	desc.add(prog_desc).add(param_desc).add(hidden_desc);
	visible_desc.add(prog_desc).add(param_desc);

	po::variables_map vm;
	try
	{
		po::store(po::command_line_parser(argc, argv).options(desc).positional(pos_desc).run(), vm);
		po::notify(vm);
		if (!config_file.empty())
		{
			std::ifstream fcfg(config_file.c_str());
			if (fcfg.good())
			{
				po::store(po::parse_config_file(fcfg, desc), vm);
				fcfg.close();
			}
			else
				std::cerr << "Warning: Can't load configuration file.\n";
		}
		po::notify(vm);
	}
	catch (po::error & e)
	{
		std::cerr << "Error while parsing program parameters: " << e.what() << std::endl;
		help(visible_desc);
		return 1;
	}

	if (vm.count("help"))
	{
		help(visible_desc);
		return 0;
	}

	if (vm.count("benchmark"))
		benchmark = true;
	if (vm.count("verbose"))
		verbose = true;
	if (vm.count("applefft"))
		use_apple_oclfft = true;
	if (vm.count("transfilt"))
		use_trans_filt_combined = true;

	if (!htk_config_file.empty())
		readhtkconfig(htk_config_file, cfg);
	if (!vm.count("wav-file") && !vm.count("scp"))
	{
		std::cerr << "Error: Missing file name.\n";
		help(visible_desc);
		return 1;
	}
	if (vm.count("wav-file") && vm.count("scp"))
	{
		std::cerr << "Options \"wav-file\" and \"scp\" are mutually exclusive.\n";
		return 1;
	}

	try
	{
		if (cfg.platform == Platform_Auto)
		{
#ifdef AFET_CUDA
			int cnt = 0;
			if (cudaGetDeviceCount(&cnt) == cudaSuccess && cnt > 0)
				cfg.platform = Platform_CUDA;
			else
#ifdef AFET_OPENCL
				cfg.platform = Platform_OpenCL;
#else
				cfg.platform = Platform_CPU;
#endif
#else
#ifdef AFET_OPENCL
			cfg.platform = Platform_OpenCL;
#else
			cfg.platform = Platform_CPU;
#endif
#endif
		}
		switch (cfg.platform)
		{
		case Platform_CUDA:
		{
#ifdef AFET_CUDA
			cudaDeviceProp prop;
			assert_cuda(cudaGetDeviceProperties(&prop, device_spec.device_id));
			std::cout << "Selecting CUDA device: " << prop.name << std::endl;
			assert_cuda(cudaSetDevice(device_spec.device_id));
			assert_cuda(cudaFree(NULL));
#else
			throw std::runtime_error("Afet not compiled with CUDA support");
#endif
			break;
		}
		case Platform_OpenCL:
		{
#ifdef AFET_OPENCL
			cl_uint nplatforms, ndevices;

			clGetPlatformIDs(0, NULL, &nplatforms);
			if (nplatforms <= device_spec.platform_id)
				throw std::runtime_error("Invalid OpenCL platform ID");
			cl_platform_id * platform_ids = (cl_platform_id *)malloc(nplatforms * sizeof(cl_platform_id));
			clGetPlatformIDs(nplatforms, platform_ids, NULL);

			int platform_id = 0,
				nonempty_platforms = 0;
			for (;; platform_id++)
			{
				if (platform_id >= nplatforms)
					throw std::runtime_error("Can't find any suitable OpenCL device");
				int ret = clGetDeviceIDs(platform_ids[platform_id], device_spec.cl_device_type, 0, NULL, &ndevices);
				if (ret == CL_DEVICE_NOT_FOUND)
					continue;
				else if (ret == CL_SUCCESS)
				{
					if (device_spec.platform_id == nonempty_platforms++)
						break;
				}
				else
					throw std::runtime_error("Can't get OpenCL device list");
			}
			if (ndevices <= device_spec.device_id)
			{
				free(platform_ids);
				throw std::runtime_error("Invalid OpenCL device ID");
			}
			cl_device_id * device_ids = (cl_device_id *)malloc(ndevices * sizeof(cl_device_id));
			clGetDeviceIDs(platform_ids[platform_id], device_spec.cl_device_type, ndevices, device_ids, NULL);

			cfg.opencl_device = device_ids[device_spec.device_id];

			const int BUFFSIZE = 128;
			char name[128];
			clGetDeviceInfo(cfg.opencl_device, CL_DEVICE_NAME, BUFFSIZE, name, NULL);
			std::cout << "Selecting OpenCL device: " << name << std::endl;
#else
			throw std::runtime_error("Afet not compiled with OpenCL support");
#endif
			break;
		}
		}

		std::vector<fs::path> input, output;
		fs::path input_dir = input_dir_arg;
		fs::path output_dir = output_dir_arg;
		if (input_dir.empty())
			input_dir = fs::current_path();
		if (output_dir.empty())
			output_dir = input_dir;

		if (!scp_file.empty())
		{
			std::ifstream fin(scp_file.c_str());
			if (fin)
			{
				for (std::string line; std::getline(fin, line);)
					input.push_back(line);
				fin.close();

				for (std::vector<fs::path>::iterator it = input.begin(); it != input.end(); ++it)
				{
					if (it->is_absolute())
					{
						fs::path out = *it;
						output.push_back(out.replace_extension(output_ext));
					}
					else
					{
						output.push_back((output_dir / *it).replace_extension(output_ext));
						*it = input_dir / *it;
					}
				}
			}
			else
			{
				std::cerr << "Can't open SCP file \"" << scp_file << "\"\n";
				return 2;
			}
		}
		else
		{
			if (fs::path(wav_file).is_absolute())
			{
				input.push_back(wav_file);
				output.push_back(fs::path(wav_file).replace_extension(output_ext));
			}
			else
			{
				input.push_back(input_dir / wav_file);
				output.push_back((output_dir / wav_file).replace_extension(output_ext));
			}
		}
		StopWatch sw;
		sw.start();

		process_files_mt(input, output, cfg, sample_limit, num_threads);

		sw.stop();
		std::cout << "Total processing time: " << sw.getTime() << " s\n";

#ifdef AFET_CUDA
		if (cfg.platform == Platform_CUDA)
			assert_cuda(cudaDeviceReset());
#endif
	}
	catch (const std::runtime_error & e)
	{
		std::cerr << "Exception thrown: " << e.what() << std::endl;
		return 3;
	}

	return 0;
}




//#pragma OPENCL EXTENSION cl_khr_fp64 : enable 
//
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//#include <tchar.h>
//#include <memory.h>
//#include <fstream>
//#include <ctime>
//#include <queue>
//#include <stdexcept>
//#include <vector>
//#include <cmath>
//#include <algorithm>
//#include <cstring>
//#include <string>
//
//#include "CL\cl.h"
//#include "utils.h"
//#include "segmentercpu.h"
//#include "mfcccpu.h"
//#include "mfccopencl.h"
//
////for perf. counters
//#include <Windows.h>
////for wav file read
//#include <sndfile.h>
//
//// Macros for OpenCL versions
//#define OPENCL_VERSION_1_2  1.2f
//#define OPENCL_VERSION_2_0  2.0f
//
///* This function helps to create informative messages in
// * case when OpenCL errors occur. It returns a string
// * representation for an OpenCL error code.
// * (E.g. "CL_DEVICE_NOT_FOUND" instead of just -1.)
// */
//const char* TranslateOpenCLError(cl_int errorCode)
//{
//    switch(errorCode)
//    {
//    case CL_SUCCESS:                            return "CL_SUCCESS";
//    case CL_DEVICE_NOT_FOUND:                   return "CL_DEVICE_NOT_FOUND";
//    case CL_DEVICE_NOT_AVAILABLE:               return "CL_DEVICE_NOT_AVAILABLE";
//    case CL_COMPILER_NOT_AVAILABLE:             return "CL_COMPILER_NOT_AVAILABLE";
//    case CL_MEM_OBJECT_ALLOCATION_FAILURE:      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
//    case CL_OUT_OF_RESOURCES:                   return "CL_OUT_OF_RESOURCES";
//    case CL_OUT_OF_HOST_MEMORY:                 return "CL_OUT_OF_HOST_MEMORY";
//    case CL_PROFILING_INFO_NOT_AVAILABLE:       return "CL_PROFILING_INFO_NOT_AVAILABLE";
//    case CL_MEM_COPY_OVERLAP:                   return "CL_MEM_COPY_OVERLAP";
//    case CL_IMAGE_FORMAT_MISMATCH:              return "CL_IMAGE_FORMAT_MISMATCH";
//    case CL_IMAGE_FORMAT_NOT_SUPPORTED:         return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
//    case CL_BUILD_PROGRAM_FAILURE:              return "CL_BUILD_PROGRAM_FAILURE";
//    case CL_MAP_FAILURE:                        return "CL_MAP_FAILURE";
//    case CL_MISALIGNED_SUB_BUFFER_OFFSET:       return "CL_MISALIGNED_SUB_BUFFER_OFFSET";                          //-13
//    case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:    return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";   //-14
//    case CL_COMPILE_PROGRAM_FAILURE:            return "CL_COMPILE_PROGRAM_FAILURE";                               //-15
//    case CL_LINKER_NOT_AVAILABLE:               return "CL_LINKER_NOT_AVAILABLE";                                  //-16
//    case CL_LINK_PROGRAM_FAILURE:               return "CL_LINK_PROGRAM_FAILURE";                                  //-17
//    case CL_DEVICE_PARTITION_FAILED:            return "CL_DEVICE_PARTITION_FAILED";                               //-18
//    case CL_KERNEL_ARG_INFO_NOT_AVAILABLE:      return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";                         //-19
//    case CL_INVALID_VALUE:                      return "CL_INVALID_VALUE";
//    case CL_INVALID_DEVICE_TYPE:                return "CL_INVALID_DEVICE_TYPE";
//    case CL_INVALID_PLATFORM:                   return "CL_INVALID_PLATFORM";
//    case CL_INVALID_DEVICE:                     return "CL_INVALID_DEVICE";
//    case CL_INVALID_CONTEXT:                    return "CL_INVALID_CONTEXT";
//    case CL_INVALID_QUEUE_PROPERTIES:           return "CL_INVALID_QUEUE_PROPERTIES";
//    case CL_INVALID_COMMAND_QUEUE:              return "CL_INVALID_COMMAND_QUEUE";
//    case CL_INVALID_HOST_PTR:                   return "CL_INVALID_HOST_PTR";
//    case CL_INVALID_MEM_OBJECT:                 return "CL_INVALID_MEM_OBJECT";
//    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:    return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
//    case CL_INVALID_IMAGE_SIZE:                 return "CL_INVALID_IMAGE_SIZE";
//    case CL_INVALID_SAMPLER:                    return "CL_INVALID_SAMPLER";
//    case CL_INVALID_BINARY:                     return "CL_INVALID_BINARY";
//    case CL_INVALID_BUILD_OPTIONS:              return "CL_INVALID_BUILD_OPTIONS";
//    case CL_INVALID_PROGRAM:                    return "CL_INVALID_PROGRAM";
//    case CL_INVALID_PROGRAM_EXECUTABLE:         return "CL_INVALID_PROGRAM_EXECUTABLE";
//    case CL_INVALID_KERNEL_NAME:                return "CL_INVALID_KERNEL_NAME";
//    case CL_INVALID_KERNEL_DEFINITION:          return "CL_INVALID_KERNEL_DEFINITION";
//    case CL_INVALID_KERNEL:                     return "CL_INVALID_KERNEL";
//    case CL_INVALID_ARG_INDEX:                  return "CL_INVALID_ARG_INDEX";
//    case CL_INVALID_ARG_VALUE:                  return "CL_INVALID_ARG_VALUE";
//    case CL_INVALID_ARG_SIZE:                   return "CL_INVALID_ARG_SIZE";
//    case CL_INVALID_KERNEL_ARGS:                return "CL_INVALID_KERNEL_ARGS";
//    case CL_INVALID_WORK_DIMENSION:             return "CL_INVALID_WORK_DIMENSION";
//    case CL_INVALID_WORK_GROUP_SIZE:            return "CL_INVALID_WORK_GROUP_SIZE";
//    case CL_INVALID_WORK_ITEM_SIZE:             return "CL_INVALID_WORK_ITEM_SIZE";
//    case CL_INVALID_GLOBAL_OFFSET:              return "CL_INVALID_GLOBAL_OFFSET";
//    case CL_INVALID_EVENT_WAIT_LIST:            return "CL_INVALID_EVENT_WAIT_LIST";
//    case CL_INVALID_EVENT:                      return "CL_INVALID_EVENT";
//    case CL_INVALID_OPERATION:                  return "CL_INVALID_OPERATION";
//    case CL_INVALID_GL_OBJECT:                  return "CL_INVALID_GL_OBJECT";
//    case CL_INVALID_BUFFER_SIZE:                return "CL_INVALID_BUFFER_SIZE";
//    case CL_INVALID_MIP_LEVEL:                  return "CL_INVALID_MIP_LEVEL";
//    case CL_INVALID_GLOBAL_WORK_SIZE:           return "CL_INVALID_GLOBAL_WORK_SIZE";                           //-63
//    case CL_INVALID_PROPERTY:                   return "CL_INVALID_PROPERTY";                                   //-64
//    case CL_INVALID_IMAGE_DESCRIPTOR:           return "CL_INVALID_IMAGE_DESCRIPTOR";                           //-65
//    case CL_INVALID_COMPILER_OPTIONS:           return "CL_INVALID_COMPILER_OPTIONS";                           //-66
//    case CL_INVALID_LINKER_OPTIONS:             return "CL_INVALID_LINKER_OPTIONS";                             //-67
//    case CL_INVALID_DEVICE_PARTITION_COUNT:     return "CL_INVALID_DEVICE_PARTITION_COUNT";                     //-68
////    case CL_INVALID_PIPE_SIZE:                  return "CL_INVALID_PIPE_SIZE";                                  //-69
////    case CL_INVALID_DEVICE_QUEUE:               return "CL_INVALID_DEVICE_QUEUE";                               //-70    
//
//    default:
//        return "UNKNOWN ERROR CODE";
//    }
//}
//
//struct SDevice
//{
//	int platform_id,
//		device_id,
//		cl_device_type;
//	SDevice() : platform_id(0), device_id(0), cl_device_type(CL_DEVICE_TYPE_ALL) {}
//};
//
//struct SVTLNAlpha
//{
//	float min_f, max_f, step_f;
//	SVTLNAlpha() : min_f(1), max_f(1), step_f(1) {}
//	SVTLNAlpha(float alpha) : min_f(alpha), max_f(alpha), step_f(1) {}
//};
//
//struct SConfig
//{
//	int method_platform;
//	SVTLNAlpha alpha;
//	float window_size,
//		shift;
//	int num_banks,
//		ceps_len,
//		norm_type,
//		dyn_type,
//		delta_l1,
//		delta_l2,
//		traps_len,
//		traps_dct_len,
//		model_order;
//	float sample_rate,
//		low_freq,
//		high_freq,
//		lift_coef;
//	bool want_c0,
//		norm_after_dyn,
//		text_output;
//	//OpenCL config
//	cl_device_id opencl_device;
//};
////
///////////////////////////////
//
//// Additional Functions to help with the Feature Extraction
//
///////////////////////////////
//
//// Segmenter Core
////////////////////
//// Shifted to segmenter.cpp
//
//// FFT Core
//////////////////////
//// implemented with ocl_dt
//
//
//// Filterbank Core
///////////////////////
//
//
//// Mel Converter
//////////////////////
//static inline float hz2mel(float f) { return 1127 * log(f / 700 + 1); }
//
//// Mel Inverse Converter
//////////////////////
//static inline float mel2hz(float f) { return 700 * (exp(f / 1127) - 1); }
//
//// DCT Converter
///////////////////////
//// Implemented as part of MFCC (Currently without OCL)
//
///* Convenient container for all OpenCL specific objects used in the sample
//*
//* It consists of two parts:
//*   - regular OpenCL objects which are used in almost each normal OpenCL applications
//*   - several OpenCL objects that are specific for this particular sample
//*
//* You collect all these objects in one structure for utility purposes
//* only, there is no OpenCL specific here: just to avoid global variables
//* and make passing all these arguments in functions easier.
//*/
//struct ocl_args_d_t
//{
//	ocl_args_d_t();
//	~ocl_args_d_t();
//
//	// Regular OpenCL objects:
//	cl_context       context;           // hold the context handler
//	cl_device_id     device;            // hold the selected device handler
//	cl_command_queue commandQueue;      // hold the commands-queue handler
//	cl_program       program;           // hold the program handler
//	cl_kernel        kernel;            // hold the kernel handler
//	float            platformVersion;   // hold the OpenCL platform version (default 1.2)
//	float            deviceVersion;     // hold the OpenCL device version (default. 1.2)
//	float            compilerVersion;   // hold the device OpenCL C version (default. 1.2)
//
//										// Objects that are specific for algorithm implemented in this sample
//	cl_mem           srcA;              // hold first source buffer
//	cl_mem           srcB;              // hold second source buffer
//	cl_mem           dstMem;            // hold destination buffer
//};
//
//ocl_args_d_t::ocl_args_d_t() :
//	context(NULL),
//	device(NULL),
//	commandQueue(NULL),
//	program(NULL),
//	kernel(NULL),
//	platformVersion(OPENCL_VERSION_1_2),
//	deviceVersion(OPENCL_VERSION_1_2),
//	compilerVersion(OPENCL_VERSION_1_2),
//	srcA(NULL),
//	srcB(NULL),
//	dstMem(NULL)
//{
//}
//
///*
//* destructor - called only once
//* Release all OpenCL objects
//* This is a regular sequence of calls to deallocate all created OpenCL resources in bootstrapOpenCL.
//*
//* You may want to call these deallocation procedures in the middle of your application execution
//* (not at the end) if you don't further need OpenCL runtime.
//* You may want to do that in order to free some memory, for example,
//* or recreate OpenCL objects with different parameters.
//*
//*/
//ocl_args_d_t::~ocl_args_d_t()
//{
//	cl_int err = CL_SUCCESS;
//
//	if (kernel)
//	{
//		err = clReleaseKernel(kernel);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseKernel returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//	if (program)
//	{
//		err = clReleaseProgram(program);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseProgram returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//	if (srcA)
//	{
//		err = clReleaseMemObject(srcA);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//	if (srcB)
//	{
//		err = clReleaseMemObject(srcB);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//	if (dstMem)
//	{
//		err = clReleaseMemObject(dstMem);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseMemObject returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//	if (commandQueue)
//	{
//		err = clReleaseCommandQueue(commandQueue);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseCommandQueue returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//	if (device)
//	{
//		err = clReleaseDevice(device);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseDevice returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//	if (context)
//	{
//		err = clReleaseContext(context);
//		if (CL_SUCCESS != err)
//		{
//			LogError("Error: clReleaseContext returned '%s'.\n", TranslateOpenCLError(err));
//		}
//	}
//
//	/*
//	* Note there is no procedure to deallocate platform
//	* because it was not created at the startup,
//	* but just queried from OpenCL runtime.
//	*/
//}
//
//
///*
//* Check whether an OpenCL platform is the required platform
//* (based on the platform's name)
//*/
//bool CheckPreferredPlatformMatch(cl_platform_id platform, const char* preferredPlatform)
//{
//	size_t stringLength = 0;
//	cl_int err = CL_SUCCESS;
//	bool match = false;
//
//	// In order to read the platform's name, we first read the platform's name string length (param_value is NULL).
//	// The value returned in stringLength
//	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &stringLength);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_NAME length returned '%s'.\n", TranslateOpenCLError(err));
//		return false;
//	}
//
//	// Now, that we know the platform's name string length, we can allocate enough space before read it
//	std::vector<char> platformName(stringLength);
//
//	// Read the platform's name string
//	// The read value returned in platformName
//	err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, stringLength, &platformName[0], NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_NAME returned %s.\n", TranslateOpenCLError(err));
//		return false;
//	}
//
//	// Now check if the platform's name is the required one
//	if (strstr(&platformName[0], preferredPlatform) != 0)
//	{
//		// The checked platform is the one we're looking for
//		match = true;
//	}
//
//	return match;
//}
//
///*
//* Find and return the preferred OpenCL platform
//* In case that preferredPlatform is NULL, the ID of the first discovered platform will be returned
//*/
//cl_platform_id FindOpenCLPlatform(const char* preferredPlatform, cl_device_type deviceType)
//{
//	cl_uint numPlatforms = 0;
//	cl_int err = CL_SUCCESS;
//
//	// Get (in numPlatforms) the number of OpenCL platforms available
//	// No platform ID will be return, since platforms is NULL
//	err = clGetPlatformIDs(0, NULL, &numPlatforms);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetplatform_ids() to get num platforms returned %s.\n", TranslateOpenCLError(err));
//		return NULL;
//	}
//	LogInfo("Number of available platforms: %u\n", numPlatforms);
//
//	if (0 == numPlatforms)
//	{
//		LogError("Error: No platforms found!\n");
//		return NULL;
//	}
//
//	std::vector<cl_platform_id> platforms(numPlatforms);
//
//	// Now, obtains a list of numPlatforms OpenCL platforms available
//	// The list of platforms available will be returned in platforms
//	err = clGetPlatformIDs(numPlatforms, &platforms[0], NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetplatform_ids() to get platforms returned %s.\n", TranslateOpenCLError(err));
//		return NULL;
//	}
//
//	// Check if one of the available platform matches the preferred requirements
//	for (cl_uint i = 0; i < numPlatforms; i++)
//	{
//		bool match = true;
//		cl_uint numDevices = 0;
//
//		// If the preferredPlatform is not NULL then check if platforms[i] is the required one
//		// Otherwise, continue the check with platforms[i]
//		if ((NULL != preferredPlatform) && (strlen(preferredPlatform) > 0))
//		{
//			// In case we're looking for a specific platform
//			match = CheckPreferredPlatformMatch(platforms[i], preferredPlatform);
//		}
//
//		// match is true if the platform's name is the required one or don't care (NULL)
//		if (match)
//		{
//			// Obtains the number of deviceType devices available on platform
//			// When the function failed we expect numDevices to be zero.
//			// We ignore the function return value since a non-zero error code
//			// could happen if this platform doesn't support the specified device type.
//			err = clGetDeviceIDs(platforms[i], deviceType, 0, NULL, &numDevices);
//			if (CL_SUCCESS != err)
//			{
//				LogError("clGetDeviceIDs() returned %s.\n", TranslateOpenCLError(err));
//			}
//
//			if (0 != numDevices)
//			{
//				// There is at list one device that answer the requirements
//				return platforms[i];
//			}
//		}
//	}
//
//	return NULL;
//}
//
//
///*
//* This function read the OpenCL platdorm and device versions
//* (using clGetxxxInfo API) and stores it in the ocl structure.
//* Later it will enable us to support both OpenCL 1.2 and 2.0 platforms and devices
//* in the same program.
//*/
//int GetPlatformAndDeviceVersion(cl_platform_id platformId, ocl_args_d_t *ocl)
//{
//	cl_int err = CL_SUCCESS;
//
//	// Read the platform's version string length (param_value is NULL).
//	// The value returned in stringLength
//	size_t stringLength = 0;
//	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, 0, NULL, &stringLength);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetPlatformInfo() to get CL_PLATFORM_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	// Now, that we know the platform's version string length, we can allocate enough space before read it
//	std::vector<char> platformVersion(stringLength);
//
//	// Read the platform's version string
//	// The read value returned in platformVersion
//	err = clGetPlatformInfo(platformId, CL_PLATFORM_VERSION, stringLength, &platformVersion[0], NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetplatform_ids() to get CL_PLATFORM_VERSION returned %s.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	if (strstr(&platformVersion[0], "OpenCL 2.0") != NULL)
//	{
//		ocl->platformVersion = OPENCL_VERSION_2_0;
//	}
//
//	// Read the device's version string length (param_value is NULL).
//	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, 0, NULL, &stringLength);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	// Now, that we know the device's version string length, we can allocate enough space before read it
//	std::vector<char> deviceVersion(stringLength);
//
//	// Read the device's version string
//	// The read value returned in deviceVersion
//	err = clGetDeviceInfo(ocl->device, CL_DEVICE_VERSION, stringLength, &deviceVersion[0], NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_VERSION returned %s.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	if (strstr(&deviceVersion[0], "OpenCL 2.0") != NULL)
//	{
//		ocl->deviceVersion = OPENCL_VERSION_2_0;
//	}
//
//	// Read the device's OpenCL C version string length (param_value is NULL).
//	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, 0, NULL, &stringLength);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION length returned '%s'.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	// Now, that we know the device's OpenCL C version string length, we can allocate enough space before read it
//	std::vector<char> compilerVersion(stringLength);
//
//	// Read the device's OpenCL C version string
//	// The read value returned in compilerVersion
//	err = clGetDeviceInfo(ocl->device, CL_DEVICE_OPENCL_C_VERSION, stringLength, &compilerVersion[0], NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetDeviceInfo() to get CL_DEVICE_OPENCL_C_VERSION returned %s.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	else if (strstr(&compilerVersion[0], "OpenCL C 2.0") != NULL)
//	{
//		ocl->compilerVersion = OPENCL_VERSION_2_0;
//	}
//
//	return err;
//}
//
//
///*
//* Generate random value for input buffers
//*/
//void generateInput(cl_int* inputArray, cl_uint arrayWidth, cl_uint arrayHeight)
//{
//	srand(12345);
//
//	// random initialization of input
//	cl_uint array_size = arrayWidth * arrayHeight;
//	for (cl_uint i = 0; i < array_size; ++i)
//	{
//		inputArray[i] = rand();
//	}
//}
//
//
///*
//* This function picks/creates necessary OpenCL objects which are needed.
//* The objects are:
//* OpenCL platform, device, context, and command queue.
//*
//* All these steps are needed to be performed once in a regular OpenCL application.
//* This happens before actual compute kernels calls are performed.
//*
//* For convenience, in this application you store all those basic OpenCL objects in structure ocl_args_d_t,
//* so this function populates fields of this structure, which is passed as parameter ocl.
//* Please, consider reviewing the fields before going further.
//* The structure definition is right in the beginning of this file.
//*/
//int SetupOpenCL(ocl_args_d_t *ocl, cl_device_type deviceType)
//{
//	// The following variable stores return codes for all OpenCL calls.
//	cl_int err = CL_SUCCESS;
//
//	// Query for all available OpenCL platforms on the system
//	// Here you enumerate all platforms and pick one which name has preferredPlatform as a sub-string
//	cl_platform_id platformId = FindOpenCLPlatform("Intel", deviceType);
//	if (NULL == platformId)
//	{
//		LogError("Error: Failed to find OpenCL platform.\n");
//		return CL_INVALID_VALUE;
//	}
//
//	// Create context with device of specified type.
//	// Required device type is passed as function argument deviceType.
//	// So you may use this function to create context for any CPU or GPU OpenCL device.
//	// The creation is synchronized (pfn_notify is NULL) and NULL user_data
//	cl_context_properties contextProperties[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platformId, 0 };
//	ocl->context = clCreateContextFromType(contextProperties, deviceType, NULL, NULL, &err);
//	if ((CL_SUCCESS != err) || (NULL == ocl->context))
//	{
//		LogError("Couldn't create a context, clCreateContextFromType() returned '%s'.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	// Query for OpenCL device which was used for context creation
//	err = clGetContextInfo(ocl->context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &ocl->device, NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clGetContextInfo() to get list of devices returned %s.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	// Read the OpenCL platform's version and the device OpenCL and OpenCL C versions
//	GetPlatformAndDeviceVersion(platformId, ocl);
//
//	// Create command queue.
//	// OpenCL kernels are enqueued for execution to a particular device through special objects called command queues.
//	// Command queue guarantees some ordering between calls and other OpenCL commands.
//	// Here you create a simple in-order OpenCL command queue that doesn't allow execution of two kernels in parallel on a target device.
//#ifdef CL_VERSION_2_0
//	if (OPENCL_VERSION_2_0 == ocl->deviceVersion)
//	{
//		const cl_command_queue_properties properties[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
//		ocl->commandQueue = clCreateCommandQueueWithProperties(ocl->context, ocl->device, properties, &err);
//	}
//	else {
//		// default behavior: OpenCL 1.2
//		cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
//		ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
//	}
//#else
//	// default behavior: OpenCL 1.2
//	cl_command_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
//	ocl->commandQueue = clCreateCommandQueue(ocl->context, ocl->device, properties, &err);
//#endif
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clCreateCommandQueue() returned %s.\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	return CL_SUCCESS;
//}
//
//
///*
//* Create and build OpenCL program from its source code
//*/
//int CreateAndBuildProgram(ocl_args_d_t *ocl)
//{
//	cl_int err = CL_SUCCESS;
//
//	// Upload the OpenCL C source code from the input file to source
//	// The size of the C program is returned in sourceSize
//	char* source = NULL;
//	size_t src_size = 0;
//	err = ReadSourceFromFile("Template.cl", &source, &src_size);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: ReadSourceFromFile returned %s.\n", TranslateOpenCLError(err));
//		goto Finish;
//	}
//
//	// And now after you obtained a regular C string call clCreateProgramWithSource to create OpenCL program object.
//	ocl->program = clCreateProgramWithSource(ocl->context, 1, (const char**)&source, &src_size, &err);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clCreateProgramWithSource returned %s.\n", TranslateOpenCLError(err));
//		goto Finish;
//	}
//
//	// Build the program
//	// During creation a program is not built. You need to explicitly call build function.
//	// Here you just use create-build sequence,
//	// but there are also other possibilities when program consist of several parts,
//	// some of which are libraries, and you may want to consider using clCompileProgram and clLinkProgram as
//	// alternatives.
//	err = clBuildProgram(ocl->program, 1, &ocl->device, "", NULL, NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clBuildProgram() for source program returned %s.\n", TranslateOpenCLError(err));
//
//		// In case of error print the build log to the standard output
//		// First check the size of the log
//		// Then allocate the memory and obtain the log from the program
//		if (err == CL_BUILD_PROGRAM_FAILURE)
//		{
//			size_t log_size = 0;
//			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
//
//			std::vector<char> build_log(log_size);
//			clGetProgramBuildInfo(ocl->program, ocl->device, CL_PROGRAM_BUILD_LOG, log_size, &build_log[0], NULL);
//
//			LogError("Error happened during the build of OpenCL program.\nBuild log:%s", &build_log[0]);
//		}
//	}
//
//Finish:
//	if (source)
//	{
//		delete[] source;
//		source = NULL;
//	}
//
//	return err;
//}
//
//
///*
//* Create OpenCL buffers from host memory
//* These buffers will be used later by the OpenCL kernel
//*/
//int CreateBufferArguments(ocl_args_d_t *ocl, cl_int* inputA, cl_int* inputB, cl_int* outputC, cl_uint arrayWidth, cl_uint arrayHeight)
//{
//	cl_int err = CL_SUCCESS;
//
//	// Create new OpenCL buffer objects
//	// As these buffer are used only for read by the kernel, you are recommended to create it with flag CL_MEM_READ_ONLY.
//	// Always set minimal read/write flags for buffers, it may lead to better performance because it allows runtime
//	// to better organize data copying.
//	// You use CL_MEM_COPY_HOST_PTR here, because the buffers should be populated with bytes at inputA and inputB.
//
//	ocl->srcA = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * arrayWidth * arrayHeight, inputA, &err);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clCreateBuffer for srcA returned %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	ocl->srcB = clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * arrayWidth * arrayHeight, inputB, &err);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clCreateBuffer for srcB returned %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	// If the output buffer is created directly on top of output buffer using CL_MEM_USE_HOST_PTR,
//	// then, depending on the OpenCL runtime implementation and hardware capabilities, 
//	// it may save you not necessary data copying.
//	// As it is known that output buffer will be write only, you explicitly declare it using CL_MEM_WRITE_ONLY.
//	ocl->dstMem = clCreateBuffer(ocl->context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, sizeof(cl_uint) * arrayWidth * arrayHeight, outputC, &err);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clCreateBuffer for dstMem returned %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//
//	return CL_SUCCESS;
//}
//
//
///*
//* Set kernel arguments
//*/
//cl_uint SetKernelArguments(ocl_args_d_t *ocl)
//{
//	cl_int err = CL_SUCCESS;
//
//	err = clSetKernelArg(ocl->kernel, 0, sizeof(cl_mem), (void *)&ocl->srcA);
//	if (CL_SUCCESS != err)
//	{
//		LogError("error: Failed to set argument srcA, returned %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	err = clSetKernelArg(ocl->kernel, 1, sizeof(cl_mem), (void *)&ocl->srcB);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: Failed to set argument srcB, returned %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	err = clSetKernelArg(ocl->kernel, 2, sizeof(cl_mem), (void *)&ocl->dstMem);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: Failed to set argument dstMem, returned %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	return err;
//}
//
//
///*
//* Execute the kernel
//*/
//cl_uint ExecuteAddKernel(ocl_args_d_t *ocl, cl_uint width, cl_uint height)
//{
//	cl_int err = CL_SUCCESS;
//
//	// Define global iteration space for clEnqueueNDRangeKernel.
//	size_t globalWorkSize[2] = { width, height };
//
//
//	// execute kernel
//	err = clEnqueueNDRangeKernel(ocl->commandQueue, ocl->kernel, 2, NULL, globalWorkSize, NULL, 0, NULL, NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: Failed to run kernel, return %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	// Wait until the queued kernel is completed by the device
//	err = clFinish(ocl->commandQueue);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clFinish return %s\n", TranslateOpenCLError(err));
//		return err;
//	}
//
//	return CL_SUCCESS;
//}
//
//
///*
//* "Read" the result buffer (mapping the buffer to the host memory address)
//*/
//bool ReadAndVerify(ocl_args_d_t *ocl, cl_uint width, cl_uint height, cl_int *inputA, cl_int *inputB)
//{
//	cl_int err = CL_SUCCESS;
//	bool result = true;
//
//	// Enqueue a command to map the buffer object (ocl->dstMem) into the host address space and returns a pointer to it
//	// The map operation is blocking
//	cl_int *resultPtr = (cl_int *)clEnqueueMapBuffer(ocl->commandQueue, ocl->dstMem, true, CL_MAP_READ, 0, sizeof(cl_uint) * width * height, 0, NULL, NULL, &err);
//
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clEnqueueMapBuffer returned %s\n", TranslateOpenCLError(err));
//		return false;
//	}
//
//	// Call clFinish to guarantee that output region is updated
//	err = clFinish(ocl->commandQueue);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clFinish returned %s\n", TranslateOpenCLError(err));
//	}
//
//	// We mapped dstMem to resultPtr, so resultPtr is ready and includes the kernel output !!!
//	// Verify the results
//	unsigned int size = width * height;
//	for (unsigned int k = 0; k < size; ++k)
//	{
//		if (resultPtr[k] != inputA[k] + inputB[k])
//		{
//			LogError("Verification failed at %d: (%d + %d = %d)\n", k, inputA[k], inputB[k], resultPtr[k]);
//			result = false;
//		}
//	}
//
//	// Unmapped the output buffer before releasing it
//	err = clEnqueueUnmapMemObject(ocl->commandQueue, ocl->dstMem, resultPtr, 0, NULL, NULL);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clEnqueueUnmapMemObject returned %s\n", TranslateOpenCLError(err));
//	}
//
//	return result;
//}
//
//int _tmain(int argc, TCHAR* argv[])
//{
//	////////////
//	// WAV READ
//	////////////
//	// Open sound file
//	SF_INFO sndInfo;
//	SNDFILE *sndFile = sf_open("a0001.wav", SFM_READ, &sndInfo);
//	if (sndFile == NULL)
//	{
//		fprintf(stderr, "Error reading source file '%s': %s\n", "a0001.wav", sf_strerror(sndFile));
//		//printf("\nThis worked\n");
//		std::getchar();
//		return 1;
//	}
//	// Check format - 16bit PCM..... EDIT: or Don't, Fucked up response
//	//if (sndInfo.format != (SF_FORMAT_PCM_16 | SF_FORMAT_NIST | SF_FORMAT_WAV | SF_FORMAT_HTK))
//	//{
//	//	printf("%X : Format num \n", sndInfo.format);
//	//	fprintf(stderr, "Input should be 16bit Wav\n");
//	//	// sf_close(sndFile);
//	//	std::getchar();
//	//}
//	//else printf("On the right track :P !\n");
//
//	// Check channels - mono
//	if (sndInfo.channels != 1)
//	{
//		fprintf(stderr, "Wrong number of channels\n");
//		sf_close(sndFile);
//		std::getchar();
//		return 1;
//	}
//	// Actual wavfile read
//	// Allocate memory
//	float *buffer = new float[sndInfo.frames * sndInfo.channels + 10];
//	if (buffer == NULL)
//	{
//		fprintf(stderr, "Could not allocate memory for data\n");
//		sf_close(sndFile);
//		return 1;
//	}
//	// Load data
//	long numFrames = (long)sf_readf_float(sndFile, buffer, sndInfo.frames);
//
//	// Check correct number of samples loaded
//	if (numFrames != sndInfo.frames)
//	{
//		printf("%ld , %ld : numframes, sndInfo.frames\n", numFrames, (long)sndInfo.frames);
//		fprintf(stderr, "Did not read enough frames for source\n");
//		sf_close(sndFile);
//		delete[]buffer;
//		std::getchar();
//	}
//	printf("Read %ld frames from %s, Sample rate: %d, Length: %fs\n", numFrames, "a0001.wav", sndInfo.samplerate, (float)numFrames / sndInfo.samplerate);
//
//	//////////////
//	// SEGMENTER
//	//////////////
//	printf("\nWant to do with CPU or OpenCL? (1/2) \n");
//	int ans;
//	std::cin >> ans;
//	//if (ans == 1)
//	//{
//
//	//}
//	/*else
//	{
//
//	}*/
//	cl_int err;
//	ocl_args_d_t ocl;
//	cl_device_type deviceType = CL_DEVICE_TYPE_GPU;
//
//	LARGE_INTEGER perfFrequency;
//	LARGE_INTEGER performanceCountNDRangeStart;
//	LARGE_INTEGER performanceCountNDRangeStop;
//
//	cl_uint arrayWidth = 1024;
//	cl_uint arrayHeight = 1024;
//
//	//initialize Open CL objects (context, queue, etc.)
//	if (CL_SUCCESS != SetupOpenCL(&ocl, deviceType))
//	{
//		return -1;
//	}
//
//	// allocate working buffers. 
//	// the buffer should be aligned with 4K page and size should fit 64-byte cached line
//	cl_uint optimizedSize = ((sizeof(cl_int) * arrayWidth * arrayHeight - 1) / 64 + 1) * 64;
//	cl_int* inputA = (cl_int*)_aligned_malloc(optimizedSize, 4096);
//	cl_int* inputB = (cl_int*)_aligned_malloc(optimizedSize, 4096);
//	cl_int* outputC = (cl_int*)_aligned_malloc(optimizedSize, 4096);
//	if (NULL == inputA || NULL == inputB || NULL == outputC)
//	{
//		LogError("Error: _aligned_malloc failed to allocate buffers.\n");
//		return -1;
//	}
//
//	//random input
//	generateInput(inputA, arrayWidth, arrayHeight);
//	generateInput(inputB, arrayWidth, arrayHeight);
//
//	// Create OpenCL buffers from host memory
//	// These buffers will be used later by the OpenCL kernel
//	if (CL_SUCCESS != CreateBufferArguments(&ocl, inputA, inputB, outputC, arrayWidth, arrayHeight))
//	{
//		return -1;
//	}
//
//	// Create and build the OpenCL program
//	if (CL_SUCCESS != CreateAndBuildProgram(&ocl))
//	{
//		return -1;
//	}
//
//	// Program consists of kernels.
//	// Each kernel can be called (enqueued) from the host part of OpenCL application.
//	// To call the kernel, you need to create it from existing program.
//	ocl.kernel = clCreateKernel(ocl.program, "Add", &err);
//	if (CL_SUCCESS != err)
//	{
//		LogError("Error: clCreateKernel returned %s\n", TranslateOpenCLError(err));
//		return -1;
//	}
//
//	// Passing arguments into OpenCL kernel.
//	if (CL_SUCCESS != SetKernelArguments(&ocl))
//	{
//		return -1;
//	}
//
//	// Regularly you wish to use OpenCL in your application to achieve greater performance results
//	// that are hard to achieve in other ways.
//	// To understand those performance benefits you may want to measure time your application spent in OpenCL kernel execution.
//	// The recommended way to obtain this time is to measure interval between two moments:
//	//   - just before clEnqueueNDRangeKernel is called, and
//	//   - just after clFinish is called
//	// clFinish is necessary to measure entire time spending in the kernel, measuring just clEnqueueNDRangeKernel is not enough,
//	// because this call doesn't guarantees that kernel is finished.
//	// clEnqueueNDRangeKernel is just enqueue new command in OpenCL command queue and doesn't wait until it ends.
//	// clFinish waits until all commands in command queue are finished, that suits your need to measure time.
//	bool queueProfilingEnable = true;
//	if (queueProfilingEnable)
//		QueryPerformanceCounter(&performanceCountNDRangeStart);
//	// Execute (enqueue) the kernel
//	if (CL_SUCCESS != ExecuteAddKernel(&ocl, arrayWidth, arrayHeight))
//	{
//		return -1;
//	}
//	if (queueProfilingEnable)
//		QueryPerformanceCounter(&performanceCountNDRangeStop);
//
//	// The last part of this function: getting processed results back.
//	// use map-unmap sequence to update original memory area with output buffer.
//	ReadAndVerify(&ocl, arrayWidth, arrayHeight, inputA, inputB);
//
//	// retrieve performance counter frequency
//	if (queueProfilingEnable)
//	{
//		QueryPerformanceFrequency(&perfFrequency);
//		LogInfo("NDRange performance counter time %f ms.\n",
//			1000.0f*(float)(performanceCountNDRangeStop.QuadPart - performanceCountNDRangeStart.QuadPart) / (float)perfFrequency.QuadPart);
//	}
//
//	_aligned_free(inputA);
//	_aligned_free(inputB);
//	_aligned_free(outputC);
//	std::getchar();
//	return 0;
//}