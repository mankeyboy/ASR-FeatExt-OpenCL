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
#include <list>
#include <sstream>
#include <direct.h>
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

//typedef int cl_device_id;

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

//namespace po = boost::program_options;
//namespace fs = boost::filesystem;
using namespace std;

bool verbose = false, use_apple_oclfft = false, use_trans_filt_combined = false;

//boost::mutex global_stream_lock, file_list_lock;

enum Method_t
{
	Method_MFCC
};

enum Platform_t
{
	Platform_Auto,
	Platform_OpenCL,
	Platform_CPU
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
	float window_size, shift;
	int num_banks,
		ceps_len,
		norm_type,
		dyn_type,
		delta_l1,
		delta_l2,
		traps_len,
		traps_dct_len,
		model_order;
	float sample_rate, low_freq, high_freq, lift_coef;
	bool want_c0, norm_after_dyn, text_output;
	//OpenCL config
	cl_device_id opencl_device;
};

struct SProcessedFile
{
	string input, output;
	SProcessedFile() {}
	SProcessedFile(const string & input, const string & output) : input(input), output(output) {}
};

void process_files_worker(std::list<SProcessedFile> * files, SConfig & cfg, int sample_limit)
{
	ParamBase * param = NULL;
	void * data = NULL;
	try
	{
		long int window_size = (long)cfg.sample_rate * cfg.window_size * 1e-3,
			shift = (long)cfg.sample_rate * cfg.shift * 1e-3;

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

		switch (cfg.platform)
		{
		case Platform_CPU:
		{
			param = new MfccCpu(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.ceps_len, cfg.want_c0, cfg.lift_coef, norm_type, dyn_type, cfg.delta_l1, cfg.delta_l2, cfg.norm_after_dyn);
			break;
		}
		case Platform_OpenCL:
		{
			param = new MfccOpenCL(sample_limit, window_size, shift, cfg.num_banks, cfg.sample_rate, cfg.low_freq, cfg.high_freq, cfg.ceps_len, cfg.want_c0, cfg.lift_coef, norm_type, dyn_type, cfg.delta_l1, cfg.delta_l2, cfg.norm_after_dyn, cfg.opencl_device);
			break;
		}
		default:
			return;
		}
		if (param == NULL)
			throw std::runtime_error("Unsupported platform");
		float * window = new float[window_size];
		for (int i = 0; i < window_size; i++)
			window[i] = (float)(0.56f - 0.46f * cos((2.0f * M_PI * i) / window_size)) / 32768.f;
		param->set_window(window);
		delete[] window;
		

		int samples_in_limit = param->get_input_buffer_size();
		int windows_out_limit = param->estimated_window_count(samples_in_limit);
		int data_out_width = param->get_output_data_width();

		int buffer_size = std::max(samples_in_limit * sizeof(short), data_out_width * windows_out_limit * sizeof(float));
		data = new char[buffer_size];

		while (true)
		{
			SProcessedFile file = files->front();
			files->pop_front();
			
			const string & file_in = file.input, &file_out = file.output;
			std::string input_file_name = file_in;
			SF_INFO info;
			info.format = 0;
			SNDFILE * f = NULL;

			f = sf_open(input_file_name.c_str(), SFM_READ, &info);

			if (f == NULL)
			{
				std::cerr << "Can't open \"" << input_file_name << "\"\n";
				continue;
			}
			sf_command(f, SFC_SET_NORM_FLOAT, NULL, SF_FALSE);
			if (info.frames < 1)
			{
				sf_close(f);
				std::cerr << "Error while loading \"" << input_file_name << "\"\n";
				continue;
			}
			if (info.samplerate != cfg.sample_rate)
			{
				sf_close(f);
				std::cerr << "File \"" << input_file_name << "\" has incorrect sample rate (" << info.samplerate << "), aborting program.\n";
				break;
			}

			std::vector<FILE *> vecfout;
			std::vector<int> vecframes;
			int fidx = 0;
			for (float alpha = cfg.alpha.min; alpha <= cfg.alpha.max; alpha = cfg.alpha.min + fidx * cfg.alpha.step)
			{
				string file_out_name;
				if (cfg.alpha.max - cfg.alpha.min < cfg.alpha.step)
					file_out_name = file_out;  //only 1 alpha
				else
				{
					string file_temp_name = file_out.substr(0, file_out.length() - 4), temp_ext = file_out.substr(file_out.length() -1 - 4, 4);
					file_out_name = file_temp_name + to_string(alpha) + temp_ext;
				}
				std::cout << "Creating output file: " << file_out_name << std::endl;
				FILE * fout = NULL;
				if (!(debug_mode))
				{
					fout = fopen(file_out_name.c_str(), cfg.text_output ? "w" : "wb");
					if (fout == NULL)
						throw std::runtime_error("Can't create output file: " + file_out_name);
				}
				vecfout.push_back(fout);
				fidx++;
			}

			int samples = (int)info.frames;
			int windows_total_in = 0,
				windows_total_out = 0,
				windows_out,
				block = 0;
			long double frame_time_step = cfg.shift / cfg.sample_rate,
				frame_time = 0.5f * cfg.window_size / cfg.sample_rate;
			while (samples > 0)
			{
				int samples_in = std::min<size_t>(samples, samples_in_limit);
				if (!(debug_mode))
					sf_readf_short(f, (short *)data, samples_in);
				//sf_readf_float(f, data, samples_in);
			
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
						std::cout << "\tBlock " << block++ << ": " << samples_in << " samples, " << windows_out << " frames, alpha = " << alpha << std::endl;
					}

					if (!(debug_mode))
					{
						FILE * fout = vecfout[fidx];
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
						if (debug_mode)
							fflush(fout);
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
						std::cout << "\tFlushing: " << windows_out << " frames, alpha = " << alpha << std::endl;
					}

					if (!(debug_mode))
					{
						FILE * fout = vecfout[fidx];
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
						if (debug_mode)
							fflush(fout);
					}
				}
			}
			else if (verbose)
			{
				std::cout << "\tNothing to flush\n";
			}
			if (verbose)
			{
				std::cout << "\tTotal frames processed per VTLN alpha: " << windows_total_in << "\n\tTotal frames processed: " << windows_total_out << std::endl;
			}

			sf_close(f);
			for (float alpha = cfg.alpha.min, fidx = 0; alpha <= cfg.alpha.max; alpha += cfg.alpha.step, fidx++)
			{
				FILE * fout = vecfout[(int)fidx];
				if (!(debug_mode) && !cfg.text_output)
				{
					fseek(fout, 0, SEEK_SET);
				}
				fclose(fout);
			}
		}
		
		delete[](char *)data;
		delete param;
	}
	catch (const std::exception & e)
	{
		delete[](char *)data;
		delete param;
		std::cerr << "Exception caught "<<e.what() << "\n" << std::endl;
	}
	catch (...)
	{
		delete[](char *)data;
		delete param;
		std::cerr << "Unknown exception caught "<< std::endl;
	}
}

void process_files_mt(const std::vector<string> & input, const std::vector<string> & output, SConfig & cfg, int sample_limit, int num_threads)
{
	if (cfg.sample_rate <= 0)
	{
		if (verbose)
			std::cout << "Sample rate not set, opening the first file: " << input.front() << "\n";
		SF_INFO info;
		info.format = 0;
		SNDFILE * f = sf_open(input.front().c_str(), SFM_READ, &info);
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

	for (int i = 0; i < num_threads; i++)
		process_files_worker( &files, cfg, sample_limit);

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

void help(std::ostream & os = std::cout)
{
	os << "Usage:\n  afet.exe [options] ([--wav-file] | --scp) file\n" << std::endl;
	os << "Afet was compiled with support for: CPU"
	   <<", OpenCL"<< "\n\n";
	printOpenCLDevices(os);
}

std::istream & operator >> (std::istream & is, Method_t & method)
{
	std::string token;
	is >> token;
	if (!(token.compare("mfcc")))
		method = Method_MFCC;
	return is;
}

std::istream & operator >> (std::istream & is, Platform_t & platform)
{
	std::string token;
	is >> token;
	if (token.compare("auto") == 0)
		platform = Platform_Auto;
	else if (token.compare("cpu") == 0)
		platform = Platform_CPU;
	else if (token.compare("opencl") == 0)
		platform = Platform_OpenCL;
	else
		throw std::runtime_error("Unknown platform");
	return is;
}

int str2clDeviceType(const std::string & t)
{
	if (t.compare("default") == 0)
		return CL_DEVICE_TYPE_DEFAULT;
	else if (t.compare("cpu") == 0)
		return CL_DEVICE_TYPE_CPU;
	else if (t.compare("gpu") == 0)
		return CL_DEVICE_TYPE_GPU;
	else if (t.compare("accelerator") == 0)
		return CL_DEVICE_TYPE_ACCELERATOR;
#ifdef CL_DEVICE_TYPE_CUSTOM
	else if (t.compare("custom") == 0)
		return CL_DEVICE_TYPE_CUSTOM;
#endif
	else if (t.compare("all") == 0)
		return CL_DEVICE_TYPE_ALL;
	else
		throw std::runtime_error("Unknown device type");
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
	/*boost::smatch sm;
	boost::regex e("([^:]+):([^:]+):([^:]+)");
	if (boost::regex_match(token, sm, e))
	{
		alpha.min = atof(sm.str(1).c_str());
		alpha.step = atof(sm.str(2).c_str());
		alpha.max = atof(sm.str(3).c_str());
	}
	else*/
		alpha = SVTLNAlpha(atof(token.c_str()));
	return is;
}
void getFilesList(string filePath, string extension, vector<string> & returnFileName)
{
	WIN32_FIND_DATA fileInfo;
	HANDLE hFind;
	string  fullPath = filePath + extension;
	char buffer[4096];
	int ret;

	const char * cs = fullPath.c_str();
	size_t wn = mbsrtowcs(NULL, &cs, 0, NULL);
	// error if wn == size_t(-1)
	wchar_t * buf = new wchar_t[wn + 1]();  // value-initialize to 0 (see below)
	wn = mbsrtowcs(buf, &cs, wn + 1, NULL);
	// error if wn == size_t(-1)

	
	hFind = FindFirstFile(buf, &fileInfo);
	if (hFind != INVALID_HANDLE_VALUE) {
		ret = wcstombs(buffer, fileInfo.cFileName, sizeof(buffer));
		returnFileName.push_back(filePath + buffer);
		while (FindNextFile(hFind, &fileInfo) != 0) {
			ret = wcstombs(buffer, fileInfo.cFileName, sizeof(buffer));
			returnFileName.push_back(filePath + buffer);
		}
	}
}

int main(int argc, char * argv[])
{
	SConfig cfg = { Method_MFCC, Platform_Auto, SVTLNAlpha(1), 25, 10, 15, 12, 2, 0, 3, 3, 31, 10, 8, 0, 64, 0, 22, true, true, true, 0 };
	SDevice device_spec;
	printOpenCLDevices();
	int sample_limit = 10000000,
		num_threads = 1;
	std::string input_dir_arg, output_dir_arg, scp_file, wav_file, config_file;
	benchmark = false;
	debug_mode = 0;
	std::cout << "In main" << endl;
/*
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

	if (vm.count("verbose"))
		verbose = true;
	if (vm.count("applefft"))
		use_apple_oclfft = true;
	if (vm.count("transfilt"))
		use_trans_filt_combined = true;

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
*/
	try
	{
		if (cfg.platform == Platform_Auto)
		{
			cfg.platform = Platform_OpenCL;
		}
		switch (cfg.platform)
		{
		case Platform_OpenCL:
		{
			cl_uint nplatforms, ndevices;
			std::cout << "Hiya Ho OpenCL" << endl;
			clGetPlatformIDs(0, NULL, &nplatforms);
			if (nplatforms <= device_spec.platform_id)
				throw std::runtime_error("Invalid OpenCL platform ID");
			cl_platform_id * platform_ids = (cl_platform_id *)malloc(nplatforms * sizeof(cl_platform_id));
			clGetPlatformIDs(nplatforms, platform_ids, NULL);

			int platform_id = 0, nonempty_platforms = 0;
			std::cout << "Who goes there: 1\n";
			for (;; platform_id++)
			{
				if (platform_id >= nplatforms)
					throw std::runtime_error("Can't find any suitable OpenCL device");
				int ret = clGetDeviceIDs(platform_ids[platform_id], device_spec.cl_device_type, 0, NULL, &ndevices);
				if (ret == CL_DEVICE_NOT_FOUND)
					continue;
				else if (ret == CL_SUCCESS)
				{
					std::cout << "Who goes there: 1 If condition ret success: "<< device_spec.platform_id << endl;
					if (device_spec.platform_id == nonempty_platforms++)
						break;
				}
				else
					throw std::runtime_error("Can't get OpenCL device list");
			}
			std::cout << "Who goes there: ndevices" << ndevices <<" <= " << device_spec.device_id << endl;
			if (ndevices <= device_spec.device_id)
			{
				free(platform_ids);
				throw std::runtime_error("Invalid OpenCL device ID");
			}
			cl_device_id * device_ids = (cl_device_id *)malloc(ndevices * sizeof(cl_device_id));
			clGetDeviceIDs(platform_ids[platform_id], device_spec.cl_device_type, ndevices, device_ids, NULL);
			for (int i = 0; i < ndevices; i++)
				std::cout << "Device ID" << i << ": "<<device_spec.device_id<< endl;
			cfg.opencl_device = device_ids[device_spec.device_id];

			const int BUFFSIZE = 128;
			char name[128];
			clGetDeviceInfo(cfg.opencl_device, CL_DEVICE_NAME, BUFFSIZE, name, NULL);
			std::cout << "Selecting OpenCL device: " << name << std::endl;
			break;
		}
		}

		std::vector<string> input, output;
		string input_dir = input_dir_arg;
		string output_dir = output_dir_arg;
		/*int maxlen = 100000;
		char * buffer = NULL;
		_getcwd(buffer, maxlen);
		string inputFolderPath(buffer);
		inputFolderPath += "soundfiles";
		string extension = ".wav";
		getFilesList(inputFolderPath, extension, wav_file);
	*/
		/*for (int i = 0; i < wav_file.size(); i++)
		if (wav_file[i].find('/'))
		{
			output.push_back(wav_file[i].substr(0, wav_file[i].length() - 4) + "_mfcc.txt");
		}*/
		/*int maxlen = 10000;
		char * buffer = NULL;
		_getcwd(buffer, maxlen);*/
		string addition = "soundfiles/a1.wav";
		input.push_back(addition);
		addition = "soundfiles/a2.wav";
		input.push_back(addition);
		addition = "soundfiles/a3.wav";
		input.push_back(addition);
		addition = "soundfiles/a4.wav";
		input.push_back(addition);
		addition = "soundfiles/a5.wav";

		
		process_files_mt(input, output, cfg, sample_limit, num_threads);

	}
	catch (const std::runtime_error & e)
	{
		std::cerr << "Exception thrown: " << e.what() << std::endl;
		return 3;
	}

	return 0;
}