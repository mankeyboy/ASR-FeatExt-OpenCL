#ifndef _OPENCLKERNELLOADER_H_
#define _OPENCLKERNELLOADER_H_

#include <string>
#include <iostream>
#include <fstream>

static bool LoadOpenCLKernelFromFile(const std::string & name, std::string & programcode)
{
	std::ifstream fin((name + ".cl").c_str());
	if (!fin)
	{
		std::cerr << "Failed to open \"" << name << ".cl\"\n";
		return false;
	}
	programcode += std::string((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
	programcode += "\n";
	fin.close();
	return true;
}

#include <algorithm>
#define NOMINMAX
#include <Windows.h>
#include "resource.h"

static bool LoadOpenCLKernelFromRes(const std::string & name, std::string & programcode)
{
	std::string resname = "IDR_OPENCLKERNEL" + name;
	std::transform(resname.begin(), resname.end(), resname.begin(), toupper);
	HRSRC res = FindResourceA(NULL, resname.c_str(), "OPENCLKERNEL");
	if (!res)
		goto err;
	HGLOBAL resMem = LoadResource(NULL, res);
	if (!resMem)
		goto err;
	char * source = (char *)LockResource(resMem);
	if (!source)
		goto err;
	DWORD resSize = SizeofResource(NULL, res);
	programcode += std::string(source, resSize);
	programcode += "\n";
	return true;
err:
	std::cerr << "Can't find kernel \"" << name << "\" in program resources.";
	if (res == NULL)
	{
		std::cerr << " FindResourceA failed with error: ";
		char * errorMsg;
		FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER, NULL, GetLastError(), 0, (LPSTR)&errorMsg, 0, NULL);
		std::cerr << errorMsg;
		LocalFree(errorMsg);
	}
	std::cerr << "Trying to load kernel from file" << std::endl;
	return LoadOpenCLKernelFromFile(name, programcode);
}

#define LoadOpenCLKernel LoadOpenCLKernelFromRes
#else

#define DECL_OCL_KERNEL(name) \
    extern const char opencl_kernel_ ## name []; \
    extern int opencl_kernel_ ## name ## _size;
#define OCL_KERNEL_PTR(name) opencl_kernel_ ## name
#define OCL_KERNEL_SIZE(name) opencl_kernel_ ## name ## _size
#define OCL_KERNEL_CHECK(n) \
    if (name.compare(#n) == 0) \
    { \
        ptr = OCL_KERNEL_PTR(n); \
        size = OCL_KERNEL_SIZE(n); \
    }

DECL_OCL_KERNEL(mfcc)
DECL_OCL_KERNEL(delta)
DECL_OCL_KERNEL(norm)

static bool LoadOpenCLKernelFromELF(const std::string & name, std::string & programcode)
{
	const char * ptr = NULL;
	int size = 0;
	OCL_KERNEL_CHECK(mfcc);
	OCL_KERNEL_CHECK(delta);
	OCL_KERNEL_CHECK(norm);
	if (ptr && size)
	{
		programcode += std::string(ptr, (size_t)size) + "\n";
		return true;
	}
	else
	{
		std::cerr << "Can't find kernel \"" << name << "\" in program executable file, trying to load file\n";
		return LoadOpenCLKernelFromFile(name, programcode);
	}
}

#define LoadOpenCLKernel LoadOpenCLKernelFromELF
#endif
