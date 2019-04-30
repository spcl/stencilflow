#include <math.h>
#include <fstream>
#include <assert.h>
#include <stdio.h>
#include <string>
#include <iostream>

#include "CL/cl.hpp"
#include "utility.h"

static const cl_uint dimX = 12;
static const cl_uint dimY = 12;
static const cl_uint vectorSize = dimX * dimY; 

int main(void)
{
	cl_int err;

	//Setup Platform

	//Get Platform ID
	std::vector<cl::Platform> PlatformList;
	err = cl::Platform::get(&PlatformList);
	assert(err==CL_SUCCESS);
	checkErr(PlatformList.size()==1 ? CL_SUCCESS : -1, "cl::Platform::get");
	print_platform_info(&PlatformList);
	
	//Setup Device
	//Get Device ID
	std::vector<cl::Device> DeviceList;
	err = PlatformList[0].getDevices(CL_DEVICE_TYPE_ALL, &DeviceList);
	assert(err==CL_SUCCESS);
	print_device_info(&DeviceList);
	
	//Create Context
	cl::Context mycontext(DeviceList);
	assert(err==CL_SUCCESS);
	
	//Create Command queue
	//TODO: Add command queues
	cl::CommandQueue queue1(mycontext, DeviceList[0]); 
	cl::CommandQueue queue2(mycontext, DeviceList[0]); 
	cl::CommandQueue queue3(mycontext, DeviceList[0]); 
	assert(err==CL_SUCCESS);

	//Create Buffers for input and output
	cl::Buffer Buffer_In(mycontext, CL_MEM_READ_ONLY, sizeof(cl_float)*vectorSize);
	cl::Buffer Buffer_Out(mycontext, CL_MEM_WRITE_ONLY, sizeof(cl_float)*vectorSize);

	//Inputs and Outputs to Kernel, X and Y are inputs, Z is output
	//The aligned attribute is used to ensure alignment
	//so that DMA could be used if we were working with a real FPGA board
	cl_float X[vectorSize]  __attribute__ ((aligned (64)));
	cl_float Z[vectorSize]  __attribute__ ((aligned (64)));

	//Allocates memory with value from 0 to 1000
	// cl_float LO= 0;   cl_float HI=1000;
	// fill_generate(X, LO, HI, vectorSize);
	for(unsigned i = 0; i < vectorSize; i++){
		X[i] = (float)1.0;
	}

	//Write data to device
	//TODO: adjust which queue data is written to
	err = queue1.enqueueWriteBuffer(Buffer_In, CL_FALSE, 0, sizeof(cl_float)*vectorSize, X);
	assert(err==CL_SUCCESS);
	queue1.finish();

	//Read in binaries from file
	//TODO: change the name of the .aocx file
	std::ifstream aocx_stream("channels.aocx", std::ios::in|std::ios::binary);
	checkErr(aocx_stream.is_open() ? CL_SUCCESS:-1, "channels.aocx");
	std::string prog(std::istreambuf_iterator<char>(aocx_stream), (std::istreambuf_iterator<char>()));
	cl::Program::Binaries mybinaries (1, std::make_pair(prog.c_str(), prog.length()+1));

	// Create the Program from the AOCX file.
	cl::Program program(mycontext, DeviceList, mybinaries);

	// build the program
	//////////////      Compile the Kernel.... For Intel FPGA, nothing is done here, but this comforms to the standard
	err=program.build(DeviceList);
	assert(err==CL_SUCCESS);

	// create the kernel
	//////////////       Find Kernel in Program
	//TODO: Create new kernels
	cl::Kernel host_reader_kernel(program, "host_reader", &err);
	cl::Kernel kernelA_kernel(program, "kernelA", &err);
	cl::Kernel host_writer_kernel(program, "host_writer", &err);
	assert(err==CL_SUCCESS);

	cl_float out = 0.0f;
	//////////////     Set Arguments to the Kernels
	//TODO: Set arguments for all kernels
	err = host_reader_kernel.setArg(0, Buffer_In);
	assert(err==CL_SUCCESS);
	err = host_writer_kernel.setArg(0, Buffer_Out);
	assert(err==CL_SUCCESS);

	printf("\nLaunching the kernel...\n");
	
	// Launch Kernel
	// TODO: launch additional kernels
	err=queue1.enqueueTask(host_reader_kernel);
	err=queue2.enqueueTask(kernelA_kernel);
	err=queue3.enqueueTask(host_writer_kernel);
	assert(err==CL_SUCCESS);

	//TODO: Wait on all the queues
	queue1.finish();
	queue2.finish();
	queue3.finish();

	// read the output
	// TODO: Change the queue for reading data
	err=queue3.enqueueReadBuffer(Buffer_Out, CL_TRUE, 0, sizeof(cl_float)*vectorSize, Z);
	assert(err==CL_SUCCESS);
	err=queue3.finish();
	assert(err==CL_SUCCESS);
	
	float CalcZ[vectorSize];

	for (int i=0; i<vectorSize; i++)
	{
		//////////////  Equivalent Code runnign on CPUs
		CalcZ[i] = X[i]*2.0; 
				
	}

	//Print Performance Results
	//verification (X, Z, CalcZ, vectorSize);

	std::cout << "Input:" << std::endl;
	for(unsigned i = 0; i < dimX; i++){
		for(unsigned j = 0; j < dimY; j++){
			std::cout << X[i*dimX + j] << " ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl << "Output:" << std::endl;
	for(unsigned i = 0; i < dimX; i++){
		for(unsigned j = 0; j < dimY; j++){
			std::cout << Z[i*dimX + j] << " ";
		}
		std::cout << std::endl;
	}

    return 1;
}
