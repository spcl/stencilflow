// This file
#include "utility.h"
#include <math.h>
#include <iostream>
#include <stdio.h>

void print_platform_info(std::vector<cl::Platform>* PlatformList)
{ 
	//Grab Platform Info for each platform
	for (int i=0; i<PlatformList->size(); i++)
	{
		printf("Platform Number: %d\n", i);
		std::cout << "Platform Name: "<<PlatformList->at(i).getInfo<CL_PLATFORM_NAME>()<<"\n";
		std::cout << "Platform Profile: "<<PlatformList->at(i).getInfo<CL_PLATFORM_PROFILE>()<<"\n";
		std::cout << "Platform Version: "<<PlatformList->at(i).getInfo<CL_PLATFORM_VERSION>()<<"\n";
		std::cout << "Platform Vendor: "<<PlatformList->at(i).getInfo<CL_PLATFORM_VENDOR>()<<"\n\n";
	}
}


void print_device_info(std::vector<cl::Device>* DeviceList)
{
	//Grab Device Info for each device
	for (int i=0; i<DeviceList->size(); i++)
	{
		printf("Device Number: %d\n", i);
		std::cout << "Device Name: "<<DeviceList->at(i).getInfo<CL_DEVICE_NAME>()<<"\n";
		std::cout << "Device Vendor: "<<DeviceList->at(i).getInfo<CL_DEVICE_VENDOR>()<<"\n";
		std::cout << "Is Device Available?: "<<DeviceList->at(i).getInfo<CL_DEVICE_AVAILABLE>()<<"\n";
		std::cout << "Is Device Little Endian?: "<<DeviceList->at(i).getInfo<CL_DEVICE_ENDIAN_LITTLE>()<<"\n";
		std::cout << "Device Max Compute Units: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>()<<"\n";
		std::cout << "Device Max Work Item Dimensions: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()<<"\n";
		std::cout << "Device Max Work Group Size: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>()<<"\n";
		std::cout << "Device Max Frequency: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>()<<"\n";
		std::cout << "Device Max Mem Alloc Size: "<<DeviceList->at(i).getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>()<<"\n\n";
	}
}

void fill_generate(cl_float X[], cl_float LO, cl_float HI, cl_uint vectorSize)
{

	//Assigns randome number from LO to HI to all locatoin of X and Y
	for (int i = 0; i < vectorSize; ++i) {
		X[i] =  LO + (cl_float)rand()/((cl_float)RAND_MAX/(HI-LO));
	}
}

bool verification (float X[], float Z[], float CalcZ[], cl_uint vectorSize)
{
	for(int i = 0; i < vectorSize-1; i++) {
		if(fabs(((float *)CalcZ)[i] - ((float *)Z)[i]) > EPSILON) {
			 printf("\nVERIFICATION FAILED! index %d, FPGA Result:%f != CPU Result %f)",
				  i,  ((float *)Z)[i], ((float *)CalcZ)[i]);
			 return false;
		}
    	}

	// Print 10 Sample Data to Standard Out
	printf("\n\nVERIFICATION PASSED!!!\n\nSome Sample of Results\n");
	printf("------------------------------------\n");
	for (int i = 0; i < ((int)vectorSize-1); i=i+((int)vectorSize)/5) {
		printf("Index %d: FPGA Result, %f, CPU result %f\n", i,  ((float*)Z)[i], ((float*)CalcZ)[i]);
	}
	return true;
}

void checkErr(cl_int err, const char * name)
{
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: " << name
                 << " (" << err << ")" << std::endl;
        exit(EXIT_FAILURE);
    }
}
