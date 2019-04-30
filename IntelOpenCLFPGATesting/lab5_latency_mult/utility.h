#ifndef __UTILITY_H
#define __UTILITY_H

#include "CL/cl.hpp"


#define EPSILON (1e-2f)

void print_platform_info(std::vector<cl::Platform>*);
void print_device_info(std::vector<cl::Device>*);
void fill_generate(cl_float X[], cl_float LO, cl_float HI, cl_uint vectorSize);
bool verification (cl_float X[], cl_float Z[], cl_float CalcZ[], cl_uint vectorSize);

void checkErr(cl_int err, const char * name);

#endif
