#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float ch_inA_A;
channel float ch_A_out;

__kernel void host_reader(__global const float * restrict inA, unsigned N) 
{
	for (unsigned i = 0; i < N; i++)
	{
		write_channel_intel(ch_inA_A, inA[i]);
	}
}

__kernel void kernelA(unsigned N)
{
	for (unsigned i = 0; i < N; i++) {
		float data =  read_channel_intel(ch_inA_A);
		float result = data * (float)1.7;
		write_channel_intel(ch_A_out, result);
	}
}

__kernel void host_writer(__global float * restrict out, unsigned N)
{
	int err;
	float value=0;
	for (unsigned i = 0; i < N; i++)
	{
		 value = read_channel_intel(ch_A_out);
		 out[i] = value;
	}
}
