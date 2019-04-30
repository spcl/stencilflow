#pragma OPENCL EXTENSION cl_intel_channels : enable

#define N 1024

channel float DATA_IN;
channel float DATA_OUT;

kernel void host_reader(__global const float * restrict data_in) {

	for(unsigned i = 0; i < N; i++){
  		write_channel_intel(DATA_IN, data_in[i]);
	}
}

kernel void kernelA() {

	for(unsigned i = 0; i < N; i++){

		float data = read_channel_intel(DATA_IN);
		float result = sin(data);
		write_channel_intel(DATA_OUT, result);
	}
}

kernel void host_writer(__global float * restrict data_out) {

	for(unsigned i = 0; i < N; i++){
  		data_out[i] = read_channel_intel(DATA_OUT);
	}
}