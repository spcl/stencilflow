#pragma OPENCL EXTENSION cl_intel_channels : enable

#define DEBUG 1
#define NEXT_DEBUG 1

#define CONSTANT_BOUNDARY_VALUE 0.0

#define BOUNDARY_LEFT 1
#define BOUNDARY_RIGHT 1
#define BOUNDARY_TOP 1
#define BOUNDARY_BOTTOM 1

#define DimX 24
#define DimY 24
#define N (DimX*DimX)

#define PADDED_X (DimX + BOUNDARY_LEFT + BOUNDARY_RIGHT)
#define PADDED_Y (DimY + BOUNDARY_BOTTOM + BOUNDARY_TOP)

#define SHIFT_REG_SIZE  ((2*PADDED_X) + 1) // for this kernel: internal buffer size [0, 2, 1]

channel float DATA_IN;
channel float DATA_OUT;


kernel void host_reader(__global const float * restrict data_in) {

	if(DEBUG){printf("hello from host_reader kernel\n");}

	for(unsigned i = 0; i < N; i++){
  		write_channel_intel(DATA_IN, data_in[i]);
		if(DEBUG){printf("host_reader: read %i-th element.\n", i);}
	}
}

float next2D(int * phase, unsigned * index, unsigned * row_index){

	switch(*phase){
		case 1:
			if(NEXT_DEBUG){printf("next2D: index: %i, phase: %i\n", *index, *phase);}
			*index += 1;
			if (*index >= BOUNDARY_TOP*PADDED_X){
				*phase = 2;
				*index = 0;
				*row_index = 0;
			}
			return CONSTANT_BOUNDARY_VALUE;
		break;
		case 2:
			
			printf("%i, %i\n", *index, *phase);

			if(*row_index >= PADDED_X){
				*row_index = 0;
			}

			*index += 1;
			if (*index >= DimY*PADDED_X){
				*phase = 3;
				*index = 0;
			}

			if(*row_index < BOUNDARY_LEFT || *row_index >= BOUNDARY_LEFT+DimX){
				if(NEXT_DEBUG){printf("next2D: phase 2: read const\n");}
				*row_index += 1;
				return CONSTANT_BOUNDARY_VALUE;
			} else {
				if(NEXT_DEBUG){printf("next2D: phase 2: read channel\n");}
				*row_index += 1;
				return read_channel_intel(DATA_IN);
			}
		break;
		case 3:
			if(NEXT_DEBUG){printf("next2D: phase3: read const\n");}
			return CONSTANT_BOUNDARY_VALUE;
		break;
	}
}


void shift(float * shift_register_buffer){

	// shift all forward and load new element into free space	
	if(DEBUG){printf("kernelA: shift data\n");}
	#pragma unroll
	for(unsigned j = SHIFT_REG_SIZE-1; j > 0 ; j--){
		shift_register_buffer[j] = shift_register_buffer[j-1];
	}
}

kernel void kernelA() {

	if(DEBUG){printf("hello from kernelA kernel\n");}

	/*
		INITIALIZATION
	*/
	if(DEBUG){printf("kernelA: initialization\n");}

	// define state of next2D method
	int phase = 1;
	unsigned index = 0;
	unsigned row_index = 0;

	// define shift register (internal buffer)
	float shift_register_buffer[SHIFT_REG_SIZE];	

	// fill initial internal buffer
	for(unsigned i = 0; i < SHIFT_REG_SIZE; i++){
		shift(shift_register_buffer);
		shift_register_buffer[0] = next2D(&phase, &index, &row_index);
	}

	/*	
		COMPUTATION
	*/
	if(DEBUG){printf("kernelA: computation\n");}
	for(unsigned i = 0; i < DimX; i++){

		// do left padding
		for(unsigned j = 0; j < BOUNDARY_LEFT; j++){
			shift(shift_register_buffer);
			shift_register_buffer[0] = next2D(&phase, &index, &row_index);
		}

		for(unsigned j = 0; j < DimY; j++){

			// do actual kernel computation
			if(DEBUG){printf("kernelA: do computation\n");}
			float result = (float)(1.0/4.0) * (shift_register_buffer[0] + shift_register_buffer[PADDED_X-1] + shift_register_buffer[PADDED_X+1] + shift_register_buffer[2*PADDED_X]);

			// write result to channel
			if(DEBUG){printf("kernelA: write result to channel\n");}
			write_channel_intel(DATA_OUT, result);

			// shift internal buffer
			shift(shift_register_buffer);
			shift_register_buffer[0] = next2D(&phase, &index, &row_index);;
		}

		// do right padding
		for(unsigned j = 0; j < BOUNDARY_RIGHT; j++){
			shift(shift_register_buffer);
			shift_register_buffer[0] = next2D(&phase, &index, &row_index);;
		}
	}
}

kernel void host_writer(__global float * restrict data_out) {

	if(DEBUG){printf("hello from host_writer kernel\n");}
	for(unsigned i = 0; i < N; i++){
  		data_out[i] = read_channel_intel(DATA_OUT);
		if(DEBUG){printf("host_writer: wrote %i-th element.\n", i);}
 	}
}