from StencilChainSimulator.code.kernel_chain_graph import KernelChainGraph


def do_estimate():

        chain = KernelChainGraph("input/dycore_upper_half_3.json")
        print_chain_info(chain)


def print_chain_info(chain):
    total_internal = [0, 0, 0]
    total_delay = [0, 0, 0]

    for node in chain.kernel_nodes:
        print("internal buffer:", node, chain.kernel_nodes[node].kernel.graph.buffer_size)
        total = [0, 0, 0]
        for entry in chain.kernel_nodes[node].kernel.graph.buffer_size:
            total = KernelChainGraph.list_add_cwise(chain.kernel_nodes[node].kernel.graph.buffer_size[entry], total)
        total_internal = KernelChainGraph.list_add_cwise(total, total_internal)
        print("path lengths:", node, chain.kernel_nodes[node].input_paths)
        print("delay buffer:", node, chain.kernel_nodes[node].delay_buffer)
        total_delay = KernelChainGraph.list_add_cwise(chain.kernel_nodes[node].delay_buffer, total_delay)
        print("latency:", node, chain.kernel_nodes[node].kernel.graph.max_latency)
        print()

    print("total internal buffer: ", total_internal)
    print("total delay buffer: ", total_delay)


if __name__ == "__main__":
    do_estimate()


    """
    average stencil program buffer estimate (average of three)
    
    fastwaves:    
    total internal buffer:  [8, 7, 6]
    total delay buffer:  [32, -12, 289]
    total: [40, -5, 295]
    
    diffusion_min:
    total internal buffer:  [16, 13, 0]
    total delay buffer:  [20, 8, 192]
    total: [36, 21, 192]
    
    advection_min:
    total internal buffer:  [8, 8, 0]
    total delay buffer:  [10, 2, 57]
    total: [18, 10, 57]
    
    average: [31, 9, 544]
    
    total dimensions: 1024x1024x64 (longitude x latitude x altitude)
    dimensions: we assume: buffering of vertical slices of size 1024*64 (to be double-checked,
     if that is possible instead of horizontal 1024*1024 elements slices)
    
    single-precision floating point (4bytes):
    4 * (544 + 9*64 + 31*64*1024) = 8130944 bytes ~ 8.1MB
    
    double-precision floating point (8bytes):
    8 * (544 + 9*64 + 31*64*1024) = 16261888 bytes ~ 16.2MB
    
    
    output of the dycore chain: 
    total internal buffer:  [0, 0, 0]
    total delay buffer:  [0, 0, 552] (
    
    """