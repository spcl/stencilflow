import argparse
import collections
import os
import re
from dapp.graph.edges import InterstateEdge
from dapp.memlet import Memlet
from dapp.sdfg import SDFG
from dapp.types import ScheduleType, StorageType
from kernel_chain_graph import KernelChainGraph, NodeType

DATA_TYPE = float
ITERATORS = ["i", "j", "k"]


def iteration_space(chain):
    if len(chain.dimensions) > 3:
        return collections.OrderedDict([
            ("i" + str(i), "0:" + str(d))
            for i, d in enumerate(chain.dimensions)
        ])
    else:
        return collections.OrderedDict([
            (ITERATORS[i], "0:" + str(d))
            for i, d in enumerate(chain.dimensions)
        ])

def stream_name(src_name, dst_name):
    return src_name + "_to_" + dst_name



def generate_sdfg(name, chain):

    sdfg = SDFG(name)

    pre_state = sdfg.add_state("initialize")
    state = sdfg.add_state("compute")
    post_state = sdfg.add_state("finalize")

    sdfg.add_edge(pre_state, state, InterstateEdge())
    sdfg.add_edge(state, post_state, InterstateEdge())

    def add_pipe(edge):

        data_name = stream_name(edge[0].name, edge[1].name)

        sdfg.add_stream(
            data_name,
            DATA_TYPE,
            buffer_size=1, # TODO: add queue size
            storage=StorageType.FPGA_Local,
            transient=True)

    def add_input(node):

        # Host-side array, which will be an input argument
        sdfg.add_array(
            node.name + "_host",
            chain.dimensions,
            DATA_TYPE)

        # Device-side copy
        sdfg.add_array(
            node.name,
            chain.dimensions,
            DATA_TYPE,
            storage=StorageType.FPGA_Global,
            transient=True)
        access_node = state.add_read(node.name)

        iterators = iteration_space(chain)

        # Copy data to the FPGA
        copy_host = pre_state.add_read(node.name + "_host")
        copy_fpga = pre_state.add_write(node.name)
        pre_state.add_memlet_path(
            copy_host,
            copy_fpga,
            memlet=Memlet.simple(copy_fpga, ", ".join(iterators.values())))

        entry, exit = state.add_map(
            "read_" + node.name, iterators, schedule=ScheduleType.FPGA_Device)

        # Sort to get deterministic output
        outputs = sorted([e[1].name for e in chain.graph.out_edges(node)])

        tasklet_code = "\n".join(["{} = memory".format(n) for n in outputs])

        tasklet = state.add_tasklet("read_" + node.name, {"memory"}, outputs,
                                    tasklet_code)

        state.add_memlet_path(
            access_node,
            entry,
            tasklet,
            dst_conn="memory",
            memlet=Memlet.simple(node.name, ", ".join(iterators.keys())))

        # Add memlets to all FIFOs connecting to compute units
        for dst_name in outputs:
            data_name = stream_name(node.name, dst_name)
            write_node = state.add_write(data_name)
            state.add_memlet_path(
                tasklet,
                exit,
                write_node,
                src_conn=dst_name,
                memlet=Memlet.simple(data_name, "0"))
    # End add_input

    def add_output(node):

        # Host-side array, which will be an output argument
        sdfg.add_array(
            node.name + "_host",
            chain.dimensions,
            DATA_TYPE)

        # Device-side copy
        sdfg.add_array(
            node.name,
            chain.dimensions,
            DATA_TYPE,
            storage=StorageType.FPGA_Global,
            transient=True)
        write_node = state.add_write(node.name)

        iterators = iteration_space(chain)

        # Copy data to the FPGA
        copy_fpga = post_state.add_read(node.name)
        copy_host = post_state.add_write(node.name + "_host")
        post_state.add_memlet_path(
            copy_fpga,
            copy_host,
            memlet=Memlet.simple(copy_host, ", ".join(iterators.values())))

        entry, exit = state.add_map(
            "write_" + node.name, iterators, schedule=ScheduleType.FPGA_Device)

        src = chain.graph.in_edges(node)
        if len(src) > 1:
            raise RuntimeError("Only one writer per output supported")
        src = next(iter(src))[0]

        tasklet_code = "memory = " + src.name

        tasklet = state.add_tasklet("write_" + node.name, {src.name},
                                    {"memory"}, tasklet_code)

        data_name = stream_name(src.name, node.name)
        read_node = state.add_read(data_name)

        state.add_memlet_path(
            read_node,
            entry,
            tasklet,
            dst_conn=src.name,
            memlet=Memlet.simple(data_name, "0"))

        state.add_memlet_path(
            tasklet,
            exit,
            write_node,
            src_conn="memory",
            memlet=Memlet.simple(node.name, ", ".join(iterators.keys())))

    def add_kernel(node):

        iterators = iteration_space(chain)

        entry, exit = state.add_map(
            node.name, iterators, schedule=ScheduleType.FPGA_Device)

        # Sort to get deterministic output
        inputs = sorted([e[0].name for e in chain.graph.in_edges(node)])
        outputs = sorted([e[1].name for e in chain.graph.out_edges(node)])

        tasklet = state.add_tasklet(node.name, inputs, outputs, "")

        for i in inputs:
            data_name = stream_name(i, node.name)
            read_node = state.add_read(data_name)
            state.add_memlet_path(
                read_node,
                entry,
                tasklet,
                dst_conn=i,
                memlet=Memlet.simple(data_name, "0"))

        for o in outputs:
            data_name = stream_name(node.name, o)
            write_node = state.add_write(data_name)
            state.add_memlet_path(
                tasklet,
                exit,
                write_node,
                src_conn=o,
                memlet=Memlet.simple(data_name, "0"))

    for link in chain.graph.edges():
        add_pipe(link)

    for node in chain.graph.nodes():
        if node.node_type == NodeType.INPUT:
            add_input(node)
        elif node.node_type == NodeType.KERNEL:
            add_kernel(node)
        elif node.node_type == NodeType.OUTPUT:
            add_output(node)
        else:
            raise RuntimeError("Unexpected node type: {}".format(
                node.node_type))

    sdfg.draw_to_file()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_file")

    args = parser.parse_args()

    name = os.path.basename(args.stencil_file)
    name = re.match("[^\.]+", name).group(0)

    chain = KernelChainGraph(args.stencil_file)

    chain.plot_graph(name + ".pdf")

    generate_sdfg(name, chain)
