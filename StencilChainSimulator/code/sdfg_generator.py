import argparse
import collections
import functools
import itertools
import operator
import numpy as np
import os
import re
from dapp.graph.edges import InterstateEdge
from dapp.memlet import Memlet
from dapp.sdfg import SDFG
from dapp.types import ScheduleType, StorageType
from kernel_chain_graph import KernelChainGraph, NodeType
from common import *

DATA_TYPE = float
ITERATORS = ["i", "j", "k"]


def make_iterators(chain):
    if len(chain.dimensions) > 3:
        return collections.OrderedDict(
            [("i" + str(i), "0:" + str(d))
             for i, d in enumerate(chain.dimensions)])
    else:
        return collections.OrderedDict(
            [(ITERATORS[i], "0:" + str(d))
             for i, d in enumerate(chain.dimensions)])


def make_stream_name(src_name, dst_name):
    return src_name + "_to_" + dst_name


def generate_sdfg(name, chain):

    sdfg = SDFG(name)

    pre_state = sdfg.add_state("initialize")
    state = sdfg.add_state("compute")
    post_state = sdfg.add_state("finalize")

    sdfg.add_edge(pre_state, state, InterstateEdge())
    sdfg.add_edge(state, post_state, InterstateEdge())

    def add_pipe(edge):

        stream_name = make_stream_name(edge[0].name, edge[1].name)

        sdfg.add_stream(
            stream_name,
            DATA_TYPE,
            buffer_size=1,  # TODO: add queue size
            storage=StorageType.FPGA_Local,
            transient=True)

    def add_input(node):

        # Host-side array, which will be an input argument
        sdfg.add_array(node.name + "_host", chain.dimensions, DATA_TYPE)

        # Device-side copy
        sdfg.add_array(
            node.name,
            chain.dimensions,
            DATA_TYPE,
            storage=StorageType.FPGA_Global,
            transient=True)
        access_node = state.add_read(node.name)

        iterators = make_iterators(chain)

        # Copy data to the FPGA
        copy_host = pre_state.add_read(node.name + "_host")
        copy_fpga = pre_state.add_write(node.name)
        pre_state.add_memlet_path(
            copy_host,
            copy_fpga,
            memlet=Memlet.simple(
                copy_fpga,
                ", ".join(iterators.values()),
                num_accesses=functools.reduce(operator.mul, chain.dimensions,
                                              1)))

        entry, exit = state.add_map(
            "read_" + node.name, iterators, schedule=ScheduleType.FPGA_Device)

        # Sort to get deterministic output
        outputs = sorted([e[1].name for e in chain.graph.out_edges(node)])

        out_memlets = ["_" + o for o in outputs]

        tasklet_code = "\n".join(
            ["{} = memory".format(o) for o in out_memlets])

        tasklet = state.add_tasklet("read_" + node.name, {"memory"},
                                    out_memlets, tasklet_code)

        state.add_memlet_path(
            access_node,
            entry,
            tasklet,
            dst_conn="memory",
            memlet=Memlet.simple(
                node.name, ", ".join(iterators.keys()), num_accesses=1))

        # Add memlets to all FIFOs connecting to compute units
        for out_name, out_memlet in zip(outputs, out_memlets):
            stream_name = make_stream_name(node.name, out_name)
            write_node = state.add_write(stream_name)
            state.add_memlet_path(
                tasklet,
                exit,
                write_node,
                src_conn=out_memlet,
                memlet=Memlet.simple(stream_name, "0", num_accesses=1))

    def add_output(node):

        # Host-side array, which will be an output argument
        sdfg.add_array(node.name + "_host", chain.dimensions, DATA_TYPE)

        # Device-side copy
        sdfg.add_array(
            node.name,
            chain.dimensions,
            DATA_TYPE,
            storage=StorageType.FPGA_Global,
            transient=True)
        write_node = state.add_write(node.name)

        iterators = make_iterators(chain)

        # Copy data to the FPGA
        copy_fpga = post_state.add_read(node.name)
        copy_host = post_state.add_write(node.name + "_host")
        post_state.add_memlet_path(
            copy_fpga,
            copy_host,
            memlet=Memlet.simple(
                copy_host,
                ", ".join(iterators.values()),
                num_accesses=functools.reduce(operator.mul, chain.dimensions,
                                              1)))

        entry, exit = state.add_map(
            "write_" + node.name, iterators, schedule=ScheduleType.FPGA_Device)

        src = chain.graph.in_edges(node)
        if len(src) > 1:
            raise RuntimeError("Only one writer per output supported")
        src = next(iter(src))[0]

        in_memlet = "_" + src.name

        tasklet_code = "memory = " + in_memlet

        tasklet = state.add_tasklet("write_" + node.name, {in_memlet},
                                    {"memory"}, tasklet_code)

        stream_name = make_stream_name(src.name, node.name)
        read_node = state.add_read(stream_name)

        state.add_memlet_path(
            read_node,
            entry,
            tasklet,
            dst_conn=in_memlet,
            memlet=Memlet.simple(stream_name, "0", num_accesses=1))

        state.add_memlet_path(
            tasklet,
            exit,
            write_node,
            src_conn="memory",
            memlet=Memlet.simple(
                node.name, ", ".join(iterators.keys()), num_accesses=1))

    def add_kernel(node):

        iterators = make_iterators(chain)

        # Extract fields to read from memory and fields to buffer
        memory_accesses = []
        buffer_sizes = []
        buffer_accesses = []
        for field, accesses in node.kernel.graph.accesses.items():
            absolute_indices = [
                stencil_memory_index(i, node.kernel.dimensions)
                for i in accesses
            ]
            first_access = np.argmax(absolute_indices)
            buffer_size = absolute_indices[first_access] - min(absolute_indices)
            memory_accesses.append(field)
            if buffer_size > 0:
                buffer_sizes.append((field, buffer_size))
            for i, access in enumerate(accesses):
                if i != first_access:
                    buffer_accesses.append((field, accesses[i]))

        # Create buffers in the SDFG and the associated read and write nodes
        buffer_reads = []
        buffer_writes = []
        for field, bs in buffer_sizes:
            buffer_name = node.name + "_" + field + "_buffer"
            if bs > 1:
                sdfg.add_array(
                    buffer_name,
                    [bs],
                    DATA_TYPE,
                    storage=StorageType.FPGA_Local,
                    transient=True)
            else:
                sdfg.add_scalar(
                    buffer_name,
                    DATA_TYPE,
                    storage=StorageType.FPGA_Registers,
                    transient=True)
            buffer_reads.append(state.add_read(buffer_name))
            buffer_writes.append(state.add_write(buffer_name))

        entry, exit = state.add_map(
            node.name, iterators, schedule=ScheduleType.FPGA_Device)

        # Sort to get deterministic output
        outputs = sorted(e[1].name for e in chain.graph.out_edges(node))

        in_memlets_memory = [i + "_in" for i in memory_accesses]
        in_memlets_buffers = [name + "_buffer_in" for name, _ in buffer_sizes]
        out_memlets_memory = [o + "_out" for o in outputs]
        out_memlets_buffers = [
            name + "_buffer_out" for name, _ in buffer_sizes
        ]

        tasklet_code = "\n".join(
            itertools.chain([o + " = 0" for o in out_memlets_memory], [
                o + "{} = 0".format("[0]" if buffer_sizes[i][1] > 1 else "")
                for i, o in enumerate(out_memlets_buffers)
            ]))

        tasklet = state.add_tasklet(
            node.name, in_memlets_memory + in_memlets_buffers,
            out_memlets_memory + out_memlets_buffers, tasklet_code)

        for in_name in memory_accesses:
            stream_name = make_stream_name(in_name, node.name)
            read_node = state.add_read(stream_name)
            state.add_memlet_path(
                read_node,
                entry,
                tasklet,
                dst_conn=in_name + "_in",
                memlet=Memlet.simple(stream_name, "0", num_accesses=1))

        for buff, (field_name, size) in zip(buffer_reads, buffer_sizes):
            state.add_memlet_path(
                buff,
                entry,
                tasklet,
                dst_conn=field_name + "_buffer_in",
                memlet=Memlet.simple(
                    buff.data, "0:{}".format(size), num_accesses=-1))

        for out_name in outputs:
            stream_name = make_stream_name(node.name, out_name)
            write_node = state.add_write(stream_name)
            state.add_memlet_path(
                tasklet,
                exit,
                write_node,
                src_conn=out_name + "_out",
                memlet=Memlet.simple(stream_name, "0", num_accesses=1))

        for buff, (field_name, size) in zip(buffer_writes, buffer_sizes):
            state.add_memlet_path(
                tasklet,
                exit,
                buff,
                src_conn=field_name + "_buffer_out",
                memlet=Memlet.simple(
                    buff.data, "0:{}".format(size), num_accesses=-1))

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

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_file")

    args = parser.parse_args()

    name = os.path.basename(args.stencil_file)
    name = re.match("[^\.]+", name).group(0)

    chain = KernelChainGraph(args.stencil_file)

    chain.plot_graph(name + ".pdf")

    sdfg = generate_sdfg(name, chain)

    sdfg.draw_to_file()

    sdfg.compile()
