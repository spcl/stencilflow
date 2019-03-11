import argparse
import collections
import functools
import helper
import itertools
import operator
import numpy as np
import os
import re
from dace.graph.edges import InterstateEdge
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.types import ScheduleType, StorageType
from kernel_chain_graph import Kernel, Input, Output, KernelChainGraph

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

    def add_pipe(sdfg, edge):

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
        buffer_accesses = {}
        for field, accesses in node.graph.accesses.items():
            indices = [
                helper.dim_to_abs_val(i, node.dimensions)
                for i in helper.unique(accesses)
            ]
            buffer_size = (
                max(indices) - min(indices) + 1)
            memory_accesses.append(field)
            buffer_sizes.append((field, buffer_size))
            buffer_accesses[field] = [buffer_size - 1 - i for i in indices]

        entry, exit = state.add_map(
            node.name, iterators, schedule=ScheduleType.FPGA_Device)

        # Sort to get deterministic output
        outputs = sorted(e[1].name for e in chain.graph.out_edges(node))

        # Add nested SDFG to do 1) shift buffers 2) read from input 3) compute
        nested_sdfg = SDFG(node.name, parent=sdfg)
        nested_sdfg_tasklet = state.add_nested_sdfg(
            nested_sdfg, sdfg,
            ([name + "_in" for name in memory_accesses] +
             [name + "_buffer_in" for name, _ in buffer_sizes]),
            ([name + "_out" for name in outputs] +
             [name + "_buffer_out" for name, _ in buffer_sizes]),
            name=node.name)

        # Shift state, which shifts all buffers by one
        shift_state = nested_sdfg.add_state(node.name + "_shift")

        # Update state, which reads new values from memory
        update_state = nested_sdfg.add_state(node.name + "_update")

        # Compute state, which reads from input channels, performs the compute,
        # and writes to the output channel(s)
        compute_state = nested_sdfg.add_state(node.name + "_compute")
        tasklet_code = ""  # TODO
        compute_tasklet = compute_state.add_tasklet(
            node.name + "_compute",
            list(
                itertools.chain.from_iterable([[
                    "{}_buffer_in_{}".format(name, access)
                    for access in accesses
                ] for name, accesses in buffer_accesses.items()])),
            [name + "_inner_out" for name in outputs], tasklet_code)

        # Connect the three nested states
        nested_sdfg.add_edge(shift_state, update_state, InterstateEdge())
        nested_sdfg.add_edge(update_state, compute_state, InterstateEdge())

        for in_name, (field_name, size) in zip(memory_accesses, buffer_sizes):

            stream_name = make_stream_name(in_name, node.name)

            # Outer memory read
            read_node_outer = state.add_read(stream_name)
            state.add_memlet_path(
                read_node_outer,
                entry,
                nested_sdfg_tasklet,
                dst_conn=in_name + "_in",
                memlet=Memlet.simple(stream_name, "0", num_accesses=1))

            # Create inner memory pipe
            stream_name_inner = stream_name + "_inner"
            stream_inner = sdfg.arrays[stream_name].clone()
            stream_inner.transient = False
            nested_sdfg.add_datadesc(stream_name_inner, stream_inner)

            buffer_name_outer = "{}_{}_buffer".format(node.name, field_name)

            # Create buffer transient in outer SDFG
            if size > 1:
                desc_outer = sdfg.add_array(
                    buffer_name_outer,
                    [size],
                    DATA_TYPE,
                    storage=StorageType.FPGA_Local,
                    transient=True)
            else:
                desc_outer = sdfg.add_scalar(
                    buffer_name_outer,
                    DATA_TYPE,
                    storage=StorageType.FPGA_Registers,
                    transient=True)

            # Create read and write nodes
            read_node_outer = state.add_read(buffer_name_outer)
            write_node_outer = state.add_write(buffer_name_outer)

            # Outer memory read
            state.add_memlet_path(
                read_node_outer,
                entry,
                nested_sdfg_tasklet,
                dst_conn=field_name + "_buffer_in",
                memlet=Memlet.simple(
                    buffer_name_outer, "0:{}".format(size), num_accesses=-1))

            # Outer memory write
            state.add_memlet_path(
                nested_sdfg_tasklet,
                exit,
                write_node_outer,
                src_conn=field_name + "_buffer_out",
                memlet=Memlet.simple(
                    write_node_outer.data,
                    "0:{}".format(size),
                    num_accesses=-1))

            # Inner copy
            buffer_name_inner = buffer_name_outer + "_inner"
            desc_inner = desc_outer.clone()
            desc_inner.name = buffer_name_inner
            desc_inner.transient = False
            nested_sdfg.add_datadesc(buffer_name_inner, desc_inner)

            # Make shift state if necessary
            if size > 1:
                shift_read = shift_state.add_read(buffer_name_inner)
                shift_write = shift_state.add_write(buffer_name_inner)
                shift_state.add_memlet_path(
                    shift_read,
                    shift_write,
                    memlet=Memlet.simple(
                        shift_read.data,
                        "1:{} - 1".format(size),
                        other_subset_str="0:{} - 2".format(size),
                        num_accesses=size - 1))

            update_read = update_state.add_read(stream_name_inner)
            update_write = update_state.add_write(buffer_name_inner)
            update_state.add_memlet_path(
                update_read,
                update_write,
                memlet=Memlet.simple(
                    update_read.data,
                    "0",
                    other_subset_str="{} - 1".format(size)
                    if size > 1 else "0",
                    num_accesses=1))

            # Make compute state
            compute_read = compute_state.add_read(buffer_name_inner)
            for index in buffer_accesses[field_name]:
                compute_state.add_memlet_path(
                    compute_read,
                    compute_tasklet,
                    dst_conn=field_name + "_buffer_in_{}".format(index),
                    memlet=Memlet.simple(
                        compute_read.data, str(index), num_accesses=1))

        for out_name in outputs:

            # Outer write
            stream_name_outer = make_stream_name(node.name, out_name)
            write_node_outer = state.add_write(stream_name_outer)
            state.add_memlet_path(
                nested_sdfg_tasklet,
                exit,
                write_node_outer,
                src_conn=out_name + "_out",
                memlet=Memlet.simple(
                    write_node_outer.data, "0", num_accesses=-1))

            # Create inner stream
            stream_name_inner = stream_name_outer + "_inner"
            stream_inner = sdfg.arrays[stream_name_outer].clone()
            stream_inner.transient = False
            nested_sdfg.add_datadesc(stream_name_inner, stream_inner)

            # Inner write
            write_node_inner = compute_state.add_write(stream_name_inner)
            compute_state.add_memlet_path(
                compute_tasklet,
                write_node_inner,
                src_conn=out_name + "_inner_out",
                memlet=Memlet.simple(
                    write_node_inner.data, "0", num_accesses=-1))

    for link in chain.graph.edges():
        add_pipe(sdfg, link)

    for node in chain.graph.nodes():
        if isinstance(node, Input):
            add_input(node)
        elif isinstance(node, Output):
            add_output(node)
        elif isinstance(node, Kernel):
            add_kernel(node)
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
