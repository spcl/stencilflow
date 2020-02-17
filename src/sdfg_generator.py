#!/usr/bin/env python3
# encoding: utf-8
"""
BSD 3-Clause License

Copyright (c) 2018-2020, Johannes de Fine Licht
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Johannes de Fine Licht"
__copyright__ = "Copyright 2018-2020, StencilFlow"
__license__ = "BSD-3-Clause"

import argparse
import collections
import functools
import itertools
import operator
import os
import re

import dace
import dace.codegen.targets.fpga
import numpy as np
from dace.graph.edges import InterstateEdge
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.dtypes import ScheduleType, StorageType, Language

import helper
from kernel_chain_graph import Kernel, Input, Output, KernelChainGraph

import stencil
from stencil.fpga import make_iterators

ITERATORS = ["i", "j", "k"]


def make_stream_name(src_name, dst_name):
    return src_name + "_to_" + dst_name


def generate_sdfg(name, chain):
    sdfg = SDFG(name)

    pre_state = sdfg.add_state("initialize")
    state = sdfg.add_state("compute")
    post_state = sdfg.add_state("finalize")

    sdfg.add_edge(pre_state, state, InterstateEdge())
    sdfg.add_edge(state, post_state, InterstateEdge())

    # TODO: For some reason, we put fake entries into the shape when the
    # dimensions in less than 3. Have to remove them here.
    dimensions_to_skip = len(chain.dimensions) - chain.kernel_dimensions
    shape = np.array(chain.dimensions)[dimensions_to_skip:]
    parameters = np.array(ITERATORS)[dimensions_to_skip:]
    # Only iterate over dimensions larger than 1, the rest will be added to the
    # SDFG as symbols that must be passed from outside.
    iterator_mask = shape > 1  # Dimensions we need to iterate over
    iterators = make_iterators(shape[iterator_mask],
                               parameters=parameters[iterator_mask])
    memlet_indices = [
        iterators[k] if iterator_mask[i] else k
        for i, k in enumerate(parameters)
    ]
    memcopy_accesses = str(
        functools.reduce(operator.mul, shape[iterator_mask], 1))

    def add_pipe(sdfg, edge):

        stream_name = make_stream_name(edge[0].name, edge[1].name)

        sdfg.add_stream(stream_name,
                        edge[0].data_type,
                        buffer_size=edge[2]["channel"]["delay_buffer"].maxsize,
                        storage=StorageType.FPGA_Local,
                        transient=True)

    def add_input(node):

        # Host-side array, which will be an input argument
        sdfg.add_array(node.name + "_host", shape, node.data_type)

        # Device-side copy
        sdfg.add_array(node.name,
                       shape,
                       node.data_type,
                       storage=StorageType.FPGA_Global,
                       transient=True)
        access_node = state.add_read(node.name)

        # Copy data to the FPGA
        copy_host = pre_state.add_read(node.name + "_host")
        copy_fpga = pre_state.add_write(node.name)
        pre_state.add_memlet_path(copy_host,
                                  copy_fpga,
                                  memlet=Memlet.simple(
                                      copy_fpga,
                                      ", ".join(memlet_indices),
                                      num_accesses=memcopy_accesses))

        entry, exit = state.add_map("read_" + node.name,
                                    iterators,
                                    schedule=ScheduleType.FPGA_Device)

        # Sort to get deterministic output
        outputs = sorted([e[1].name for e in chain.graph.out_edges(node)])

        out_memlets = ["_" + o for o in outputs]

        tasklet_code = "\n".join(
            ["{} = memory".format(o) for o in out_memlets])

        tasklet = state.add_tasklet("read_" + node.name, {"memory"},
                                    out_memlets, tasklet_code)

        state.add_memlet_path(access_node,
                              entry,
                              tasklet,
                              dst_conn="memory",
                              memlet=Memlet.simple(node.name,
                                                   ", ".join(parameters),
                                                   num_accesses=1))

        # Add memlets to all FIFOs connecting to compute units
        for out_name, out_memlet in zip(outputs, out_memlets):
            stream_name = make_stream_name(node.name, out_name)
            write_node = state.add_write(stream_name)
            state.add_memlet_path(tasklet,
                                  exit,
                                  write_node,
                                  src_conn=out_memlet,
                                  memlet=Memlet.simple(stream_name,
                                                       "0",
                                                       num_accesses=1))

    def add_output(node):

        # Host-side array, which will be an output argument
        sdfg.add_array(node.name + "_host", shape, node.data_type)

        # Device-side copy
        sdfg.add_array(node.name,
                       shape,
                       node.data_type,
                       storage=StorageType.FPGA_Global,
                       transient=True)
        write_node = state.add_write(node.name)

        # Copy data to the FPGA
        copy_fpga = post_state.add_read(node.name)
        copy_host = post_state.add_write(node.name + "_host")
        post_state.add_memlet_path(copy_fpga,
                                   copy_host,
                                   memlet=Memlet.simple(
                                       copy_host,
                                       ", ".join(memlet_indices),
                                       num_accesses=memcopy_accesses))

        entry, exit = state.add_map("write_" + node.name,
                                    iterators,
                                    schedule=ScheduleType.FPGA_Device)

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

        state.add_memlet_path(read_node,
                              entry,
                              tasklet,
                              dst_conn=in_memlet,
                              memlet=Memlet.simple(stream_name,
                                                   "0",
                                                   num_accesses=1))

        state.add_memlet_path(tasklet,
                              exit,
                              write_node,
                              src_conn="memory",
                              memlet=Memlet.simple(node.name,
                                                   ", ".join(parameters),
                                                   num_accesses=1))

    def add_kernel(node):

        # Enrich accesses with the names of the corresponding input connectors
        input_to_connector = collections.OrderedDict(
            (k, "_" + k) for k in node.graph.accesses)
        accesses = collections.OrderedDict(
            (k, [x[dimensions_to_skip:] for x in v]) for k, v in zip(
                input_to_connector.values(), node.graph.accesses.values()))

        # Map output field to output connector
        output_field = [e[1].name for e in chain.graph.out_edges(node)]
        if len(output_field) != 1:
            raise ValueError(
                "Expected single output edge, got: {}".format(output_field))
        output_field = output_field[0]
        output_connector = "_" + node.name
        output_dict = collections.OrderedDict([(output_connector,
                                                [0] * len(shape))])

        # Generate the C code. We unfortunately cannot pass the Python
        # directly, because OpenCL doesn't support auto.
        output_ctype = node.outputs[output_field]["data_type"].ctype
        code = "\n".join([
            output_ctype + " " + re.sub(
                r"\b{}\b".format(node.name), "{}[{}]".format(
                    output_connector, ", ".join("0" for _ in shape)),
                expr.strip()) + ";"
            for expr in node.generate_relative_access_kernel_string(
                relative_to_center=True,
                flatten_index=False,
                output_dimensions=len(shape)).split(";")
        ])
        # Replace input fields with the connector name.
        for f, c in input_to_connector.items():
            code = re.sub(r"\b{}\b".format(f), c, code)

        # Replace input fields with the connector name.
        boundary_conditions = {
            input_to_connector[f]: bc
            for f, bc in node.boundary_conditions.items()
        }

        # Replace "type" with "btype" to avoid problems with DaCe deserialize
        for field, bc in boundary_conditions.items():
            if "type" in bc:
                bc["btype"] = bc["type"]
                del bc["type"]

        stencil_node = stencil.Stencil(node.name, tuple(shape), accesses,
                                       output_dict, boundary_conditions, code)
        stencil_node.implementation = "FPGA"
        state.add_node(stencil_node)

        # Add read nodes and memlets
        for field_name, connector in input_to_connector.items():

            stream_name = make_stream_name(field_name, node.name)

            # Outer memory read
            read_node = state.add_read(stream_name)
            state.add_memlet_path(read_node,
                                  stencil_node,
                                  dst_conn=connector,
                                  memlet=Memlet.simple(stream_name,
                                                       "0",
                                                       num_accesses=-1))

        # Add write node and memlet
        stream_name = make_stream_name(node.name, output_field)

        # Outer write
        write_node = state.add_write(stream_name)
        state.add_memlet_path(stencil_node,
                              write_node,
                              src_conn=output_connector,
                              memlet=Memlet.simple(stream_name,
                                                   "0",
                                                   num_accesses=-1))

    # First generate all connections between kernels and memories
    for link in chain.graph.edges(data=True):
        add_pipe(sdfg, link)

    # Now generate all memory access functions so arrays are registered
    for node in chain.graph.nodes():
        if isinstance(node, Input):
            add_input(node)
        elif isinstance(node, Output):
            add_output(node)
        elif isinstance(node, Kernel):
            # Generate these separately after
            pass
        else:
            raise RuntimeError("Unexpected node type: {}".format(
                node.node_type))

    # Finally generate the compute kernels
    for node in chain.graph.nodes():
        if isinstance(node, Kernel):
            add_kernel(node)

    return sdfg


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_input",
                        help="Stencil description file (.json)")
    parser.add_argument("sdfg_output", help="Output SDFG file (.sdfg)")
    parser.add_argument("--plot-graph", dest="plot-graph", action="store_true")
    parser.add_argument("--plot-sdfg", dest="plot-sdfg", action="store_true")
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Compile the SDFG for verification/debugging purposes.")

    args = parser.parse_args()

    name = os.path.basename(args.stencil_input)
    name = re.match("[^\.]+", name).group(0)

    chain = KernelChainGraph(args.stencil_input)

    if getattr(args, "plot-graph"):
        chain.plot_graph(name + ".pdf")

    sdfg = generate_sdfg(name, chain)

    if getattr(args, "plot-sdfg"):
        chain.plot_graph(name + ".pdf")

    sdfg.save(args.sdfg_output)
    print("SDFG saved to: " + args.sdfg_output)

    if args.compile:
        dace.Config.set("compiler", "fpga_vendor", value="intel_fpga")
        sdfg.compile()
