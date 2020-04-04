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
import ast
import astunparse
import collections
import copy
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

from stencilflow.kernel import Kernel
from stencilflow.input import Input
from stencilflow.output import Output

import stencilflow.stencil as stencil
from stencilflow.stencil.fpga import make_iterators

import networkx as nx

ITERATORS = ["i", "j", "k"]


def make_stream_name(src_name, dst_name):
    return src_name + "_to_" + dst_name


def _generate_init(chain):

    # TODO: For some reason, we put fake entries into the shape when the
    # dimensions in less than 3. Have to remove them here.
    dimensions_to_skip = len(chain.dimensions) - chain.kernel_dimensions
    shape = np.array(chain.dimensions)[dimensions_to_skip:]
    parameters = np.array(ITERATORS)[dimensions_to_skip:]
    # Only iterate over dimensions larger than 1, the rest will be added to the
    # SDFG as symbols that must be passed from outside.
    iterator_mask = shape > 1  # Dimensions we need to iterate over
    iterators = make_iterators(
        shape[iterator_mask], parameters=parameters[iterator_mask])
    memlet_indices = [
        iterators[k] if iterator_mask[i] else k
        for i, k in enumerate(parameters)
    ]
    memcopy_accesses = str(
        functools.reduce(operator.mul, shape[iterator_mask], 1))

    return (dimensions_to_skip, shape, parameters, iterators, memlet_indices,
            memcopy_accesses)


def _generate_stencil(node, chain, shape, dimensions_to_skip):

    # Enrich accesses with the names of the corresponding input connectors
    input_to_connector = collections.OrderedDict(
        (k, "_" + k) for k in node.graph.accesses)
    accesses = collections.OrderedDict(
        (k, ([True] * len(shape), [tuple(x[dimensions_to_skip:]) for x in v]))
        for k, v in zip(input_to_connector.values(),
                        node.graph.accesses.values()))

    # Map output field to output connector
    output_to_connector = collections.OrderedDict(
        (e[1].name, "_" + e[1].name) for e in chain.graph.out_edges(node))
    output_dict = collections.OrderedDict(
        [(oc, [0] * len(shape)) for oc in output_to_connector.values()])

    # Grab code from StencilFlow
    code = node.generate_relative_access_kernel_string(
        relative_to_center=True,
        flatten_index=False,
        python_syntax=True,
        output_dimensions=len(shape))

    # Add writes to each output
    code += "\n" + "\n".join(
        "{}[{}] = {}".format(oc, ", ".join(["0"] * len(shape)), node.name)
        for oc in output_to_connector.values())

    # We need to rename field accesses to their input connectors
    class _StencilFlowVisitor(ast.NodeTransformer):
        def __init__(self, input_to_connector):
            self.input_to_connector = input_to_connector

        def visit_Subscript(self, node: ast.Subscript):
            field = node.value.id
            if field in self.input_to_connector:
                # Rename to connector name
                node.value.id = input_to_connector[field]
            return node

    # Transform the code using the visitor above
    ast_visitor = _StencilFlowVisitor(input_to_connector)
    old_ast = ast.parse(code)
    new_ast = ast_visitor.visit(old_ast)
    code = astunparse.unparse(new_ast)

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

    return stencil_node, input_to_connector, output_to_connector


def generate_sdfg(name, chain):
    sdfg = SDFG(name)

    pre_state = sdfg.add_state("initialize")
    state = sdfg.add_state("compute")
    post_state = sdfg.add_state("finalize")

    sdfg.add_edge(pre_state, state, InterstateEdge())
    sdfg.add_edge(state, post_state, InterstateEdge())

    (dimensions_to_skip, shape, parameters, iterators, memlet_indices,
     memcopy_accesses) = _generate_init(chain)

    def add_pipe(sdfg, edge):

        stream_name = make_stream_name(edge[0].name, edge[1].name)

        sdfg.add_stream(
            stream_name,
            edge[0].data_type,
            buffer_size=edge[2]["channel"]["delay_buffer"].maxsize,
            storage=StorageType.FPGA_Local,
            transient=True)

    def add_input(node):

        # Host-side array, which will be an input argument
        sdfg.add_array(node.name + "_host", shape, node.data_type)

        # Device-side copy
        sdfg.add_array(
            node.name,
            shape,
            node.data_type,
            storage=StorageType.FPGA_Global,
            transient=True)
        access_node = state.add_read(node.name)

        # Copy data to the FPGA
        copy_host = pre_state.add_read(node.name + "_host")
        copy_fpga = pre_state.add_write(node.name)
        pre_state.add_memlet_path(
            copy_host,
            copy_fpga,
            memlet=Memlet.simple(
                copy_fpga,
                ", ".join(memlet_indices),
                num_accesses=memcopy_accesses))

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
                node.name, ", ".join(parameters), num_accesses=1))

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
        sdfg.add_array(node.name + "_host", shape, node.data_type)

        # Device-side copy
        sdfg.add_array(
            node.name,
            shape,
            node.data_type,
            storage=StorageType.FPGA_Global,
            transient=True)
        write_node = state.add_write(node.name)

        # Copy data to the FPGA
        copy_fpga = post_state.add_read(node.name)
        copy_host = post_state.add_write(node.name + "_host")
        post_state.add_memlet_path(
            copy_fpga,
            copy_host,
            memlet=Memlet.simple(
                copy_host,
                ", ".join(memlet_indices),
                num_accesses=memcopy_accesses))

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
                node.name, ", ".join(parameters), num_accesses=1))

    def add_kernel(node):

        (stencil_node,
         input_to_connector, output_to_connector) = _generate_stencil(
             node, chain, shape, dimensions_to_skip)
        stencil_node.implementation = "FPGA"
        state.add_node(stencil_node)

        # Add read nodes and memlets
        for field_name, connector in input_to_connector.items():

            stream_name = make_stream_name(field_name, node.name)

            # Outer memory read
            read_node = state.add_read(stream_name)
            state.add_memlet_path(
                read_node,
                stencil_node,
                dst_conn=connector,
                memlet=Memlet.simple(
                    stream_name, "0", num_accesses=memcopy_accesses))

        # Add read nodes and memlets
        for output_name, connector in output_to_connector.items():

            # Add write node and memlet
            stream_name = make_stream_name(node.name, output_name)

            # Outer write
            write_node = state.add_write(stream_name)
            state.add_memlet_path(
                stencil_node,
                write_node,
                src_conn=connector,
                memlet=Memlet.simple(
                    stream_name, "0", num_accesses=memcopy_accesses))

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


def generate_reference(name, chain):
    """Generates a simple, unoptimized SDFG to run on the CPU, for verification
       purposes."""

    sdfg = SDFG(name)

    (dimensions_to_skip, shape, parameters, iterators, memlet_indices,
     memcopy_accesses) = _generate_init(chain)

    prev_state = sdfg.add_state("init")

    shape = tuple(map(int, shape))

    for node in chain.graph.nodes():
        if isinstance(node, Input) or isinstance(node, Output):
            sdfg.add_array(node.name, shape, node.data_type)

    for link in chain.graph.edges(data=True):
        if link[0].name not in sdfg.arrays:
            sdfg.add_array(
                link[0].name,
                shape,
                link[0].data_type,
                transient=True)

    # Enforce dependencies via topological sort
    for node in nx.topological_sort(chain.graph):

        if not isinstance(node, Kernel):
            continue

        state = sdfg.add_state(node.name)
        sdfg.add_edge(prev_state, state, dace.InterstateEdge())

        (stencil_node,
         input_to_connector, output_to_connector) = _generate_stencil(
             node, chain, shape, dimensions_to_skip)
        stencil_node.implementation = "CPU"

        for field, connector in input_to_connector.items():

            # Outer memory read
            read_node = state.add_read(field)
            state.add_memlet_path(
                read_node,
                stencil_node,
                dst_conn=connector,
                memlet=Memlet.simple(
                    field, ", ".join(
                        "0:{}".format(s) for s in sdfg.data(field).shape)))

        for _, connector in output_to_connector.items():

            # Outer write
            write_node = state.add_write(node.name)
            state.add_memlet_path(
                stencil_node,
                write_node,
                src_conn=connector,
                memlet=Memlet.simple(
                    node.name, ", ".join(
                        "0:{}".format(s) for s in sdfg.data(field).shape)))

        prev_state = state

    return sdfg


def _nodes_reachable_from(graph,
                          node,
                          split_data,
                          ancestors=True,
                          descendants=True):
    q = [node]
    seen = set()
    while len(q) > 0:
        n = q.pop()
        if n in seen:
            continue
        seen.add(n)
        # Don't cross splitting point
        is_split = hasattr(n, "data") and n.data == split_data
        if not is_split or ancestors:
            for reachable in graph.predecessors(n):
                if reachable not in seen:
                    q.append(reachable)
        if not is_split or descendants:
            for reachable in graph.successors(n):
                if reachable not in seen:
                    q.append(reachable)
    return seen


def _nodes_before_or_after(sdfg, split_state, split_data, after):
    import networkx as nx
    states = set()
    nodes = set()
    states_to_search = collections.deque([split_state])
    seen = set()
    data_names = {split_data}
    while len(states_to_search) > 0:
        state = states_to_search.popleft()
        if state in seen:
            continue
        seen.add(state)
        fixed_point = False
        while not fixed_point:
            num_names = len(data_names)
            for n in state.data_nodes():
                if n.data in data_names:
                    states.add(state)
                    if after:
                        local_nodes = _nodes_reachable_from(
                            state, n, split_data, False, True)
                    else:
                        local_nodes = _nodes_reachable_from(
                            state, n, split_data, True, False)
                    for la in local_nodes:
                        if isinstance(la, dace.graph.nodes.AccessNode):
                            data_names.add(la.data)
                    nodes |= set((state, ln) for ln in local_nodes)
            fixed_point = num_names == len(data_names)
        next_states = itertools.chain(sdfg.successors(state),
                                      sdfg.predecessors(state))
        for state in next_states:
            if state not in seen:
                states_to_search.append(state)
    return states, nodes


def _remove_nodes_and_states(sdfg, states_to_keep, nodes_to_keep):
    node_to_id = {}
    # Create mapping that will not change as we modify the graph
    for state in sdfg:
        state_id = sdfg.node_id(state)
        node_to_id[state] = state_id
        for node in state:
            node_to_id[node] = (state_id, state.node_id(node))
    # Now remove the nodes that (in the original mapping) should not be kept
    for state in list(sdfg.states()):
        state_id = node_to_id[state]
        if state_id not in states_to_keep:
            sdfg.remove_node(state)
        else:
            for node in list(state.nodes()):
                node_id = node_to_id[node]
                if node_id not in nodes_to_keep:
                    state.remove_node(node)


def split_sdfg(sdfg, remote_stream, send_rank, receive_rank, port):
    """Split the input SDFG into two SDFGs connected by remote streams,
       to be executed in a multi-FPGA setup using SMI.
        :param sdfg: SDFG to split into two SDFGs.
        :param remote_stream: Stream data name (not node) to split on
        :param send_rank: Rank that will send
        :param receive_rank: Rank that will receive
        :param port: Port identifier
        :return: The two resulting SDFGs
    """

    # Locate read and write nodes in SDFG
    read_node = None
    read_state = None
    write_node = None
    write_state = None
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.graph.nodes.AccessNode):
            if node.data != remote_stream:
                continue
            if node.access == dace.AccessType.ReadOnly:
                if read_state is not None:
                    raise ValueError("Multiple reads found for: {}".format(
                        node.data))
                read_node = node
                read_state = state
            elif node.access == dace.AccessType.WriteOnly:
                if write_state is not None:
                    raise ValueError("Multiple writes found for: {}".format(
                        node.data))
                write_node = node
                write_state = state
            else:
                raise ValueError("Unsupported access type: {}".format(
                    node.access))
    if read_node is None or write_node is None:
        raise ValueError("Remote stream {} not found.".format(remote_stream))

    # Classify nodes into whether they appear before or after the split
    states_before, nodes_before = (_nodes_before_or_after(
        sdfg, read_state, remote_stream, False))
    states_after, nodes_after = (_nodes_before_or_after(
        sdfg, write_state, remote_stream, True))
    nodes_before.remove((read_state, read_node))
    nodes_after.remove((write_state, write_node))
    intersection = nodes_before & nodes_after
    if len(intersection) != 0:
        raise ValueError(
            "Node does not perfectly split SDFG, intersection is: {}".format(
                intersection))

    # Turn splitting stream into remote access nodes
    sdfg.data(read_node.data).storage = dace.dtypes.StorageType.FPGA_Remote
    sdfg.data(read_node.data).location["snd_rank"] = send_rank
    sdfg.data(read_node.data).location["port"] = port
    sdfg.data(write_node.data).storage = dace.dtypes.StorageType.FPGA_Remote
    sdfg.data(write_node.data).location["rcv_rank"] = receive_rank
    sdfg.data(write_node.data).location["port"] = port

    # Now duplicate the SDFG, and remove all nodes that don't belong in the
    # respectively side of the split
    name = sdfg.name
    sdfg_before = copy.deepcopy(sdfg)
    sdfg_after = copy.deepcopy(sdfg)
    sdfg_before._name = name + "_before"
    sdfg_after._name = name + "_after"
    nodes_before = set(
        (sdfg.node_id(s), s.node_id(n)) for s, n in nodes_before)
    nodes_after = set((sdfg.node_id(s), s.node_id(n)) for s, n in nodes_after)
    states_before = set(sdfg.node_id(s) for s in states_before)
    states_after = set(sdfg.node_id(s) for s in states_after)
    _remove_nodes_and_states(sdfg_before, states_before, nodes_before)
    _remove_nodes_and_states(sdfg_after, states_after, nodes_after)

    sdfg_before.validate()
    sdfg_after.validate()

    return sdfg_before, sdfg_after
