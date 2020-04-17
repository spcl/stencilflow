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

import stencilflow
from stencilflow.kernel import Kernel
from stencilflow.input import Input
from stencilflow.output import Output

import stencilflow.stencil as stencil
from stencilflow.stencil.fpga import make_iterators

import networkx as nx

MINIMUM_CHANNEL_DEPTH = 32


def make_stream_name(src_name, dst_name):
    return src_name + "_to_" + dst_name


def _generate_init(chain):

    # TODO: For some reason, we put fake entries into the shape when the
    # dimensions in less than 3. Have to remove them here.
    dimensions_to_skip = len(chain.dimensions) - chain.kernel_dimensions
    shape = chain.dimensions[dimensions_to_skip:]
    vector_length = chain.vectorization
    if vector_length > 1:
        if shape[-1] % vector_length != 0:
            raise ValueError("Shape not divisible by vectorization width")
    parameters = stencilflow.ITERATORS[dimensions_to_skip:]
    # Only iterate over dimensions larger than 1, the rest will be added to the
    # SDFG as symbols that must be passed from outside.
    iterator_mask = [s > 1 for s in shape]  # Dimensions to iterate over
    iterators = make_iterators(
        [shape[i] for i, m in enumerate(iterator_mask) if m],
        parameters=[parameters[i] for i, m in enumerate(iterator_mask) if m])
    memcopy_indices = [
        iterators[k] if iterator_mask[i] else k
        for i, k in enumerate(parameters)
    ]
    if vector_length > 1:
        iterators[parameters[-1]] += "/{}".format(vector_length)
    memcopy_accesses = str(
        functools.reduce(operator.mul,
                         [shape[i] for i, m in enumerate(iterator_mask) if m],
                         1))

    return (dimensions_to_skip, shape, vector_length, parameters, iterators,
            memcopy_indices, memcopy_accesses)


def _generate_stencil(node, chain, shape, dimensions_to_skip):

    # Enrich accesses with the names of the corresponding input connectors
    input_dims = {
        k: [i in (node.inputs[k]["input_dim"])
            for i in stencilflow.ITERATORS] if "input_dim" in node.inputs[k]
        and node.inputs[k]["input_dim"] is not None else [True] * len(shape)
        for k in node.graph.accesses
    }
    input_to_connector = collections.OrderedDict(
        (k, "_" + k if any(dims) else k) for k, dims in input_dims.items())
    accesses = collections.OrderedDict((conn, (input_dims[name], [
        tuple(np.array(x[dimensions_to_skip:])[input_dims[name]])
        for x in node.graph.accesses[name]
    ])) for name, conn in zip(input_to_connector.keys(),
                              input_to_connector.values()))

    # Map output field to output connector
    output_to_connector = collections.OrderedDict(
        (e[1].name, "_" + e[1].name) for e in chain.graph.out_edges(node))
    output_dict = collections.OrderedDict([
        (oc, [0] * len(shape)) for oc in output_to_connector.values()
    ])

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


def _get_input_parameters(input_node, global_parameters, global_vector_length):
    """Determines the iterators and vector length for a given input."""
    for output in input_node.outputs.values():
        try:
            input_pars = output["input_dim"][:]
            vector_length = (global_vector_length
                             if input_pars[-1] == global_parameters[-1] else 1)
            # Just needed any output to retrieve the dimensions
            return input_pars, vector_length
        except (KeyError, TypeError):
            pass  # input_dim is not defined or is None
    return global_parameters, global_vector_length


def _add_pipe(sdfg, edge, parameters, vector_length):

    stream_name = make_stream_name(edge[0].name, edge[1].name)

    if isinstance(edge[0], stencilflow.input.Input):
        parameters, vector_length = _get_input_parameters(
            edge[0], parameters, vector_length)

    sdfg.add_stream(
        stream_name,
        edge[0].data_type,
        # Always maintain some channel depth to have greater stall tolerance
        buffer_size=max(MINIMUM_CHANNEL_DEPTH,
                        edge[2]["channel"]["delay_buffer"].maxsize),
        storage=StorageType.FPGA_Local,
        transient=True,
        veclen=vector_length)


def generate_sdfg(name, chain):
    sdfg = SDFG(name)

    for k, v in chain.constants.items():
        sdfg.add_constant(k, v["value"], dace.data.Scalar(v["data_type"]))

    pre_state = sdfg.add_state("initialize")
    state = sdfg.add_state("compute")
    post_state = sdfg.add_state("finalize")

    sdfg.add_edge(pre_state, state, InterstateEdge())
    sdfg.add_edge(state, post_state, InterstateEdge())

    (dimensions_to_skip, shape, vector_length, parameters, iterators,
     memcopy_indices, memcopy_accesses) = _generate_init(chain)

    def add_input(node):

        # Collapse iterators and shape if input is lower dimensional
        for output in node.outputs.values():
            try:
                input_pars = output["input_dim"][:]
            except (KeyError, TypeError):
                input_pars = list(parameters)  # Copy
            break  # Just needed any output to retrieve the dimensions
        else:
            raise ValueError("Input {} is not connected to anything.".format(
                node.name))
        # If scalar, just add a symbol
        if len(input_pars) == 0:
            sdfg.add_symbol(node.name, node.data_type, override_dtype=True)
            return  # We're done
        input_shape = [shape[list(parameters).index(i)] for i in input_pars]
        input_accesses = str(functools.reduce(operator.mul, input_shape, 1))
        # Only vectorize the read if the innermost dimensions is read
        input_vector_length = (vector_length
                               if input_pars[-1] == parameters[-1] else 1)
        input_iterators = collections.OrderedDict(
            (k, v) for k, v in iterators.items() if k in input_pars)

        # Host-side array, which will be an input argument
        sdfg.add_array(node.name + "_host", shape, node.data_type)

        # Device-side copy
        sdfg.add_array(node.name,
                       input_shape,
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
                                      ", ".join(memcopy_indices),
                                      num_accesses=input_accesses,
                                      veclen=input_vector_length))

        entry, exit = state.add_map("read_" + node.name,
                                    input_iterators,
                                    schedule=ScheduleType.FPGA_Device)

        # Sort to get deterministic output
        outputs = sorted([e[1].name for e in chain.graph.out_edges(node)])

        out_memlets = ["_" + o for o in outputs]

        tasklet_code = "\n".join(
            ["{} = memory".format(o) for o in out_memlets])

        tasklet = state.add_tasklet("read_" + node.name, {"memory"},
                                    out_memlets, tasklet_code)

        vectorized_pars = input_pars
        if input_vector_length > 1:
            vectorized_pars[-1] = "{}*{}".format(input_vector_length,
                                                 vectorized_pars[-1])
        state.add_memlet_path(access_node,
                              entry,
                              tasklet,
                              dst_conn="memory",
                              memlet=Memlet.simple(node.name,
                                                   ", ".join(vectorized_pars),
                                                   num_accesses=1,
                                                   veclen=input_vector_length))

        # Add memlets to all FIFOs connecting to compute units
        for out_name, out_memlet in zip(outputs, out_memlets):
            stream_name = make_stream_name(node.name, out_name)
            write_node = state.add_write(stream_name)
            state.add_memlet_path(tasklet,
                                  exit,
                                  write_node,
                                  src_conn=out_memlet,
                                  memlet=Memlet.simple(
                                      stream_name,
                                      "0",
                                      num_accesses=1,
                                      veclen=input_vector_length))

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

        # Copy data to the host
        copy_fpga = post_state.add_read(node.name)
        copy_host = post_state.add_write(node.name + "_host")
        post_state.add_memlet_path(copy_fpga,
                                   copy_host,
                                   memlet=Memlet.simple(
                                       copy_host,
                                       ", ".join(memcopy_indices),
                                       num_accesses=memcopy_accesses,
                                       veclen=vector_length))

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

        vectorized_pars = parameters
        if vector_length > 1:
            vectorized_pars[-1] = "{}*{}".format(vector_length,
                                                 vectorized_pars[-1])

        stream_name = make_stream_name(src.name, node.name)
        read_node = state.add_read(stream_name)

        state.add_memlet_path(read_node,
                              entry,
                              tasklet,
                              dst_conn=in_memlet,
                              memlet=Memlet.simple(stream_name,
                                                   "0",
                                                   num_accesses=1,
                                                   veclen=vector_length))

        state.add_memlet_path(tasklet,
                              exit,
                              write_node,
                              src_conn="memory",
                              memlet=Memlet.simple(node.name,
                                                   ", ".join(parameters),
                                                   num_accesses=1,
                                                   veclen=vector_length))

    def add_kernel(node):

        (stencil_node, input_to_connector,
         output_to_connector) = _generate_stencil(node, chain, shape,
                                                  dimensions_to_skip)
        stencil_node.implementation = "FPGA"
        state.add_node(stencil_node)

        # Add read nodes and memlets
        for field_name, connector in input_to_connector.items():

            # Scalars are symbols rather than data nodes
            input_vector_length = vector_length
            try:
                if len(node.inputs[field_name]["input_dim"]) == 0:
                    continue
                else:
                    # If the innermost dimension of this field is not the
                    # vectorized one, read it as scalars
                    if (node.inputs[field_name]["input_dim"][-1] !=
                            parameters[-1]):
                        input_vector_length = 1
            except (KeyError, TypeError):
                pass  # input_dim is not defined or is None

            stream_name = make_stream_name(field_name, node.name)

            # Outer memory read
            read_node = state.add_read(stream_name)
            state.add_memlet_path(read_node,
                                  stencil_node,
                                  dst_conn=connector,
                                  memlet=Memlet.simple(
                                      stream_name,
                                      "0",
                                      num_accesses=memcopy_accesses,
                                      veclen=input_vector_length))

        # Add read nodes and memlets
        for output_name, connector in output_to_connector.items():

            # Add write node and memlet
            stream_name = make_stream_name(node.name, output_name)

            # Outer write
            write_node = state.add_write(stream_name)
            state.add_memlet_path(stencil_node,
                                  write_node,
                                  src_conn=connector,
                                  memlet=Memlet.simple(
                                      stream_name,
                                      "0",
                                      num_accesses=memcopy_accesses,
                                      veclen=vector_length))

    # First generate all connections between kernels and memories
    for link in chain.graph.edges(data=True):
        _add_pipe(sdfg, link, parameters, vector_length)

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

    (dimensions_to_skip, shape, vector_length, parameters, iterators,
     memcopy_indices, memcopy_accesses) = _generate_init(chain)

    prev_state = sdfg.add_state("init")

    # Throw vectorization in the bin for the reference code
    vector_length = 1

    shape = tuple(map(int, shape))

    for node in chain.graph.nodes():
        if isinstance(node, Input) or isinstance(node, Output):
            sdfg.add_array(node.name, shape, node.data_type)

    for link in chain.graph.edges(data=True):
        if link[0].name not in sdfg.arrays:
            sdfg.add_array(link[0].name,
                           shape,
                           link[0].data_type,
                           transient=True)

    # Enforce dependencies via topological sort
    for node in nx.topological_sort(chain.graph):

        if not isinstance(node, Kernel):
            continue

        state = sdfg.add_state(node.name)
        sdfg.add_edge(prev_state, state, dace.InterstateEdge())

        (stencil_node, input_to_connector,
         output_to_connector) = _generate_stencil(node, chain, shape,
                                                  dimensions_to_skip)
        stencil_node.implementation = "CPU"

        for field, connector in input_to_connector.items():

            # Outer memory read
            read_node = state.add_read(field)
            state.add_memlet_path(read_node,
                                  stencil_node,
                                  dst_conn=connector,
                                  memlet=Memlet.simple(
                                      field, ", ".join(
                                          "0:{}".format(s)
                                          for s in sdfg.data(field).shape)))

        for _, connector in output_to_connector.items():

            # Outer write
            write_node = state.add_write(node.name)
            state.add_memlet_path(stencil_node,
                                  write_node,
                                  src_conn=connector,
                                  memlet=Memlet.simple(
                                      node.name, ", ".join(
                                          "0:{}".format(s)
                                          for s in sdfg.data(field).shape)))

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
