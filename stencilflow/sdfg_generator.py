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
import warnings

import dace
import dace.codegen.targets.fpga
import numpy as np
from dace.sdfg.sdfg import InterstateEdge
from dace.memlet import Memlet
from dace.sdfg import SDFG
from dace.dtypes import ScheduleType, StorageType, Language

import stencilflow
from stencilflow.kernel import Kernel
from stencilflow.input import Input
from stencilflow.output import Output

import stencilflow.stencil as stencil
from stencilflow.stencil._common import make_iterators

import networkx as nx

MINIMUM_CHANNEL_DEPTH = 2048

NUM_BANKS = 4


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
    if vector_length > 1:
        iterators[parameters[-1]] += "//{}".format(vector_length)
    memcopy_indices = [
        iterators[k] if iterator_mask[i] else k
        for i, k in enumerate(parameters)
    ]
    memcopy_accesses = str(
        functools.reduce(operator.mul,
                         [shape[i] for i, m in enumerate(iterator_mask) if m],
                         1))

    return (dimensions_to_skip, shape, vector_length, parameters, iterators,
            memcopy_indices, memcopy_accesses)


def _generate_stencil(node, chain, shape, dimensions_to_skip):

    # Enrich accesses with the names of the corresponding input connectors
    input_dims = {
        k: [
            i in node.inputs[k]["input_dims"]
            for i in stencilflow.ITERATORS[dimensions_to_skip:]
        ] if "input_dims" in node.inputs[k]
        and node.inputs[k]["input_dims"] is not None else [True] * len(shape)
        for k in node.graph.accesses
    }
    input_to_connector = collections.OrderedDict(
        (k, k + "_in" if any(dims) else k) for k, dims in input_dims.items())
    accesses = collections.OrderedDict()
    for name, access_list in node.graph.accesses.items():
        indices = input_dims[name]
        conn = input_to_connector[name]
        accesses[conn] = (indices, [])
        num_dims = len(indices)
        for access in access_list:
            if len(access) > len(indices):
                access = access[len(access) - num_dims:]  # Trim
            accesses[conn][1].append(
                tuple(a for a, v in zip(access, indices) if v))

    # Map output field to output connector
    output_to_connector = collections.OrderedDict(
        (e[1].name, e[1].name + "_out") for e in chain.graph.out_edges(node))
    output_dict = collections.OrderedDict([
        (oc, [0] * len(shape)) for oc in output_to_connector.values()
    ])

    # Grab code from StencilFlow
    code = node.kernel_string

    # Add writes to each output
    code += "\n" + "\n".join("{}[{}] = {}".format(
        oc, ", ".join(stencilflow.ITERATORS[:len(shape)]), node.name)
                             for oc in output_to_connector.values())

    # We need to replace indices with relative ones, and rename field accesses
    # to their input connectors
    class _StencilFlowVisitor(ast.NodeTransformer):
        def __init__(self, input_to_connector, num_dims):
            self.input_to_connector = input_to_connector
            self.num_dims = num_dims

        @staticmethod
        def _index_to_offset(node):
            if isinstance(node, ast.Name):
                return 0
            elif isinstance(node, ast.BinOp):
                if isinstance(node.op, ast.Sub):
                    return -int(node.right.n)
                elif isinstance(node.op, ast.Add):
                    return int(node.right.n)
            raise TypeError("Unrecognized offset: {}".format(
                astunparse.unparse(node)))

        def visit_Subscript(self, node: ast.Subscript):
            # Rename to connector name
            field = node.value.id
            if field in self.input_to_connector:
                node.value.id = input_to_connector[field]
            # Convert [i, j + 1, k - 1] to [0, 1, -1]
            if isinstance(node.slice.value, ast.Tuple):
                indices = node.slice.value.elts
                if len(indices) > self.num_dims:
                    # Cut off extra dimensions added by analysis
                    indices = indices[len(indices) - self.num_dims:]
                # Negative indices show up as a UnaryOp, others as Num
                offsets = tuple(
                    map(_StencilFlowVisitor._index_to_offset, indices))
            else:
                # One dimensional access doesn't show up as a tuple
                offsets = (_StencilFlowVisitor._index_to_offset(
                    node.slice.value), )
            t = "({})".format(", ".join(map(str, offsets)))
            node.slice.value = ast.parse(t).body[0].value
            self.generic_visit(node)
            return node

    # Transform the code using the visitor above
    ast_visitor = _StencilFlowVisitor(input_to_connector, len(shape))
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

    # Truncate name if too long
    stencil_name = node.name
    if len(node.name) + 7 > 42:
        stencil_name = stencil_name[35:]

    stencil_node = stencil.Stencil(stencil_name, tuple(shape), accesses,
                                   output_dict, boundary_conditions, code)

    return stencil_node, input_to_connector, output_to_connector


def _get_input_parameters(input_node, global_parameters, global_vector_length):
    """Determines the iterators and vector length for a given input."""
    for output in input_node.outputs.values():
        try:
            input_pars = output["input_dims"][:]
            vector_length = (global_vector_length
                             if input_pars[-1] == global_parameters[-1] else 1)
            # Just needed any output to retrieve the dimensions
            return input_pars, vector_length
        except (KeyError, TypeError):
            pass  # input_dim is not defined or is None
    return global_parameters, global_vector_length


def _add_pipe(sdfg, edge, parameters, vector_length):

    src_name = edge[0].name
    dst_name = edge[1].name
    if isinstance(edge[0], stencilflow.input.Input):
        try:
            parameters, vector_length = _get_input_parameters(
                edge[0], parameters, vector_length)
        except IndexError:
            return  # This is a 0D input, don't create any pipe
        src_name = "read_" + src_name
    if isinstance(edge[1], stencilflow.output.Output):
        dst_name = "write_" + dst_name

    stream_name = "{}_to_{}".format(src_name, dst_name)

    sdfg.add_stream(
        stream_name,
        dace.dtypes.vector(edge[0].data_type, vector_length),
        # Always maintain some channel depth to have greater stall tolerance
        buffer_size=max(MINIMUM_CHANNEL_DEPTH,
                        edge[2]["channel"]["delay_buffer"].maxsize),
        storage=StorageType.FPGA_Local,
        transient=True)


def generate_sdfg(name, chain, synthetic_reads=False, specialize_scalars=False):
    sdfg = SDFG(name)

    for k, v in chain.constants.items():
        sdfg.add_constant(k, v["value"], dace.data.Scalar(v["data_type"]))

    if specialize_scalars:
        for k, v in chain.inputs.items():
            if len(v["input_dims"]) == 0:
                try:
                    val = stencilflow.load_array(v)
                except FileNotFoundError:
                    continue
                print(f"Specialized constant {k} to {val}.")
                sdfg.add_constant(k, val)

    pre_state = sdfg.add_state("initialize")
    state = sdfg.add_state("compute")
    post_state = sdfg.add_state("finalize")

    sdfg.add_edge(pre_state, state, InterstateEdge())
    sdfg.add_edge(state, post_state, InterstateEdge())

    (dimensions_to_skip, shape, vector_length, parameters, iterators,
     memcopy_indices, memcopy_accesses) = _generate_init(chain)
    vshape = list(shape)  # Copy
    if vector_length > 1:
        vshape[-1] //= vector_length

    def add_input(node, bank):

        # Collapse iterators and shape if input is lower dimensional
        for output in node.outputs.values():
            try:
                input_pars = output["input_dims"][:]
            except (KeyError, TypeError):
                input_pars = list(parameters)  # Copy
            break  # Just needed any output to retrieve the dimensions
        else:
            raise ValueError("Input {} is not connected to anything.".format(
                node.name))
        # If scalar, just add a symbol
        if len(input_pars) == 0:
            sdfg.add_symbol(node.name, node.data_type)
            return  # We're done
        input_shape = [shape[list(parameters).index(i)] for i in input_pars]
        is_lower_dim = len(input_shape) != len(shape)
        input_accesses = str(functools.reduce(operator.mul, input_shape, 1))
        # Only vectorize the read if the innermost dimensions is read
        input_vector_length = (vector_length
                               if input_pars[-1] == parameters[-1] else 1)
        input_vtype = (dace.dtypes.vector(node.data_type, input_vector_length)
                       if input_vector_length > 1 else node.data_type)
        # Always read 512-bit vectors from memory
        memory_veclen = 64 // node.data_type.bytes
        gearbox_factor = memory_veclen // input_vector_length
        memory_dtype = (input_vtype if is_lower_dim else dace.dtypes.vector(node.data_type, memory_veclen))

        input_vshape = list(input_shape)
        if input_vector_length > 1:
            input_vshape[-1] //= input_vector_length
        memory_shape = list(input_shape)
        memory_shape[-1] //= memory_veclen

        # Sort to get deterministic output
        outputs = sorted([e[1].name for e in chain.graph.out_edges(node)])

        out_memlets = ["_" + o for o in outputs]

        entry, exit = state.add_map("read_" + node.name,
                                    iterators,
                                    schedule=ScheduleType.FPGA_Device)

        if not synthetic_reads:  # Generate synthetic inputs without memory

            # Host-side array, which will be an input argument
            sdfg.add_array(node.name + "_host", input_shape, node.data_type)

            # Device-side copy
            _, array = sdfg.add_array(node.name,
                                      memory_shape,
                                      memory_dtype,
                                      storage=StorageType.FPGA_Global,
                                      transient=True)
            array.location["bank"] = bank
            access_node = state.add_read(node.name)

            # Copy data to the FPGA
            copy_host = pre_state.add_read(node.name + "_host")
            copy_fpga = pre_state.add_write(node.name)
            pre_state.add_memlet_path(copy_host,
                                      copy_fpga,
                                      memlet=Memlet(f"{copy_fpga.data}[{', '.join(f'0:{s}' for s in memory_shape)}]"))

            tasklet_code = "\n".join([f"{o} = memory" for o in out_memlets])

            tasklet = state.add_tasklet("read_" + node.name, {"memory"}, out_memlets, tasklet_code)

            vectorized_pars = input_pars
            # if input_vector_length > 1:
            #     vectorized_pars[-1] = "{}*{}".format(input_vector_length,
            #                                          vectorized_pars[-1])

            # Lower-dimensional arrays should buffer values and send them
            # multiple times
            if is_lower_dim:
                buffer_name = node.name + "_buffer"
                sdfg.add_array(buffer_name,
                               input_shape,
                               input_vtype,
                               storage=StorageType.FPGA_Local,
                               transient=True)
                buffer_node = state.add_access(buffer_name)
                buffer_entry, buffer_exit = state.add_map(
                    "buffer_" + node.name, {
                        k: "0:{}".format(v)
                        for k, v in zip(input_pars, input_shape)
                    },
                    schedule=dace.ScheduleType.FPGA_Device)
                buffer_tasklet = state.add_tasklet("buffer_" + node.name,
                                                   {"memory"}, {"buffer"},
                                                   "buffer = memory")
                state.add_memlet_path(access_node,
                                      buffer_entry,
                                      buffer_tasklet,
                                      dst_conn="memory",
                                      memlet=dace.Memlet.simple(
                                          access_node.data,
                                          ", ".join(vectorized_pars),
                                          num_accesses=1))
                state.add_memlet_path(buffer_tasklet,
                                      buffer_exit,
                                      buffer_node,
                                      src_conn="buffer",
                                      memlet=dace.Memlet.simple(
                                          buffer_node.data,
                                          ", ".join(input_pars),
                                          num_accesses=1))
                state.add_memlet_path(buffer_node,
                                      entry,
                                      tasklet,
                                      dst_conn="memory",
                                      memlet=dace.Memlet.simple(
                                          buffer_node.data,
                                          ", ".join(input_pars),
                                          num_accesses=1))
            else:

                # Read 512-bit vectors into a buffered stream
                buffer_iterators = copy.copy(iterators)
                buffer_iterators[parameters[-1]] += f"/{gearbox_factor}"
                buffer_entry, buffer_exit = state.add_map("buffer_" + node.name,
                                                          buffer_iterators,
                                                          schedule=dace.ScheduleType.FPGA_Device)
                buffer_tasklet = state.add_tasklet(f"buffer_{node.name}",
                                                   {"memory"}, {"to_gearbox"}, "to_gearbox = memory")
                gearbox_in_stream_name = f"{node.name}_gearbox_in"
                gearbox_out_stream_name = f"{node.name}_gearbox_out"
                sdfg.add_stream(gearbox_in_stream_name, memory_dtype, 512, storage=dace.StorageType.FPGA_Local, transient=True);
                gearbox_in_stream_write = state.add_write(gearbox_in_stream_name)
                state.add_memlet_path(access_node,
                                      buffer_entry,
                                      buffer_tasklet,
                                      dst_conn="memory",
                                      memlet=Memlet(f"{access_node.data}[{', '.join(input_pars)}]"))
                state.add_memlet_path(buffer_tasklet,
                                      buffer_exit,
                                      gearbox_in_stream_write,
                                      src_conn="to_gearbox",
                                      memlet=Memlet(f"{gearbox_in_stream_name}[0]"))

                # Gearbox into the expected vector width
                gearbox_buffer_name = f"{node.name}_gearbox_buffer"
                gearbox_in_stream_read = state.add_read(gearbox_in_stream_name)
                sdfg.add_array(gearbox_buffer_name, [1], memory_dtype, storage=dace.StorageType.FPGA_Local, transient=True)
                gearbox_read = state.add_read(gearbox_buffer_name)
                gearbox_write = state.add_write(gearbox_buffer_name)
                gearbox_iterators = copy.copy(buffer_iterators)
                gearbox_iterators["gb"] = f"0:{gearbox_factor}"
                gearbox_entry, gearbox_exit = state.add_map(f"gearbox_{node.name}",
                                                            gearbox_iterators,
                                                            schedule=dace.ScheduleType.FPGA_Device)
                gearbox_tasklet = state.add_tasklet(f"gearbox_{node.name}", {"from_memory", "buffer_in"}, {"to_compute", "buffer_out"},
                                                    f"""
const auto flit = (gb == 0) ? from_memory.pop() : buffer_in;
dace::vec<{node.data_type.base_type.ctype}, {input_vector_length}> val;
for (unsigned w = 0; w < {input_vector_length}; ++w) {{
  val[w] = flit[gb * {input_vector_length} + w];
}}
{gearbox_out_stream_name}.push(val);
buffer_out = flit;""",
                                                    language=dace.Language.CPP)
                state.add_memlet_path(gearbox_in_stream_read,
                                      gearbox_entry,
                                      gearbox_tasklet,
                                      dst_conn="from_memory",
                                      memlet=Memlet(f"{gearbox_in_stream_name}[0]", dynamic=True))
                state.add_memlet_path(gearbox_read,
                                      gearbox_entry,
                                      gearbox_tasklet,
                                      dst_conn="buffer_in",
                                      memlet=Memlet(f"{gearbox_buffer_name}[0]"))
                state.add_memlet_path(gearbox_tasklet,
                                      gearbox_exit,
                                      gearbox_write,
                                      src_conn="buffer_out",
                                      memlet=Memlet(f"{gearbox_buffer_name}[0]"))
                sdfg.add_stream(gearbox_out_stream_name, input_vtype, 16, storage=dace.StorageType.FPGA_Local, transient=True);
                gearbox_out_stream_write = state.add_write(gearbox_out_stream_name)
                state.add_memlet_path(gearbox_tasklet,
                                      gearbox_exit,
                                      gearbox_out_stream_write,
                                      src_conn="to_compute",
                                      memlet=Memlet(f"{gearbox_out_stream_name}[0]"))

                gearbox_out_stream_read = state.add_read(gearbox_out_stream_name)
                state.add_memlet_path(gearbox_out_stream_read,
                                      entry,
                                      tasklet,
                                      dst_conn="memory",
                                      memlet=Memlet(f"{gearbox_out_stream_name}[0]"))

        else:

            tasklet_code = "\n".join([
                "{} = {}".format(o, float(synthetic_reads)) for o in out_memlets
            ])

            tasklet = state.add_tasklet("read_" + node.name, {}, out_memlets,
                                        tasklet_code)

            state.add_memlet_path(entry, tasklet, memlet=dace.Memlet())

        # Add memlets to all FIFOs connecting to compute units
        for out_name, out_memlet in zip(outputs, out_memlets):
            stream_name = "read_{}_to_{}".format(node.name, out_name)
            write_node = state.add_write(stream_name)
            state.add_memlet_path(tasklet,
                                  exit,
                                  write_node,
                                  src_conn=out_memlet,
                                  memlet=Memlet.simple(stream_name,
                                                       "0",
                                                       num_accesses=1))

    def add_output(node, bank):

        # Always write 512-bit vectors to memory
        memory_veclen = 64 // node.data_type.bytes
        gearbox_factor = memory_veclen // vector_length
        memory_dtype = dace.dtypes.vector(node.data_type, memory_veclen)
        memory_shape = list(shape)
        memory_shape[-1] //= memory_veclen

        # Host-side array, which will be an output argument
        try:
            sdfg.add_array(node.name + "_host", shape, node.data_type)
            _, array = sdfg.add_array(node.name, memory_shape, memory_dtype, storage=StorageType.FPGA_Global, transient=True)
            array.location["bank"] = bank
        except NameError:
            # This array is also read
            sdfg.data(node.name + "_host").access = dace.AccessType.ReadWrite
            sdfg.data(node.name).access = dace.AccessType.ReadWrite

        # Device-side copy
        write_node = state.add_write(node.name)

        # Copy data to the host
        copy_fpga = post_state.add_read(node.name)
        copy_host = post_state.add_write(node.name + "_host")
        post_state.add_memlet_path(copy_fpga,
                                   copy_host,
                                   memlet=Memlet(f"{copy_fpga.data}[{', '.join(f'0:{s}' for s in memory_shape)}]"))

        # Stream from compute
        src = chain.graph.in_edges(node)
        if len(src) > 1:
            raise RuntimeError("Only one writer per output supported")
        src = next(iter(src))[0]
        stream_name = "{}_to_write_{}".format(src.name, node.name)
        read_node = state.add_read(stream_name)

        # Gearbox into the expected vector width
        gearbox_out_stream_name = f"{node.name}_gearbox_out"
        gearbox_buffer_name = f"{node.name}_gearbox_buffer"
        sdfg.add_array(gearbox_buffer_name, [1], memory_dtype, storage=dace.StorageType.FPGA_Local, transient=True)
        sdfg.add_stream(gearbox_out_stream_name, memory_dtype, 512, storage=dace.StorageType.FPGA_Local, transient=True);
        gearbox_read = state.add_read(gearbox_buffer_name)
        gearbox_write = state.add_write(gearbox_buffer_name)
        gearbox_out_stream_write = state.add_write(gearbox_out_stream_name)
        buffer_iterators = copy.copy(iterators)
        buffer_iterators[parameters[-1]] += f"/{gearbox_factor}"
        gearbox_iterators = copy.copy(buffer_iterators)
        gearbox_iterators["gb"] = f"0:{gearbox_factor}"
        gearbox_entry, gearbox_exit = state.add_map(f"gearbox_{node.name}",
                                                    gearbox_iterators,
                                                    schedule=dace.ScheduleType.FPGA_Device)
        gearbox_tasklet = state.add_tasklet(f"gearbox_{node.name}", {"from_compute", "buffer_in"}, {"to_memory", "buffer_out"},
                                            f"""
const auto val = from_compute;
for (unsigned w = 0; w < {vector_length}; ++w) {{
    buffer_in[gb * {vector_length} + w] = val[w];
}}
buffer_out = buffer_in;
if (gb == {gearbox_factor} - 1) {{
  to_memory.push(buffer_out);
}}""",
                                            language=dace.Language.CPP)
        state.add_memlet_path(read_node,
                              gearbox_entry,
                              gearbox_tasklet,
                              dst_conn="from_compute",
                              memlet=Memlet(f"{read_node.data}[0]"))
        state.add_memlet_path(gearbox_tasklet,
                              gearbox_exit,
                              gearbox_out_stream_write,
                              src_conn="to_memory",
                              memlet=Memlet(f"{gearbox_out_stream_name}[0]", dynamic=True))
        state.add_memlet_path(gearbox_read,
                              gearbox_entry,
                              gearbox_tasklet,
                              dst_conn="buffer_in",
                              memlet=Memlet(f"{gearbox_buffer_name}[0]"))
        state.add_memlet_path(gearbox_tasklet,
                              gearbox_exit,
                              gearbox_write,
                              src_conn="buffer_out",
                              memlet=Memlet(f"{gearbox_buffer_name}[0]"))

        # Write 512-bit vectors from a buffered stream
        buffer_entry, buffer_exit = state.add_map("buffer_" + node.name,
                                                  buffer_iterators,
                                                  schedule=dace.ScheduleType.FPGA_Device)
        buffer_tasklet = state.add_tasklet(f"buffer_{node.name}",
                                           {"from_gearbox"}, {"to_memory"}, "to_memory = from_gearbox")
        gearbox_out_stream_read = state.add_read(gearbox_out_stream_name)
        state.add_memlet_path(gearbox_out_stream_read,
                              buffer_entry,
                              buffer_tasklet,
                              dst_conn="from_gearbox",
                              memlet=Memlet(f"{gearbox_out_stream_name}[0]"))
        state.add_memlet_path(buffer_tasklet,
                              buffer_exit,
                              write_node,
                              src_conn="to_memory",
                              memlet=Memlet(f"{write_node.data}[{', '.join(parameters)}]"))

    def add_kernel(node):

        (stencil_node, input_to_connector,
         output_to_connector) = _generate_stencil(node, chain, shape,
                                                  dimensions_to_skip)

        if len(stencil_node.output_fields) == 0:
            if len(input_to_connector) == 0:
                warnings.warn("Ignoring orphan stencil: {}".format(node.name))
            else:
                raise ValueError("Orphan stencil with inputs: {}".format(
                    node.name))
            return

        vendor_str = dace.config.Config.get("compiler", "fpga_vendor")
        if vendor_str == "intel_fpga":
            stencil_node.implementation = "Intel FPGA"
        elif vendor_str == "xilinx":
            stencil_node.implementation = "Xilinx"
        else:
            raise ValueError(f"Unsupported FPGA backend: {vendor_str}")
        state.add_node(stencil_node)

        is_from_memory = {
            e[0].name: not isinstance(e[0], stencilflow.kernel.Kernel)
            for e in chain.graph.in_edges(node)
        }
        is_to_memory = {
            e[1].name: not isinstance(e[1], stencilflow.kernel.Kernel)
            for e in chain.graph.out_edges(node)
        }

        # Add read nodes and memlets
        for field_name, connector in input_to_connector.items():

            input_vector_length = vector_length
            try:
                # Scalars are symbols rather than data nodes
                if len(node.inputs[field_name]["input_dims"]) == 0:
                    continue
                else:
                    # If the innermost dimension of this field is not the
                    # vectorized one, read it as scalars
                    if (node.inputs[field_name]["input_dims"][-1] !=
                            parameters[-1]):
                        input_vector_length = 1
            except (KeyError, TypeError):
                pass  # input_dim is not defined or is None

            if is_from_memory[field_name]:
                stream_name = "read_{}_to_{}".format(field_name, node.name)
            else:
                stream_name = "{}_to_{}".format(field_name, node.name)

            # Outer memory read
            read_node = state.add_read(stream_name)
            state.add_memlet_path(read_node,
                                  stencil_node,
                                  dst_conn=connector,
                                  memlet=Memlet.simple(
                                      stream_name,
                                      "0",
                                      num_accesses=memcopy_accesses))

        # Add read nodes and memlets
        for output_name, connector in output_to_connector.items():

            # Add write node and memlet
            if is_to_memory[output_name]:
                stream_name = "{}_to_write_{}".format(node.name, output_name)
            else:
                stream_name = "{}_to_{}".format(node.name, output_name)

            # Outer write
            write_node = state.add_write(stream_name)
            state.add_memlet_path(stencil_node,
                                  write_node,
                                  src_conn=connector,
                                  memlet=Memlet.simple(
                                      stream_name,
                                      "0",
                                      num_accesses=memcopy_accesses))

    # First generate all connections between kernels and memories
    for link in chain.graph.edges(data=True):
        _add_pipe(sdfg, link, parameters, vector_length)

    bank = 0
    # Now generate all memory access functions so arrays are registered
    for node in chain.graph.nodes():
        if isinstance(node, Input):
            add_input(node, bank)
            bank = (bank + 1) % NUM_BANKS
        elif isinstance(node, Output):
            # Generate these separately after
            pass
        elif isinstance(node, Kernel):
            # Generate these separately after
            pass
        else:
            raise RuntimeError("Unexpected node type: {}".format(
                node.node_type))

    # Generate the compute kernels
    for node in chain.graph.nodes():
        if isinstance(node, Kernel):
            add_kernel(node)

    # Finally generate the output components
    for node in chain.graph.nodes():
        if isinstance(node, Output):
            add_output(node, bank)
            bank = (bank + 1) % NUM_BANKS

    return sdfg


def generate_reference(name, chain):
    """Generates a simple, unoptimized SDFG to run on the CPU, for verification
       purposes."""

    sdfg = SDFG(name)

    for k, v in chain.constants.items():
        sdfg.add_constant(k, v["value"], dace.data.Scalar(v["data_type"]))

    (dimensions_to_skip, shape, vector_length, parameters, iterators,
     memcopy_indices, memcopy_accesses) = _generate_init(chain)

    prev_state = sdfg.add_state("init")

    # Throw vectorization in the bin for the reference code
    vector_length = 1

    shape = tuple(map(int, shape))

    input_shapes = {}  # Maps inputs to their shape tuple

    for node in chain.graph.nodes():
        if isinstance(node, Input) or isinstance(node, Output):
            if isinstance(node, Input):
                for output in node.outputs.values():
                    pars = tuple(
                        output["input_dims"]
                    ) if "input_dims" in output and output[
                        "input_dims"] is not None else tuple(parameters)
                    arr_shape = tuple(s for s, p in zip(shape, parameters)
                                      if p in pars)
                    input_shapes[node.name] = arr_shape
                    break
                else:
                    raise ValueError("No outputs found for input node.")
            else:
                arr_shape = shape
            if len(arr_shape) > 0:
                try:
                    sdfg.add_array(node.name, arr_shape, node.data_type)
                except NameError:
                    sdfg.data(
                        node.name).access = dace.dtypes.AccessType.ReadWrite
            else:
                sdfg.add_symbol(node.name, node.data_type)

    for link in chain.graph.edges(data=True):
        name = link[0].name
        if name not in sdfg.arrays and name not in sdfg.symbols:
            sdfg.add_array(name, shape, link[0].data_type, transient=True)
            input_shapes[name] = tuple(shape)

    input_iterators = {
        k: tuple("0:{}".format(s) for s in v)
        for k, v in input_shapes.items()
    }

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

            if len(input_iterators[field]) == 0:
                continue  # Scalar variable

            # Outer memory read
            read_node = state.add_read(field)
            state.add_memlet_path(read_node,
                                  stencil_node,
                                  dst_conn=connector,
                                  memlet=Memlet.simple(
                                      field, ", ".join(input_iterators[field])))

        for _, connector in output_to_connector.items():

            # Outer write
            write_node = state.add_write(node.name)
            state.add_memlet_path(stencil_node,
                                  write_node,
                                  src_conn=connector,
                                  memlet=Memlet.simple(
                                      node.name, ", ".join("0:{}".format(s)
                                                           for s in shape)))

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
                        if isinstance(la, dace.sdfg.nodes.AccessNode):
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


def _num_channels_needed(desc, bytes_per_channel, ports):
    data_width = desc.veclen * dace.dtypes._BYTES[desc.dtype.type]
    if bytes_per_channel is None:
        return data_width, 1
    if data_width <= bytes_per_channel:
        return data_width, 1
    num_streams = int(np.ceil(data_width / bytes_per_channel))
    print("Using {} parallel streams to transmit data.".format(num_streams))
    try:
        num_ports = len(ports)
    except TypeError:
        num_ports = 1
    if num_ports != num_streams:
        raise ValueError("{} port numbers required, got {}.".format(
            num_streams, num_ports))
    if desc.veclen % num_streams != 0:
        raise ValueError("Can only split into even chunks.")
    return data_width, num_streams


def split_sdfg(sdfg,
               remote_stream,
               send_rank,
               receive_rank,
               ports,
               bytes_per_channel=None):
    """Split the input SDFG into two SDFGs connected by remote streams,
       to be executed in a multi-FPGA setup using SMI.
        :param sdfg: SDFG to split into two SDFGs.
        :param remote_stream: Stream data name (not node) to split on
        :param send_rank: Rank that will send
        :param receive_rank: Rank that will receive
        :param ports: One or more port identifiers
        :param bytes_per_channel: Maximum bytes to write on a channel per cycle.
        :return: The two resulting SDFGs
    """

    # Locate read and write nodes in SDFG
    read_node = None
    read_state = None
    write_node = None
    write_state = None
    for node, state in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.AccessNode):
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
    states_after, nodes_after = (_nodes_before_or_after(sdfg, write_state,
                                                        remote_stream, True))
    nodes_before.remove((read_state, read_node))
    nodes_after.remove((write_state, write_node))
    intersection = nodes_before & nodes_after
    if len(intersection) != 0:
        raise ValueError(
            "Node does not perfectly split SDFG, intersection is: {}".format(
                intersection))

    read_desc = sdfg.data(read_node.data)
    write_desc = sdfg.data(write_node.data)

    data_width, num_channels = _num_channels_needed(read_desc,
                                                    bytes_per_channel, ports)
    veclen_split = read_desc.veclen // num_channels

    if num_channels == 1:
        # Turn splitting stream into remote access nodes
        sdfg.data(read_node.data).storage = dace.dtypes.StorageType.FPGA_Remote
        sdfg.data(read_node.data).location["snd_rank"] = send_rank
        sdfg.data(read_node.data).location["port"] = ports
        sdfg.data(write_node.data).storage = dace.dtypes.StorageType.FPGA_Remote
        sdfg.data(write_node.data).location["rcv_rank"] = receive_rank
        sdfg.data(write_node.data).location["port"] = ports
    else:
        #######################################################################
        # Reading part
        #######################################################################
        num_accesses = sum(
            (e.data.num_accesses
             for e in read_state.out_edges(read_node)), 0) // read_desc.veclen
        read_split_name = read_node.data + "__read_split"
        read_entry, read_exit = read_state.add_map(
            "{}_map".format(read_split_name),
            {"split": "0:{}".format(num_accesses)},
            schedule=dace.dtypes.ScheduleType.FPGA_Device)
        buffer_to_stream = read_state.add_write(read_node.data)
        code = ""
        for i, p in enumerate(ports):
            for v in range(veclen_split):
                code += "_{}[{}] = _port_{}[{}];\n".format(
                    read_node.data, i * veclen_split + v, p, v)
        read_gearbox_tasklet = read_state.add_tasklet(
            "{}_gearbox".format(read_split_name),
            {"_port_{}".format(p)
             for p in ports}, {"_{}".format(read_node.data)},
            code,
            language=dace.dtypes.Language.CPP)
        nodes_after |= {(read_state, read_entry), (read_state, read_exit),
                        (read_state, buffer_to_stream),
                        (read_state, read_gearbox_tasklet)}
        for i, p in enumerate(ports):
            port_stream_name = "{}_port_{}".format(read_split_name, p)
            sdfg.add_stream(port_stream_name,
                            dace.dtypes.vector(read_desc.dtype, veclen_split),
                            storage=dace.dtypes.StorageType.FPGA_Remote,
                            transient=True)
            sdfg.data(port_stream_name).location = {
                "snd_rank": send_rank,
                "rcv_rank": receive_rank,
                "port": p
            }
            port_read = read_state.add_read(port_stream_name)
            nodes_after.add((read_state, port_read))
            read_state.add_memlet_path(port_read,
                                       read_entry,
                                       read_gearbox_tasklet,
                                       dst_conn="_port_{}".format(p),
                                       memlet=dace.Memlet.simple(
                                           port_read.data, "0", num_accesses=1))
        # Tasklet to stream
        read_state.add_memlet_path(read_gearbox_tasklet,
                                   read_exit,
                                   buffer_to_stream,
                                   src_conn="_{}".format(buffer_to_stream.data),
                                   memlet=dace.Memlet.simple(
                                       buffer_to_stream.data,
                                       "0",
                                       num_accesses=1))
        #######################################################################
        # Writing part
        #######################################################################
        num_accesses = sum(
            (e.data.num_accesses
             for e in write_state.in_edges(write_node)), 0) // write_desc.veclen
        write_split_name = write_node.data + "__write_split"
        write_entry, write_exit = write_state.add_map(
            "{}_map".format(write_split_name),
            {"split": "0:{}".format(num_accesses)},
            schedule=dace.dtypes.ScheduleType.FPGA_Device)
        stream_to_buffer = write_state.add_read(write_node.data)
        code = ""
        for i, p in enumerate(ports):
            for v in range(veclen_split):
                code += "_port_{}[{}] = _{}[{}];\n".format(
                    p, v, write_node.data, i * veclen_split + v)
        write_gearbox_tasklet = write_state.add_tasklet(
            "{}_gearbox".format(write_split_name),
            {"_{}".format(write_node.data)},
            {"_port_{}".format(p)
             for p in ports},
            code,
            language=dace.dtypes.Language.CPP)
        nodes_before |= {(write_state, write_entry), (write_state, write_exit),
                         (write_state, stream_to_buffer),
                         (write_state, write_gearbox_tasklet)}
        # Stream to tasklet
        write_state.add_memlet_path(stream_to_buffer,
                                    write_entry,
                                    write_gearbox_tasklet,
                                    dst_conn="_{}".format(write_node.data),
                                    memlet=dace.Memlet.simple(
                                        stream_to_buffer.data,
                                        "0",
                                        num_accesses=1))
        for i, p in enumerate(ports):
            port_stream_name = "{}_port_{}".format(write_split_name, p)
            sdfg.add_stream(port_stream_name,
                            dace.dtypes.vector(write_desc.dtype, veclen_split),
                            storage=dace.dtypes.StorageType.FPGA_Remote,
                            transient=True)
            sdfg.data(port_stream_name).location = {
                "snd_rank": send_rank,
                "rcv_rank": receive_rank,
                "port": p
            }
            port_write = write_state.add_write(port_stream_name)
            nodes_before.add((write_state, port_write))
            write_state.add_memlet_path(write_gearbox_tasklet,
                                        write_exit,
                                        port_write,
                                        src_conn="_port_{}".format(p),
                                        memlet=dace.Memlet.simple(
                                            port_write.data, "0"))

    # Keep track of containers
    containers_before = {
        n.data
        for _, n in nodes_before if isinstance(n, dace.sdfg.nodes.AccessNode)
    }
    containers_after = {
        n.data
        for _, n in nodes_after if isinstance(n, dace.sdfg.nodes.AccessNode)
    }

    # Now duplicate the SDFG, and remove all nodes that don't belong in the
    # respectively side of the split
    name = sdfg.name
    sdfg_before = copy.deepcopy(sdfg)
    sdfg_after = copy.deepcopy(sdfg)
    sdfg_before._name = name
    sdfg_after._name = name
    nodes_before = set((sdfg.node_id(s), s.node_id(n)) for s, n in nodes_before)
    nodes_after = set((sdfg.node_id(s), s.node_id(n)) for s, n in nodes_after)
    states_before = set(sdfg.node_id(s) for s in states_before)
    states_after = set(sdfg.node_id(s) for s in states_after)
    _remove_nodes_and_states(sdfg_before, states_before, nodes_before)
    _remove_nodes_and_states(sdfg_after, states_after, nodes_after)

    # Purge unused containers
    for arr in list(sdfg_before.arrays.keys()):
        if arr not in containers_before:
            sdfg_before.remove_data(arr)
    for arr in list(sdfg_after.arrays.keys()):
        if arr not in containers_after:
            sdfg_after.remove_data(arr)

    sdfg_before.validate()
    sdfg_after.validate()

    return sdfg_before, sdfg_after
