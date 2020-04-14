import ast
import astunparse
import collections
import functools
import itertools
import operator
import re

import dace
import numpy as np
from .subscript_converter import SubscriptConverter

JUNK_VAL = -100000


def dim_to_abs_val(input, dimensions):
    """Compute scalar number from independent dimension unit."""
    vec = [
        functools.reduce(operator.mul, dimensions[i + 1:], 1)
        for i in range(len(dimensions))
    ]
    return functools.reduce(operator.add, map(operator.mul, input, vec), 0)


def make_iterators(dimensions, halo_sizes=None, parameters=None):
    def add_halo(i):
        if i == len(dimensions) - 1 and halo_sizes is not None:
            return " + " + str(-halo_sizes[0] + halo_sizes[1])
        else:
            return ""

    if parameters is None:
        return collections.OrderedDict([("i" + str(i),
                                         "0:" + str(d) + add_halo(i))
                                        for i, d in enumerate(dimensions)])
    else:
        return collections.OrderedDict([(parameters[i],
                                         "0:" + str(d) + add_halo(i))
                                        for i, d in enumerate(dimensions)])


# Value inserted when the output is junk and should not be used
JUNK_VAL = -1000


@dace.library.expansion
class ExpandStencilFPGA(dace.library.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_outer")
        state = sdfg.add_state(node.label + "_outer")

        shape = np.array(node.shape)

        # Extract which fields to read from streams and what to buffer
        field_to_memlet = {}  # Maps fields to which memlet carries it
        field_to_data = {}  # Maps fields to which data object represents it
        buffer_sizes = collections.OrderedDict()
        buffer_accesses = collections.OrderedDict()
        in_edges = parent_state.in_edges(node)
        scalars = {}  # {name: type}
        vector_lengths = {}  # {name: vector length}
        for field_name, (dim_mask, relative) in node.accesses.items():
            relative = dace.dtypes.deduplicate(
                [tuple(int(x) for x in r) for r in relative])
            if not any(dim_mask):
                # This is a scalar, no buffer needed. Instead, the SDFG must
                # take this as a symbol
                scalars[field_name] = parent_sdfg.symbols[field_name]
                sdfg.add_symbol(field_name,
                                parent_sdfg.symbols[field_name],
                                override_dtype=True)
                continue
            # Find the corresponding input memlets
            for e in in_edges:
                if e.dst_connector == field_name:
                    field_to_memlet[field_name] = e
                    field_to_data[field_name] = dace.sdfg.find_input_arraynode(
                        parent_state, e).data
                    vector_lengths[field_name] = e.data.veclen
                    break
            else:
                raise KeyError(
                    "Input connector {} was not found for {}".format(
                        connector, field_name, node.label))
            # Deduplicate, as we can have multiple accesses to the same index
            abs_indices = (
                [dim_to_abs_val(i, shape[dim_mask]) for i in relative] +
                ([0] if node.boundary_conditions[field_name]["btype"] == "copy"
                 else []))
            max_access = max(abs_indices)
            min_access = min(abs_indices)
            buffer_size = max_access - min_access + vector_lengths[field_name]
            buffer_sizes[field_name] = buffer_size
            # (indices relative to center, buffer indices, buffer center index)
            buffer_accesses[field_name] = ([tuple(r) for r in relative], [
                i - min_access for i in abs_indices
            ], -min_access)

        # Find output connectors
        for field_name, offset in node.output_fields.items():
            for e in parent_state.out_edges(node):
                if e.src_connector == field_name:
                    field_to_memlet[field_name] = e
                    field_to_data[
                        field_name] = dace.sdfg.find_output_arraynode(
                            parent_state, e).data
                    vector_lengths[field_name] = e.data.veclen
                    break
            else:
                raise KeyError(
                    "Output connector {} was not found for {}".format(
                        field_name, node.label))

        # Assert that we support the given vectorization widths
        vector_length = max(vector_lengths.values())
        for field_name in node.output_fields:
            if vector_lengths[field_name] != vector_length:
                raise ValueError(
                    "{} has vector length {}, should be {}".format(
                        field_name, vector_lengths[field_name], vector_length))

        # All inputs must be vectorized if they read the innermost dimension,
        # and cannot be vectorized if they don't
        for field_name, (dim_mask, _) in node.accesses.items():
            if dim_mask[-1] == True:
                if vector_lengths[field_name] != vector_length:
                    raise ValueError("Input {} has vectorization width, "
                                     "expected {}.".format(
                                         field_name,
                                         vector_lengths[field_name],
                                         vector_length))
            else:
                if vector_lengths[field_name] != 1:
                    raise ValueError(
                        "Input {} cannot be vectorized, "
                        "because it doesn't read the innermost dimension.".
                        format(field_name))

        # Create a initialization phase corresponding to the highest distance
        # to the center
        init_sizes = [
            (buffer_sizes[key] - vector_length - val[2]) // vector_length
            for key, val in buffer_accesses.items()
        ]
        init_size_max = int(np.max(init_sizes))

        parameters = np.array(["i", "j", "k"])[:len(shape)]
        iterator_mask = shape > 1  # Dimensions we need to iterate over
        iterators = make_iterators(shape[iterator_mask],
                                   parameters=parameters[iterator_mask])
        if vector_length > 1:
            iterators[parameters[-1]] += "/{}".format(vector_length)

        # Manually add pipeline entry and exit nodes
        pipeline_range = dace.properties.SubsetProperty.from_string(', '.join(
            iterators.values()))
        pipeline = dace.codegen.targets.fpga.Pipeline(
            "compute_" + node.label,
            list(iterators.keys()),
            pipeline_range,
            dace.dtypes.ScheduleType.FPGA_Device,
            False,
            init_size=init_size_max,
            init_overlap=False,
            drain_size=init_size_max,
            drain_overlap=True)
        entry = dace.codegen.targets.fpga.PipelineEntry(pipeline)
        exit = dace.codegen.targets.fpga.PipelineExit(pipeline)
        state.add_nodes_from([entry, exit])

        # Add nested SDFG to do 1) shift buffers 2) read from input 3) compute
        nested_sdfg = dace.SDFG(node.label + "_inner", parent=state)
        nested_sdfg._parent_sdfg = sdfg  # TODO: This should not be necessary
        nested_sdfg_tasklet = dace.graph.nodes.NestedSDFG(
            nested_sdfg.label,
            nested_sdfg,
            # Input connectors
            [k + "_in" for k, v in node.accesses.items() if any(v[0])] +
            [name + "_buffer_in" for name, _ in buffer_sizes.items()],
            # Output connectors
            [k + "_out" for k in node.output_fields.keys()] +
            [name + "_buffer_out" for name, _ in buffer_sizes.items()],
            schedule=dace.ScheduleType.FPGA_Device)
        state.add_node(nested_sdfg_tasklet)

        # Shift state, which shifts all buffers by one
        shift_state = nested_sdfg.add_state(node.label + "_shift")

        # Update state, which reads new values from memory
        update_state = nested_sdfg.add_state(node.label + "_update")

        #######################################################################
        # Tasklet code generation
        #######################################################################

        code = node.code.as_string

        # Replace relative indices with memlet names
        converter = SubscriptConverter()

        # Add copy boundary conditions
        for field in node.boundary_conditions:
            btype = node.boundary_conditions[field]["btype"]
            if btype == "copy":
                center_index = tuple(
                    0 for _ in range(sum(accesses[field][0], 0)))
                # This will register the renaming
                converter.convert(field, center_index)

        new_ast = converter.visit(ast.parse(code))
        code = astunparse.unparse(new_ast)
        code_memlet_names = converter.names

        #######################################################################
        # Implement boundary conditions
        #######################################################################

        boundary_code = ""
        # Loop over each input
        for (field_name, (accesses, accesses_buffer,
                          center)) in buffer_accesses.items():
            # Loop over each access to this data
            for indices in accesses:
                # Loop over each index of this access
                try:
                    memlet_name = code_memlet_names[field_name][indices]
                except KeyError:
                    raise KeyError("Missing access in code: {}[{}]".format(
                        field_name, ", ".join(map(str, indices))))
                cond = []
                for i, offset in enumerate(indices):
                    if vector_length > 1 and i == len(indices) - 1:
                        par = "{}*{} + i_unroll".format(
                            vector_length, parameters[i])
                    else:
                        par = parameters[i]
                    if offset < 0:
                        cond.append(par + " < " + str(-offset))
                    elif offset > 0:
                        cond.append(par + " >= " + str(shape[i] - offset))
                ctype = parent_sdfg.data(field_to_data[field_name]).dtype.ctype
                if len(cond) == 0:
                    boundary_code += "{} = _{}\n".format(
                        memlet_name, memlet_name)
                else:
                    bc = node.boundary_conditions[field_name]
                    btype = bc["btype"]
                    if btype == "copy":
                        center_memlet = code_memlet_names[field_name][center]
                        boundary_val = "_{}".format(center_memlet)
                    elif btype == "constant":
                        boundary_val = bc["value"]
                    elif btype == "shrink":
                        # We don't need to do anything here, it's up to the
                        # user to not use the junk output
                        boundary_val = JUNK_VAL
                        pass
                    else:
                        raise ValueError(
                            "Unsupported boundary condition type: {}".format(
                                node.boundary_conditions[field_name]["btype"]))
                    boundary_code += ("{} = {} if {} else _{}\n".format(
                        memlet_name, boundary_val, " or ".join(cond),
                        memlet_name))

        #######################################################################
        # Only write if we're in bounds
        #######################################################################

        write_code = ("\n".join([
            "{}_inner_out = {}\n".format(
                output,
                code_memlet_names[output][tuple(0 for _ in range(len(shape)))])
            for output in node.output_fields
        ]))
        write_condition = ("if not {}:\n\t".format("".join(
            pipeline.init_condition())) if init_size_max > 0 else "")

        code = boundary_code + "\n" + code + "\n" + write_code

        #######################################################################
        # Create DaCe compute state
        #######################################################################

        # Compute state, which reads from input channels, performs the compute,
        # and writes to the output channel(s)
        compute_state = nested_sdfg.add_state(node.label + "_compute")
        compute_inputs = list(
            itertools.chain.from_iterable(
                [["_" + v for v in code_memlet_names[f].values()]
                 for f, a in node.accesses.items() if any(a[0])]))
        compute_tasklet = compute_state.add_tasklet(
            node.label + "_compute",
            compute_inputs,
            [name + "_inner_out" for name in node.output_fields],
            code,
            language=dace.dtypes.Language.Python)
        if vector_length > 1:
            compute_unroll_entry, compute_unroll_exit = compute_state.add_map(
                compute_state.label + "_unroll",
                {"i_unroll": "0:{}".format(vector_length)},
                schedule=dace.ScheduleType.FPGA_Device,
                unroll=True)

        # Connect the three nested states
        nested_sdfg.add_edge(shift_state, update_state,
                             dace.graph.edges.InterstateEdge())
        nested_sdfg.add_edge(update_state, compute_state,
                             dace.graph.edges.InterstateEdge())

        # First, grab scalar variables
        for scalar, scalar_type in scalars.items():
            nested_sdfg.add_symbol(scalar, scalar_type, True)

        for (field_name, size), init_size in zip(buffer_sizes.items(),
                                                 init_sizes):

            data_name = field_to_data[field_name]
            connector = field_to_memlet[field_name].dst_connector

            # Outer memory read
            stream_name_outer = connector
            stream_name_inner = field_name + "_in"
            stream_outer = parent_sdfg.arrays[data_name].clone()
            stream_outer.transient = False
            sdfg.add_datadesc(stream_name_outer, stream_outer)
            read_node_outer = state.add_read(stream_name_outer)
            state.add_memlet_path(read_node_outer,
                                  entry,
                                  nested_sdfg_tasklet,
                                  dst_conn=stream_name_inner,
                                  memlet=dace.memlet.Memlet.simple(
                                      stream_name_outer,
                                      "0",
                                      num_accesses=-1,
                                      veclen=vector_lengths[field_name]))

            # Create inner memory pipe
            stream_inner = stream_outer.clone()
            nested_sdfg.add_datadesc(stream_name_inner, stream_inner)

            buffer_name_outer = "{}_{}_buffer".format(node.label, field_name)
            buffer_name_inner_read = "{}_buffer_in".format(field_name)
            buffer_name_inner_write = "{}_buffer_out".format(field_name)

            # Create buffer transient in outer SDFG
            field_dtype = parent_sdfg.data(data_name).dtype
            _, desc_outer = sdfg.add_array(
                buffer_name_outer, (size, ),
                field_dtype,
                storage=dace.dtypes.StorageType.FPGA_Local,
                transient=True)

            # Create read and write nodes
            read_node_outer = state.add_read(buffer_name_outer)
            write_node_outer = state.add_write(buffer_name_outer)

            # Outer buffer read
            state.add_memlet_path(read_node_outer,
                                  entry,
                                  nested_sdfg_tasklet,
                                  dst_conn=buffer_name_inner_read,
                                  memlet=dace.memlet.Memlet.simple(
                                      buffer_name_outer,
                                      "0:{}".format(size),
                                      num_accesses=-1,
                                      veclen=vector_lengths[field_name]))

            # Outer buffer write
            state.add_memlet_path(nested_sdfg_tasklet,
                                  exit,
                                  write_node_outer,
                                  src_conn=buffer_name_inner_write,
                                  memlet=dace.memlet.Memlet.simple(
                                      write_node_outer.data,
                                      "0:{}".format(size),
                                      num_accesses=-1,
                                      veclen=vector_lengths[field_name]))

            # Inner copy
            desc_inner_read = desc_outer.clone()
            desc_inner_read.transient = False
            desc_inner_read.name = buffer_name_inner_read
            desc_inner_write = desc_inner_read.clone()
            desc_inner_write.name = buffer_name_inner_write
            nested_sdfg.add_datadesc(buffer_name_inner_read, desc_inner_read)
            nested_sdfg.add_datadesc(buffer_name_inner_write, desc_inner_write)

            # Make shift state if necessary
            if size > 1:
                shift_read = shift_state.add_read(buffer_name_inner_read)
                shift_write = shift_state.add_write(buffer_name_inner_write)
                shift_entry, shift_exit = shift_state.add_map(
                    "shift_{}".format(field_name), {
                        "i_shift":
                        "0:{} - {}".format(size, vector_lengths[field_name])
                    },
                    schedule=dace.dtypes.ScheduleType.FPGA_Device,
                    unroll=True)
                shift_tasklet = shift_state.add_tasklet(
                    "shift_{}".format(field_name),
                    {"{}_shift_in".format(field_name)},
                    {"{}_shift_out".format(field_name)},
                    "{field}_shift_out = {field}_shift_in".format(
                        field=field_name))
                shift_state.add_memlet_path(
                    shift_read,
                    shift_entry,
                    shift_tasklet,
                    dst_conn=field_name + "_shift_in",
                    memlet=dace.memlet.Memlet.simple(
                        shift_read.data,
                        "i_shift + {}".format(vector_lengths[field_name]),
                        num_accesses=1))
                shift_state.add_memlet_path(shift_tasklet,
                                            shift_exit,
                                            shift_write,
                                            src_conn=field_name + "_shift_out",
                                            memlet=dace.memlet.Memlet.simple(
                                                shift_write.data,
                                                "i_shift",
                                                num_accesses=1))

            # Begin reading according to this field's own buffer size, which is
            # translated to an index by subtracting it from the maximum buffer
            # size
            begin_reading = (init_size_max - init_size)
            end_reading = (
                functools.reduce(operator.mul, shape, 1) / vector_length +
                init_size_max - init_size)

            update_read = update_state.add_read(stream_name_inner)
            update_write = update_state.add_write(buffer_name_inner_write)
            update_tasklet = update_state.add_tasklet(
                "read_wavefront", {"wavefront_in"}, {"buffer_out"},
                "if {it} >= {begin} and {it} < {end}:\n"
                "\tbuffer_out = wavefront_in\n".format(
                    it=pipeline.iterator_str(),
                    begin=begin_reading,
                    end=end_reading),
                language=dace.dtypes.Language.Python)
            update_state.add_memlet_path(
                update_read,
                update_tasklet,
                memlet=dace.memlet.Memlet.simple(
                    update_read.data,
                    "0",
                    num_accesses=-1,
                    veclen=vector_lengths[field_name]),
                dst_conn="wavefront_in")
            update_state.add_memlet_path(
                update_tasklet,
                update_write,
                memlet=dace.memlet.Memlet.simple(
                    update_write.data,
                    "{} - {}".format(size, vector_lengths[field_name])
                    if size > 1 else "0",
                    num_accesses=vector_lengths[field_name],
                    veclen=vector_lengths[field_name]),
                src_conn="buffer_out")

            # Make compute state
            compute_read = compute_state.add_read(buffer_name_inner_read)
            for relative, offset in zip(buffer_accesses[field_name][0],
                                        buffer_accesses[field_name][1]):
                memlet_name = code_memlet_names[field_name][tuple(relative)]
                if vector_length > 1:
                    offset = "{} + i_unroll".format(offset)
                    path = [
                        compute_read, compute_unroll_entry, compute_tasklet
                    ]
                else:
                    offset = str(offset)
                    path = [compute_read, compute_tasklet]
                compute_state.add_memlet_path(
                    *path,
                    dst_conn="_" + memlet_name,
                    memlet=dace.memlet.Memlet.simple(
                        compute_read.data,
                        offset,
                        num_accesses=1,
                        veclen=vector_lengths[field_name]))

        for field_name, offset in node.output_fields.items():

            if offset is not None and list(offset) != [0] * len(offset):
                raise NotImplementedError("Output offsets not implemented")

            data_name = field_to_data[field_name]

            # Outer write
            stream_name_outer = field_name
            stream_name_inner = field_name + "_out"
            stream_outer = parent_sdfg.arrays[data_name].clone()
            stream_outer.transient = False
            sdfg.add_datadesc(stream_name_outer, stream_outer)
            write_node_outer = state.add_write(stream_name_outer)
            state.add_memlet_path(nested_sdfg_tasklet,
                                  exit,
                                  write_node_outer,
                                  src_conn=stream_name_inner,
                                  memlet=dace.memlet.Memlet.simple(
                                      stream_name_outer,
                                      "0",
                                      num_accesses=-1,
                                      veclen=vector_lengths[field_name]))

            # Create inner stream
            stream_inner = stream_outer.clone()
            nested_sdfg.add_datadesc(stream_name_inner, stream_inner)

            # Inner write
            write_node_inner = compute_state.add_write(stream_name_inner)

            # Intermediate buffer, mostly relevant for vectorization
            output_buffer_name = field_name + "_output_buffer"
            nested_sdfg.add_array(output_buffer_name, (vector_length, ),
                                  stream_inner.dtype,
                                  storage=dace.StorageType.FPGA_Registers,
                                  transient=True)
            output_buffer = compute_state.add_access(output_buffer_name)

            # Condition write tasklet
            output_tasklet = compute_state.add_tasklet(
                field_name + "_conditional_write",
                {"_{}".format(output_buffer_name)},
                {"_{}".format(stream_name_inner)},
                (write_condition +
                 "_{} = _{}".format(stream_name_inner, output_buffer_name)))

            # If vectorized, we need to pass through the unrolled scope
            if vector_length > 1:
                compute_state.add_memlet_path(
                    compute_tasklet,
                    compute_unroll_exit,
                    output_buffer,
                    src_conn=field_name + "_inner_out",
                    memlet=dace.memlet.Memlet.simple(output_buffer_name,
                                                     "i_unroll",
                                                     num_accesses=1))
            else:
                compute_state.add_memlet_path(
                    compute_tasklet,
                    output_buffer,
                    src_conn=field_name + "_inner_out",
                    memlet=dace.memlet.Memlet.simple(output_buffer_name,
                                                     "0",
                                                     num_accesses=1))

            # Final memlet to the output
            compute_state.add_memlet_path(
                output_buffer,
                output_tasklet,
                dst_conn="_{}".format(output_buffer_name),
                memlet=dace.Memlet.simple(output_buffer.data,
                                          "0",
                                          num_accesses=vector_length,
                                          veclen=vector_length))
            compute_state.add_memlet_path(
                output_tasklet,
                write_node_inner,
                src_conn="_{}".format(stream_name_inner),
                memlet=dace.memlet.Memlet.simple(write_node_inner.data,
                                                 "0",
                                                 num_accesses=-1,
                                                 veclen=vector_length))

        sdfg.parent = parent_state
        sdfg._parent_sdfg = parent_sdfg  # TODO: this should not be necessary

        return sdfg
