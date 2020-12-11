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
from ._common import JUNK_VAL, dim_to_abs_val, make_iterators


@dace.library.expansion
class ExpandStencilXilinx(dace.library.ExpandTransformation):

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
            if not any(dim_mask):
                # This is a scalar, no buffer needed. Instead, the SDFG must
                # take this as a symbol
                scalars[field_name] = parent_sdfg.symbols[field_name]
                vector_lengths[field_name] = 1
                sdfg.add_symbol(field_name, parent_sdfg.symbols[field_name])
                continue
            # Find the corresponding input memlets
            for e in in_edges:
                if e.dst_conn == field_name:
                    field_to_memlet[field_name] = e
                    data = dace.sdfg.find_input_arraynode(parent_state, e).data
                    field_to_data[field_name] = data
                    vector_length = parent_sdfg.data(data).veclen
                    vector_lengths[field_name] = vector_length
                    break
            else:
                raise KeyError("Input connector {} was not found for {}".format(connector, field_name, node.label))
            relative = [tuple(int(x) for x in r) for r in relative]
            # Extend every access with the vector length, then deduplicate
            _relative = []
            for x in relative:
                _relative.extend(x[:-1] + (x[-1] + i, ) for i in range(vector_length))
            relative = dace.dtypes.deduplicate([tuple(t) for t in sorted(_relative)])
            # Deduplicate, as we can have multiple accesses to the same index
            abs_indices = ([dim_to_abs_val(i, shape[dim_mask]) for i in relative] +
                           ([0] if node.boundary_conditions[field_name]["btype"] == "copy" else []))
            max_access = max(abs_indices)
            min_access = min(abs_indices)
            buffer_indices = [i - min_access for i in abs_indices]
            buffer_indices_major = [i // vector_length for i in buffer_indices]
            buffer_indices_minor = [i % vector_length for i in buffer_indices]
            buffer_accesses[field_name] = {
                "relative": relative,
                "major": buffer_indices_major,
                "major_unique": dace.dtypes.deduplicate(buffer_indices_major),
                "minor": buffer_indices_minor,
                "center": -min_access
            }
            # Each buffer size is the relative difference in absolute size
            # between this index and the next. Leave out the last element, as
            # it is set directly from the wavefront
            buffer_sizes[field_name] = [
                buffer_accesses[field_name]["major_unique"][i + 1] - buffer_accesses[field_name]["major_unique"][i]
                for i in range(len(buffer_accesses[field_name]["major_unique"]) - 1)
            ]

        # Find output connectors
        for field_name, offset in node.output_fields.items():
            for e in parent_state.out_edges(node):
                if e.src_conn == field_name:
                    data = dace.sdfg.find_output_arraynode(parent_state, e).data
                    field_to_data[field_name] = data
                    vector_lengths[field_name] = parent_sdfg.data(data).veclen
                    break
            else:
                raise KeyError("Output connector {} was not found for {}".format(field_name, node.label))

        # Assert that we support the given vectorization widths
        vector_length = max(vector_lengths.values())
        for field_name in node.output_fields:
            if vector_lengths[field_name] != vector_length:
                raise ValueError("{} has vector length {}, should be {}".format(field_name, vector_lengths[field_name],
                                                                                vector_length))

        # All inputs must be vectorized if they read the innermost dimension,
        # and cannot be vectorized if they don't
        for field_name, (dim_mask, _) in node.accesses.items():
            if dim_mask[-1] == True:
                if vector_lengths[field_name] != vector_length:
                    raise ValueError("Input {} has vectorization width, "
                                     "expected {}.".format(field_name, vector_lengths[field_name], vector_length))
            else:
                if vector_lengths[field_name] != 1:
                    raise ValueError("Input {} cannot be vectorized, "
                                     "because it doesn't read the innermost dimension.".format(field_name))

        # Create a initialization phase corresponding to the highest distance
        # to the center
        init_sizes = [
            sum(buffer_sizes[key], 0) - (val["center"] + vector_length - 1) // vector_length
            for key, val in buffer_accesses.items()
        ]
        init_size_max = int(np.max(init_sizes))

        parameters = np.array(["i", "j", "k"])[:len(shape)]
        iterator_mask = shape > 1  # Dimensions we need to iterate over
        iterators = make_iterators(shape[iterator_mask], parameters=parameters[iterator_mask])
        if vector_length > 1:
            iterators[parameters[-1]] += "/{}".format(vector_length)

        # Manually add pipeline entry and exit nodes
        pipeline_range = dace.properties.SubsetProperty.from_string(', '.join(iterators.values()))
        pipeline = dace.sdfg.nodes.Pipeline("compute_" + node.label,
                                            list(iterators.keys()),
                                            pipeline_range,
                                            dace.dtypes.ScheduleType.FPGA_Device,
                                            False,
                                            init_size=init_size_max,
                                            init_overlap=False,
                                            drain_size=init_size_max,
                                            drain_overlap=True)
        entry = dace.sdfg.nodes.PipelineEntry(pipeline)
        exit = dace.sdfg.nodes.PipelineExit(pipeline)
        state.add_nodes_from([entry, exit])

        #######################################################################
        # Tasklet code generation
        #######################################################################

        input_code = node.code.as_string

        # Replace relative indices with memlet names

        # Add copy boundary conditions
        for field in node.boundary_conditions:
            btype = node.boundary_conditions[field]["btype"]
            if btype == "copy":
                raise NotImplementedError("Copy boundary condition not implemented for Xilinx.")

        code = ""
        code_memlet_names = {f: {} for f in itertools.chain(node.accesses, node.output_fields)}
        for i in range(vector_length):
            converter = SubscriptConverter(offset=i)
            new_ast = converter.visit(ast.parse(input_code))
            code += astunparse.unparse(new_ast)
            for k, v in converter.names.items():
                code_memlet_names[k].update(v)

        #######################################################################
        # Implement boundary conditions
        #######################################################################

        boundary_code = ""
        # Loop over each input
        for (field_name, accesses) in buffer_accesses.items():
            # Loop over each access to this data
            for indices in accesses["relative"]:
                # Loop over each index of this access
                try:
                    memlet_name = code_memlet_names[field_name][indices]
                except KeyError:
                    raise KeyError("Missing access in code: {}[{}]".format(field_name, ", ".join(map(str, indices))))
                cond = []
                for j, offset in enumerate(indices):
                    if vector_length > 1 and j == len(indices) - 1:
                        par = "{}*{}".format(vector_length, parameters[j])
                    else:
                        par = parameters[j]
                    if offset < 0:
                        cond.append(par + " < " + str(-offset))
                    elif offset > 0:
                        cond.append(par + " >= " + str(shape[j] - offset))
                ctype = parent_sdfg.data(field_to_data[field_name]).dtype.ctype
                if len(cond) == 0:
                    boundary_code += f"{memlet_name} = {memlet_name}_in\n"
                else:
                    bc = node.boundary_conditions[field_name]
                    btype = bc["btype"]
                    if btype == "copy":
                        raise NotImplementedError
                    elif btype == "constant":
                        boundary_val = bc["value"]
                    elif btype == "shrink":
                        # We don't need to do anything here, it's up to the
                        # user to not use the junk output
                        boundary_val = JUNK_VAL
                        pass
                    else:
                        raise ValueError("Unsupported boundary condition type: {}".format(
                            node.boundary_conditions[field_name]["btype"]))
                    boundary_code += (f"{memlet_name} = {boundary_val}"
                                      f"if {' or '.join(cond)}"
                                      f"else {memlet_name}_in\n")

        #######################################################################
        # Only write if we're in bounds
        #######################################################################

        write_code = ""
        result_out_connectors = []
        for output in node.output_fields:
            for i in range(vector_length):
                index = tuple(0 for _ in range(len(shape) - 1)) + (i, )
                memlet_name = code_memlet_names[output][index]
                conn = f"{memlet_name}_out"
                result_out_connectors.append(conn)
                write_code += f"{conn} = {memlet_name}\n"

        # Concatenate everything
        code = (boundary_code + "\n" + code + "\n" + write_code)

        #######################################################################
        # Create nested SDFG
        #######################################################################

        in_connectors_outer = [f"_{x}" for x in buffer_accesses]
        out_connectors_outer = [f"_{x}" for x in node.output_fields]
        in_connectors_inner = []
        for name, accesses in buffer_accesses.items():
            connectors = [code_memlet_names[name][a] for a in accesses["relative"]]
            in_connectors_inner.extend(f"{c}_in" for c in connectors)
            buffer_conns = [f"{name}_buffer_{i}" for i in range(len(buffer_sizes[name]))]
            in_connectors_outer += buffer_conns
            out_connectors_outer += buffer_conns
        out_connectors_inner = list(result_out_connectors)

        nested_sdfg = dace.SDFG(f"{node.label}_inner")
        nested_sdfg_node = state.add_nested_sdfg(nested_sdfg, sdfg, in_connectors_outer, out_connectors_outer)
        for s in itertools.chain(pipeline.params, [pipeline.iterator_str()]):
            nested_sdfg.add_symbol(s, dace.dtypes.int64)
            nested_sdfg_node.symbol_mapping[s] = s
        try:
            s = pipeline.init_condition()
            nested_sdfg.add_symbol(s, dace.dtypes.bool)
            nested_sdfg_node.symbol_mapping[s] = s
        except VmlueError:
            pass
        try:
            s = pipeline.drain_condition()
            nested_sdfg.add_symbol(s, dace.dtypes.bool)
            nested_sdfg_node.symbol_mapping[s] = s
        except ValueError:
            pass

        compute_state = nested_sdfg.add_state(f"{node.label}_compute")
        update_state = nested_sdfg.add_state(f"{node.label}_update")
        nested_sdfg.add_edge(compute_state, update_state, dace.InterstateEdge())

        # Create compute tasklet
        compute_tasklet = compute_state.add_tasklet(node.label, in_connectors_inner, out_connectors_inner, code)

        for (field_name, sizes), init_size in zip(buffer_sizes.items(), init_sizes):

            data_name = field_to_data[field_name]
            connector = field_to_memlet[field_name].dst_conn

            # Outer memory read
            stream_name_outer = connector
            stream_outer = parent_sdfg.arrays[data_name].clone()
            stream_outer.transient = False
            sdfg.add_datadesc(stream_name_outer, stream_outer)
            read_node_outer = state.add_read(stream_name_outer)

            # Inner memory read
            stream_name_inner = f"_{connector}"
            stream_inner = parent_sdfg.arrays[data_name].clone()
            stream_inner.transient = False
            nested_sdfg.add_datadesc(stream_name_inner, stream_inner)
            read_node_inner = compute_state.add_read(stream_name_inner)

            # Connect to nested SDFG
            state.add_memlet_path(read_node_outer,
                                  entry,
                                  nested_sdfg_node,
                                  dst_conn=f"_{connector}",
                                  memlet=dace.Memlet(f"{stream_name_outer}[0]", dynamic=True))

            # Intermediate buffers to unpack vector types
            buffer_name_vector = f"{field_name}_vector"
            _, buffer_desc = nested_sdfg.add_array(buffer_name_vector, (1, ),
                                                   dace.vector(stream_outer.dtype.base_type, vector_length),
                                                   storage=dace.StorageType.FPGA_Registers,
                                                   transient=True)
            vector_access = compute_state.add_access(buffer_name_vector)

            buffer_name_scalar = f"{field_name}_scalar"
            _, buffer_desc = nested_sdfg.add_array(buffer_name_scalar, (vector_length, ),
                                                   stream_outer.dtype.base_type,
                                                   storage=dace.StorageType.FPGA_Registers,
                                                   transient=True)
            scalar_access = compute_state.add_access(buffer_name_scalar)

            # Begin reading according to this field's own buffer size, which is
            # translated to an index by subtracting it from the maximum buffer
            # size
            begin_reading = (init_size_max - init_size)
            end_reading = (functools.reduce(operator.mul, shape, 1) / vector_length + init_size_max - init_size)

            read_tasklet = compute_state.add_tasklet(
                f"read_{field_name}", {f"{field_name}_in"}, {f"{field_name}_out"}, """\
if {it} >= {begin} and {it} < {end}:
    {conn_out} = {conn_in}
else:
    {conn_out} = {junk}""".format(it=pipeline.iterator_str(),
                                  begin=begin_reading,
                                  end=end_reading,
                                  conn_in=f"{field_name}_in",
                                  conn_out=f"{field_name}_out",
                                  junk=JUNK_VAL))

            compute_state.add_memlet_path(read_node_inner,
                                          read_tasklet,
                                          dst_conn=f"{field_name}_in",
                                          memlet=dace.Memlet(f"{stream_name_inner}[0]", dynamic=True))

            compute_state.add_memlet_path(read_tasklet,
                                          vector_access,
                                          src_conn=f"{field_name}_out",
                                          memlet=dace.Memlet(f"{buffer_name_vector}[0]"))

            compute_state.add_memlet_path(vector_access,
                                          scalar_access,
                                          memlet=dace.Memlet(f"{buffer_name_vector}[0]",
                                                             other_subset=f"0:{vector_length}"))

            for i, acc in enumerate(buffer_accesses[field_name]["relative"][-vector_length:]):
                connector = f"{code_memlet_names[field_name][acc]}_in"
                compute_state.add_memlet_path(scalar_access,
                                              compute_tasklet,
                                              dst_conn=connector,
                                              memlet=dace.Memlet(f"{buffer_name_scalar}[{i}]"))

            pipeline_iterator = entry.pipeline.iterator_str()

            for i, (size, acc) in enumerate(zip(sizes, buffer_accesses[field_name]["major_unique"][:-1])):

                buffer_name = f"{field_name}_buffer_{i}"
                input_cast_name = f"{field_name}_input_cast_{i}"

                field_dtype = parent_sdfg.data(data_name).dtype
                nested_sdfg.add_array(buffer_name, (1, ),
                                      dace.vector(field_dtype.base_type, vector_length),
                                      storage=dace.dtypes.StorageType.FPGA_Registers,
                                      transient=False)
                nested_sdfg.add_array(input_cast_name, (vector_length, ),
                                      field_dtype.base_type,
                                      storage=dace.dtypes.StorageType.FPGA_Registers,
                                      transient=True)

                # Create outer version and connect it
                buffer_name_outer = f"{buffer_name}_outer"
                sdfg.add_array(buffer_name_outer, (size, ),
                               dace.vector(field_dtype.base_type, vector_length),
                               storage=dace.dtypes.StorageType.FPGA_Registers,
                               transient=True)
                buffer_read_outer = state.add_read(buffer_name_outer)
                buffer_write_outer = state.add_write(buffer_name_outer)
                buffer_subset_outer = f"{buffer_name_outer}[{pipeline_iterator}%{size}]"
                state.add_memlet_path(buffer_read_outer,
                                      entry,
                                      nested_sdfg_node,
                                      dst_conn=buffer_name,
                                      memlet=dace.Memlet(buffer_subset_outer))
                state.add_memlet_path(nested_sdfg_node,
                                      exit,
                                      buffer_write_outer,
                                      src_conn=buffer_name,
                                      memlet=dace.Memlet(buffer_subset_outer))

                # Create read and write nodes
                buffer_read = compute_state.add_read(buffer_name)
                input_cast = compute_state.add_access(input_cast_name)

                buffer_subset = f"{buffer_name}[0]"
                cast_subset = f"0:{vector_length}"

                # Find the memlets that need to access this buffer
                index_mask = [
                    buffer_accesses[field_name]["major"][i] == acc
                    for i in range(len(buffer_accesses[field_name]["major"]))
                ]
                relative = [b for b, i in zip(buffer_accesses[field_name]["relative"], index_mask) if i]
                minor = [b for b, i in zip(buffer_accesses[field_name]["minor"], index_mask) if i]
                for r, m in zip(relative, minor):
                    conn = code_memlet_names[field_name][r]
                    compute_state.add_memlet_path(input_cast,
                                                  compute_tasklet,
                                                  dst_conn=f"{conn}_in",
                                                  memlet=dace.Memlet(f"{input_cast_name}[{m}]"))

                # Buffer to cast
                compute_state.add_memlet_path(buffer_read,
                                              input_cast,
                                              memlet=dace.Memlet(buffer_subset, other_subset=cast_subset))

                # Update buffer
                buffer_write = update_state.add_write(buffer_name)
                if i < len(buffer_accesses[field_name]["major_unique"]) - 2:
                    next_buffer_name = f"{field_name}_buffer_{i+1}"
                    next_buffer_read = update_state.add_read(next_buffer_name)
                else:
                    # Last buffer comes from the input
                    next_buffer_name = f"{field_name}_vector"
                    next_buffer_read = update_state.add_read(next_buffer_name)
                next_buffer_subset = f"{next_buffer_name}[0]"
                update_state.add_memlet_path(next_buffer_read, buffer_write, memlet=dace.Memlet(buffer_subset))

        for field_name, offset in node.output_fields.items():

            if offset is not None and list(offset) != [0] * len(offset):
                raise NotImplementedError("Output offsets not implemented")

            data_name = field_to_data[field_name]

            # Outer write
            stream_name_outer = field_name
            stream_name = f"_{stream_name_outer}"
            stream_outer = parent_sdfg.arrays[data_name].clone()
            stream_outer.transient = False
            try:
                sdfg.add_datadesc(stream_name_outer, stream_outer)
            except NameError:  # Already an input
                parent_sdfg.arrays[data_name].access = (dace.AccessType.ReadWrite)
            write_node_outer = state.add_write(stream_name_outer)
            state.add_memlet_path(nested_sdfg_node,
                                  exit,
                                  write_node_outer,
                                  src_conn=stream_name,
                                  memlet=dace.Memlet(f"{stream_name_outer}[0]"))

            # Inner write
            stream = stream_outer.clone()
            try:
                nested_sdfg.add_datadesc(stream_name, stream)
            except NameError:  # Already an input
                pass
            write_node = compute_state.add_write(stream_name)

            # Add intermediate buffer to mediate vector length conversion
            output_buffer_scalar_name = f"{field_name}_output_buffer_scalar"
            _, output_buffer_scalar_desc = nested_sdfg.add_array(output_buffer_scalar_name, (vector_length, ),
                                                                 stream.dtype.base_type,
                                                                 storage=dace.StorageType.FPGA_Registers,
                                                                 transient=True)
            output_buffer_scalar_access = compute_state.add_access(output_buffer_scalar_name)

            # Add intermediate buffer to mediate vector length conversion
            output_buffer_vector_name = f"{field_name}_output_buffer_vector"
            _, output_buffer_vector_desc = nested_sdfg.add_array(output_buffer_vector_name, (1, ),
                                                                 dace.vector(stream.dtype.base_type, vector_length),
                                                                 storage=dace.StorageType.FPGA_Registers,
                                                                 transient=True)
            output_buffer_vector_access = compute_state.add_access(output_buffer_vector_name)

            for i, o in enumerate(result_out_connectors):
                compute_state.add_memlet_path(compute_tasklet,
                                              output_buffer_scalar_access,
                                              src_conn=o,
                                              memlet=dace.Memlet(f"{output_buffer_scalar_name}[{i}]"))

            # Memory width conversion
            compute_state.add_memlet_path(output_buffer_scalar_access,
                                          output_buffer_vector_access,
                                          memlet=dace.Memlet(f"{output_buffer_vector_name}[0]",
                                                             other_subset=f"0:{vector_length}"))

            # Conditional write
            if init_size_max > 0:
                init_cond = pipeline.init_condition()
                write_condition = f"if not {init_cond}:\n\t"
            else:
                write_condition = ""
            write_condition += "pipe = buffer"

            write_tasklet = compute_state.add_tasklet(f"write_{field_name}", {"buffer"}, {"pipe"}, write_condition)

            compute_state.add_memlet_path(output_buffer_vector_access,
                                          write_tasklet,
                                          dst_conn="buffer",
                                          memlet=dace.Memlet(f"{output_buffer_vector_name}[0]"))

            compute_state.add_memlet_path(write_tasklet,
                                          write_node,
                                          src_conn="pipe",
                                          memlet=dace.Memlet(f"{stream_name}[0]"))

        sdfg.parent = parent_state
        sdfg._parent_sdfg = parent_sdfg  # TODO: this should not be necessary

        return sdfg
