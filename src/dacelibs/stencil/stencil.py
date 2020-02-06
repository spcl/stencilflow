import collections
import functools
import itertools
import operator
import re

import dace
import numpy as np


def dim_to_abs_val(input, dimensions):
    """Compute scalar number from independent dimension unit."""
    vec = [
        functools.reduce(operator.mul, dimensions[i + 1:], 1)
        for i in range(len(dimensions))
    ]
    return functools.reduce(operator.add, map(operator.mul, input, vec))


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


def relative_to_buffer_index(buffer_size, index):
    return buffer_size - 1 - abs(index)


@dace.library.expansion
class ExpandStencilFPGA(dace.library.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_outer")
        state = sdfg.add_state(node.label + "_outer")

        # Extract which fields to read from streams and what to buffer
        field_to_memlet = {}  # Maps fields to which memlet carries it
        field_to_data = {}  # Maps fields to which data object represents it
        buffer_sizes = collections.OrderedDict()
        buffer_accesses = collections.OrderedDict()
        in_edges = parent_state.in_edges(node)
        for field_name, (connector, relative) in node.accesses.items():
            # Deduplicate, as we can have multiple accesses to the same index
            abs_indices = dace.dtypes.deduplicate(
                [dim_to_abs_val(i, node.shape) for i in relative] +
                ([0] if node.boundary_conditions[field_name]["type"] == "copy"
                 else []))
            max_access = max(abs_indices)
            min_access = min(abs_indices)
            buffer_size = max_access - min_access + 1
            buffer_sizes[field_name] = buffer_size
            # (indices relative to center, buffer indices, buffer center index)
            buffer_accesses[field_name] = (relative, [
                i - min_access for i in abs_indices
            ], -min_access)
            # Find the corresponding input memlets
            for e in in_edges:
                if e.dst_connector == connector:
                    field_to_memlet[field_name] = e
                    field_to_data[field_name] = dace.sdfg.find_input_arraynode(
                        parent_state, e).data
                    break
            else:
                raise KeyError(
                    "Input connector {} for {} was not found for {}".format(
                        connector, field_name, node.label))

        # Find output connectors
        for field_name, connector in node.output_fields.items():
            for e in parent_state.out_edges(node):
                if e.src_connector == connector:
                    field_to_memlet[field_name] = e
                    field_to_data[
                        field_name] = dace.sdfg.find_output_arraynode(
                            parent_state, e).data
                    break
            else:
                raise KeyError(
                    "Output connector {} for {} was not found for {}".format(
                        connector, field_name, node.label))

        # Create a initialization phase corresponding to the highest distance
        # to the center
        init_sizes = [
            buffer_sizes[key] - 1 - val[2]
            for key, val in buffer_accesses.items()
        ]
        init_size_max = np.max(init_sizes)

        parameters = np.array(node.iterators)  # All iterator parameters
        shape = np.array(node.shape)
        iterator_mask = shape > 1  # Dimensions we need to iterate over
        iterators = make_iterators(
            shape[iterator_mask], parameters=parameters[iterator_mask])

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
            [k + "_in" for k in node.accesses.keys()] +  # Input connectors
            [name + "_buffer_in" for name, _ in buffer_sizes.items()],
            [k + "_out" for k in node.output_fields.keys()] +  # Output connectors
            [name + "_buffer_out" for name, _ in buffer_sizes.items()])
        state.add_node(nested_sdfg_tasklet)

        # Shift state, which shifts all buffers by one
        shift_state = nested_sdfg.add_state(node.label + "_shift")

        # Update state, which reads new values from memory
        update_state = nested_sdfg.add_state(node.label + "_update")

        #######################################################################
        # Implement boundary conditions
        #######################################################################

        boundary_code = ""
        # Loop over each input
        for (field_name, (accesses, accesses_buffer,
                         center)) in buffer_accesses.items():
            # Loop over each access to this data
            for indices, offset_buffer in zip(accesses, accesses_buffer):
                # Loop over each index of this access
                cond = []
                for i, offset in enumerate(indices):
                    if offset < 0:
                        cond.append(parameters[i] + " < " + str(-offset))
                    elif offset > 0:
                        cond.append(parameters[i] + " >= " +
                                    str(shape[i] - offset))
                ctype = parent_sdfg.data(field_to_data[field_name]).dtype.ctype
                if len(cond) == 0:
                    boundary_code += "{} {}_{} = _{}_{};\n".format(
                        ctype, field_name, offset_buffer, field_name,
                        offset_buffer)
                else:
                    if node.boundary_conditions[field_name]["type"] == "copy":
                        boundary_val = "_{}_{}".format(field_name, center)
                    elif node.boundary_conditions[field_name][
                            "type"] == "constant":
                        boundary_val = node.boundary_conditions[field_name][
                            "value"]
                    boundary_code += (
                        "{} {}_{} = ({}) ? ({}) : (_{}_{});\n".format(
                            ctype, field_name, offset_buffer, " || ".join(cond),
                            boundary_val, field_name, offset_buffer))

        #######################################################################
        # Only write if we're in bounds
        #######################################################################

        write_code = (
            ("if (!{}) {{\n".format("".join(pipeline.init_condition()))
             if init_size_max > 0 else "") + ("\n".join([
                 "write_channel_intel({}_inner_out, {});".format(output, output)
                 for output in node.output_fields
             ])) + ("\n}" if init_size_max > 0 else "\n"))

        #######################################################################
        # Tasklet code generation
        #######################################################################

        code = boundary_code + "\n" + node.code

        # Replace array accesses with memlet names
        pattern = re.compile("(([a-zA-Z_][a-zA-Z0-9_]*)\[([^\]]+)\])")
        for full_str, field, index in re.findall(pattern, code):
            buffer_index = relative_to_buffer_index(buffer_sizes[field],
                                                    int(index))
            if int(index) > 0:
                raise ValueError("Received positive index " + full_str + ".")
            code = code.replace(full_str, "{}_{}".format(field, buffer_index))
        code += "\n" + write_code

        #######################################################################
        # Create DaCe compute state
        #######################################################################

        # Compute state, which reads from input channels, performs the compute,
        # and writes to the output channel(s)
        compute_state = nested_sdfg.add_state(node.label + "_compute")
        compute_inputs = list(
            itertools.chain.from_iterable(
                [["_{}_{}".format(name, offset) for offset in offsets]
                 for name, (_, offsets, _) in buffer_accesses.items()]))
        compute_tasklet = compute_state.add_tasklet(
            node.label + "_compute",
            compute_inputs,
            [name + "_inner_out" for name in node.output_fields],
            code,
            language=dace.dtypes.Language.CPP)

        # Connect the three nested states
        nested_sdfg.add_edge(shift_state, update_state,
                             dace.graph.edges.InterstateEdge())
        nested_sdfg.add_edge(update_state, compute_state,
                             dace.graph.edges.InterstateEdge())

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
            state.add_memlet_path(
                read_node_outer,
                entry,
                nested_sdfg_tasklet,
                dst_conn=stream_name_inner,
                memlet=dace.memlet.Memlet.simple(
                    stream_name_outer, "0", num_accesses=-1))

            # Create inner memory pipe
            stream_inner = stream_outer.clone()
            nested_sdfg.add_datadesc(stream_name_inner, stream_inner)

            buffer_name_outer = "{}_{}_buffer".format(node.label, field_name)
            buffer_name_inner_read = "{}_buffer_in".format(field_name)
            buffer_name_inner_write = "{}_buffer_out".format(field_name)

            # Create buffer transient in outer SDFG
            field_dtype = parent_sdfg.data(field_name).dtype
            if size > 1:
                _, desc_outer = sdfg.add_array(
                    buffer_name_outer, [size],
                    field_dtype,
                    storage=dace.dtypes.StorageType.FPGA_Local,
                    transient=True)
            else:
                _, desc_outer = sdfg.add_scalar(
                    buffer_name_outer,
                    field_dtype,
                    storage=dace.dtypes.StorageType.FPGA_Registers,
                    transient=True)

            # Create read and write nodes
            read_node_outer = state.add_read(buffer_name_outer)
            write_node_outer = state.add_write(buffer_name_outer)

            # Outer buffer read
            state.add_memlet_path(
                read_node_outer,
                entry,
                nested_sdfg_tasklet,
                dst_conn=buffer_name_inner_read,
                memlet=dace.memlet.Memlet.simple(
                    buffer_name_outer, "0:{}".format(size), num_accesses=-1))

            # Outer buffer write
            state.add_memlet_path(
                nested_sdfg_tasklet,
                exit,
                write_node_outer,
                src_conn=buffer_name_inner_write,
                memlet=dace.memlet.Memlet.simple(
                    write_node_outer.data,
                    "0:{}".format(size),
                    num_accesses=-1))

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
                    "shift_{}".format(field_name),
                    {"i_shift": "0:{} - 1".format(size)},
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
                        shift_read.data, "i_shift + 1", num_accesses=1))
                shift_state.add_memlet_path(
                    shift_tasklet,
                    shift_exit,
                    shift_write,
                    src_conn=field_name + "_shift_out",
                    memlet=dace.memlet.Memlet.simple(
                        shift_write.data, "i_shift", num_accesses=1))

            # Begin reading according to this field's own buffer size, which is
            # translated to an index by subtracting it from the maximum buffer
            # size
            begin_reading = (init_size_max - init_size)
            end_reading = (functools.reduce(operator.mul, shape, 1) +
                           init_size_max - init_size)

            update_read = update_state.add_read(stream_name_inner)
            update_write = update_state.add_write(buffer_name_inner_write)
            update_tasklet = update_state.add_tasklet(
                "read_wavefront", {"wavefront_in"}, {"buffer_out"},
                "if ({it} >= {begin} && {it} < {end}) {{\n"
                "buffer_out = read_channel_intel(wavefront_in);\n"
                "}} else {{\n"
                "buffer_out = -1000;\n"
                "}}\n".format(
                    it=pipeline.iterator_str(),
                    begin=begin_reading,
                    end=end_reading),
                language=dace.dtypes.Language.CPP)
            update_state.add_memlet_path(
                update_read,
                update_tasklet,
                memlet=dace.memlet.Memlet.simple(
                    update_read.data, "0", num_accesses=-1),
                dst_conn="wavefront_in")
            update_state.add_memlet_path(
                update_tasklet,
                update_write,
                memlet=dace.memlet.Memlet.simple(
                    update_write.data,
                    "{} - 1".format(size) if size > 1 else "0",
                    num_accesses=1),
                src_conn="buffer_out")

            # Make compute state
            compute_read = compute_state.add_read(buffer_name_inner_read)
            for offset in buffer_accesses[field_name][1]:
                compute_state.add_memlet_path(
                    compute_read,
                    compute_tasklet,
                    dst_conn="_" + field_name + "_" + str(offset),
                    memlet=dace.memlet.Memlet.simple(
                        compute_read.data, str(offset), num_accesses=1))

        for field_name, connector in node.output_fields.items():

            data_name = field_to_data[field_name]

            # Outer write
            stream_name_outer = connector
            stream_name_inner = field_name + "_out"
            stream_outer = parent_sdfg.arrays[data_name].clone()
            stream_outer.transient = False
            sdfg.add_datadesc(stream_name_outer, stream_outer)
            write_node_outer = state.add_write(stream_name_outer)
            state.add_memlet_path(
                nested_sdfg_tasklet,
                exit,
                write_node_outer,
                src_conn=stream_name_inner,
                memlet=dace.memlet.Memlet.simple(
                    stream_name_outer, "0", num_accesses=-1))

            # Create inner stream
            stream_inner = stream_outer.clone()
            nested_sdfg.add_datadesc(stream_name_inner, stream_inner)

            # Inner write
            write_node_inner = compute_state.add_write(stream_name_inner)
            compute_state.add_memlet_path(
                compute_tasklet,
                write_node_inner,
                src_conn=field_name + "_inner_out",
                memlet=dace.memlet.Memlet.simple(
                    write_node_inner.data, "0", num_accesses=-1))

        sdfg.parent = parent_state
        sdfg._parent_sdfg = parent_sdfg  # TODO: this should not be necessary

        return sdfg


@dace.library.node
class Stencil(dace.library.LibraryNode):
    """Represents applying a stencil to a full input domain."""

    implementations = {"FPGA": ExpandStencilFPGA}
    default_implementation = "FPGA"

    # Iteration space definition
    iterators = dace.properties.ListProperty(
        element_type=str, desc="Iterators mapping the dimensions")
    shape = dace.properties.ListProperty(
        dace.symbolic.pystr_to_symbolic, desc="Shape of stencil dimensions")

    # Definition of stencil computation
    code = dace.properties.CodeProperty(
        desc="Stencil code using all inputs to produce all outputs",
        default="")
    code._language = dace.dtypes.Language.CPP
    accesses = dace.properties.Property(
        dtype=dict,
        desc=("For each input field, the corresponding input connector "
              "and a list of field accesses"),
        default={})
    output_fields = dace.properties.Property(
        dtype=dict,
        desc="List of output fields and their corresponding output connectors",
        default={})
    boundary_conditions = dace.properties.Property(
        dtype=dict,
        desc="Boundary condition specifications for each accessed field",
        default={})

    def __init__(self,
                 label,
                 iterators=[],
                 shape=[],
                 accesses={},
                 output_fields={},
                 boundary_conditions={},
                 code=""):
        in_connectors = [v[0] for v in accesses.values()]
        out_connectors = list(output_fields.values())
        super().__init__(label, inputs=in_connectors, outputs=out_connectors)
        self.iterators = iterators
        self.shape = shape
        self.accesses = accesses
        self.output_fields = output_fields
        self.boundary_conditions = boundary_conditions
        self.code = type(self).code.from_string(code, dace.dtypes.Language.CPP)

    def validate(state, sdfg):
        return True  # NYI


@dace.library.library
class Stencils:

    nodes = [Stencil]
    transformations = []
    default_implementation = None
