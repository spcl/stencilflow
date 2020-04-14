import ast
import astunparse
import collections
import itertools

import dace
import numpy as np
from .subscript_converter import SubscriptConverter

JUNK_VAL = -100000


@dace.library.expansion
class ExpandStencilCPU(dace.library.ExpandTransformation):

    environments = []

    @staticmethod
    def expansion(node, parent_state, parent_sdfg):

        sdfg = dace.SDFG(node.label + "_outer")
        state = sdfg.add_state(node.label + "_outer")

        shape = np.array(node.shape)

        parameters = np.array(["i", "j", "k"])[:len(shape)]

        # Find outer data descriptor
        field_dtype = {}
        for e in parent_state.in_edges(node):
            field = e.dst_connector
            if field in node.accesses:
                field_dtype[field] = parent_sdfg.data(
                    dace.sdfg.find_input_arraynode(parent_state, e).data).dtype
        for e in parent_state.out_edges(node):
            field = e.src_connector
            if field in node.output_fields:
                field_dtype[field] = parent_sdfg.data(
                    dace.sdfg.find_output_arraynode(parent_state,
                                                    e).data).dtype

        #######################################################################
        # Tasklet code generation
        #######################################################################

        code = node.code.as_string

        # Replace relative indices with memlet names
        converter = SubscriptConverter()
        new_ast = converter.visit(ast.parse(code))
        code = astunparse.unparse(new_ast)
        code_memlet_names = converter.names

        #######################################################################
        # Implement boundary conditions
        #######################################################################

        boundary_code = ""
        # Loop over each input
        for field_name, (iterators, accesses) in node.accesses.items():
            # Loop over each access to this data
            for indices in accesses:
                try:
                    memlet_name = code_memlet_names[field_name][indices]
                except KeyError:
                    raise KeyError("Missing access in code: {}[{}]".format(
                        field_name, ", ".join(map(str, indices))))
                cond = []
                # Loop over each index of this access
                for i, offset in enumerate(indices):
                    if offset < 0:
                        cond.append(parameters[i] + " < " + str(-offset))
                    elif offset > 0:
                        cond.append(parameters[i] + " >= " +
                                    str(shape[i] - offset))
                ctype = field_dtype[field_name]
                if len(cond) == 0:
                    boundary_code += "{} = {}_in\n".format(
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
                    boundary_code += ("{} = {} if {} else {}_in\n".format(
                        memlet_name, boundary_val, " or ".join(cond),
                        memlet_name))

        #######################################################################
        # Write all output memlets
        #######################################################################

        write_code = "\n".join("{}_out = {}".format(
            code_memlet_names[output][tuple(
                0
                for _ in range(len(shape)))], code_memlet_names[output][tuple(
                    0 for _ in range(len(shape)))], output)
                               for output in node.output_fields)

        code = boundary_code + "\n" + code + "\n" + write_code

        input_memlets = sum(
            [["{}_in".format(c) for c in v.values()]
             for k, v in code_memlet_names.items() if k in node.accesses], [])
        output_memlets = sum(
            [["{}_out".format(c) for c in v.values()]
             for k, v in code_memlet_names.items() if k in node.output_fields],
            [])

        #######################################################################
        # Create tasklet
        #######################################################################

        tasklet = state.add_tasklet(node.label + "_compute",
                                    input_memlets,
                                    output_memlets,
                                    code,
                                    language=dace.dtypes.Language.Python)

        #######################################################################
        # Build dataflow state
        #######################################################################

        entry, exit = state.add_map(
            node.name + "_map",
            collections.OrderedDict((parameters[i], "0:" + str(shape[i]))
                                    for i in range(len(shape))))

        for field in code_memlet_names:

            dtype = field_dtype[field]
            data = sdfg.add_array(field, shape, dtype)

            if field in node.accesses:
                read_node = state.add_read(field)
                field_parameters = parameters[node.accesses[field][0]]
                for indices, connector in code_memlet_names[field].items():
                    access_str = ", ".join(
                        "{} + ({})".format(p, i)
                        for p, i in zip(field_parameters, indices))
                    memlet = dace.Memlet.simple(field,
                                                access_str,
                                                num_accesses=-1)
                    memlet.allow_oob = True
                    state.add_memlet_path(read_node,
                                          entry,
                                          tasklet,
                                          dst_conn=connector + "_in",
                                          memlet=memlet)
            else:
                write_node = state.add_write(field)
                for indices, connector in code_memlet_names[field].items():
                    state.add_memlet_path(tasklet,
                                          exit,
                                          write_node,
                                          src_conn=connector + "_out",
                                          memlet=dace.Memlet.simple(
                                              field, ", ".join(parameters)))

        #######################################################################

        sdfg.parent = parent_state
        sdfg._parent_sdfg = parent_sdfg  # TODO: this should not be necessary

        return sdfg
