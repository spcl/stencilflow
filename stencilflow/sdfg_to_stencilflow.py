#!/usr/bin/env python3
import ast
import astunparse
import json
import re
import warnings
from typing import Optional, Tuple

import dace
from stencilflow.stencil import stencil
from stencilflow.stencil.nestk import NestK
from dace.data import Array
from dace.frontend.python.astutils import unparse
from dace.transformation.dataflow import MapFission
from dace.transformation.interstate import LoopUnroll, InlineSDFG


def _specialize_symbols(iterable, symbols):
    specialized = ", ".join(map(str, iterable))
    for k, v in symbols.items():
        specialized = re.sub(r"\b{}\b".format(k), str(v), specialized)
    return type(iterable)(int(dace.sympy.sympify(v))
                          for v in specialized.split(", "))


def _permute_array(array: Array, perm: Tuple[int, int, int], sdfg: dace.SDFG,
                   array_name: str):
    array.shape = [array.shape[i] for i in perm]
    array.strides = [array.strides[i] for i in perm]
    array.offset = [array.offset[i] for i in perm]
    # Modify all edges coming in/out of the array
    for state in sdfg.nodes():
        for e in state.edges():
            if e.data.data == array_name:
                e.data.subset = type(
                    e.data.subset)([e.data.subset[i] for i in perm])


def standardize_data_layout(sdfg):
    I, J, K = tuple(dace.symbol(sym) for sym in 'IJK')

    for nsdfg in sdfg.all_sdfgs_recursive():
        for aname, array in nsdfg.arrays.items():
            if K in array.free_symbols or len(array.shape) == 3:
                i_index = next((i for i, s in enumerate(array.shape)
                                if I in s.free_symbols), -1)
                j_index = next((i for i, s in enumerate(array.shape)
                                if J in s.free_symbols), -1)
                # The K index is the remainder after I and J have been detected
                # (this helps work with subranges of the vertical domain)
                k_index = next(i for i, _ in enumerate(array.shape)
                               if i not in (i_index, j_index))
                # NOTE: We use the J, K, I format here. To change, permute
                # the order below.
                order = tuple(dim for dim in (j_index, k_index, i_index)
                              if dim >= 0)
                _permute_array(array, order, nsdfg, aname)


def remove_unused_sinks(top_sdfg: dace.SDFG):
    """ Remove unused transient sink nodes and their generating
        computation. """
    for sdfg in top_sdfg.all_sdfgs_recursive():
        for state in sdfg.nodes():
            toremove = set()
            map_sink_nodes = [
                n for n in state.nodes() if state.out_degree(n) == 1
                and state.out_edges(n)[0].data.data is None
            ]
            for node in state.sink_nodes() + map_sink_nodes:
                if (isinstance(node, dace.nodes.AccessNode)
                        and sdfg.arrays[node.data].transient):
                    if len([
                            n for s in sdfg.nodes() for n in s.nodes()
                            if isinstance(n, dace.nodes.AccessNode)
                            and n.data == node.data
                    ]) == 1:
                        if state.in_degree(node) == 1:
                            predecessor = state.in_edges(node)[0].src
                            # Only remove the node (and its predecessor) if it
                            # has one unique predecessor that is not connected
                            # to anything else
                            if (state.out_degree(predecessor) == 1
                                    and isinstance(predecessor,
                                                   dace.nodes.CodeNode)):
                                # Also remove potentially isolated input nodes
                                for e in state.in_edges(predecessor):
                                    if len(state.all_edges(e.src)) == 1:
                                        toremove.add(e.src)
                                toremove.add(predecessor)
                                toremove.add(node)

            state.remove_nodes_from(toremove)


def remove_scalar_transients(top_sdfg: dace.SDFG):
    """ Clean up tasklet->scalar-transient, replacing them with symbols. """
    dprint = print  # lambda *args: pass
    removed_transients = 0
    for sdfg in top_sdfg.all_sdfgs_recursive():
        transients_to_remove = {}
        for dname, desc in sdfg.arrays.items():
            skip = False
            if isinstance(desc, dace.data.Scalar) and desc.transient:
                # Find node where transient is instantiated
                init_tasklet: Optional[dace.nodes.Tasklet] = None
                itstate = None
                for state in sdfg.nodes():
                    if skip:
                        break
                    for node in state.nodes():
                        if (isinstance(node, dace.nodes.AccessNode)
                                and node.data == dname):
                            if state.in_degree(node) > 1:
                                dprint('Cannot remove scalar', dname,
                                       '(more than one input)')
                                skip = True
                                break
                            elif state.in_degree(node) == 1:
                                if init_tasklet is not None:
                                    dprint('Cannot remove scalar', dname,
                                           '(initialized multiple times)')
                                    skip = True
                                    break
                                init_tasklet = state.in_edges(node)[0].src
                                itstate = state
                if init_tasklet is None:
                    dprint('Cannot remove scalar', dname, '(uninitialized)')
                    skip = True
                if skip:
                    continue

                # We can remove transient, find value from tasklet
                if len(init_tasklet.code) != 1:
                    dprint('Cannot remove scalar', dname, '(complex tasklet)')
                    continue
                if not isinstance(init_tasklet.code[0], ast.Assign):
                    dprint('Cannot remove scalar', dname, '(complex tasklet2)')
                    continue
                val = float(unparse(init_tasklet.code[0].value))

                dprint('Converting', dname, 'to constant with value', val)
                transients_to_remove[dname] = val
                # Remove initialization tasklet
                itstate.remove_node(init_tasklet)

        # Remove transients
        for dname, val in transients_to_remove.items():
            # Add constant, remove data descriptor
            del sdfg.arrays[dname]
            sdfg.add_constant(dname, val)

            for state in sdfg.nodes():
                for node in state.nodes():
                    if (isinstance(node, dace.nodes.AccessNode)
                            and node.data == dname):
                        # For all access node instances, remove
                        # outgoing edge connectors from subsequent nodes,
                        # then remove access nodes
                        for edge in state.out_edges(node):
                            edge.dst._in_connectors.remove(edge.dst_conn)
                            # If dst is a NestedSDFG, add the dst_connector as
                            # a constant too and remove internal nodes
                            # TODO(later): refactor into a function
                            if isinstance(edge.dst, dace.nodes.NestedSDFG):
                                nsdfg: dace.SDFG = edge.dst.sdfg
                                nsdfg.add_constant(edge.dst_conn, val)
                                del nsdfg.arrays[edge.dst_conn]
                                for nstate in nsdfg.nodes():
                                    for nnode in nstate.nodes():
                                        if (isinstance(nnode,
                                                       dace.nodes.AccessNode)
                                                and
                                                nnode.data == edge.dst_conn):
                                            for nedge in nstate.out_edges(
                                                    nnode):
                                                nedge.dst._in_connectors.remove(
                                                    nedge.dst_conn)
                                            nstate.remove_node(nnode)

                        # Lastly, remove the node itself
                        state.remove_node(node)

        removed_transients += len(transients_to_remove)
    print('Cleaned up %d extra scalar transients' % removed_transients)


def canonicalize_sdfg(sdfg, symbols={}):
    # Clean up unnecessary subgraphs
    remove_scalar_transients(sdfg)

    # Fuse and nest parallel K-loops
    sdfg.apply_transformations_repeated(MapFission, validate=False)
    standardize_data_layout(sdfg)
    sdfg.apply_transformations_repeated([NestK, InlineSDFG], validate=False)

    # Specialize symbols
    sdfg.specialize(symbols)
    undefined_symbols = sdfg.undefined_symbols(False)
    if len(undefined_symbols) != 0:
        raise ValueError("Missing symbols: {}".format(
            ", ".join(undefined_symbols)))
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, stencil.Stencil):
            node.shape = _specialize_symbols(node.shape, symbols)
        if isinstance(node, dace.graph.nodes.MapEntry):
            ranges = []
            for r in node.map.range:
                ranges.append(_specialize_symbols(r, symbols))
            node.map.range = ranges

    # Unroll sequential K-loops
    sdfg.apply_transformations_repeated([LoopUnroll], validate=False)

    return sdfg


class _OutputTransformer(ast.NodeTransformer):
    def __init__(self):
        self._offset = (0, 0, 0)
        self._stencil_name = None

    @property
    def stencil_name(self):
        return self._stencil_name

    @property
    def offset(self):
        return self._offset

    def visit_Assign(self, node: ast.Assign):
        if len(node.targets) != 1:
            raise ValueError  # Sanity check
        for i, subscript_node in enumerate(node.targets):
            try:
                indices = ast.literal_eval(subscript_node.slice.value)
                self._stencil_name = subscript_node.value.id
            except AttributeError:
                # This is an unsubscripted Name node, just grab the name
                self._stencil_name = subscript_node.id
                break
            if any(indices):
                self._offset = indices
            # Remove subscript
            node.targets[i] = subscript_node.value
        self.generic_visit(node)
        return node


class _RenameTransformer(ast.NodeTransformer):
    def __init__(self, rename_map, offset, accesses):
        self._rename_map = rename_map
        self._offset = offset
        self._accesses = accesses
        self._operation_count = 0

    @property
    def operation_count(self):
        return self._operation_count

    def visit_Subscript(self, node: ast.Subscript):
        # Convert [0, 1, -1] to [i, j + 1, k - 1]
        offsets = [
            offset for offset, valid in zip(self._offset, self._accesses[
                node.value.id][0]) if valid
        ]
        if isinstance(node.slice.value, ast.Tuple):
            # Negative indices show up as a UnaryOp, others as Num
            indices = (x.n if isinstance(x, ast.Num) else -x.operand.n
                       for x in node.slice.value.elts)
        else:
            # One dimensional access doesn't show up as a tuple
            indices = (node.slice.value.n if isinstance(
                node.slice.value, ast.Num) else -node.slice.value.operand.n, )
        indices = tuple(x - o for x, o in zip(indices, offsets))
        t = "(" + ", ".join(
            stencilflow.ITERATORS[i] +
            (" + " + str(o) if o > 0 else (" - " + str(-o) if o < 0 else ""))
            for i, o in enumerate(indices)) + ")"
        node.slice.value = ast.parse(t).body[0].value
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.left, ast.Subscript) or isinstance(
                node.left, ast.BinOp) or isinstance(
                    node.right, ast.Subscript) or isinstance(
                        node.right, ast.BinOp):
            self._operation_count += 1
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self._rename_map:
            node.id = self._rename_map[node.id]
        self.generic_visit(node)
        return node


def sdfg_to_stencilflow(sdfg, output_path, data_directory=None):

    if not isinstance(sdfg, dace.SDFG):
        sdfg = dace.SDFG.from_file(sdfg)

    undefined_symbols = sdfg.undefined_symbols(False)
    if len(undefined_symbols) != 0:
        raise ValueError("Undefined symbols in SDFG: {}".format(
            ", ".join(undefined_symbols)))

    reads = {}
    writes = set()

    result = {"inputs": {}, "outputs": [], "dimensions": None, "program": {}}

    versions = {}  # {field: count}

    shape = []

    # Has to be a list so we can pass by reference
    operation_count = [0]

    def _visit(sdfg, reads, writes, result, versions, shape, operation_count):

        for state in dace.graph.nxutil.dfs_topological_sort(
                sdfg, sdfg.source_nodes()):

            for node in dace.graph.nxutil.dfs_topological_sort(
                    state, state.source_nodes()):

                if isinstance(node, stencil.Stencil):

                    stencil_json = {}

                    in_edges = {e.dst_conn: e for e in state.in_edges(node)}
                    out_edges = {e.src_conn: e for e in state.out_edges(node)}

                    current_sdfg = sdfg

                    rename_map = {}

                    for connector, accesses in node.accesses.items():
                        read_node = dace.sdfg.find_input_arraynode(
                            state, in_edges[connector])
                        # Use array node instead of connector name
                        field = read_node.data
                        dtype = current_sdfg.data(
                            read_node.data).dtype.type.__name__
                        # Do versioning
                        if field not in versions:
                            versions[field] = 0
                        if versions[field] == 0:
                            name = field
                        else:
                            name = "{}_{}".format(field, versions[field])
                            print("Versioned {} to {}.".format(field, name))
                        rename_map[connector] = name
                        if name in reads:
                            if reads[name] != dtype:
                                raise ValueError(
                                    "Type mismatch: {} vs. {}".format(
                                        reads[name], dtype))
                        else:
                            reads[name] = dtype

                    if len(node.output_fields) != 1:
                        raise ValueError(
                            "Only 1 output per stencil is supported, "
                            "but {} has {} outputs.".format(
                                node.label, len(node.output_fields)))

                    for connector in node.output_fields:
                        write_node = dace.sdfg.find_output_arraynode(
                            state, out_edges[connector])
                        # Rename to array node
                        field = write_node.data
                        # Add new version
                        if field not in versions:
                            versions[field] = 0
                            rename = field
                        else:
                            versions[field] = versions[field] + 1
                            rename = "{}_{}".format(field, versions[field])
                            print("Versioned {} to {}.".format(field, rename))
                        stencil_json["data_type"] = current_sdfg.data(
                            write_node.data).dtype.type.__name__
                        rename_map[connector] = rename
                        output = rename
                        break  # Grab first and only element

                    if output in writes:
                        raise KeyError(
                            "Multiple writes to field: {}".format(field))
                    writes.add(output)

                    # Now we need to go rename versioned variables in the
                    # stencil code
                    output_transformer = _OutputTransformer()
                    old_ast = ast.parse(node.code.as_string)
                    new_ast = output_transformer.visit(old_ast)
                    output_offset = output_transformer.offset
                    rename_transformer = _RenameTransformer(
                        rename_map, output_offset, node.accesses)
                    new_ast = rename_transformer.visit(new_ast)
                    operation_count[0] += rename_transformer.operation_count
                    code = astunparse.unparse(new_ast)
                    stencil_name = output
                    stencil_json["computation_string"] = code

                    # Also rename boundary conditions
                    # TODO: remove workaround due to wrongly named BCs
                    stencil_json["boundary_conditions"] = {}
                    for k, v in node.boundary_conditions.items():
                        if k not in rename_map:
                            k += "_in"
                        k = rename_map[k]
                        stencil_json["boundary_conditions"][k] = v

                    result["program"][stencil_name] = stencil_json

                    # Extract stencil shape from stencil
                    s = list(node.shape)
                    if len(shape) == 0:
                        shape += s
                    else:
                        if s != shape:
                            raise ValueError(
                                "Stencil shape mismatch: {} vs. {}".format(
                                    shape, s))

                elif isinstance(node, dace.graph.nodes.Tasklet):
                    warnings.warn("Ignored tasklet {}".format(node.label))

                elif isinstance(node, dace.graph.nodes.AccessNode):
                    pass

                elif isinstance(node, dace.graph.nodes.NestedSDFG):
                    _visit(node.sdfg, reads, writes, result, versions, shape,
                           operation_count)

                elif isinstance(node, dace.sdfg.SDFGState):
                    pass

                else:
                    raise TypeError("Unsupported node type in {}: {}".format(
                        state.label,
                        type(node).__name__))

                # End node loop

    _visit(sdfg, reads, writes, result, versions, shape, operation_count)

    print("Found {} arithmetic operations.".format(operation_count[0]))

    unroll_factor = 1
    for count in versions.values():
        if count > unroll_factor:
            unroll_factor = count

    shape = tuple(map(int, shape))
    result["dimensions"] = shape

    inputs = reads.keys() - writes
    outputs = writes - reads.keys()

    result["outputs"] = list(sorted(outputs))
    for field in inputs:
        dtype = reads[field]
        path = "{}_{}_{}.dat".format(field,
                                     "x".join(map(str, result["dimensions"])),
                                     dtype)
        if data_directory is not None:
            path = os.path.join(data_directory, path)
        result["inputs"][field] = {"data": path, "data_type": dtype}

    with open(output_path, "w") as out_file:
        out_file.write(json.dumps(result, indent=True))
