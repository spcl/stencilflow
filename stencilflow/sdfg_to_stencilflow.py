#!/usr/bin/env python3
import ast
import astunparse
import collections
import copy
import functools
import json
import re
import os
import warnings
from typing import Dict, Optional, Tuple

import dace
import stencilflow
from stencilflow.stencil import stencil
from stencilflow.stencil.nestk import NestK
from stencilflow.stencil.remove_loop import RemoveLoop
from stencilflow.stencil.stencilfusion import StencilFusion, ReplaceSubscript
from dace.data import Array
from dace.frontend.python.astutils import unparse, ASTFindReplace
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
                i_index = next(
                    (i
                     for i, s in enumerate(array.shape) if I in s.free_symbols),
                    -1)
                j_index = next(
                    (i
                     for i, s in enumerate(array.shape) if J in s.free_symbols),
                    -1)
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
                if len(init_tasklet.code.code) != 1:
                    dprint('Cannot remove scalar', dname, '(complex tasklet)')
                    continue
                if not isinstance(init_tasklet.code.code[0], ast.Assign):
                    dprint('Cannot remove scalar', dname, '(complex tasklet2)')
                    continue
                val = float(unparse(init_tasklet.code.code[0].value))

                dprint('Converting', dname, 'to constant with value', val)
                transients_to_remove[dname] = val
                # Remove initialization tasklet
                itstate.remove_node(init_tasklet)

        _remove_transients(sdfg, transients_to_remove)
        removed_transients += len(transients_to_remove)
    print('Cleaned up %d extra scalar transients' % removed_transients)


def remove_constant_stencils(top_sdfg: dace.SDFG):
    dprint = print  # lambda *args: pass
    removed_transients = 0
    for sdfg in top_sdfg.all_sdfgs_recursive():
        transients_to_remove = {}
        for state in sdfg.nodes():
            for node in state.nodes():
                if (isinstance(node, stencil.Stencil)
                        and state.in_degree(node) == 0
                        and state.out_degree(node) == 1):
                    # We can remove transient, find value from tasklet
                    if len(node.code.code) != 1:
                        dprint('Cannot remove scalar stencil', node.name,
                               '(complex code)')
                        continue
                    if not isinstance(node.code.code[0], ast.Assign):
                        dprint('Cannot remove scalar stencil', node.name,
                               '(complex code2)')
                        continue
                    # Ensure no one else is writing to it
                    onode = state.memlet_path(state.out_edges(node)[0])[-1].dst
                    dname = state.out_edges(node)[0].data.data
                    if any(
                            s.in_degree(n) > 0 for s in sdfg.nodes()
                            for n in s.nodes() if n != onode and isinstance(
                                n, dace.nodes.AccessNode) and n.data == dname):

                        continue
                    val = float(eval(unparse(node.code.code[0].value)))

                    dprint('Converting scalar stencil result', dname,
                           'to constant with value', val)
                    transients_to_remove[dname] = val
                    # Remove initialization tasklet
                    state.remove_node(node)

        _remove_transients(sdfg, transients_to_remove, ReplaceSubscript)
        removed_transients += len(transients_to_remove)
    print('Cleaned up %d extra scalar stencils' % removed_transients)


def _remove_transients(sdfg: dace.SDFG,
                       transients_to_remove: Dict[str, float],
                       replacer: ast.NodeTransformer = ASTFindReplace):
    """ Replaces transients with constants, removing associated access
        nodes. """
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
                        for e in state.memlet_tree(edge):
                            # Do not break scopes if there are no other edges
                            if len(state.edges_between(e.src, e.dst)) == 1:
                                state.add_edge(e.src, None, e.dst, None,
                                               dace.Memlet())
                            state.remove_edge_and_connectors(e)
                            # If tasklet, replace connector name with constant
                            if isinstance(e.dst, dace.nodes.Tasklet):
                                replacer({
                                    e.dst_conn: dname
                                }).visit(e.dst.code.code)
                            # If stencil, handle similarly
                            elif isinstance(e.dst, stencil.Stencil):
                                del e.dst.accesses[e.dst_conn]
                                for i, stmt in enumerate(e.dst.code.code):
                                    e.dst.code.code[i] = replacer({
                                        e.dst_conn:
                                        dname
                                    }).visit(stmt)
                            # If dst is a NestedSDFG, add the dst_connector as
                            # a constant and remove internal nodes
                            elif isinstance(e.dst, dace.nodes.NestedSDFG):
                                nsdfg: dace.SDFG = e.dst.sdfg
                                _remove_transients(nsdfg, {dname: val})

                    # Lastly, remove the node itself
                    state.remove_node(node)


def split_condition_interstate_edges(sdfg: dace.SDFG):
    edges_to_split = set()
    for isedge in sdfg.edges():
        if (not isedge.data.is_unconditional()
                and len(isedge.data.assignments) > 0):
            edges_to_split.add(isedge)

    for ise in edges_to_split:
        sdfg.remove_edge(ise)
        interim = sdfg.add_state()
        sdfg.add_edge(ise.src, interim, dace.InterstateEdge(ise.data.condition))
        sdfg.add_edge(interim, ise.dst,
                      dace.InterstateEdge(assignments=ise.data.assignments))


def canonicalize_sdfg(sdfg: dace.SDFG, symbols={}):
    # Clean up unnecessary subgraphs
    remove_scalar_transients(sdfg)
    remove_unused_sinks(sdfg)
    remove_constant_stencils(sdfg)
    split_condition_interstate_edges(sdfg)

    # Fuse and nest parallel K-loops
    sdfg.apply_transformations_repeated(MapFission, validate=False)
    standardize_data_layout(sdfg)
    sdfg.apply_transformations_repeated([NestK, InlineSDFG], validate=False)
    sdfg.apply_transformations_repeated([StencilFusion])

    # Remove loops
    loops_removed = sdfg.apply_transformations_repeated([RemoveLoop],
                                                        validate=False)
    if loops_removed > 0:
        raise ValueError("Control flow loops not supported.")

    from dace.transformation.interstate import StateFusion
    sdfg.apply_transformations_repeated(StateFusion)
    sdfg.apply_strict_transformations()

    # Specialize symbols and constants
    sdfg.specialize(symbols)
    symbols.update(sdfg.constants)
    undefined_symbols = sdfg.free_symbols
    if len(undefined_symbols) != 0:
        raise ValueError("Missing symbols: {}".format(
            ", ".join(undefined_symbols)))
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, stencil.Stencil):
            node.shape = _specialize_symbols(node.shape, symbols)
        if isinstance(node, dace.sdfg.nodes.MapEntry):
            ranges = []
            for r in node.map.range:
                ranges.append(_specialize_symbols(r, symbols))
            node.map.range = ranges

        # Make transformation passes on tasklets and stencil libnodes
        if hasattr(node, 'code'):

            new_code = [_Predicator().visit(stmt) for stmt in node.code.code]

            # min/max predication requires multiple passes (nested expressions)
            minmax_predicated = 1
            while minmax_predicated > 0:
                pred = _MinMaxPredicator()
                tmp_code = [pred.visit(stmt) for stmt in new_code]
                minmax_predicated = pred.count

                # Some of the outputs may be lists, flatten
                new_code = []

                def flatten(val):
                    for v in val:
                        if isinstance(v, list):
                            flatten(v)
                        else:
                            new_code.append(v)

                flatten(tmp_code)

            node.code.code = new_code

    return sdfg


class _Predicator(ast.NodeTransformer):
    def visit_If(self, node: ast.If):
        if len(node.body) == 1 and len(node.orelse) == 1:
            if not isinstance(node.body[0], ast.Assign):
                return self.generic_visit(node)
            if not isinstance(node.orelse[0], ast.Assign):
                return self.generic_visit(node)
            if_assign: ast.Assign = node.body[0]
            else_assign: ast.Assign = node.orelse[0]
            if len(if_assign.targets) != 1 or len(else_assign.targets) != 1:
                return self.generic_visit(node)

            # Replace the condition with a predicated ternary expression
            if astunparse.unparse(if_assign.targets[0]) == astunparse.unparse(
                    else_assign.targets[0]):
                new_node = ast.Assign(targets=if_assign.targets,
                                      value=ast.IfExp(test=node.test,
                                                      body=if_assign.value,
                                                      orelse=else_assign.value))
                return ast.copy_location(new_node, node)
        return self.generic_visit(node)


class _MinMaxPredicator(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_Assign(self, node: ast.Assign):
        if not isinstance(node.value, ast.Call):
            return self.generic_visit(node)
        if len(node.targets) != 1:
            return self.generic_visit(node)

        target = node.targets[0]
        if isinstance(target, ast.Subscript):
            target = target.value
        if not isinstance(target, ast.Name):
            return self.generic_visit(node)
        tname: str = target.id

        callnode: ast.Call = node.value
        fname = astunparse.unparse(callnode.func)[:-1]
        if fname not in ('min', 'max'):
            return self.generic_visit(node)
        if len(callnode.args) != 2:
            raise NotImplementedError('Arguments to min/max (%d) != 2' %
                                      len(callnode.args))

        result = []
        names = []
        for i, arg in enumerate(callnode.args):
            newname = '__dace_%s%d_%s' % (fname, i, tname)
            names.append(newname)
            result.append(ast.Assign(targets=[ast.Name(id=newname)], value=arg))

        result.append(
            ast.Assign(
                targets=node.targets,
                value=ast.IfExp(
                    test=ast.Compare(left=ast.Name(id=names[0]),
                                     ops=[ast.Lt()],
                                     comparators=[ast.Name(id=names[1])]),
                    body=ast.Name(id=names[0]) if fname == 'min' else ast.Name(
                        id=names[1]),
                    orelse=ast.Name(id=names[1])
                    if fname == 'min' else ast.Name(id=names[0]))))
        self.count += 1
        return result


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
    def __init__(self, rename_map, offset, accesses, constants):
        self._rename_map = rename_map
        self._offset = offset
        self._accesses = accesses
        self._constants = constants
        self._defined = set()
        self._operation_count = 0

    @property
    def operation_count(self):
        return self._operation_count

    @staticmethod
    def _offset_to_index(node, offset, iterator):
        if isinstance(node, ast.Num):
            num = int(node.n)
            if num == 0:
                return iterator
            else:
                return "{} + {}".format(iterator, num)
        elif isinstance(node, ast.UnaryOp):
            return "{} - {}".format(iterator, int(node.operand.n))
        raise TypeError("Unrecognized offset: {}".format(
            astunparse.unparse(node)))

    def visit_Subscript(self, node: ast.Subscript):
        # Convert [0, 1, -1] to [i, j + 1, k - 1]
        offsets = [
            o for o, v in zip(self._offset, self._accesses[node.value.id][0])
            if v
        ]
        iterators = [
            i for i, v in zip(stencilflow.ITERATORS, self._accesses[
                node.value.id][0]) if v
        ]
        if isinstance(node.slice.value, ast.Tuple):
            # Negative indices show up as a UnaryOp, others as Num
            indices = tuple(
                _RenameTransformer._offset_to_index(n, o, i)
                for n, o, i in zip(node.slice.value.elts, offsets, iterators))
        else:
            # One dimensional access doesn't show up as a tuple
            indices = (_RenameTransformer._offset_to_index(
                node.slice.value, offsets[0], iterators[0]), )
        indices = "({})".format(", ".join(map(str, indices)))
        node.slice.value = ast.parse(str(indices)).body[0].value
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

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            self._defined.add(target.id)
        self.generic_visit(node)
        return node

    def visit_Call(self, node: ast.Call):
        self._defined.add(node.func.id)
        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self._rename_map:
            node.id = self._rename_map[node.id]
        elif node.id in self._constants:
            pass
        elif node.id in stencilflow.ITERATORS:
            pass
        elif node.id in self._defined:
            pass
        else:
            raise ValueError("Unrecognized variable: {}".format(node.id))
        self.generic_visit(node)
        return node


def sdfg_to_stencilflow(sdfg,
                        output_path,
                        vector_length=1,
                        data_directory=None):

    if not isinstance(sdfg, dace.SDFG):
        sdfg = dace.SDFG.from_file(sdfg)

    undefined_symbols = sdfg.free_symbols
    if len(undefined_symbols) != 0:
        raise ValueError("Undefined symbols in SDFG: {}".format(
            ", ".join(undefined_symbols)))

    reads = {}
    writes = set()
    global_data = {k for k, v in sdfg.arrays.items() if not v.transient}

    result = {
        "inputs": {},
        "outputs": [],
        "dimensions": None,
        "vectorization": vector_length,
        "program": {}
    }

    versions = {}  # {field: count}

    shape = []

    # Has to be a list so we can pass by reference
    operation_count = [0]

    # Retrieve topological order of all stencils present in the program
    def _make_topological_order(sdfg, topological_order):
        for state in dace.sdfg.utils.dfs_topological_sort(
                sdfg, sdfg.source_nodes()):
            for node in dace.sdfg.utils.dfs_topological_sort(
                    state, state.source_nodes()):
                if isinstance(node, stencil.Stencil):
                    if len(node.output_fields) != 1:
                        raise ValueError(
                            "Only 1 output per stencil is supported, "
                            "but {} has {} outputs.".format(
                                node.label, len(node.output_fields)))
                    out_edges = {e.src_conn: e for e in state.out_edges(node)}
                    for connector in node.output_fields:
                        write_node = dace.sdfg.find_output_arraynode(
                            state, out_edges[connector])
                        output = write_node.data
                        break  # Grab first and only element
                    writes.add(output)  # Can have duplicates
                    topological_order.append((node, state, sdfg, output))
                elif isinstance(node, dace.sdfg.nodes.Tasklet):
                    warnings.warn("Ignored tasklet {}".format(node.label))
                elif isinstance(node, dace.sdfg.nodes.AccessNode):
                    pass
                elif isinstance(node, dace.sdfg.nodes.NestedSDFG):
                    _make_topological_order(node.sdfg, topological_order)
                elif isinstance(node, dace.sdfg.SDFGState):
                    pass
                else:
                    raise TypeError("Unsupported node type in {}: {}".format(
                        state.label,
                        type(node).__name__))

    topological_order = []  # [(node, state, sdfg, name of output field)]
    _make_topological_order(sdfg, topological_order)

    # Do versioning if writes, so that the last output has the original name
    output_versions = {}  # {node: output_name}
    output_fields = global_data & writes
    temp_fields = writes - output_fields
    if len(output_fields) == 0:
        raise ValueError("Program has no outputs.")
    for field in output_fields:
        writing_nodes = [n for (n, _, _, o) in topological_order if o == field]
        for i, n in enumerate(writing_nodes):
            if i < len(writing_nodes) - 1:
                rename = "{}__{}".format(field, i + 1)
                print("Versioned write to {} to {} in {}.".format(
                    field, rename, n))
            else:
                # Last write should have original field name
                rename = field
            output_versions[n] = rename
    for field in temp_fields:
        w = 0
        for node, _, _, output in topological_order:
            if output == field:
                if w == 0:
                    rename = field
                else:
                    rename = "{}__{}".format(field, w)
                    print("Versioned {} to {} in {}.".format(
                        field, rename, node))
                output_versions[node] = rename
                w += 1
    # Do versioning of inputs
    input_versions = {}  # {(node, input): input_name}
    current_name = {}
    for node, state, sdfg, output in topological_order:
        in_edges = {e.dst_conn: e for e in state.in_edges(node)}
        for connector in node.accesses:
            field = dace.sdfg.find_input_arraynode(state,
                                                   in_edges[connector]).data
            if field in current_name:
                rename = current_name[field]  # Use version if any
            else:
                rename = field
            input_versions[(node, field)] = rename
        current_name[output] = output_versions[node]  # Progress renaming

    # Now we can start doing a pass
    for node, state, sdfg, _ in topological_order:

        stencil_json = {}

        boundary_conditions = collections.OrderedDict()

        in_edges = {e.dst_conn: e for e in state.in_edges(node)}
        out_edges = {e.src_conn: e for e in state.out_edges(node)}

        current_sdfg = sdfg

        rename_map = {}

        for connector, accesses in node.accesses.items():
            read_node = dace.sdfg.find_input_arraynode(state,
                                                       in_edges[connector])
            # Use array node instead of connector name
            field = read_node.data
            dtype = current_sdfg.data(read_node.data).dtype.type.__name__
            name = input_versions[(node, field)]
            rename_map[connector] = name
            boundary_conditions[name] = (node.boundary_conditions[connector]
                                         if connector
                                         in node.boundary_conditions else None)
            if name in reads:
                if reads[name][0] != dtype:
                    raise ValueError("Type mismatch: {} vs. {}".format(
                        reads[name][0], dtype))
            else:
                reads[name] = (dtype, accesses[0])

        for connector in node.output_fields:
            break  # Get first and only element
        write_node = dace.sdfg.find_output_arraynode(state,
                                                     out_edges[connector])
        # Rename to array node
        output_connector = connector
        field = write_node.data
        # Add new version
        rename = output_versions[node]
        rename_map[connector] = rename
        stencil_json["data_type"] = current_sdfg.data(
            write_node.data).dtype.type.__name__
        rename_map[connector] = rename
        output = rename

        for field, bc in boundary_conditions.items():
            if bc is None:
                # Use output boundary condition
                boundary_conditions[field] = node.boundary_conditions[
                    output_connector]

        # Now we need to go rename versioned variables in the
        # stencil code
        output_transformer = _OutputTransformer()
        old_ast = ast.parse(node.code.as_string)
        new_ast = output_transformer.visit(old_ast)
        output_offset = output_transformer.offset
        rename_transformer = _RenameTransformer(rename_map, output_offset,
                                                node.accesses, sdfg.constants)
        new_ast = rename_transformer.visit(new_ast)
        operation_count[0] += rename_transformer.operation_count
        code = astunparse.unparse(new_ast)
        stencil_name = output
        stencil_json["computation_string"] = code
        stencil_json["boundary_conditions"] = boundary_conditions

        if stencil_name in result["program"]:
            raise ValueError("Duplicate stencil: {}".format(stencil_name))

        result["program"][stencil_name] = stencil_json

        # Extract stencil shape from stencil
        s = list(node.shape)
        if len(shape) == 0:
            shape += s
        else:
            if s != shape:
                prod_old = functools.reduce(lambda a, b: a * b, shape)
                prod_new = functools.reduce(lambda a, b: a * b, s)
                if prod_new > prod_old:
                    updated = s
                else:
                    updated = shape
                warnings.warn("Stencil shape mismatch: {} vs. {}. "
                              "Setting to maximum {}.".format(
                                  shape, s, updated))
                shape = updated

    print("Found {} arithmetic operations.".format(operation_count[0]))

    unroll_factor = 1
    for count in versions.values():
        if count > unroll_factor:
            unroll_factor = count

    shape = tuple(map(int, shape))
    result["dimensions"] = shape

    result["constants"] = {}
    for k, (container, val) in sdfg.constants_prop.items():
        result["constants"][k] = {
            "value": str(val),
            "data_type": container.dtype.type.__name__
        }

    result["outputs"] = list(sorted([i for i in writes if i in global_data]))
    if len(result["outputs"]) == 0:
        raise ValueError("SDFG has no non-transient outputs.")
    for field, (dtype, dimensions) in reads.items():
        if field not in global_data:
            continue  # This is not an input
        path = "{}_{}_{}.dat".format(
            field, "x".join(
                map(str, [
                    d
                    for i, d in enumerate(result["dimensions"]) if dimensions[i]
                ])), dtype)
        if data_directory is not None:
            path = os.path.join(data_directory, path)
        result["inputs"][field] = {
            "data":
            path,
            "data_type":
            dtype,
            "input_dims":
            [p for i, p in enumerate(stencilflow.ITERATORS) if dimensions[i]]
        }
    if len(result["inputs"]) == 0:
        raise ValueError("SDFG has no inputs.")

    with open(output_path, "w") as out_file:
        out_file.write(json.dumps(result, indent=True))
