#!/usr/bin/env python3
import ast
import astunparse
import json
import re
import warnings

import dace
from stencilflow.stencil import stencil
from stencilflow.stencil.nestk import NestK
from dace.transformation.pattern_matching import Transformation
from dace.transformation.dataflow import MapFission
from dace.transformation.interstate import LoopUnroll

PARAMETERS = ["i", "j", "k"]


def canonicalize_sdfg(sdfg, symbols={}):

    # Specialize symbols
    sdfg.specialize(symbols)
    undefined_symbols = sdfg.undefined_symbols(False)
    if len(undefined_symbols) != 0:
        raise ValueError("Missing symbols: {}".format(
            ", ".join(undefined_symbols)))

    # Unroll sequential K-loops
    sdfg.apply_transformations_repeated([LoopUnroll], validate=False)

    # Fuse and nest parallel K-loops
    strict = [
        k for k, v in Transformation.extensions().items()
        if v.get('strict', False)
    ]
    extra = [NestK, MapFission]

    return sdfg.apply_transformations_repeated(strict + extra, validate=False)


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
    def __init__(self, rename_map, offset):
        self._rename_map = rename_map
        self._offset = offset
        self._output_count = 0

    def visit_Index(self, node: ast.Subscript):
        # Convert [0, 1, -1] to [i, j + 1, k - 1]
        offsets = tuple(x.n - self._offset[i]
                        for i, x in enumerate(node.value.elts))
        t = "(" + ", ".join(PARAMETERS[i] +
                            (" + " + str(o) if o > 0 else
                             (" - " + str(o) if o < 0 else ""))
                            for i, o in enumerate(offsets)) + ")"
        node.value = ast.parse(t).body[0].value
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

    def _visit(sdfg, reads, writes, result, versions, shape):

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
                    old_ast = ast.parse(node.code)
                    new_ast = output_transformer.visit(old_ast)
                    output_offset = output_transformer.offset
                    rename_transformer = _RenameTransformer(
                        rename_map, output_offset)
                    new_ast = rename_transformer.visit(new_ast)
                    code = astunparse.unparse(new_ast)
                    stencil_name = output
                    stencil_json["computation_string"] = code

                    # Also rename boundary conditions
                    stencil_json["boundary_conditions"] = {
                        rename_map[k]: v
                        for k, v in node.boundary_conditions.items()
                    }

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
                    _visit(node.sdfg, reads, writes, result, versions, shape)

                elif isinstance(node, dace.sdfg.SDFGState):
                    pass

                else:
                    raise TypeError("Unsupported node type in {}: {}".format(
                        state.label,
                        type(node).__name__))

                # End node loop

    _visit(sdfg, reads, writes, result, versions, shape)

    unroll_factor = 1
    for count in versions.values():
        if count > unroll_factor:
            unroll_factor = count

    shape = ", ".join(map(str, shape))
    for k, v in symbols.items():
        shape = re.sub(r"\b{}\b".format(k), str(v), shape)
    shape = tuple(int(v) for v in shape.split(", "))
    if result["dimensions"] is None:
        result["dimensions"] = shape
    else:
        if shape != result["dimensions"]:
            raise ValueError(
                "Conflicting shapes found: {} vs. {}".format(
                    shape, result["dimensions"]))

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
