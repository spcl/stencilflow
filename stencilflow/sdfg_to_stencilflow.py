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

def _canonicalize_sdfg(sdfg):
    strict = [
        k for k, v in Transformation.extensions().items()
        if v.get('strict', False)
    ]
    extra = [NestK, MapFission]
    return sdfg.apply_transformations_repeated(strict + extra, validate=False)

class _RemoveOutputSubscript(ast.NodeTransformer):

    def visit_Assign(self, node: ast.Assign):
        for i, subscript_node in enumerate(node.targets):
            indices = ast.literal_eval(subscript_node.slice.value)
            if any(indices):
                raise ValueError("Output offset not yet supported.")
            # Remove subscript
            node.targets[i] = subscript_node.value
        return node


def sdfg_to_stencilflow(sdfg, output_path, data_directory=None, symbols={}):

    if not isinstance(sdfg, dace.SDFG):
        sdfg = dace.SDFG.from_file(sdfg)

    _canonicalize_sdfg(sdfg)

    sdfg.specialize(symbols)

    undefined_symbols = sdfg.undefined_symbols(False)
    if len(undefined_symbols) != 0:
        raise ValueError("Undefined symbols in SDFG: {}".format(
            ", ".join(undefined_symbols)))

    reads = {}
    writes = set()

    result = {"inputs": {}, "outputs": [], "dimensions": None, "program": {}}

    for node, parent in sdfg.all_nodes_recursive():

        if isinstance(node, stencil.Stencil):

            if node.label in result["program"]:
                raise KeyError("Duplicate stencil: " + node.label)

            remover = _RemoveOutputSubscript()
            old_ast = ast.parse(node.code)
            new_ast = remover.visit(old_ast)
            code = astunparse.unparse(new_ast)

            stencil_json = {}
            stencil_json["computation_string"] = code
            stencil_json["boundary_conditions"] = node.boundary_conditions

            in_edges = {e.dst_conn: e for e in parent.in_edges(node)}
            out_edges = {e.src_conn: e for e in parent.out_edges(node)}

            current_sdfg = parent.parent

            for field, accesses in node.accesses.items():
                dtype = current_sdfg.data(
                    dace.sdfg.find_input_arraynode(
                        parent, in_edges[field]).data).dtype.type.__name__
                if field in reads:
                    if reads[field] != dtype:
                        raise ValueError("Type mismatch: {} vs. {}".format(
                            reads[field], dtype))
                else:
                    reads[field] = dtype

            if len(node.output_fields) != 1:
                raise ValueError("Only 1 output per stencil is supported, "
                                 "but {} has {} outputs.".format(
                                     node.label, len(node.output_fields)))

            for output in node.output_fields:
                break  # Grab first and only element
            if output in writes:
                raise KeyError("Multiple writes to field: {}".format(field))
            writes.add(output)

            stencil_json["data_type"] = current_sdfg.data(
                dace.sdfg.find_output_arraynode(
                    parent, out_edges[output]).data).dtype.type.__name__

            result["program"][node.label] = stencil_json

            # Extract stencil shape from stencil
            shape = " ".join(map(str, node.shape))
            for k, v in symbols.items():
                shape = re.sub(r"\b{}\b".format(k), str(v), shape)
            shape = tuple(int(v) for v in shape.split(" "))
            if result["dimensions"] is None:
                result["dimensions"] = shape
            else:
                if shape != result["dimensions"]:
                    raise ValueError(
                        "Conflicting shapes found: {} vs. {}".format(
                            shape, result["dimensions"]))

        elif isinstance(node, dace.graph.nodes.Tasklet):
            warnings.warn("Ignored tasklet {}".format(node.label))

        elif isinstance(node, dace.graph.nodes.AccessNode):
            pass

        elif isinstance(node, dace.graph.nodes.NestedSDFG):
            pass

        elif isinstance(node, dace.sdfg.SDFGState):
            pass

        else:
            raise TypeError("Unsupported node type in {}: {}".format(
                parent.label,
                type(node).__name__))

        # End node loop

    inputs = reads.keys() - writes
    outputs = writes - reads.keys()

    result["outputs"] = list(sorted(outputs))
    for field in inputs:
        dtype = reads[field]
        path = "{}_{}_{}.dat".format(field, "x".join(
            map(str, result["dimensions"])), dtype)
        if data_directory is not None:
            path = os.path.join(data_directory, path)
        result["inputs"][field] = {"data": path, "data_type": dtype}

    with open(output_path, "w") as out_file:
        out_file.write(json.dumps(result, indent=True))
