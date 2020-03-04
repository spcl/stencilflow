#!/usr/bin/env python3
import json

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

def sdfg_to_stencilflow(sdfg, output_path, data_directory=None):

    if not isinstance(sdfg, dace.SDFG):
        sdfg = dace.SDFG.from_file(sdfg)

    _canonicalize_sdfg(sdfg)

    sdfg.save("canonical.sdfg")

    reads = {}
    writes = set()

    result = {"inputs": {}, "outputs": [], "dimensions": None, "program": {}}

    for node, parent in sdfg.all_nodes_recursive():

        if isinstance(node, stencil.Stencil):

            if node.label in result["program"]:
                raise KeyError("Duplicate stencil: " + node.label)

            stencil_json = {}
            stencil_json["computation_string"] = node.code
            stencil_json["boundary_conditions"] = node.boundary_conditions

            in_edges = {e.dst_conn: e for e in parent.in_edges(node)}
            out_edges = {e.src_conn: e for e in parent.out_edges(node)}

            for field, accesses in node.accesses.items():
                dtype = sdfg.data(
                    dace.sdfg.find_input_arraynode(
                        parent, in_edges[field]).data).dtype.ctype
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

            stencil_json["data_type"] = sdfg.data(
                dace.sdfg.find_output_arraynode(
                    parent, out_edges[output]).data).dtype.ctype

            result["program"][node.label] = stencil_json

            # Extract stencil shape from stencil
            shape = tuple(map(str, node.shape))
            if result["dimensions"] is None:
                result["dimensions"] = node.shape
            else:
                if node.shape != result["dimensions"]:
                    raise ValueError(
                        "Conflicting shapes found: {} vs. {}".format(
                            shape, result["dimensions"]))

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
