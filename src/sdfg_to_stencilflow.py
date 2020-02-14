#!/usr/bin/env python3
import json

import dace
import stencil

def sdfg_to_stencilflow(sdfg, output_path, data_directory=None):

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
                if field in reads:
                    raise KeyError(
                        "Multiple reads from field: {}".format(field))

                dtype = sdfg.data(
                    dace.sdfg.find_input_arraynode(
                        parent, in_edges[field]).data).dtype.ctype
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

        elif isinstance(node, dace.graph.nodes.AccessNode):
            pass

        elif isinstance(node, dace.graph.nodes.Tasklet):
            print("Skipping tasklet {}.".format(node.label))

        elif isinstance(node, dace.graph.nodes.MapEntry) or isinstance(
                node, dace.graph.nodes.MapExit):

            # Extract stencil shape from map entry/exit
            shape = []
            for begin, end, step in node.map.range:
                if begin != 0:
                    raise ValueError("Ranges must start at 0.")
                if step != 1:
                    raise ValueError("Step size must be 1.")
                shape.append(str(end + 1))

            if result["dimensions"] is None:
                result["dimensions"] = shape
            else:
                if shape != result["dimensions"]:
                    raise ValueError(
                        "Conflicting shapes found: {} vs. {}".format(
                            shape, result["dimensions"]))

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

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input_sdfg")
    parser.add_argument("output_json")
    parser.add_argument("-data-directory")
    args = parser.parse_args()

    sdfg = dace.sdfg.SDFG.from_file(args.input_sdfg)

    sdfg_to_stencilflow(sdfg, args.output_json, args.data_directory)

    print("Saved StencilFlow program to: {}".format(args.output_json))
