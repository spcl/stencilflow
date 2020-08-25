#!/usr/bin/env python3
import argparse
import click
import itertools
import json
import numpy as np


ITERATORS = ["i", "j", "k"]

def make_field_name(index):
    return "a{}".format(index)


def make_stage_name(index):
    return "b{}".format(index)


def make_extent_accesses(size, extent, stencil_shape):
    if size > 0:
        if extent != 0:
            if stencil_shape == "box":
                indices = np.arange(-extent, extent + 1)
            else:
                indices = np.hstack((np.arange(-extent,
                                               0), np.arange(1, extent + 1)))
        else:
            indices = [0]
    else:
        indices = []
    return indices


@click.command()
@click.argument("data_type", type=click.Choice(["float32", "float64"]))
@click.argument("num_stages", type=int)
@click.argument("num_fields_spatial", type=float)
@click.argument("size_x", type=int)
@click.argument("size_y", type=int)
@click.argument("size_z", type=int)
@click.argument("extent_x", type=int)
@click.argument("extent_y", type=int)
@click.argument("extent_z", type=int)
@click.option("-fork_frequency",
              help="At what rate forks should be generated.",
              type=float,
              default=0)
@click.option("-fork_length_left",
              help="Number of stencils in left branch of each fork.",
              type=int,
              default=2)
@click.option("-fork_length_right",
              help="Number of stencils in right branch of each fork.",
              type=int,
              default=2)
@click.option("-stencil_shape",
              type=click.Choice(["cross", "box", "diffusion", "hotspot"]),
              default="cross")
@click.option("-vectorize", help="Vectorization factor.", type=int, default=1)
def synthesize_stencil(data_type, num_stages, num_fields_spatial, size_x,
                       size_y, size_z, extent_x, extent_y, extent_z,
                       fork_frequency, fork_length_left, fork_length_right,
                       stencil_shape, vectorize):
    """\b
    num_fields_spatial: Number of fields per stencil that are read from external memory (i.e., not shared with others). Fractional numbers are allowed.
    size_x: Size of domain in first dimension (can be zero).
    size_y: Size of domain in second dimension (can be zero).
    size_z: Size of domain in third dimension (can be zero).
    extent_x: Extent of stencil in the x-dimension.
    extent_y: Extent of stencil in the y-dimension (2D and 3D only).
    extent_z: Extent of stencil in the z-dimension (3D only).
    """

    shape = []
    for s in [size_x, size_y, size_z]:
        if s > 0:
            shape.append(s)

    program = {
        "inputs": {},
        "outputs": [],
        "dimensions": None,
        "program": {},
        "dimensions": shape,
        "vectorization": vectorize
    }

    iterators = ITERATORS[3 - len(shape):]

    # Generate all indices that each field is accessed with
    dimensions = [
        make_extent_accesses(size_x, extent_x, stencil_shape),
        make_extent_accesses(size_y, extent_y, stencil_shape),
        make_extent_accesses(size_z, extent_z, stencil_shape)
    ][:len(shape)]
    if stencil_shape in ["cross", "diffusion", "hotspot"]:
        indices = []
        for i in range(len(dimensions)):
            indices += itertools.product(
                *[d if j == i else [0] for j, d in enumerate(dimensions)])
    elif stencil_shape == "box":
        indices = itertools.product(*dimensions)

    _indices = []
    for t in indices:
        s = []
        for i, val in enumerate(t):
            s.append(iterators[i] + ("+" + str(val) if val > 0 else "-" +
                                     str(abs(val)) if val < 0 else ""))
        _indices.append(", ".join(s))
    indices = _indices

    prev_name = "a"
    field_counter = 0
    spatial_to_insert = 0

    # Add first field
    name = "a"
    program["inputs"][name] = {
        "data": "constant:1",
        "data_type": data_type,
        "dimensions": ["i", "j", "k"][3 - len(shape):]
    }
    field_counter += 1

    fork_ends = []
    fork_to_insert = 0

    def make_code(name, fields, indices):
        code = f"{name} = "
        if stencil_shape == "hotspot":
            if len(fields) != 1:
                raise ValueError("Hotspot only supports a single input field")
            field = fields[0]
            if len(shape) == 3:
                # Hotspot 3D
                code += (f"cc * {field}[i, j, k] + "
                         f"cn * {field}[i, j-1, k] + "
                         f"cs * {field}[i, j+1, k] + "
                         f"cw * {field}[i, j, k-1] + "
                         f"ce * {field}[i, j, k+1] + "
                         f"ca * {field}[i-1, j, k] + "
                         f"cb * {field}[i+1, j, k] + "
                         f"sdc * power[i, j, k] + "
                         f"ca * amb")
            elif len(shape) == 2:
                # Hotspot 2D
                code += (
                    f"{field}[j, k] + "
                    f"sdc * (power[j, k] + "
                    f"({field}[j-1, k] + {field}[j+1, k] - 2.0 * {field}[j, k]) * r_y + "
                    f"({field}[j, k-1] + {field}[j, k+1] - 2.0 * {field}[j, k]) * r_x + "
                    f"(amb - {field}[j, k]) * r_z)")
            else:
                raise ValueError("Unsupported number of indices for hotspot.")
        else:
            operands = []
            for field in fields:
                operands += ["{}[{}]".format(field, i) for i in indices]
            if stencil_shape == "diffusion":
                operands = [
                    "c{}*{}".format(i, o) for i, o in enumerate(operands)
                ]
                code += " + ".join(operands)
            else:
                code + "{}*({})".format(1 / len(operands), " + ".join(operands))
        return code

    def insert_stencil(prev_name, name, fork_ends, spatial_to_insert,
                       field_counter):

        stencil_json = {}
        stencil_json["data_type"] = data_type
        stencil_json["boundary_conditions"] = {}

        stage_spatials = []

        spatial_to_insert += num_fields_spatial
        while spatial_to_insert >= 1:
            field = make_field_name(field_counter)
            stage_spatials.append(field)
            program["inputs"][field] = {
                "data": "constant:0.5",
                "data_type": data_type
            }
            field_counter += 1
            spatial_to_insert -= 1

        if len(fork_ends) > 0:
            inputs = fork_ends + stage_spatials
        else:
            inputs = [prev_name] + stage_spatials

        for i in inputs:
            stencil_json["boundary_conditions"][i] = {
                "type": "constant",
                "value": 0
            }
        if stencil_shape == "hotspot":
            stencil_json["boundary_conditions"]["power"] = {
                "type": "constant",
                "value": 0
            }

        stencil_json["computation_string"] = make_code(name, inputs, indices)

        program["program"][name] = stencil_json

        return spatial_to_insert, field_counter

    for stage in range(num_stages):

        name = make_stage_name(stage)

        spatial_to_insert, field_counter = insert_stencil(
            prev_name, name, fork_ends, spatial_to_insert, field_counter)

        fork_ends = []

        fork_to_insert += fork_frequency
        if stage < num_stages - 1 and fork_to_insert >= 1:

            prev_name_fork = name
            index_fork = 0
            for i in range(fork_length_left):
                name_fork = "{}a{}".format(name, index_fork)
                spatial_to_insert, field_counter = insert_stencil(
                    prev_name_fork, name_fork, [], spatial_to_insert,
                    field_counter)
                prev_name_fork = name_fork
                index_fork += 1
            fork_ends.append(name_fork)

            prev_name_fork = name
            index_fork = 0
            for i in range(fork_length_right):
                name_fork = "{}b{}".format(name, index_fork)
                spatial_to_insert, field_counter = insert_stencil(
                    prev_name_fork, name_fork, [], spatial_to_insert,
                    field_counter)
                prev_name_fork = name_fork
                index_fork += 1
            fork_ends.append(name_fork)

            fork_to_insert = 0

        prev_name = name

    if stencil_shape == "hotspot":
        program["inputs"]["power"] = {
            "data": "constant:0.5",
            "data_type": data_type
        }
        if len(shape) == 2:  # Hotspot 2D
            scalars = ["sdc", "r_x", "r_y", "r_z", "amb"]
        elif len(shape) == 3:  # Hotspot 3D
            scalars = ["cc", "cn", "cs", "cw", "ce", "ca", "cb", "sdc", "amb"]
        else:
            raise NotImplementedError
        for s in scalars:
            program["inputs"][s] = {
                "data": "constant:0.5",
                "data_type": data_type,
                "dimensions": []
            }
    elif stencil_shape == "diffusion":
        for i in range(len(indices)):
            program["inputs"][f"c{i}"] = {
                "data": "constant:0.5",
                "data_type": data_type,
                "dimensions": []
            }

    program["outputs"].append(name)

    args = [
        data_type, num_stages, num_fields_spatial, size_x, size_y, size_z,
        extent_x, extent_y, extent_z, fork_frequency, fork_length_left,
        fork_length_right, stencil_shape, vectorize
    ]
    output_path = "_".join(map(str, args)).replace(".", "p") + ".json"

    with open(output_path, "w") as out_file:
        out_file.write(json.dumps(program, indent=True))

    print("Wrote synthetic stencil to: {}".format(output_path))


if __name__ == "__main__":
    synthesize_stencil()
