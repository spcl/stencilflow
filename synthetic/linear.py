import argparse
import itertools
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("data_type", type=str, choices=["float32", "float64"])
parser.add_argument("num_stages", type=int)
parser.add_argument(
    "num_fields_spatial",
    help=("Number of fields per stencil "
          "that are read from external memory (i.e., not shared with others)."
          "Fractional numbers are allowed."),
    type=float)
parser.add_argument(
    "size_x",
    help="Size of domain in first dimension (can be zero).",
    type=int)
parser.add_argument(
    "size_y",
    help="Size of domain in second dimension (can be zero).",
    type=int)
parser.add_argument(
    "size_z",
    help="Size of domain in third dimension (can be zero).",
    type=int)
parser.add_argument(
    "extent_x", help="Extent of stencil in the x-dimension.", type=int)
parser.add_argument(
    "extent_y",
    help="Extent of stencil in the y-dimension (2D and 3D only).",
    type=int)
parser.add_argument(
    "extent_z",
    help="Extent of stencil in the z-dimension (3D only).",
    type=int)
args = parser.parse_args()

shape = []
for s in ["size_x", "size_y", "size_z"]:
    val = getattr(args, s)
    if val > 0:
        shape.append(val)

program = {
    "inputs": {},
    "outputs": [],
    "dimensions": None,
    "program": {},
    "dimensions": shape
}

field_counter = 0
spatial_to_insert = 0

# Add first field
name = "a"
program["inputs"][name] = {"data": "constant:0.5", "data_type": args.data_type}
field_counter += 1


def make_field_name(index):
    return "a{}".format(index)


def make_stage_name(index):
    return "b{}".format(index)


prev_name = "a"

for stage in range(args.num_stages):

    name = make_stage_name(stage)

    stencil_json = {}
    stencil_json["data_type"] = args.data_type
    stencil_json["boundary_conditions"] = {
        prev_name: {
            "type": "constant",
            "value": 0
        }
    }

    stage_spatials = []

    spatial_to_insert += args.num_fields_spatial
    while spatial_to_insert >= 1:
        field = make_field_name(field_counter)
        stage_spatials.append(field)
        program["inputs"][field] = {
            "data": "constant:0.5",
            "data_type": args.data_type
        }
        stencil_json["boundary_conditions"][field] = {
            "type": "constant",
            "value": 0
        }
        field_counter += 1
        spatial_to_insert -= 1

    # Generate all indices that each field is accessed with
    dimensions = []
    if args.size_x > 0:
        if args.extent_x != 0:
            xs = np.hstack((np.arange(-args.extent_x, 0),
                            np.arange(1, args.extent_x + 1)))
        else:
            xs = [0]
        dimensions.append(xs)
    if args.size_y > 0:
        if args.extent_y != 0:
            ys = np.hstack((np.arange(-args.extent_y, 0),
                            np.arange(1, args.extent_y + 1)))
        else:
            ys = [0]
        dimensions.append(ys)
    if args.size_z > 0:
        if args.extent_z != 0:
            zs = np.hstack((np.arange(-args.extent_z, 0),
                            np.arange(1, args.extent_z + 1)))
        else:
            zs = [0]
        dimensions.append(zs)
    indices = itertools.product(*dimensions)
    indices = [
        "i{}, j{}, k{}".format(
            ("+" + str(i) if i > 0 else "-" + str(abs(i)) if i < 0 else ""),
            ("+" + str(j) if j > 0 else "-" + str(abs(j)) if j < 0 else ""),
            ("+" + str(k) if k > 0 else "-" + str(abs(k)) if k < 0 else ""))
            for (i, j, k) in indices
    ]

    # Synthesize code
    operands = []
    for field in itertools.chain([prev_name], stage_spatials):
        operands += ["{}[{}]".format(field, i) for i in indices]
    code = "{} = {}*({})".format(name, 1 / len(operands), " + ".join(operands))

    stencil_json["computation_string"] = code

    program["program"][name] = stencil_json

    prev_name = name

program["outputs"].append(name)

vals = []
for arg in vars(args):
    vals.append(getattr(args, arg))
output_path = "_".join(map(str, vals)) + ".json"

with open(output_path, "w") as out_file:
    out_file.write(json.dumps(program, indent=True))

print("Wrote synthetic linear stencil to: {}".format(output_path))
