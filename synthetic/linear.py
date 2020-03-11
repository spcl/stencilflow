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
parser.add_argument(
    "-fork_frequency",
    help="At what rate forks should be generated.",
    type=float,
    default=0)
parser.add_argument(
    "-fork_length_left",
    help="Number of stencils in left branch of each fork.",
    type=int,
    default=2)
parser.add_argument(
    "-fork_length_right",
    help="Number of stencils in right branch of each fork.",
    type=int,
    default=2)
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


def make_field_name(index):
    return "a{}".format(index)


def make_stage_name(index):
    return "b{}".format(index)


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
        ("+" + str(j) if j > 0 else "-" + str(abs(j))
         if j < 0 else ""), ("+" + str(k) if k > 0 else "-" + str(abs(k))
                             if k < 0 else "")) for (i, j, k) in indices
]


def make_code(name, fields):
    operands = []
    for field in fields:
        operands += ["{}[{}]".format(field, i) for i in indices]
    return "{} = {}*({})".format(name, 1 / len(operands), " + ".join(operands))


prev_name = "a"
field_counter = 0
spatial_to_insert = 0

# Add first field
name = "a"
program["inputs"][name] = {"data": "constant:1", "data_type": args.data_type}
field_counter += 1

def insert_stencil(prev_name, name, fork_ends):

    stencil_json = {}
    stencil_json["data_type"] = args.data_type
    stencil_json["boundary_conditions"] = {}

    stage_spatials = []

    global spatial_to_insert
    global field_counter

    spatial_to_insert += args.num_fields_spatial
    while spatial_to_insert >= 1:
        field = make_field_name(field_counter)
        stage_spatials.append(field)
        program["inputs"][field] = {
            "data": "constant:0.5",
            "data_type": args.data_type
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

    stencil_json["computation_string"] = make_code(name, inputs)

    program["program"][name] = stencil_json

fork_ends = []
fork_to_insert = 0

for stage in range(args.num_stages):

    name = make_stage_name(stage)

    insert_stencil(prev_name, name, fork_ends)

    fork_ends = []

    fork_to_insert += args.fork_frequency
    if stage < args.num_stages - 1 and fork_to_insert >= 1:

        prev_name_fork = name
        index_fork = 0
        for i in range(args.fork_length_left):
            name_fork = "{}a{}".format(name, index_fork)
            insert_stencil(prev_name_fork, name_fork, [])
            prev_name_fork = name_fork
            index_fork += 1
        fork_ends.append(name_fork)

        prev_name_fork = name
        index_fork = 0
        for i in range(args.fork_length_right):
            name_fork = "{}b{}".format(name, index_fork)
            insert_stencil(prev_name_fork, name_fork, [])
            prev_name_fork = name_fork
            index_fork += 1
        fork_ends.append(name_fork)

        fork_to_insert = 0

    prev_name = name

program["outputs"].append(name)

vals = []
for arg in vars(args):
    vals.append(getattr(args, arg))
output_path = "_".join(map(str, vals)) + ".json"

with open(output_path, "w") as out_file:
    out_file.write(json.dumps(program, indent=True))

print("Wrote synthetic linear stencil to: {}".format(output_path))
