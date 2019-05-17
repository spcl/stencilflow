#!/usr/bin/env python3
import argparse
import helper
import numpy as np
import dace
import itertools
import os
import re
import shutil
import subprocess as sp
from sdfg_generator import generate_sdfg
from kernel_chain_graph import KernelChainGraph

parser = argparse.ArgumentParser()
parser.add_argument("stencil_file")
parser.add_argument("mode", choices=["emulation", "hardware"])
args = parser.parse_args()

# Load program file
program_description = helper.parse_json(args.stencil_file)
name = os.path.basename(args.stencil_file)
name = re.match("[^\.]+", name).group(0)

# Create SDFG
chain = KernelChainGraph(args.stencil_file)
sdfg = generate_sdfg(name, chain)

# Configure and compile SDFG
dace.config.Config.set("compiler", "fpga_vendor", value="intel_fpga")
dace.config.Config.set("compiler", "use_cache", value=True)
dace.config.Config.set("optimizer", "interface", value="")
if args.mode == "emulation":
    dace.config.Config.set("compiler", "intel_fpga", "mode", value="emulator")
else:
    dace.config.Config.set("compiler", "intel_fpga", "mode", value="hardware")
program = sdfg.compile()

# Load data from disk
input_arrays = helper.load_input_arrays(program_description)

# Initialize output arrays
output_arrays = {
    arr_name: np.empty(program_description["dimensions"], dtype=float)
    for arr_name in program_description["outputs"]
}

# Compile program (if running emulation)
build_folder = os.path.join(".dacecache", name, "build")
if args.mode == "emulation":
    print("Compiling emulation kernel...")
    sp.run(
        ["make", "intelfpga_compile_" + name + "_emulator"], cwd=build_folder)
elif args.mode == "hardware":
    if not os.path.exists(os.path.join(build_folder, name + "_hardware.aocx")):
        raise FileNotFoundError("Hardware kernel has not been built.")

# Run program
args = {
    key + "_host": val
    for key, val in itertools.chain(input_arrays.items(),
                                    output_arrays.items())
}
print("Executing DaCe program...")
program(**args)
print("Finished running program.")

# Write results to file
output_folder = os.path.join("results", name)
os.makedirs(output_folder, exist_ok=True)
helper.save_output_arrays(output_arrays, output_folder)
print("Results saved to " + output_folder)
