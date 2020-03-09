#!/usr/bin/env python3
"""
    Utility program for splitting, code generation and execution of a stencil program.

    Note: the program is split in two components around a given stream (it relies on split_sdfg)

    Command line arguments:
    - stencil_file: JSON file containing stencil description
    - mode: emulation or hardware. The former triggers code generation and compilation for the emulation toolchain
    - split_stream: the stream around which the original stencil SDFG is partitioned
    - send_rank, receive_rank, port: SMI needed parameters
    - which_to_execute: which part of the program execute, before/after

    If DACE_compiler_use_cache is set to 1, the code is not generated/recompiled
"""
import argparse
import os
import sys
import argparse
import itertools
import os
import re
import subprocess as sp
import sys

import dace
import numpy as np

from stencilflow import *
from stencilflow.log_level import LogLevel
import stencilflow.helper as helper
from stencilflow.sdfg_generator import split_sdfg

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("sdfg_file")
    parser.add_argument("stencil_file")
    parser.add_argument("mode", choices=["emulation", "hardware"])
    parser.add_argument("rank")
    parser.add_argument("num_ranks")

    args = parser.parse_args()
    stencil_file = args.stencil_file
    sdfg_file = args.sdfg_file
    mode = args.mode
    my_rank = int(args.rank)
    num_ranks = int(args.num_ranks)

    # Load program description
    program_description = helper.parse_json(stencil_file)
    stencil_name = os.path.basename(stencil_file)
    stencil_name = re.match("[^\.]+", stencil_name).group(0)

    # Load SDFG
    sdfg = dace.SDFG.from_file(sdfg_file)

    # ----------------------------------------------
    # Configure
    # ----------------------------------------------
    dace.config.Config.set("compiler", "fpga_vendor", value="intel_fpga")
    dace.config.Config.set("optimizer", "interface", value="")

    use_cache = dace.config.Config.get_bool("compiler", "use_cache")
    if mode == "emulation":
        dace.config.Config.set(
            "compiler", "intel_fpga", "mode", value="emulator")
    elif mode == "hardware":
        dace.config.Config.set(
            "compiler", "intel_fpga", "mode", value="hardware")
    else:
        raise ValueError("Unrecognized execution mode: {}".format(mode))

    # ----------------------------------------------
    # Specialize and compile the program
    # ----------------------------------------------


    # Specialize SDFG

    sdfg_name = sdfg.name
    as_json = sdfg.to_json()
    as_json["attributes"]["name"] = sdfg_name + "_" +str(my_rank)
    sdfg_name = sdfg_name + "_" +str(my_rank)
    sdfg = dace.SDFG.from_json(as_json)
    sdfg.expand_library_nodes()
    sdfg.specialize(dict(smi_rank=my_rank, smi_num_ranks=num_ranks))

    # Compile
    try:
        program = sdfg.compile()
    except dace.codegen.compiler.CompilationError as ex:
        # Fresh new...we need to build everythin
        print("Captured Compilation Error exception")

        # build
        build_folder = os.path.join(".dacecache", sdfg_name,
                                    "build")

        # codegen host
        sp.run(
            [
                "make",
                "intelfpga_smi_" + sdfg_name + "_codegen_host", "-j4"
            ],
            cwd=build_folder,
            check=True)
        # make host program
        sp.run(["make"], cwd=build_folder, check=True)
        if mode == "emulation":
            # emulated bitstream
            sp.run(
                [
                    "make", "intelfpga_smi_compile_" + sdfg_name +
                            "_emulator"
                ],
                cwd=build_folder,
                check=True)
        elif mode == "hardware":
            if not os.path.exists(
                    os.path.join(build_folder, name + "_hardware.aocx")):
                raise FileNotFoundError(
                    "Hardware kernel has not been built (" + os.path.join(
                        build_folder, name + "_hardware.aocx") + ").")

        # reload the program
        program = sdfg.compile()

    # ----------------------------------------------
    # Create Input/Output data for this SDFG
    # ----------------------------------------------

    dace_args = {}

    # Create inputs
    input_data = sdfg.source_nodes()[0].source_nodes()
    sdfg_input_data = {}
    for n in input_data:
        if isinstance(n, dace.graph.nodes.AccessNode) and sdfg.arrays[n.data].storage == dace.dtypes.StorageType.Default:
            # remove trailing "_host" and get the input parameters from program description
            if n.data.endswith("_host"):
                data_name = n.data[:-5]
                if data_name in program_description["inputs"]:
                    sdfg_input_data[data_name] = program_description["inputs"][data_name]
            else:
                raise ValueError("Uhm...strange, what kind of data is " + n.data + "?")


    # Load data from disk (if any)
    if sdfg_input_data:
        print("Loading input arrays...")
        input_directory = os.path.dirname(stencil_file)
        input_arrays = helper.load_input_arrays(
            sdfg_input_data, prefix=input_directory)
        for key,val in input_arrays.items():
            dace_args[key+"_host"] = val

    # Create outputs

    save_outputs = False
    output_data = sdfg.sink_nodes()[0].sink_nodes()
    sdfg_output_data = []
    for n in output_data:
        if isinstance(n, dace.graph.nodes.AccessNode) and sdfg.arrays[n.data].storage == dace.dtypes.StorageType.Default:
            #remove trailing "_host" and check if this is an output parameter
            if n.data.endswith("_host"):
                data_name = n.data[:-5]
                if data_name in program_description["outputs"]:
                    sdfg_output_data += [data_name]
            else:
                raise ValueError("Uhm...strange, what kind of data is " + n.data +"?")
    if sdfg_output_data:
        save_outputs = True
        print("Initializing output arrays...")
        output_arrays = {
            arr_name: helper.aligned(
                np.zeros(
                    program_description["dimensions"],
                    dtype=program_description["program"][arr_name]["data_type"]
                        .type), 64)
            for arr_name in sdfg_output_data
        }
        for key, val in output_arrays.items():
            dace_args[key+"_host"] = val


    print("Executing DaCe program...")
    program(**dace_args)
    print("Finished running program.")

    if save_outputs:
        # Save output
        output_folder = os.path.join("results", stencil_name)
        os.makedirs(output_folder, exist_ok=True)
        helper.save_output_arrays(output_arrays, output_folder)
        print("Results saved to " + output_folder)
