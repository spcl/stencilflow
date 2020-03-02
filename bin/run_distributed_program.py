#!/usr/bin/env python3
"""
    Utility program for splitting, code generation and execution of a stencil program.

    Note: the program is splitted in two components around a given stream (it relies on split_sdfg)

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
    parser.add_argument("stencil_file")
    parser.add_argument("mode", choices=["emulation", "hardware"])
    parser.add_argument("split_stream")
    parser.add_argument("send_rank")
    parser.add_argument("receive_rank")
    parser.add_argument("port", type=int)
    parser.add_argument("which_to_execute", choices=["before", "after"])

    args = parser.parse_args()
    stencil_file = args.stencil_file
    mode = args.mode

    ## Borrowed from run_program

    # The splitter starts from the generated sdfg, which  contains copies to/from host
    # Load program file
    print("Creating original SDFG...")

    program_description = helper.parse_json(stencil_file)
    name = os.path.basename(stencil_file)
    name = re.match("[^\.]+", name).group(0)

    # Create SDFG
    chain = KernelChainGraph(
        path=stencil_file,
        plot_graph=False,
        log_level=int(LogLevel.BASIC.value))

    sdfg = generate_sdfg(name, chain)

    # Split the SDFG (borrowed from split_sdfg)
    print("Splitting SDFG...")
    sdfg_before, sdfg_after = split_sdfg(
        sdfg, args.split_stream, args.send_rank, args.receive_rank, args.port)

    input_file = os.path.splitext(stencil_file)[0]
    sdfg.save("original.sdfg")
    before_path = "before.sdfg"
    after_path = "after.sdfg"
    sdfg_before.save(before_path)
    sdfg_after.save(after_path)
    print("Split SDFGs saved to:\n\t{}\n\t{}".format(before_path, after_path))

    # Configure and compile SDFG
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

    # At this point we should decide which one we would like to run
    if args.which_to_execute == "before":
        print("Compiling SDFG (before)...")
        sdfg_before_name = sdfg_before.name
        sdfg_before.expand_library_nodes()

        # Specialize SDFG indicating current rank and total number of ranks
        sdfg_before.specialize(dict(smi_rank=0, smi_num_ranks=2))
        try:
            program = sdfg_before.compile()
        except dace.codegen.compiler.CompilationError as ex:
            print("Captured Compilation Error exception")

            #build
            build_folder = os.path.join(".dacecache", sdfg_before_name,
                                        "build")

            #codegen host
            sp.run(
                [
                    "make",
                    "intelfpga_smi_" + sdfg_before_name + "_codegen_host", "-j"
                ],
                cwd=build_folder,
                check=True)
            # make host program
            sp.run(["make"], cwd=build_folder, check=True)
            if mode == "emulation":
                # emulated bitstream
                sp.run(
                    [
                        "make", "intelfpga_smi_compile_" + sdfg_before_name +
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
            program = sdfg_before.compile()

        # Load data from disk
        print("Loading input arrays...")
        input_directory = os.path.dirname(stencil_file)
        input_arrays = helper.load_input_arrays(
            program_description["inputs"], prefix=input_directory)
        dace_args = {key + "_host": val for key, val in input_arrays.items()}
    else:
        print("Compiling SDFG (after)...")
        sdfg_after_name = sdfg_after.name
        sdfg_after.expand_library_nodes()

        # Specialize the SDFG
        sdfg_after.specialize(dict(smi_rank=1, smi_num_ranks=2))
        try:
            program = sdfg_after.compile()
        except dace.codegen.compiler.CompilationError as ex:
            print("Captured Compilation Error exception")

            # build
            build_folder = os.path.join(".dacecache", sdfg_after_name, "build")

            # codegen host
            sp.run(
                [
                    "make",
                    "intelfpga_smi_" + sdfg_after_name + "_codegen_host", "-j"
                ],
                cwd=build_folder,
                check=True)
            # make host program
            sp.run(["make"], cwd=build_folder, check=True)
            if mode == "emulation":
                # emulated bitstream
                sp.run(
                    [
                        "make", "intelfpga_smi_compile_" + sdfg_after_name +
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

            # Reload the program
            program = sdfg_after.compile()

        print("Initializing output arrays...")
        output_arrays = {
            arr_name: helper.aligned(
                np.zeros(
                    program_description["dimensions"],
                    dtype=program_description["program"][arr_name]["data_type"]
                    .type), 64)
            for arr_name in program_description["outputs"]
        }
        dace_args = {key + "_host": val for key, val in output_arrays.items()}

    print("Executing DaCe program...")
    program(**dace_args)
    print("Finished running program.")

    if args.which_to_execute == "after":
        # Save output
        output_folder = os.path.join("results", name)
        os.makedirs(output_folder, exist_ok=True)
        helper.save_output_arrays(output_arrays, output_folder)
        print("Results saved to " + output_folder)
