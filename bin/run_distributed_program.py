#!/usr/bin/env python3
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

    #TODO restore other arguments
    args = parser.parse_args()

    stencil_file = args.stencil_file
    mode = args.mode


    ## Borrowed from run_program

    # The splitter starts from the generated sdfg, which  contains copies to/from host
    # Load program file

    program_description = helper.parse_json(stencil_file)
    name = os.path.basename(stencil_file)
    name = re.match("[^\.]+", name).group(0)

    # Create SDFG
    chain = KernelChainGraph(path=stencil_file,
                             plot_graph=False,
                             log_level=int(LogLevel.BASIC.value))


    print("Splitting SDFG...")
    sdfg = generate_sdfg(name, chain)

    sdfg_before, sdfg_after = split_sdfg(sdfg, args.split_stream,
                                         args.send_rank, args.receive_rank,
                                         args.port)

    input_file = os.path.splitext(stencil_file)[0]
    sdfg.save("original.sdfg")
    before_path = "before.sdfg"
    after_path = "after.sdfg"
    sdfg_before.save(before_path)
    sdfg_after.save(after_path)
    print("Split SDFGs saved to:\n\t{}\n\t{}".format(before_path, after_path))

    # At this point we should decide which one we would like to run

    # Configure and compile SDFG
    dace.config.Config.set("compiler", "fpga_vendor", value="intel_fpga")
    # dace.config.Config.set("compiler", "use_cache", value=True)
    dace.config.Config.set("optimizer", "interface", value="")

    use_cache = dace.config.Config.get_bool("compiler", "use_cache")
    if mode == "emulation":
        dace.config.Config.set("compiler",
                               "intel_fpga",
                               "mode",
                               value="emulator")
    elif mode == "hardware":
        dace.config.Config.set("compiler",
                               "intel_fpga",
                               "mode",
                               value="hardware")
    else:
        raise ValueError("Unrecognized execution mode: {}".format(mode))


    if args.which_to_execute == "before":
        print("Compiling SDFG...")
        sdfg_before_name = sdfg_before.name
        sdfg_before.expand_library_nodes()
        try:
            sdfg_before.specialize(dict(smi_rank=0, smi_num_ranks=2))
            program = sdfg_before.compile()
        except dace.codegen.compiler.CompilationError as ex:
            print("Captured Compilation Error exception")

        if not use_cache:
            #build
            build_folder = os.path.join(".dacecache", sdfg_before_name, "build")

            #codegen host
            sp.run(["make", "intelfpga_smi_" + sdfg_before_name + "_codegen_host", "-j"],
                   cwd=build_folder,
                   check=True)
            # make host program
            sp.run(["make"],
                   cwd=build_folder,
                   check=True)
            # emulated bitstream
            sp.run(["make", "intelfpga_smi_compile_" + sdfg_before_name + "_emulator"],
                   cwd=build_folder,
                   check=True)
        else:
            print("Using cached compilations")

        #load the data
        # Load data from disk
        print("Loading input arrays...")
        input_directory = os.path.dirname(stencil_file)
        input_arrays = helper.load_input_arrays(program_description["inputs"],
                                                prefix=input_directory)
        dace_args = {
            key + "_host": val
            for key, val in input_arrays.items()
        }
    else:
        print("Compiling SDFG...")
        sdfg_after_name = sdfg_after.name
        sdfg_after.expand_library_nodes()

        try:
            sdfg_after.specialize(dict(smi_rank=1, smi_num_ranks=2))

            program = sdfg_after.compile()
        except dace.codegen.compiler.CompilationError as ex:
            print("Captured Compilation Error exception")
        if not use_cache:
            # build
            build_folder = os.path.join(".dacecache", sdfg_after_name, "build")

            # codegen host
            sp.run(["make", "intelfpga_smi_" + sdfg_after_name + "_codegen_host", "-j"],
                   cwd=build_folder,
                   check=True)
            # make host program
            sp.run(["make"],
                   cwd=build_folder,
                   check=True)
            # emulated bitstream
            sp.run(["make", "intelfpga_smi_compile_" + sdfg_after_name + "_emulator"],
                   cwd=build_folder,
                   check=True)
        else:
            print("Using cached compilations")

        print("Initializing output arrays...")
        output_arrays = {
            arr_name: helper.aligned(
                np.zeros(program_description["dimensions"],
                         dtype=program_description["program"][arr_name]
                         ["data_type"].type), 64)
            for arr_name in program_description["outputs"]
        }
        dace_args = {
            key + "_host": val
            for key, val in output_arrays.items()
        }
        import pdb
        pdb.set_trace()

    print(dace_args)

    print("Executing DaCe program...")
    program(**dace_args)
    print("Finished running program.")

    if args.which_to_execute == "after":
        output_folder = os.path.join("results", name)
        os.makedirs(output_folder, exist_ok=True)
        helper.save_output_arrays(output_arrays, output_folder)
        print("Results saved to " + output_folder)



