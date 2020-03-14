#!/usr/bin/env python3
"""
    Utility program for splitting, code generation and execution of a stencil program.

    Note: the program is split in two components around a given stream (it relies on split_sdfg)

    Command line arguments:
    - sdfg_file: file containing the SDFG to execute
    - stencil_file: JSON file containing the entire stencil description
    - mode: emulation/hardware
    - rank: the rank id (used for routing)
    - num_ranks: total number of ranks (used by SMI for routing)
    - -compare_to_reference: if you want to run the CPU reference implementation


"""
import argparse
import os
import sys
import argparse
import itertools
import os
import glob
import re
import subprocess as sp
import sys
import copy
import dace
import numpy as np
from mpi4py import MPI
from stencilflow import *
from stencilflow.log_level import LogLevel
from stencilflow.sdfg_generator import generate_sdfg, generate_reference
import stencilflow.helper as helper

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_with_rank(rank, message, color = bcolors.HEADER):
    # Utility functions: print a message indicating the rank
    # By default, the message is printed using purple color
    print(color + "[Rank {}] {}".format(rank,message) +bcolors.ENDC)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("sdfgs_dir", help = "Directory containing the SDFG to execute")
    parser.add_argument("stencil_file", help = "JSON description of the stencil")
    parser.add_argument("mode", choices=["emulation", "hardware"], help = "Execution mode")
    parser.add_argument("-compare-to-reference", action="store_true", help = "Flag for comparing the result with reference")
    parser.add_argument("-recompute-routes", action="store", help = "Recompute routes by using a topology file (meaningful for hardware execution mode)")
    parser.add_argument("-sequential-compile", action="store_true", help = "If specified compile everything sequential. Useful for CI to not overload Pauli")



    args = parser.parse_args()
    stencil_file = args.stencil_file
    sdfgs_dir = args.sdfgs_dir
    mode = args.mode
    compare_to_reference = args.compare_to_reference
    topology_file = args.recompute_routes
    sequential_compile=args.sequential_compile


    # MPI: get current size, rank and name
    num_ranks = MPI.COMM_WORLD.Get_size()
    my_rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

       # Load program description
    program_description = helper.parse_json(stencil_file)
    stencil_name = os.path.basename(stencil_file)
    stencil_name = re.match("[^\.]+", stencil_name).group(0)

    # Load SDFG
    # Load the corresponding SDFG. Please note that it will look to a file with name "*_<rank>.sdfg"
    sdfg_file = glob.glob("{}*{}.sdfg".format(sdfgs_dir, my_rank))
    if len(sdfg_file) != 1:
        print("[Rank {}] SDFG not found ".format(my_rank))
        exit(-1)
    sdfg = dace.SDFG.from_file(sdfg_file[0])

    print_with_rank(my_rank, "executing SDFG {} on {}".format(sdfg_file,name))

    # ----------------------------------------------
    # Configure
    # ----------------------------------------------

    dace.config.Config.set("compiler", "fpga_vendor", value="intel_fpga")
    dace.config.Config.set("optimizer", "interface", value="")
    dace.config.Config.set("compiler",
                          "intel_fpga",
                          "smi_ranks",
                          value=num_ranks)
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

    # ----------------------------------------------
    # Specialize and compile the program
    # ----------------------------------------------

    # Change SDFG name to have different folder in `.dacecache`

    sdfg_name = sdfg.name
    as_json = sdfg.to_json()
    as_json["attributes"]["name"] = sdfg_name + "_" + str(my_rank)
    sdfg_name = sdfg_name + "_" + str(my_rank)
    sdfg = dace.SDFG.from_json(as_json)
    sdfg.expand_library_nodes()

    # Compile
    try:
        program = sdfg.compile()
    except dace.codegen.compiler.CompilationError as ex:
        # Fresh new...we need to build everything
        print_with_rank(my_rank, "Captured Compilation Error exception", bcolors.WARNING)

        # build
        build_folder = os.path.join(".dacecache", sdfg_name, "build")

        # codegen host
        if sequential_compile:
            # Otherwise Pauli explodes
            sp.run(["make", "intelfpga_smi_" + sdfg_name + "_codegen_host"],
                   cwd=build_folder,
                   check=True)
        else:
            sp.run(["make", "intelfpga_smi_" + sdfg_name + "_codegen_host", "-j4"],
               cwd=build_folder,
               check=True)
        # make host program
        sp.run(["make"], cwd=build_folder, check=True)
        if mode == "emulation":
            # emulated bitstream
            sp.run(
                ["make", "intelfpga_smi_compile_" + sdfg_name + "_emulator_"+str(my_rank)],
                cwd=build_folder,
                check=True)

        # reload the program
        program = sdfg.compile()

    if mode == "hardware":
        build_folder = os.path.join(".dacecache", sdfg_name, "build")
        if topology_file:
            print_with_rank(my_rank, "Recompute routes using " + topology_file)
            #copy the topology file in the build folder
            os.system("cp {} {}/topology.json".format(topology_file, build_folder))
            sp.run(["make", "intelfpga_smi_recompute_table_" + sdfg_name],
                   cwd=build_folder,
                   check=True)


        if not os.path.exists(
                os.path.join(build_folder, sdfg_name + "_hardware.aocx")):
            raise FileNotFoundError(
                "Hardware kernel has not been built (" +
                os.path.join(build_folder, sdfg_name + "_hardware.aocx") + ").")

    # ----------------------------------------------
    # Create Input/Output data for this SDFG
    # ----------------------------------------------

    dace_args = {}

    # Create inputs
    input_data = sdfg.source_nodes()[0].source_nodes()
    sdfg_input_data = {}
    for n in input_data:
        if isinstance(n, dace.graph.nodes.AccessNode) and sdfg.arrays[
                n.data].storage == dace.dtypes.StorageType.Default:
            # remove trailing "_host" and get the input parameters from program description
            if n.data.endswith("_host"):
                data_name = n.data[:-5]
                if data_name in program_description["inputs"]:
                    sdfg_input_data[data_name] = program_description["inputs"][
                        data_name]
            else:
                raise ValueError("Uhm...strange, what kind of data is " +
                                 n.data + "?")

    # Load data from disk (if any)
    if sdfg_input_data:
        print_with_rank(my_rank, "Loading input arrays...")
        input_directory = os.path.dirname(stencil_file)
        input_arrays = helper.load_input_arrays(sdfg_input_data,
                                                prefix=input_directory)
        for key, val in input_arrays.items():
            dace_args[key + "_host"] = val

    # Create outputs

    save_outputs = False
    output_data = sdfg.sink_nodes()[0].sink_nodes()
    sdfg_output_data = []
    for n in output_data:
        if isinstance(n, dace.graph.nodes.AccessNode) and sdfg.arrays[
                n.data].storage == dace.dtypes.StorageType.Default:
            #remove trailing "_host" and check if this is an output parameter
            if n.data.endswith("_host"):
                data_name = n.data[:-5]
                if data_name in program_description["outputs"]:
                    sdfg_output_data += [data_name]
            else:
                raise ValueError("Uhm...strange, what kind of data is " +
                                 n.data + "?")
    if sdfg_output_data:
        save_outputs = True
        print_with_rank(my_rank,"Initializing output arrays...")
        output_arrays = {
            arr_name: helper.aligned(
                np.zeros(program_description["dimensions"],
                         dtype=program_description["program"][arr_name]
                         ["data_type"].type), 64)
            for arr_name in sdfg_output_data
        }
        for key, val in output_arrays.items():
            dace_args[key + "_host"] = val



    # ----------------------------------------------
    # Execute
    # ----------------------------------------------

    MPI.COMM_WORLD.Barrier()
    print_with_rank(my_rank, "Executing DaCe program...")
    program(**dace_args)
    print(my_rank, "Finished running program.")

    if save_outputs:
        # Save output
        output_folder = os.path.join("results", stencil_name)
        os.makedirs(output_folder, exist_ok=True)
        helper.save_output_arrays(output_arrays, output_folder)
        print_with_rank(my_rank, "Results saved to " + output_folder)

    # Clean up
    if mode == "emulation" and my_rank == 0:
        os.system("rm emulated_channel*")

    MPI.COMM_WORLD.Barrier()
    # ----------------------------------------------
    # Compare to reference (only meaningful on ranks that output something)
    # ----------------------------------------------
    # TODO: this is currently done on the last node only. Fix this
    if compare_to_reference and my_rank == num_ranks - 1:
        chain = KernelChainGraph(path=stencil_file)
        reference_sdfg = generate_reference(stencil_name + "_reference", chain)
        reference_sdfg.expand_library_nodes()
        reference_program = reference_sdfg.compile()

        # Load input data
        input_directory = os.path.dirname(stencil_file)
        reference_input_arrays = helper.load_input_arrays(
            program_description["inputs"], prefix=input_directory)
        reference_output_arrays = copy.deepcopy(output_arrays)

        dace_args = {
            key: val
            for key, val in itertools.chain(reference_input_arrays.items(),
                                            reference_output_arrays.items())
        }
        print("Executing reference DaCe program...")
        reference_program(**dace_args)
        print("Finished running program.")

        reference_folder = os.path.join(output_folder, "reference")
        os.makedirs(reference_folder, exist_ok=True)
        helper.save_output_arrays(reference_output_arrays, reference_folder)
        print("Reference results saved to " + reference_folder)
        print("Comparing to reference SDFG...")
        for outp in output_arrays:
            if not helper.arrays_are_equal(
                    np.ravel(output_arrays[outp]),
                    np.ravel(reference_output_arrays[outp])):
                print(bcolors.FAIL + bcolors.BOLD + "Result mismatch." + bcolors.ENDC)
                exit(1)
        print(bcolors.OKGREEN + bcolors.BOLD + "Results verified." + bcolors.ENDC)
        exit(0)
    exit(0)
