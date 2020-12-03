import argparse
import copy
import itertools
import os
import re
import subprocess as sp
import sys

import dace
import numpy as np

import stencilflow
from stencilflow.simulator import Simulator
from stencilflow.kernel_chain_graph import KernelChainGraph
from stencilflow.sdfg_generator import generate_sdfg, generate_reference
from stencilflow.log_level import LogLevel


def run_program(stencil_file,
                mode,
                run_simulation=False,
                compare_to_reference=False,
                input_directory=None,
                use_cached_sdfg=None,
                skip_execution=False,
                generate_input=False,
                synthetic_reads=None,
                specialize_scalars=False,
                plot=False,
                halo=0,
                repetitions=1,
                log_level=LogLevel.BASIC,
                print_result=False,
                xilinx=False):

    # Load program file
    program_description = stencilflow.parse_json(stencil_file)
    name = os.path.basename(stencil_file)
    name = re.match("(.+)\.[^\.]+", name).group(1).replace(".", "_")

    # Create SDFG
    if log_level >= LogLevel.BASIC:
        print("Creating kernel graph...")
    chain = KernelChainGraph(path=stencil_file,
                             plot_graph=plot,
                             log_level=log_level)

    # do simulation
    if run_simulation:
        if log_level >= LogLevel.BASIC:
            print("Running simulation...")
        sim = Simulator(program_name=name,
                        program_description=program_description,
                        input_nodes=chain.input_nodes,
                        kernel_nodes=chain.kernel_nodes,
                        output_nodes=chain.output_nodes,
                        dimensions=chain.dimensions,
                        write_output=False,
                        log_level=log_level)
        sim.simulate()
        simulation_result = sim.get_result()

    if use_cached_sdfg:
        if log_level >= LogLevel.BASIC:
            print("Loading cached SDFG...")
        sdfg_path = os.path.join(".dacecache", name, "program.sdfg")
        sdfg = dace.SDFG.from_file(sdfg_path)
    else:
        if log_level >= LogLevel.BASIC:
            print("Generating SDFG...")
        sdfg = generate_sdfg(name,
                             chain,
                             synthetic_reads=synthetic_reads,
                             specialize_scalars=specialize_scalars)

    if compare_to_reference:
        if use_cached_sdfg:
            if log_level >= LogLevel.BASIC:
                print("Loading cached reference SDFG...")
            sdfg_path = os.path.join(".dacecache", name + "_reference",
                                     "program.sdfg")
            reference_sdfg = dace.SDFG.from_file(sdfg_path)
        else:
            if log_level >= LogLevel.BASIC:
                print("Generating reference SDFG...")
            reference_sdfg = generate_reference(name + "_reference", chain)

    # Configure and compile SDFG
    if not xilinx:
        dace.config.Config.set("compiler", "fpga_vendor", value="intel_fpga")
    else:
        dace.config.Config.set("compiler", "fpga_vendor", value="xilinx")
    dace.config.Config.set("optimizer", "transform_on_call", value=False)
    dace.config.Config.set(
        "compiler",
        "intel_fpga",
        "kernel_flags",
        value="-fp-relaxed -cl-no-signed-zeros -no-interleaving=default"
        " -global-ring -duplicate-ring -cl-fast-relaxed-math -cl-single-precision-constant"
    )
    if mode == "emulation":
        dace.config.Config.set("compiler",
                               "intel_fpga",
                               "mode",
                               value="emulator")
        dace.config.Config.set("compiler", "xilinx", "mode", value="simulation")
    elif mode == "hardware":
        dace.config.Config.set("compiler",
                               "intel_fpga",
                               "mode",
                               value="hardware")
        dace.config.Config.set("compiler", "xilinx", "mode", value="hardware")
    else:
        raise ValueError("Unrecognized execution mode: {}".format(mode))
    if log_level >= LogLevel.BASIC:
        print("Expanding library nodes...")
    sdfg.expand_library_nodes()
    if log_level >= LogLevel.BASIC:
        print("Compiling SDFG...")
    program = sdfg.compile()
    if compare_to_reference:
        if log_level >= LogLevel.BASIC:
            print("Compiling reference SDFG...")
        reference_sdfg.expand_library_nodes()
        reference_program = reference_sdfg.compile()

    if skip_execution or repetitions == 0:
        if log_level >= LogLevel.BASIC:
            print("Skipping execution and exiting.")
        return

    # Load data from disk
    if log_level >= LogLevel.BASIC:
        print("Loading input arrays...")
    if input_directory is None:
        input_directory = os.path.dirname(stencil_file)
    input_description = copy.copy(program_description["inputs"])
    if generate_input:
        # Generate some input so we don't load files off the disk
        for k in input_description:
            input_description[k]["data"] = "constant:0.5"
    input_arrays = stencilflow.load_input_arrays(
        input_description,
        prefix=input_directory,
        shape=program_description["dimensions"])

    # Initialize output arrays
    if log_level >= LogLevel.BASIC:
        print("Initializing output arrays...")
    output_arrays = {
        arr_name: stencilflow.aligned(
            np.zeros(program_description["dimensions"],
                     dtype=program_description["program"][arr_name]
                     ["data_type"].type), 64)
        for arr_name in program_description["outputs"]
    }
    if compare_to_reference:
        reference_output_arrays = copy.deepcopy(output_arrays)

    # Run program
    dace_args = {
        (key +
         "_host" if hasattr(val, "shape") and len(val.shape) > 0 else key): val
        for key, val in itertools.chain(input_arrays.items(),
                                        output_arrays.items())
    }
    if repetitions == 1:
        print("Executing DaCe program...")
        program(**dace_args)
        print("Finished running program.")
    else:
        for i in range(repetitions):
            print("Executing repetition {}/{}...".format(i + 1, repetitions))
            program(**dace_args)
            print("Finished running program.")

    if print_result:
        for key, val in output_arrays.items():
            print(key + ":", val)

    # Run reference program
    if compare_to_reference:
        dace_args = {
            key: val
            for key, val in itertools.chain(input_arrays.items(),
                                            reference_output_arrays.items())
        }
        print("Executing reference DaCe program...")
        reference_program(**dace_args)
        print("Finished running program.")

    if print_result:
        for key, val in reference_output_arrays.items():
            print(key + ":", val)

    # Write results to file
    output_folder = os.path.join("results", name)
    os.makedirs(output_folder, exist_ok=True)
    if halo > 0:
        # Prune halos
        for k, v in output_arrays.items():
            output_arrays[k] = v[tuple(slice(halo, -halo) for _ in v.shape)]
        if compare_to_reference:
            for k, v in reference_output_arrays.items():
                reference_output_arrays[k] = v[tuple(
                    slice(halo, -halo) for _ in v.shape)]
    stencilflow.save_output_arrays(output_arrays, output_folder)
    print("Results saved to " + output_folder)
    if compare_to_reference:
        reference_folder = os.path.join(output_folder, "reference")
        os.makedirs(reference_folder, exist_ok=True)
        stencilflow.save_output_arrays(reference_output_arrays,
                                       reference_folder)
        print("Reference results saved to " + reference_folder)

    if compare_to_reference:
        print("Comparing to reference SDFG...")
        for outp in output_arrays:
            got = output_arrays[outp]
            expected = reference_output_arrays[outp]
            if not stencilflow.arrays_are_equal(np.ravel(got),
                                                np.ravel(expected)):
                print("Expected: {}".format(expected))
                print("Got:      {}".format(got))
                raise ValueError("Result mismatch.")
        print("Results verified.")
        return 0

    # Compare simulation result to fpga result
    if run_simulation:
        print("Comparing results...")
        all_match = True
        for outp in output_arrays:
            print("FPGA result:")
            print("\t{}".format(np.ravel(output_arrays[outp])))
            print("Simulation result:")
            print("\t{}".format(np.ravel(simulation_result[outp])))
            if not stencilflow.arrays_are_equal(
                    np.ravel(output_arrays[outp]),
                    np.ravel(simulation_result[outp])):
                all_match = False
        if all_match:
            print("Results verified.")
            return 0
        else:
            print("Result mismatch.")
            return 1
