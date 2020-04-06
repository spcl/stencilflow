#!/usr/bin/env python3
# encoding: utf-8
"""
BSD 3-Clause License

Copyright (c) 2018-2020, Johannes de Fine Licht, Andreas Kuster
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

__author__ = "Johannes de Fine Licht, Andreas Kuster"
__copyright__ = "Copyright 2018-2020, StencilFlow"
__license__ = "BSD-3-Clause"

import argparse
import copy
import itertools
import os
import re
import subprocess as sp
import sys

import dace
import numpy as np

from stencilflow import *
from stencilflow.sdfg_generator import generate_sdfg, generate_reference
from stencilflow.log_level import LogLevel
import stencilflow.helper as helper


def run_program(stencil_file,
                mode,
                run_simulation=False,
                compare_to_reference=False,
                input_directory=None,
                use_cached_sdfg=None,
                skip_execution=False,
                plot=False,
                log_level=LogLevel.BASIC,
                print_result=False):

    # Load program file
    program_description = helper.parse_json(stencil_file)
    name = os.path.basename(stencil_file)
    name = re.match("([^\.]+)\.[^\.]+", name).group(1)

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
        sdfg = generate_sdfg(name, chain)

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
    dace.config.Config.set("compiler", "fpga_vendor", value="intel_fpga")
    # dace.config.Config.set("compiler", "use_cache", value=True)
    dace.config.Config.set("optimizer", "interface", value="")
    dace.config.Config.set(
        "compiler",
        "intel_fpga",
        "kernel_flags",
        value="-fp-relaxed -cl-no-signed-zeros -no-interleaving=default"
        "-global-ring -cl-fast-relaxed-math -cl-single-precision-constant")
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

    # Load data from disk
    if log_level >= LogLevel.BASIC:
        print("Loading input arrays...")
    if input_directory is None:
        input_directory = os.path.dirname(stencil_file)
    input_arrays = helper.load_input_arrays(
        program_description["inputs"],
        prefix=input_directory,
        shape=program_description["dimensions"])

    # Initialize output arrays
    if log_level >= LogLevel.BASIC:
        print("Initializing output arrays...")
    output_arrays = {
        arr_name: helper.aligned(
            np.zeros(program_description["dimensions"],
                     dtype=program_description["program"][arr_name]
                     ["data_type"].type), 64)
        for arr_name in program_description["outputs"]
    }
    if compare_to_reference:
        reference_output_arrays = copy.deepcopy(output_arrays)

    if skip_execution:
        exit()

    # Run program
    dace_args = {
        key + "_host": val
        for key, val in itertools.chain(input_arrays.items(),
                                        output_arrays.items())
    }
    print("Executing DaCe program...")
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
    helper.save_output_arrays(output_arrays, output_folder)
    print("Results saved to " + output_folder)
    if compare_to_reference:
        reference_folder = os.path.join(output_folder, "reference")
        os.makedirs(reference_folder, exist_ok=True)
        helper.save_output_arrays(reference_output_arrays, reference_folder)
        print("Reference results saved to " + reference_folder)

    if compare_to_reference:
        print("Comparing to reference SDFG...")
        for outp in output_arrays:
            if not helper.arrays_are_equal(
                    np.ravel(output_arrays[outp]),
                    np.ravel(reference_output_arrays[outp])):
                print("Result mismatch.")
                return 1
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
            if not helper.arrays_are_equal(np.ravel(output_arrays[outp]),
                                           np.ravel(simulation_result[outp])):
                all_match = False
        if all_match:
            print("Results verified.")
            return 0
        else:
            print("Result mismatch.")
            return 1
