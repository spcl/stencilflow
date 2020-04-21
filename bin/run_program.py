#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import stencilflow
from stencilflow.run_program import run_program

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_file")
    parser.add_argument("mode", choices=["emulation", "hardware"])
    parser.add_argument("-run-simulation", action="store_true")
    parser.add_argument("-compare-to-reference", action="store_true")
    parser.add_argument("-input-directory")
    parser.add_argument("-use-cached-sdfg",
                        dest="use_cached_sdfg",
                        action="store_true")
    parser.add_argument("-skip-execution",
                        dest="skip_execution",
                        action="store_true")
    parser.add_argument("-generate-input", action="store_true")
    parser.add_argument("-halo", type=int, default=0)
    parser.add_argument("-repetitions", type=int, default=1)
    parser.add_argument("-synthetic-reads", type=float, default=None)
    parser.add_argument("-plot", action="store_true")
    parser.add_argument("-log-level", choices=["0", "1", "2", "3"], default=3)
    parser.add_argument("-print-result",
                        dest="print_result",
                        action="store_true")
    args = parser.parse_args()

    args.log_level = stencilflow.log_level.LogLevel(args.log_level)

    sys.exit(run_program(**vars(args)))
