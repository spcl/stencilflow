#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow import run_program

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_file")
    parser.add_argument("mode", choices=["emulation", "hardware"])
    parser.add_argument("-run-simulation", action="store_true")
    parser.add_argument("-input-directory")
    parser.add_argument("-skip-execution",
                        dest="skip_execution",
                        action="store_true")
    parser.add_argument("-plot", action="store_true")
    parser.add_argument("-log-level", choices=["0", "1", "2", "3"], default=3)
    parser.add_argument("-print-result",
                        dest="print_result",
                        action="store_true")
    args = parser.parse_args()

    sys.exit(run_program(**vars(args)))
