#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow import generate_sdfg, KernelChainGraph

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_input",
                        help="Stencil description file (.json)")
    parser.add_argument("sdfg_output", help="Output SDFG file (.sdfg)")
    parser.add_argument("--plot-graph", dest="plot-graph", action="store_true")
    parser.add_argument("--plot-sdfg", dest="plot-sdfg", action="store_true")
    parser.add_argument(
        "--compile",
        dest="compile",
        action="store_true",
        help="Compile the SDFG for verification/debugging purposes.")

    args = parser.parse_args()

    name = os.path.basename(args.stencil_input)
    name = re.match("[^\.]+", name).group(0)

    chain = KernelChainGraph(args.stencil_input)

    if getattr(args, "plot-graph"):
        chain.plot_graph(name + ".pdf")

    sdfg = generate_sdfg(name, chain)

    if getattr(args, "plot-sdfg"):
        chain.plot_graph(name + ".pdf")

    sdfg.save(args.sdfg_output)
    print("SDFG saved to: " + args.sdfg_output)

    if args.compile:
        dace.Config.set("compiler", "fpga_vendor", value="intel_fpga")
        sdfg.compile()
