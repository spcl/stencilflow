#!/usr/bin/env python3
import argparse
import os
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dace import SDFG
from stencilflow import canonicalize_sdfg

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_sdfg")
    parser.add_argument("output_sdfg")
    parser.add_argument("-symbols")

    args = parser.parse_args()

    symbols = {}
    if args.symbols:
        for m in re.finditer(r"(\w+)[\s\W]*=[\s\W]*(\d+)", args.symbols):
            symbols[m.group(1)] = int(m.group(2))

    sdfg = SDFG.from_file(args.input_sdfg)

    canonicalize_sdfg(sdfg, symbols=symbols)
    sdfg.save(args.output_sdfg)

    print("Saved canonicalized SDFG to: {}".format(args.output_sdfg))
