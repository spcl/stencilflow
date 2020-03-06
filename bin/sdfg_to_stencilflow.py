#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow import sdfg_to_stencilflow

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_sdfg")
    parser.add_argument("output_json")
    parser.add_argument("-data-directory")
    parser.add_argument("-symbols")

    args = parser.parse_args()

    symbols = {}
    for m in re.finditer("(\w+)[\s\W]*=[\s\W]*(\d+)", args.symbols):
        symbols[m.group(1)] = int(m.group(2))

    sdfg_to_stencilflow(
        args.input_sdfg,
        args.output_json,
        args.data_directory,
        symbols=symbols)

    print("Saved StencilFlow program to: {}".format(args.output_json))
