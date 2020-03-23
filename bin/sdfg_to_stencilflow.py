#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow import sdfg_to_stencilflow

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_sdfg")
    parser.add_argument("output_json")
    parser.add_argument("-data-directory")

    args = parser.parse_args()

    sdfg_to_stencilflow(args.input_sdfg, args.output_json, args.data_directory)

    print("Saved StencilFlow program to: {}".format(args.output_json))
