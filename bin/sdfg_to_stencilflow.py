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
    parser.add_argument("-vector-length", type=int, default=1)

    args = parser.parse_args()

    sdfg_to_stencilflow(args.input_sdfg,
                        args.output_json,
                        vector_length=args.vector_length,
                        data_directory=args.data_directory)

    print("Saved StencilFlow program to: {}".format(args.output_json))
