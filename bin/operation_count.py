#!/usr/bin/env python3
import argparse
import ast
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow.helper import operation_count

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
args = parser.parse_args()

operations = operation_count(args.input_file)

op_sum = 0
for name, count in operations.items():
    print("{}: {}".format(name, count))
    op_sum += count
print("Total: {}".format(op_sum))
