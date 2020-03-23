#!/usr/bin/env python3
import argparse
import ast
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow.helper import operation_count, minimum_communication_volume

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("-frequency", default=200)
args = parser.parse_args()

operations = operation_count(args.input_file)

op_sum = 0
op_sum_total = 0
for name, (count, count_total) in operations.items():
    print("{}: {} per cycle ({} for program)".format(name, count, count_total))
    op_sum += count
    op_sum_total += count_total
print("Total: {} per cycle ({} for program)".format(op_sum, op_sum_total))
print("Minimum runtime at {} MHz: {} s".format(
    args.frequency, op_sum_total / op_sum / (args.frequency * 1e6)))
print("Maximum performance at {} MHz: {} GOp/s".format(
    args.frequency, 1e-9 * (op_sum * args.frequency * 1e6)))
print("Minimum communication volume: {} MB".format(
    1e-6 * minimum_communication_volume(args.input_file)))
