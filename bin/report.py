#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow.kernel_chain_graph import KernelChainGraph
from stencilflow.log_level import LogLevel

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("-frequency", default=200, type=float)
args = parser.parse_args()

chain = KernelChainGraph(path=args.input_file, log_level=LogLevel.NO_LOG)

operations = chain.operation_count()
min_runtime = chain.runtime_lower_bound()
min_comm_volume = chain.minimum_communication_volume()

op_sum = 0
op_sum_total = 0
for name, (count, count_total) in operations.items():
    print("{}: {} per cycle ({} for program)".format(name, count, count_total))
    op_sum += count
    op_sum_total += count_total
print("Total: {} per cycle ({} for program)".format(op_sum, op_sum_total))
print("Lower bound on runtime: {} cycles ({} seconds at {} MHz)".format(
    min_runtime, min_runtime / (args.frequency * 1e6), args.frequency))
print("Upper bound on performance at {} MHz: {} GOp/s".format(
    args.frequency, 1e-9 * op_sum_total / min_runtime * args.frequency * 1e6))
print("Peak performance at {} MHz: {} GOp/s".format(
    args.frequency, 1e-9 * (op_sum * args.frequency * 1e6)))
print("Lower bound communication volume: {} MB".format(1e-6 * min_comm_volume))
print("Upper bound bandwidth: {} GB/s".format(
    1e-9 * min_comm_volume / (min_runtime / (1e6 * args.frequency))))
