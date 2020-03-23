#!/usr/bin/env python3
import argparse
import ast
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class _OpCounter(ast.NodeVisitor):
    def __init__(self):
        self._operation_count = {}

    @property
    def operation_count(self):
        return self._operation_count

    def visit_BinOp(self, node: ast.BinOp):
        if isinstance(node.left, ast.Subscript) or isinstance(
                node.left, ast.BinOp) or isinstance(
                    node.right, ast.Subscript) or isinstance(
                        node.right, ast.BinOp):
            op_name = type(node.op).__name__
            if op_name not in self._operation_count:
                  self._operation_count[op_name] = 0
            self._operation_count[op_name] += 1
        self.generic_visit(node)

parser = argparse.ArgumentParser()
parser.add_argument("input_file")
args = parser.parse_args()

with open(args.input_file, "r") as in_file:
    program_description = json.loads(in_file.read())

operations = {}

for name, stencil in program_description["program"].items():

    stencil_ast = ast.parse(stencil["computation_string"])

    counter = _OpCounter()
    counter.visit(stencil_ast)
    num_ops = counter.operation_count
    for name, count in num_ops.items():
        if name not in operations:
            operations[name] = count
        else:
            operations[name] += count

op_sum = 0
for name, count in operations.items():
    print("{}: {}".format(name, count))
    op_sum += count
print("Total: {}".format(op_sum))
