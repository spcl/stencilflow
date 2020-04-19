#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from stencilflow.sdfg_generator import split_sdfg
import dace

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("sdfg_file")
    args.add_argument("split_stream")
    args.add_argument("send_rank")
    args.add_argument("receive_rank")
    args.add_argument("port", type=str)
    args.add_argument("-bytes_per_channel",
                      type=int,
                      help="Maximum channel width per cycle in bytes.")
    args = args.parse_args()

    ports = [int(x) for x in args.port.split(",")]
    if len(ports) == 0:
        ports = ports[0]

    sdfg = dace.SDFG.from_file(args.sdfg_file)
    sdfg_before, sdfg_after = split_sdfg(
        sdfg,
        args.split_stream,
        args.send_rank,
        args.receive_rank,
        ports,
        bytes_per_channel=args.bytes_per_channel)

    input_file = os.path.splitext(args.sdfg_file)[0]
    before_path = input_file + "_before.sdfg"
    after_path = input_file + "_after.sdfg"
    sdfg_before.save(before_path)
    sdfg_after.save(after_path)
    print("Split SDFGs saved to:\n\t{}\n\t{}".format(before_path, after_path))
