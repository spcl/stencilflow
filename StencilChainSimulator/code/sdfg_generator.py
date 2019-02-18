import argparse
import collections
import os
import re
from dapp.memlet import Memlet
from dapp.sdfg import SDFG
from dapp.types import ScheduleType, StorageType
from kernel_chain_graph import KernelChainGraph, NodeType

DATA_TYPE = float
ITERATORS = ["i", "j", "k"]


def iteration_space(chain):
    if len(chain.dimensions) > 3:
        return collections.OrderedDict([
            ("i" + str(i), "0:" + str(d))
            for i, d in enumerate(chain.dimensions)
        ])
    else:
        return collections.OrderedDict([
            (ITERATORS[i], "0:" + str(d))
            for i, d in enumerate(chain.dimensions)
        ])

def stream_name(src_name, dst_name):
    return src_name + "_to_" + dst_name


def add_array(sdfg, state, node, chain):

    sdfg.add_array(
        node.name,
        chain.dimensions,
        DATA_TYPE,
        storage=StorageType.FPGA_Global,
        transient=True)

    access_node = state.add_read(node.name)

    iterators = iteration_space(chain)

    entry, exit = state.add_map(
        "read_" + node.name, iterators, schedule=ScheduleType.FPGA_Device)

    # Sort to get deterministic output
    outputs = sorted(node.outputs.keys())

    tasklet_code = "\n".join(["{} = memory".format(n) for n in outputs])

    tasklet = state.add_tasklet("read_" + node.name, {"memory"}, outputs,
                                tasklet_code)

    state.add_memlet_path(
        access_node,
        entry,
        tasklet,
        dst_conn="memory",
        memlet=Memlet.simple(node.name, ", ".join(iterators.keys())))

    for dst_name in outputs:
        data_name = stream_name(node.name, dst_name)
        sdfg.add_stream(
            data_name,
            DATA_TYPE,
            buffer_size=node.outputs[dst_name].maxsize,
            storage=StorageType.FPGA_Local,
            transient=True)
        write_node = state.add_write(data_name)
        state.add_memlet_path(
            tasklet,
            exit,
            write_node,
            src_conn=dst_name,
            memlet=Memlet.simple(data_name, "0"))



def generate_sdfg(name, chain):

    sdfg = SDFG(name)

    state = sdfg.add_state("compute")

    memories = {}

    # Proce
    for node in chain.graph.nodes():
        if node.node_type == NodeType.INPUT:
            add_array(sdfg, state, node, chain)

    sdfg.draw_to_file()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_file")

    args = parser.parse_args()

    name = os.path.basename(args.stencil_file)
    name = re.match("[^\.]+", name).group(0)

    chain = KernelChainGraph(args.stencil_file)

    # chain.plot_graph(name + ".pdf")

    generate_sdfg(name, chain)
