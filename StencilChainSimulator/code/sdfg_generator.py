import argparse
import os
import re
from dapp.sdfg import SDFG
from kernel_chain_graph import KernelChainGraph


def generate_sdfg(name, chain):

    sdfg = SDFG(name)

    for node in chain.graph.nodes():
        import pdb; pdb.set_trace()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("stencil_file")

    args = parser.parse_args()

    name = os.path.basename(args.stencil_file)
    name = re.match("[^\.]+", name).group(0)

    chain = KernelChainGraph(args.stencil_file)

    chain.plot_graph(name + ".pdf")

    # generate_sdfg(name, chain)
