import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from StencilChainSimulator.code.helper import Helper
from StencilChainSimulator.code.kernel import Kernel
from StencilChainSimulator.code.bounded_queue import BoundedQueue


class KernelChainGraph:

    def __init__(self, path, dimensions):
        self.path = path
        self.dimensions = dimensions
        self.program = None  # type: dict  # program[stencil_name] = stencil expression
        self.kernels = None
        self.kernel_latency = None  # TODO: adapt channel size according to min/max latency to the kernel (over all
        # inputs)
        self.channels = None  # each channel is an edge between two kernels or a kernel and an input
        self.graph = nx.DiGraph()

        self.import_program()
        self.create_kernels()
        self.compute_kernel_latency()
        self.connect_kernels()
        self.plot_graph()

    def plot_graph(self, save_path=None):

        # create drawing area
        plt.figure(figsize=(20, 20))
        plt.axis('off')

        # generate positions
        positions = nx.nx_pydot.graphviz_layout(self.graph, prog='dot')

        # divide nodes into different lists for colouring purpose
        nums = list()
        names = list()
        ops = list()
        outs = list()

        # add nodes to list
        for node in self.graph.nodes:
            names.append(node)

        # create dictionary of labels
        labels = dict()
        for node in self.graph.nodes:
            labels[node] = node.generate_label()

        # add nodes and edges
        nx.draw_networkx_nodes(self.graph, positions, nodelist=names, node_color='orange',
                               node_size=3000, node_shape='s', edge_color='black')

        nx.draw_networkx_nodes(self.graph, positions, nodelist=outs, node_color='green',
                               node_size=3000, node_shape='s')

        nx.draw_networkx_nodes(self.graph, positions, nodelist=nums, node_color='#007acc',
                               node_size=3000, node_shape='s')

        nx.draw_networkx(self.graph, positions, nodelist=ops, node_color='red', node_size=3000,
                         node_shape='o', font_weight='bold', font_size=16, edge_color='black', arrows=True,
                         arrowsize=36,
                         arrowstyle='-|>', width=6, linwidths=1, with_labels=False)

        nx.draw_networkx_labels(self.graph, positions, labels=labels, font_weight='bold', font_size=16)

        # save plot to file if save_path has been specified
        if save_path is not None:
            plt.savefig(save_path)

        # plot it
        plt.show()

    def connect_kernels(self):

        self.channels = dict()

        for src in self.graph.nodes:
            for dest in self.graph.nodes:
                if src is not dest:  # skip src == dest
                    for inp in dest.graph.inputs:
                        if src.name == inp.name:
                            # add edge
                            self.graph.add_edge(src, dest)
                            # create channel
                            name = src.name + "_" + dest.name
                            channel = BoundedQueue(name, 1)
                            self.channels[name] = channel
                            # add channel to both endpoints
                            src.outputs[dest.name] = channel
                            dest.inputs[src.name] = channel

    def import_program(self):
        self.program = Helper.parse_json(self.path)

    def create_kernels(self):

        # create dict
        self.kernels = dict()

        # create all kernel objects and add them to the graph
        for kernel in self.program:
            self.graph.add_node(Kernel(name=kernel,
                                       kernel_string=self.program[kernel],
                                       dimensions=self.dimensions))

    def compute_kernel_latency(self):

        # create dict
        self.kernel_latency = dict()

        # compute kernel latency of each kernel
        for kernel in self.kernels:
            self.kernel_latency[kernel] = self.kernels[kernel].graph.max_latency
    '''
        simple test stencil program for debugging
    '''


if __name__ == "__main__":
    chain = KernelChainGraph("simple_input1.json", [10, 10, 10])

