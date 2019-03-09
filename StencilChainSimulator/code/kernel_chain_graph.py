import argparse
import networkx as nx
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from enum import Enum
import operator
import functools
from helper import Helper
from kernel import Kernel
from bounded_queue import BoundedQueue


class NodeType(Enum):
    INPUT = 1,
    KERNEL = 2,
    OUTPUT = 3


class Node:
    def __init__(self, name, node_type, kernel=None, data_queue=None):
        self.name = name
        self.node_type = node_type
        self.kernel = kernel  # only used for type: KERNEL
        self.outputs = dict()
        self.data_queue = data_queue  # only used for type: INPUT, OUTPUT
        self.input_paths = dict()
        self.delay_buffer = dict()
        self.total_delay_buffer = [0, 0, 0]

    def generate_label(self):
        return self.name


class KernelChainGraph:
    def __init__(self, path):
        self.inputs = None  # type: dict # inputs[name] = input_array_data
        self.outputs = None
        self.path = path
        self.dimensions = None
        self.program = None  # type: dict  # program[stencil_name] = stencil expression
        self.kernels = None
        self.kernel_latency = None  # TODO: adapt channel size according to min/max latency to the kernel (over all
        # inputs)
        self.channels = None  # each channel is an edge between two kernels or a kernel and an input
        self.graph = nx.DiGraph()

        self.input_nodes = dict()
        self.output_nodes = dict()
        self.kernel_nodes = dict()

        self.import_input()
        self.create_kernels()
        self.compute_kernel_latency()
        self.connect_kernels()
        self.compute_delay_buffer()
        self.add_channels()
        # self.plot_graph("overview.png")

    def plot_graph(self, save_path=None):

        # create drawing area
        fig, ax = plt.subplots()
        fig.set_size_inches(25, 25)
        ax.set_axis_off()

        # generate positions
        positions = nx.nx_pydot.graphviz_layout(self.graph, prog='dot')

        # divide nodes into different lists for colouring purpose
        nums = list()
        names = list()
        ops = list()
        outs = list()

        # add nodes to list
        for node in self.graph.nodes:
            if node.node_type == NodeType.KERNEL:
                ops.append(node)
            elif node.node_type == NodeType.INPUT:
                names.append(node)
            elif node.node_type == NodeType.OUTPUT:
                outs.append(node)

        # create dictionary of labels
        labels = dict()
        for node in self.graph.nodes:
            labels[node] = node.generate_label()

        # add nodes and edges
        nx.draw_networkx_nodes(
            self.graph,
            positions,
            nodelist=names,
            node_color='orange',
            node_size=3000,
            node_shape='s',
            edge_color='black')

        nx.draw_networkx_nodes(
            self.graph,
            positions,
            nodelist=outs,
            node_color='green',
            node_size=3000,
            node_shape='s')

        nx.draw_networkx_nodes(
            self.graph,
            positions,
            nodelist=nums,
            node_color='#007acc',
            node_size=3000,
            node_shape='s')

        nx.draw_networkx(
            self.graph,
            positions,
            nodelist=ops,
            node_color='red',
            node_size=3000,
            node_shape='o',
            font_weight='bold',
            font_size=16,
            edge_color='black',
            arrows=True,
            arrowsize=36,
            arrowstyle='-|>',
            width=6,
            linwidths=1,
            with_labels=False)

        nx.draw_networkx_labels(
            self.graph,
            positions,
            labels=labels,
            font_weight='bold',
            font_size=16)

        # save plot to file if save_path has been specified
        if save_path is not None:
            fig.savefig(save_path)
        else:
            # plot it
            fig.show()

    def channel_size(self, dest_node, src_node):
        del_buf = self.kernel_nodes[dest_node.name].delay_buffer[src_node.name]
        int_buf = self.kernel_nodes[dest_node.name].kernel.graph.buffer_size[src_node.name]

        if del_buf >= int_buf:  # return max # TODO: check if this is always the case
            return del_buf
        else:
            return int_buf

    def connect_kernels(self):

        for src in self.graph.nodes:
            for dest in self.graph.nodes:
                if src is not dest:  # skip src == dest
                    if src.node_type == NodeType.KERNEL and dest.node_type == NodeType.KERNEL:  # case: KERNEL -> KERNEL
                        for inp in dest.kernel.graph.inputs:
                            if src.name == inp.name:
                                # add edge
                                self.graph.add_edge(src, dest, channel=None)
                                break
                    elif src.node_type == NodeType.INPUT and dest.node_type == NodeType.KERNEL:  # case: INPUT -> KERNEL
                        for inp in dest.kernel.graph.inputs:
                            if src.name == inp.name:
                                # add edge
                                self.graph.add_edge(src, dest, channel=None)
                                break
                    elif dest.node_type == NodeType.OUTPUT:  # case: INPUT/KERNEL -> OUTPUT
                        if src.name == dest.name:
                            # add edge
                            self.graph.add_edge(src, dest, channel=None)
                    else:
                        pass  # Are there reasons for existence of those combinations?

    def add_channels(self):
        self.channels = dict()

        for src in self.graph.nodes:
            for dest in self.graph.nodes:
                if src is not dest:  # skip src == dest
                    if src.node_type == NodeType.KERNEL and dest.node_type == NodeType.KERNEL:  # case: KERNEL -> KERNEL
                        for inp in dest.kernel.graph.inputs:
                            if src.name == inp.name:
                                # create channel
                                name = src.name + "_" + dest.name
                                channel = BoundedQueue(name, 1 + src.kernel.convert_3d_to_1d(self.channel_size(dest, src)))
                                # TODO: move convert_3d_to_1d into higher hierarchical level
                                self.channels[name] = channel
                                # add channel to both endpoints
                                src.kernel.outputs[dest.name] = channel
                                dest.kernel.inputs[src.name] = channel
                                # add to edge
                                self.graph[src][dest]['channel'] = channel
                                break
                    elif src.node_type == NodeType.INPUT and dest.node_type == NodeType.KERNEL:  # case: INPUT -> KERNEL
                        for inp in dest.kernel.graph.inputs:
                            if src.name == inp.name:
                                # create channel
                                name = src.name + "_" + dest.name
                                channel = BoundedQueue(name, 1 + dest.kernel.convert_3d_to_1d(self.channel_size(dest, src)))
                                self.channels[name] = channel
                                # add channel to both endpoints
                                src.outputs[dest.name] = channel
                                dest.kernel.inputs[src.name] = channel
                                # add to edge
                                self.graph[src][dest]['channel'] = channel
                                break
                    elif dest.node_type == NodeType.OUTPUT:  # case: INPUT/KERNEL -> OUTPUT
                        if src.name == dest.name:
                            # create channel
                            name = src.name + "_" + dest.name
                            channel = BoundedQueue(name, 1)  # no buffer
                            self.channels[name] = channel
                            # add channel to both endpoints
                            src.outputs[dest.name] = channel
                            dest.data_queue = channel
                            # add to edge
                            self.graph[src][dest]['channel'] = channel
                    else:
                        pass  # Are there reasons for existence of those combinations?

    def import_input(self):
        inp = Helper.parse_json(self.path)
        self.program = inp["program"]
        self.inputs = inp["inputs"]
        self.outputs = inp["outputs"]
        self.dimensions = inp["dimensions"]

    def total_elements(self):
        return functools.reduce(operator.mul, self.dimensions,
                                1)  # foldl (*) 1 [...]

    def create_kernels(self):

        # create dict
        self.kernels = dict()

        # create all kernel objects and add them to the graph
        for kernel in self.program:
            new_node = Node(
                kernel, NodeType.KERNEL,
                Kernel(
                    name=kernel,
                    kernel_string=self.program[kernel],
                    dimensions=self.dimensions))
            self.graph.add_node(new_node)
            self.kernel_nodes[kernel] = new_node

        # create all input nodes
        for inp in self.inputs:
            if len(self.inputs[inp]) == self.total_elements():  # if the input data list is of size total_elements,
                #  it is a valid input for simulation
                self.graph.add_node(
                    Node(
                        name=inp,
                        node_type=NodeType.INPUT,
                        data_queue=BoundedQueue(inp, self.total_elements(),
                                                self.inputs[inp])))
            else:
                self.graph.add_node(Node(name=inp, node_type=NodeType.INPUT))

        # create all output nodes
        for out in self.outputs:
            self.graph.add_node(
                Node(name=out, node_type=NodeType.OUTPUT, data_queue=None))

    def compute_kernel_latency(self):

        # create dict
        self.kernel_latency = dict()

        # compute kernel latency of each kernel
        for kernel in self.kernels:
            self.kernel_latency[kernel] = self.kernels[
                kernel].graph.max_latency

    '''
    delay buffer entries should be of the format:

    kernel.input_paths:

    {
        "in1": [[a,b,c, pred1], [d,e,f, pred2], ...],
        "in2": [ ... ],
        ...
    }

    where inX are input arrays to the stencil chain and predY are the kernel predecessors/inputs
    '''

    def compute_delay_buffer(self):

        # get topological order for top-down walk through of the graph
        try:
            order = nx.topological_sort(self.graph)
        except nx.exception.NetworkXUnfeasible:
            raise ValueError("Cycle detected, cannot be sorted topologically!")

        for node in order:

            # process delay buffer (no additional delay buffer will appear because of the topological order)
            for inp in node.input_paths:
                # min_delay = Helper.min_list_entry(node.input_paths[inp])
                max_delay = max(node.input_paths[inp])
                for entry in node.input_paths[inp]:
                    # node.delay_buffer[entry[3]] = Helper.list_subtract_cwise(entry[0:3], min_delay[0:3])
                    node.delay_buffer[entry[3]] = Helper.list_subtract_cwise(max_delay[0:3], entry[0:3])
            if node.node_type == NodeType.INPUT:
                node.delay_buffer = [0, 0, 0, node.name]

            for succ in self.graph.successors(node):

                if node.node_type == NodeType.INPUT:  # add INPUT node to all as direct input (=0 delay buffer)
                    # add emtpy list dictionary entry for enabling list append()
                    if node.name not in succ.input_paths:
                        succ.input_paths[node.name] = []
                    # succ.input_paths[node.name].append((node.name, [0, 0, 0])) ##
                    succ.input_paths[node.name].append([0, 0, 0, node.name])

                elif node.node_type == NodeType.KERNEL:  # add KERNEL

                    # add latency, internal_buffer, delay_buffer
                    internal_buffer = self.kernel_nodes[node.name].kernel.graph.buffer_size[Helper.max_dict_entry_key(
                        self.kernel_nodes[node.name].kernel.graph.buffer_size)]
                    latency = self.kernel_nodes[node.name].kernel.graph.max_latency

                    for entry in node.input_paths:

                        if entry not in succ.input_paths:
                            succ.input_paths[entry] = []

                        delay_buffer = max(node.input_paths[entry][0:3])
                        # max determines 'longest path'

                        total = [
                            internal_buffer[0] + delay_buffer[0],
                            internal_buffer[1] + delay_buffer[1],
                            internal_buffer[2] + latency + delay_buffer[2],
                            node.name
                        ]

                        succ.input_paths[entry].append(total)

                else:  # NodeType.OUTPUT: do nothing
                    continue

    '''
        simple test stencil program for debugging
    '''


if __name__ == "__main__":

    # usage: python3 kernel_chain_graph.py --stencil_file simple_input_delay_buf.json --plot False --report True
    parser = argparse.ArgumentParser()
    parser.add_argument("--stencil_file")
    parser.add_argument("--plot")
    parser.add_argument("--report")

    args = parser.parse_args()

    chain = KernelChainGraph(args.stencil_file)

    if args.plot == "True":
        chain.plot_graph("output.png")

    if args.report == "True":

        print("Report of {}\n".format(args.stencil_file))

        total_internal = [0, 0, 0]
        total_delay = [0, 0, 0]

        print("dimensions of data array: {}\n".format(chain.dimensions))

        print("channel info:")
        for u, v, channel in chain.graph.edges(data='channel'):
            if channel is not None:
                print("channel name: {}, max channel size: {}".format(channel.name, channel.maxsize))
        print()

        print("field access info:")
        for node in chain.kernel_nodes:
            print("node name: {}, field accesses: {}".format(node, chain.kernel_nodes[node].kernel.graph.accesses))
        print()

        print("internal buffer size info:")
        for node in chain.kernel_nodes:
            print("node name: {}, internal buffer size: {}".format(node,
                                                                   chain.kernel_nodes[node].kernel.graph.buffer_size))
        print()

        print("delay buffer size info:")
        for node in chain.kernel_nodes:
            print("node name: {}, delay buffer size: {}".format(node, chain.kernel_nodes[node].delay_buffer))
        print()

        print("path length info:")
        for node in chain.kernel_nodes:
            print("node name: {}, path lengths: {}".format(node, chain.kernel_nodes[node].input_paths))
        print()

        print("latency info:")
        for node in chain.kernel_nodes:
            print("node name: {}, node latency: {}".format(node, chain.kernel_nodes[node].kernel.graph.max_latency))
        print()

        print("total buffer info:")
        total = 0
        for node in chain.kernel_nodes:
            for u, v, channel in chain.graph.edges(data='channel'):
                if channel is not None:
                    total += channel.maxsize
        print("total buffer size: {}\n".format(total))
