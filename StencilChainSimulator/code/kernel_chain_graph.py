import argparse
import operator
import functools
import networkx as nx
import matplotlib.pyplot as plt
import helper
from kernel import Kernel
from bounded_queue import BoundedQueue
from base_node_class import BaseKernelNodeClass


class Input(BaseKernelNodeClass):

    def __init__(self, name, data_queue=None):
        super().__init__(name)
        self.data_queue = data_queue


class Output(BaseKernelNodeClass):

    def __init__(self, name, data_queue=None):
        super().__init__(name)
        self.data_queue = data_queue


class KernelChainGraph:

    def __init__(self, path, plot_graph=False):
        self.inputs = None  # type: dict # inputs[name] = input_array_data
        self.outputs = None
        self.path = path
        self.dimensions = None
        self.program = None  # type: dict  # program[stencil_name] = stencil expression
        self.kernels = None
        self.kernel_latency = None
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
        if plot_graph:
            # plot kernel chain graph
            self.plot_graph()
            # plot all compute graphs
            for compute_kernel in self.kernel_nodes:
                self.kernel_nodes[compute_kernel].graph.plot_graph()

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
            if isinstance(node, Kernel):
                ops.append(node)
            elif isinstance(node, Input):
                names.append(node)
            elif isinstance(node, Output):
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
        int_buf = self.kernel_nodes[dest_node.name].graph.buffer_size[src_node.name]

        if del_buf >= int_buf:  # return max
            return del_buf
        else:
            return int_buf

    def connect_kernels(self):

        for src in self.graph.nodes:
            for dest in self.graph.nodes:
                if src is not dest:  # skip src == dest
                    if isinstance(src, Kernel) and isinstance(dest, Kernel):  # case: KERNEL -> KERNEL
                        for inp in dest.graph.inputs:
                            if src.name == inp.name:
                                # add edge
                                self.graph.add_edge(src, dest, channel=None)
                                break
                    elif isinstance(src, Input) and isinstance(dest, Kernel):  # case: INPUT -> KERNEL
                        for inp in dest.graph.inputs:
                            if src.name == inp.name:
                                # add edge
                                self.graph.add_edge(src, dest, channel=None)
                                break
                    elif isinstance(dest, Output):  # case: INPUT/KERNEL -> OUTPUT
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
                    if isinstance(src, Kernel) and isinstance(dest, Kernel):  # case: KERNEL -> KERNEL
                        for inp in dest.graph.inputs:
                            if src.name == inp.name:
                                # create channel
                                name = src.name + "_" + dest.name
                                channel = BoundedQueue(name, 1 + src.convert_3d_to_1d(self.channel_size(dest, src)))
                                # TODO: move convert_3d_to_1d into higher hierarchical level
                                self.channels[name] = channel
                                # add channel to both endpoints
                                src.outputs[dest.name] = channel
                                dest.inputs[src.name] = channel
                                # add to edge
                                self.graph[src][dest]['channel'] = channel
                                break
                    elif isinstance(src, Input) and isinstance(dest, Kernel):  # case: INPUT -> KERNEL
                        for inp in dest.graph.inputs:
                            if src.name == inp.name:
                                # create channel
                                name = src.name + "_" + dest.name
                                channel = BoundedQueue(name, 1 + dest.convert_3d_to_1d(self.channel_size(dest, src)))
                                self.channels[name] = channel
                                # add channel to both endpoints
                                src.outputs[dest.name] = channel
                                dest.inputs[src.name] = channel
                                # add to edge
                                self.graph[src][dest]['channel'] = channel
                                break
                    elif isinstance(dest, Output):  # case: INPUT/KERNEL -> OUTPUT
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
        inp = helper.parse_json(self.path)
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
            new_node = Kernel(name=kernel, kernel_string=self.program[kernel], dimensions=self.dimensions)
            self.graph.add_node(new_node)
            self.kernel_nodes[kernel] = new_node

        # create all input nodes
        for inp in self.inputs:
            new_node = Input(name=inp, data_queue=BoundedQueue(name=inp, maxsize=self.total_elements(),
                                                               collection=self.inputs[inp]))
            self.graph.add_node(new_node)

        # create all output nodes
        for out in self.outputs:
            new_node = Output(name=out, data_queue=None)
            self.graph.add_node(new_node)

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
                max_delay = max(node.input_paths[inp])
                for entry in node.input_paths[inp]:
                    node.delay_buffer[entry[3]] = helper.list_subtract_cwise(max_delay[0:3], entry[0:3])
            if isinstance(node, Input):  # NodeType.INPUT:
                node.delay_buffer = [0, 0, 0, node.name]

            for succ in self.graph.successors(node):

                if isinstance(node, Input):  # add INPUT node to all as direct input (=0 delay buffer)
                    # add emtpy list dictionary entry for enabling list append()
                    if node.name not in succ.input_paths:
                        succ.input_paths[node.name] = []
                    succ.input_paths[node.name].append([0, 0, 0, node.name])

                elif isinstance(node, Kernel):  # add KERNEL

                    # add latency, internal_buffer, delay_buffer
                    internal_buffer = self.kernel_nodes[node.name].graph.buffer_size[helper.max_dict_entry_key(
                        self.kernel_nodes[node.name].graph.buffer_size)]
                    latency = self.kernel_nodes[node.name].graph.max_latency

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
    Since we know the output nodes as well as the path lengths the critical path is just 
    max { latency(node) + max { path_length(node) | node in output nodes }
    '''
    def compute_critical_path(self):

        critical_path_length = [0, 0, 0]
        for output in self.outputs:
            a = self.kernel_nodes[output].graph.max_latency
            b = max(self.kernel_nodes[output].input_paths)
            c = max(self.kernel_nodes[output].input_paths[b])
            c[2] += a
            critical_path_length = max(critical_path_length, c)
        return helper.dim_to_abs_val(c[0:3], self.dimensions)

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

    chain = KernelChainGraph(path=args.stencil_file, plot_graph=args.plot)

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
            print("node name: {}, field accesses: {}".format(node, chain.kernel_nodes[node].graph.accesses))
        print()

        print("internal buffer size info:")
        for node in chain.kernel_nodes:
            print("node name: {}, internal buffer size: {}".format(node,
                                                                   chain.kernel_nodes[node].graph.buffer_size))
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
            print("node name: {}, node latency: {}".format(node, chain.kernel_nodes[node].graph.max_latency))
        print()

        print("critical path info:")
        print("critical path length is {}\n".format(chain.compute_critical_path()))

        print("total buffer info:")
        total = 0
        for node in chain.kernel_nodes:
            for u, v, channel in chain.graph.edges(data='channel'):
                if channel is not None:
                    total += channel.maxsize
        print("total buffer size: {}\n".format(total))
