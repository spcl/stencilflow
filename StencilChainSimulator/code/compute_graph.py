import ast
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from StencilChainSimulator.code.helper import Helper


class NodeType(Enum):
    NAME = 1,
    NUM = 2,
    BINOP = 3,
    CALL = 4,
    OUTPUT = 5


class Node:

    def __init__(self, node, number):
        self.number = number
        self.latency = None
        if node is not None:
            self.node_type = self.get_type(node)
            self.name = self.generate_name(node)

    @staticmethod
    def get_type(node):

        if isinstance(node, ast.Name):
            return NodeType.NAME
        elif isinstance(node, ast.Num):
            return NodeType.NUM
        elif isinstance(node, ast.BinOp):
            return NodeType.BINOP
        elif isinstance(node, ast.Call):
            return NodeType.CALL
        else:
            return None

    _OP_NAME_MAP = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mult",
        ast.Div: "div",
        ast.Invert: "neg"
    }

    def generate_name(self, node):

        if self.node_type == NodeType.NAME:
            return node.id
        elif self.node_type == NodeType.NUM:
            return node.n
        elif self.node_type == NodeType.BINOP:
            return Node._OP_NAME_MAP[type(node.op)]
        elif self.node_type == NodeType.CALL:
            return node.func.id

    def generate_label(self):
        return str(self.name)


class ComputeGraph:

    def __init__(self):

        # read static parameters from config
        self.config = Helper.parse_config("compute_graph.config")

        self.graph = nx.DiGraph()
        self.tree = None
        self.root = None

    def generate_graph(self, computation_string):

        # generate abstract syntax tree
        self.tree = ast.parse(computation_string)

        # TODO: support of the following input: "(a + out) * cos(out); out = a + b"

        last = self.ast_tree_walk(self.tree.body[0].value, 1)

        # add output node
        outp = Node(None, 0)
        outp.name = "out"
        outp.node_type = NodeType.OUTPUT
        self.graph.add_node(outp)

        # add root to class
        self.root = outp

        # add edge to first op
        self.graph.add_edge(last, outp)

        return self.graph

    def ast_tree_walk(self, node, number):

        # create node
        new_node = Node(node, number)

        # add node to graph
        self.graph.add_node(new_node)

        if isinstance(node, ast.BinOp):

            # do tree-walk recursively and get references to children (to create the edges to them)
            left = self.ast_tree_walk(node.left, ComputeGraph.child_left_number(number))
            right = self.ast_tree_walk(node.right, ComputeGraph.child_right_number(number))

            # add edges from parent to children
            self.graph.add_edge(left, new_node)
            self.graph.add_edge(right, new_node)

        elif isinstance(node, ast.Call):

            # do tree-walk for all arguments
            if len(node.args) > 2:
                raise NotImplementedError("Current implementation does not support more than two arguments due"
                                          " to the binary tree numbering convention")

            # process first argument
            first = self.ast_tree_walk(node.args[0], ComputeGraph.child_left_number(number))
            self.graph.add_edge(first, new_node)

            # check if second argument exist
            if len(node.args) >= 2:
                second = self.ast_tree_walk(node.args[1], ComputeGraph.child_right_number(number))
                self.graph.add_edge(second, new_node)

        elif isinstance(node, ast.Name):
            # nothing to do
            pass
        elif isinstance(node, ast.Num):
            # nothing to do
            pass

        # return node for reference purpose
        return new_node

    # tree.body[i] : i-th computation_string
    # tree.body[i].value = BinOp    -> subtree: .left, .right, .op {Add, Mult, Name, Call}
    # tree.body[i].value = Name     -> subtree: .id (name)
    # tree.body[i].value = Call     -> subtree: .func.id (function), .args[i] (i-th argument)

    @staticmethod
    def child_left_number(n):
        return 2*n + 1

    @staticmethod
    def child_right_number(n):
        return 2*n

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

        for node in self.graph.nodes:
            if node.node_type == NodeType.NUM:
                nums.append(node)
            elif node.node_type == NodeType.NAME:
                names.append(node)
            elif node.node_type == NodeType.BINOP or node.node_type == NodeType.CALL:
                ops.append(node)
            elif node.node_type == NodeType.OUTPUT:
                outs.append(node)

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

        '''
            merge draw nodes together with draw edges: using the hack of adding nodes/edges together in nx.draw_networkx(), the arrows are 
            getting aligned correctly 

            nx.draw_networkx_edges(G, positions, edge_color='black', arrows=True, arrowsize=48,
                                   arrowstyle='-|>', width=6, linewidths=0)                                             
        '''

        # save plot to file if save_path has been specified
        if save_path is not None:
            plt.savefig(save_path)

        # plot it
        plt.show()

    def calculate_latency(self):
        # idea: do a longest-path tree-walk (since the graph is a DAG (directed acyclic graph) we can do that
        # efficiently
        self.root.latency = 0
        self.latency_tree_walk(self.root)

    def latency_tree_walk(self, node):

        # check node type
        if node.node_type == NodeType.NAME or node.node_type == NodeType.NUM:
            return
        elif node.node_type == NodeType.BINOP or node.node_type == NodeType.CALL:

            # get op latency
            op_latency = self.config["op_latency"][node.name]

            for child in self.graph.pred[node]:
                child.latency = node.latency + op_latency
                self.latency_tree_walk(child)

        elif node.node_type == NodeType.OUTPUT:

            for child in self.graph.pred[node]:
                child.latency = node.latency
                self.latency_tree_walk(child)


'''

    Creation of a proper graph representation for the computation data flow graph.
    
    Credits for node-visitor: https://stackoverflow.com/questions/33029168/how-to-calculate-an-equation-in-a-string-python
    
    More info: https://networkx.github.io/
    
'''

if __name__ == "__main__":

    '''
        simple example for debugging purpose
    '''

    computation = "(a + 5) * cos(a + b)"
    graph = ComputeGraph()
    graph.generate_graph(computation)
    graph.calculate_latency()
    graph.plot_graph()

