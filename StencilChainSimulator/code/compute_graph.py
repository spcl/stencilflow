import ast
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from helper import Helper
from calculator import Calculator


class NodeType(Enum):
    NAME = 1,
    NUM = 2,
    BINOP = 3,
    CALL = 4,
    OUTPUT = 5,
    SUBSCRIPT = 6


class Node:

    def __init__(self, node, number):
        self.number = number
        self.latency = -1
        self.index = list()
        if node is not None:
            self.node_type = self.get_type(node)
            self.name = self.generate_name(node)

    @staticmethod
    def get_type(node):

        if isinstance(node, ast.Name):  # variables or array access
            return NodeType.NAME
        elif isinstance(node, ast.Num):  # static value
            return NodeType.NUM
        elif isinstance(node, ast.BinOp):  # binary operation
            return NodeType.BINOP
        elif isinstance(node, ast.Call):  # function (e.g. sin, cos,..)
            return NodeType.CALL
        elif isinstance(node, ast.Assign):  # assign operator (var = expr;)
            return NodeType.OUTPUT
        elif isinstance(node, ast.Subscript):  # array access (form: arr[i,j,k])
            return NodeType.SUBSCRIPT
        else:
            return None

    _OP_NAME_MAP = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mult",
        ast.Div: "div",
        ast.Invert: "neg"
        # TODO: support operation: res = (a < b) ? c : d
    }

    _OP_SYM_MAP = {
        ast.Add: "+",
        ast.Sub: "-"
    }

    _VAR_MAP = {
        "i": 0,
        "j": 0,
        "k": 0
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
        elif self.node_type == NodeType.OUTPUT:
            return node.targets[0].id
        elif self.node_type == NodeType.SUBSCRIPT:
            # create index
            for slice in node.slice.value.elts:
                if isinstance(slice, ast.Name):
                    self.index.append(Node._VAR_MAP[slice.id])
                elif isinstance(slice, ast.BinOp):
                    # note: only support for index variations [i, j+3,..]
                    # read index expression
                    expression = str(slice.left.id) + Node._OP_SYM_MAP[type(slice.op)] + str(slice.right.n)

                    # convert [i+1,j, k-1] into [1, 0, -1]
                    calculator = Calculator()
                    self.index.append(calculator.eval_expr(Node._VAR_MAP, expression))
            # merge SUBSCRIPT nodes with NAME nodes, since they are handled identically from this point of time on
            self.node_type = NodeType.NAME

            return node.value.id

    def generate_label(self):

        if self.node_type == NodeType.NAME:
            return str(self.name) + str(self.index)
        else:
            return str(self.name)


class ComputeGraph:

    def __init__(self):

        # read static parameters from config
        self.config = Helper.parse_json("compute_graph.config")
        self.graph = nx.DiGraph()
        self.tree = None
        self.max_latency = -1
        self.inputs = None
        self.outputs = None
        self.min_index = None
        self.max_index = None
        self.buffer_size = None

    @staticmethod
    def compare_to(index_a, index_b):  # A >= B ?
        if index_a[0] > index_b[0]:
            return True
        elif index_a[1] == index_b[1]:
            if index_a[1] > index_b[1]:
                return True
            elif index_a[1] == index_b[1]:
                if index_a[2] > index_b[2]:
                    return True
                elif index_a[2] == index_b[2]:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def setup_internal_buffers(self):

        # init dicts
        self.min_index = dict()  # min_index["buffer_name"] = [i_min, j_min, k_min]
        self.max_index = dict()
        self.buffer_size = dict()  # buffer_size["buffer_name"] = size

        # find min and max index
        for inp in self.inputs:
            if inp.node_type == NodeType.NAME:
                if inp.name in self.min_index:
                    if not self.compare_to(inp.index, self.min_index[inp.name]):
                        self.min_index[inp.name] = inp.index

                    if self.compare_to(inp.index, self.max_index[inp.name]):
                        self.max_index[inp.name] = inp.index
                else:  # first entry
                    self.min_index[inp.name] = inp.index
                    self.max_index[inp.name] = inp.index

        # set buffer_size = max_index - min_index
        for buffer_name in self.min_index:
            self.buffer_size[buffer_name] = [abs(a_i - b_i) for a_i, b_i in zip(self.max_index[buffer_name], self.min_index[buffer_name])]

    def determine_inputs_outputs(self):

        # create empty sets
        self.inputs = set()
        self.outputs = set()

        # idea: do a tree-walk: all node with cardinality(predecessor)=0 are inputs, all nodes with cardinality(
        # successor)=0 are outputs
        for node in self.graph.nodes:
            if len(self.graph.pred[node]) == 0:
                self.inputs.add(node)
            if len(self.graph.succ[node]) == 0:
                self.outputs.add(node)

    def contract_edge(self, u, v):

        # add edges of v to u
        for edge in self.graph.succ[v]:
            self.graph.add_edge(u, edge)
        for edge in self.graph.pred[v]:
            self.graph.add_edge(edge, u)
        # remove v
        self.graph.remove_node(v)

    def generate_graph(self, computation_string):

        # generate abstract syntax tree
        self.tree = ast.parse(computation_string)

        for equation in self.tree.body:

            # check if base node is of type Expr or Assign
            if isinstance(equation, ast.Assign):
                lhs = Node(equation, 0)
                rhs = self.ast_tree_walk(equation.value, 1)
                self.graph.add_edge(rhs, lhs)

        # merge ambiguous variables in tree (implies: merge of ast.Assign trees into a single tree)
        '''
            NOTE: This nested loop runs in O(n^2), which is NOT the optimal solution (e.g. HashMap would be more 
            appropriate).
        '''
        outp_nodes = list(self.graph.nodes)
        for outp in outp_nodes:
            if outp.node_type == NodeType.NAME or outp.node_type == NodeType.SUBSCRIPT: # TODO: check if this makes sense
                inp_nodes = list(self.graph.nodes)
                for inp in inp_nodes:
                    if outp is not inp and outp.name == inp.name and outp.index == inp.index: # only contract if the indices match too
                        # contract nodes
                        outp.node_type = NodeType.NAME
                        self.contract_edge(outp, inp)

        # test if graph is now one component (for directed graph: each non-output must have at least one successor)
        for node in self.graph.nodes:
            if node.node_type != NodeType.OUTPUT and len(self.graph.succ[node]) == 0:
                raise RuntimeError("Kernel-internal data flow is not single component (must be connected in the sense "
                                   "of a DAG).")

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

    '''
    ast structure:
        tree.body[i] : i-th expression
        tree.body[i] = Assign: of type: x = Expr
        tree.body[i].targets          ->
        tree.body[i].value = {BinOp, Name, Call}
        tree.body[i] = Expr:
        tree.body[i].value = BinOp    -> subtree: .left, .right, .op {Add, Mult, Name, Call}
        tree.body[i].value = Name     -> subtree: .id (name)
        tree.body[i].value = Call     -> subtree: .func.id (function), .args[i] (i-th argument)
        tree.body[i].value = Subscript -> subtree: .slice.value.elts[i]: i-th parameter in [i, j, k, ...]
    '''

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
            elif node.node_type == NodeType.NAME or node.node_type == NodeType.SUBSCRIPT:
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

        # save plot to file if save_path has been specified
        if save_path is not None:
            plt.savefig(save_path)

        # plot it
        plt.show()

    def try_set_max_latency(self, new_val):
        if self.max_latency < new_val:
            self.max_latency = new_val

    def calculate_latency(self):

        # idea: do a longest-path tree-walk (since the graph is a DAG (directed acyclic graph) we can do that
        # efficiently
        for node in self.graph.nodes:
            if node.node_type == NodeType.OUTPUT:
                node.latency = 0
                self.try_set_max_latency(node.latency)
                self.latency_tree_walk(node)

    def latency_tree_walk(self, node):

        # check node type
        if node.node_type == NodeType.NAME or node.node_type == NodeType.NUM:
            for child in self.graph.pred[node]:
                child.latency = node.latency
                self.latency_tree_walk(child)
        elif node.node_type == NodeType.BINOP or node.node_type == NodeType.CALL:

            # get op latency
            op_latency = self.config["op_latency"][node.name]

            for child in self.graph.pred[node]:
                child.latency = max(child.latency, node.latency + op_latency)
                self.latency_tree_walk(child)

        elif node.node_type == NodeType.OUTPUT:

            for child in self.graph.pred[node]:
                child.latency = node.latency
                self.latency_tree_walk(child)

        self.try_set_max_latency(node.latency)


'''
    Creation of a proper graph representation for the computation data flow graph.
    
    Credits for node-visitor: https://stackoverflow.com/questions/33029168/how-to-calculate-an-equation-in-a-string-python
    
    More info: https://networkx.github.io/  
'''

if __name__ == "__main__":

    '''
        simple example for debugging purpose
    '''
    computation = "res = (A[i,j,k] + out) * cos(out); out = A[i,j,k-1] + B[i-1,j,k+1]"
    graph = ComputeGraph()
    graph.generate_graph(computation)
    graph.calculate_latency()
    graph.plot_graph("compute_graph_example.png")  # write graph to file
    # graph.plot_graph()
