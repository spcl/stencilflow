import ast
import operator
import networkx as nx
import matplotlib.pyplot as plt
import helper
from calculator import Calculator
from base_node_class import BaseOperationNodeClass


class Name(BaseOperationNodeClass):

    def __init__(self, ast_node, number):
        super().__init__(ast_node, number)

    def generate_name(self, ast_node):
        return ast_node.id


class Num(BaseOperationNodeClass):

    def __init__(self, ast_node, number):
        super().__init__(ast_node, number)

    def generate_name(self, ast_node):
        return ast_node.n


class Binop(BaseOperationNodeClass):

    _OP_NAME_MAP = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mult",
        ast.Div: "div",
        ast.Invert: "neg"
        # TODO: support operation: res = (a < b) ? c : d
    }

    _OP_SYM_MAP = {
        "add": "+",
        "sub": "-",
        "mult": "*",
        "div": "/",
        "neg": "-"
        # TODO: support operation: res = (a < b) ? c : d
    }

    def __init__(self, ast_node, number):
        super().__init__(ast_node, number)

    def generate_name(self, ast_node):
        return self._OP_NAME_MAP[type(ast_node.op)]

    def generate_op_sym(self):
        return self._OP_SYM_MAP[self.name]


class Call(BaseOperationNodeClass):

    def __init__(self, ast_node, number):
        super().__init__(ast_node, number)

    def generate_name(self, ast_node):
        return ast_node.func.id


class Output(BaseOperationNodeClass):

    def __init__(self, ast_node, number):
        super().__init__(ast_node, number)

    def generate_name(self, ast_node):
        return ast_node.targets[0].id


class Subscript(BaseOperationNodeClass):

    _VAR_MAP = {
        "i": 0,
        "j": 0,
        "k": 0
    }

    _OP_SYM_MAP = {
        ast.Add: "+",
        ast.Sub: "-"
    }

    def __init__(self, ast_node, number):
        self.index = None
        self.create_index(ast_node=ast_node)
        super().__init__(ast_node, number)

    def create_index(self, ast_node):
        # create index
        self.index = list()
        for slice in ast_node.slice.value.elts:
            if isinstance(slice, ast.Name):
                self.index.append(self._VAR_MAP[slice.id])
            elif isinstance(slice, ast.BinOp):
                # note: only support for index variations [i, j+3,..]
                # read index expression
                expression = str(slice.left.id) + self._OP_SYM_MAP[type(slice.op)] + str(slice.right.n)

                # convert [i+1,j, k-1] into [1, 0, -1]
                calculator = Calculator()
                self.index.append(calculator.eval_expr(self._VAR_MAP, expression))

    def generate_name(self, ast_node):
        return ast_node.value.id

    def generate_label(self):
        return str(self.name) + str(self.index)


class Ternary(BaseOperationNodeClass):

    def __init__(self, ast_node, number):
        super().__init__(ast_node, number)

    def generate_name(self, ast_node):
        return "?"


class Compare(BaseOperationNodeClass):

    _COMP_MAP = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq
    }

    _COMP_SYM = {
        operator.lt: "<",
        operator.le: "<=",
        operator.gt: ">",
        operator.ge: ">=",
        operator.eq: "=="
    }

    def __init__(self, ast_node, number):
        self.op = self._COMP_MAP[type(ast_node.ops[0])]
        super().__init__(ast_node, number)

    def generate_name(self, ast_node):
        return self._COMP_SYM[self.op]


class ComputeGraph:

    def __init__(self):

        # read static parameters from config
        self.config = helper.parse_json("compute_graph.config")
        self.graph = nx.DiGraph()
        self.tree = None
        self.max_latency = -1
        self.inputs = None
        self.outputs = None
        self.min_index = None
        self.max_index = None
        self.buffer_size = None
        self.accesses = dict()  # dictionary containing all field accesses for a specific resource e.g.
        # {"A":{[0,0,0],[0,-1,0]}} for the stencil "res = A[i,j,k] + A[i,j+1,k]"

    @staticmethod
    def create_operation_node(node, number):

        if isinstance(node, ast.Name):  # variables or array access
            return Name(node, number)
        elif isinstance(node, ast.Num):  # static value
            return Num(node, number)
        elif isinstance(node, ast.BinOp):  # binary operation
            return Binop(node, number)
        elif isinstance(node, ast.Call):  # function (e.g. sin, cos,..)
            return Call(node, number)
        elif isinstance(node, ast.Assign):  # assign operator (var = expr;)
            return Output(node, number)
        elif isinstance(node, ast.Subscript):  # array access (form: arr[i,j,k])
            return Subscript(node, number)
        elif isinstance(node, ast.IfExp):  # ternary operation
            return Ternary(node, number)
        elif isinstance(node, ast.Compare):
            return Compare(node, number)  # TODO: test if correct
        else:
            raise Exception("Unknown AST type {}".format(type(node)))

    def setup_internal_buffers(self):

        # init dicts
        self.min_index = dict()  # min_index["buffer_name"] = [i_min, j_min, k_min]
        self.max_index = dict()
        self.buffer_size = dict()  # buffer_size["buffer_name"] = size

        # find min and max index
        for inp in self.inputs:
            if isinstance(inp, Subscript): # TODO: isinstance(inp, Name) or # inp.node_type == NodeType.NAME: #
                if inp.name in self.min_index:
                    if inp.index < self.min_index[inp.name]:
                        self.min_index[inp.name] = inp.index
                    if inp.index >= self.max_index[inp.name]:
                        self.max_index[inp.name] = inp.index
                else:  # first entry
                    self.min_index[inp.name] = inp.index
                    self.max_index[inp.name] = inp.index

                if inp.name not in self.accesses:
                    self.accesses[inp.name] = list()
                self.accesses[inp.name].append(inp.index)
        # set buffer_size = max_index - min_index
        for buffer_name in self.min_index:
            self.buffer_size[buffer_name] = [abs(a_i - b_i) for a_i, b_i in zip(self.max_index[buffer_name], self.min_index[buffer_name])]

        # update access to have [0,0,0] for the max_index (subtract it from all)
        for field in self.accesses:
            updated_entries = list()
            for entry in self.accesses[field]:
                updated_entries.append(helper.list_subtract_cwise(entry, self.max_index[field]))
            self.accesses[field] = updated_entries

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
                lhs = self.create_operation_node(equation, 0)  # lhs = Node(equation, 0)
                rhs = self.ast_tree_walk(equation.value, 1)
                self.graph.add_edge(rhs, lhs)

        # merge ambiguous variables in tree (implies: merge of ast.Assign trees into a single tree)
        outp_nodes = list(self.graph.nodes)
        for outp in outp_nodes:
            if isinstance(outp, Name):
                inp_nodes = list(self.graph.nodes)
                for inp in inp_nodes:
                    if isinstance(outp, Subscript) and outp is not inp and outp.name == inp.name and outp.index == inp.index:
                        # only contract if the indices match too contract nodes if index matches
                        self.contract_edge(outp, inp)
                    elif isinstance(outp, Name) and outp is not inp and outp.name == inp.name:
                        # contract nodes if index matches
                        self.contract_edge(outp, inp)
        # test if graph is now one component (for directed graph: each non-output must have at least one successor)
        for node in self.graph.nodes:
            if not isinstance(node, Output) and len(self.graph.succ[node]) == 0:
                raise RuntimeError("Kernel-internal data flow is not single component (must be connected in the sense "
                                   "of a DAG).")
        return self.graph

    def ast_tree_walk(self, node, number):

        # create node
        new_node = self.create_operation_node(node, number)  # new_node = Node(node, number)

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
        elif isinstance(node, ast.Compare):

            left = self.ast_tree_walk(node.left, ComputeGraph.child_left_number(number))
            right = self.ast_tree_walk(node.comparators[0], ComputeGraph.child_right_number(number))

            self.graph.add_edge(left, new_node)
            self.graph.add_edge(right, new_node)

        elif isinstance(node, ast.IfExp):

            test = self.ast_tree_walk(node.test, 0)
            true_path = self.ast_tree_walk(node.body, ComputeGraph.child_left_number(number))
            false_path = self.ast_tree_walk(node.orelse, ComputeGraph.child_right_number(number))

            self.graph.add_edge(true_path, new_node)
            self.graph.add_edge(false_path, new_node)
            self.graph.add_edge(test, new_node)

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
        comp = list()

        for node in self.graph.nodes:
            if isinstance(node, Num):  # node.node_type == NodeType.NUM:
                nums.append(node)
            elif isinstance(node, Name) or isinstance(node, Subscript):
                names.append(node)
            elif isinstance(node, Binop) or isinstance(node, Call):
                ops.append(node)
            elif isinstance(node, Output):
                outs.append(node)
            elif isinstance(node, Ternary) or isinstance(node, Compare):
                comp.append(node)

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

        nx.draw_networkx_nodes(self.graph, positions, nodelist=comp, node_color='#009999',
                               node_size=3000, node_shape='o')

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
        for node in self.graph.nodes:
            if isinstance(node, Output):
                node.latency = 1
                self.try_set_max_latency(node.latency)
                self.latency_tree_walk(node)

    def latency_tree_walk(self, node):

        # check node type
        if isinstance(node, Name) or isinstance(node, Num):
            for child in self.graph.pred[node]:
                child.latency = node.latency
                self.latency_tree_walk(child)
        elif isinstance(node, Binop) or isinstance(node, Call):

            # get op latency
            op_latency = self.config["op_latency"][node.name]

            for child in self.graph.pred[node]:
                child.latency = max(child.latency, node.latency + op_latency)
                self.latency_tree_walk(child)

        elif isinstance(node, Output):

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
    # computation = "res = (A[i,j,k] + out) * cos(out); out = A[i,j,k-1] + B[i-1,j,k+1]"
    computation = "res = a if (a+1 > b-c) else b"
    graph = ComputeGraph()
    graph.generate_graph(computation)
    graph.calculate_latency()
    graph.plot_graph("compute_graph_example.png")  # write graph to file
    # graph.plot_graph()
