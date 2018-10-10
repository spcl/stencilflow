import ast
import networkx as nx
import matplotlib.pyplot as plt


class ComputeGraph:

    def __init__(self, path):
        self.path = path
        self.graph = nx.DiGraph()


    def generate_graph(self, computation_string):
        return


class GraphGenerator(ast.NodeVisitor):

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        # return Calculator._OP_MAP[type(node.op)](left, right)

    def visit_Num(self, node):
        return node.n

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        return
        # TODO return variables[node.id];

    def visit_Call(self, node):
        return
        # TODO return Calculator._CALL_MAP[node.func.id](self.visit(node.args[0]))

    @classmethod
    def evaluate(cls, expression):
        tree = ast.parse(expression)
        calc = cls()
        return calc.visit(tree.body[0])


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

    G = nx.DiGraph()
    G.add_node(0)
    G.add_node(1)
    G.add_edge(0,1)
    nx.draw(G)
    plt.savefig("path.png")