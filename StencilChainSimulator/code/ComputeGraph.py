import ast
import networkx as nx

class ComputeGraph:

    def __init__(self, path):
        self.path = path


'''
    Port from the non-class implementation in StencilFPGA to a class version using kernel classes instead of the
    dictionary.
'''