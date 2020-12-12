import ast
import astunparse
from collections import defaultdict

class SubscriptConverter(ast.NodeTransformer):
    def __init__(self, offset=0, dtype=None):
        self.names = defaultdict(dict)
        self.offset = offset
        self.dtype = dtype

    def convert(self, varname, index_tuple):

        # Add offset to last index
        index_tuple = index_tuple[:-1] + (index_tuple[-1] + self.offset, )

        # Remove extraneous symbols
        index_str = ''.join(c for c in str(index_tuple) if c not in '( )')

        # Replace tuple and negative symbols
        index_str = index_str.replace(',', '_')
        index_str = index_str.replace('-', 'm')

        # Add variable name
        index_str = varname + '_' + index_str

        self.names[varname][index_tuple] = index_str

        return index_str

    def visit_Subscript(self, node: ast.Subscript):
        if not isinstance(node.value, ast.Name):
            raise TypeError('Only subscripts of variables are supported')

        varname = node.value.id
        index_tuple = ast.literal_eval(node.slice.value)
        try:
            len(index_tuple)
        except TypeError:
            # Turn into a tuple
            index_tuple = (index_tuple, )

        index_str = self.convert(varname, index_tuple)

        return ast.copy_location(ast.Name(id=index_str), node)

    def visit_Constant(self, node: ast.Constant):
        if self.dtype is not None:
            return ast.copy_location(ast.Name(id=f"dace.{self.dtype.type.__name__}({node.value})"), node)
        else:
            return self.generic_visit(node)


if __name__ == '__main__':
    converter = SubscriptConverter()
    code = '''
output = inp[-1,0,1] + fc[1,2] + inp[-1,0, 1] + inp[2,1,2] * fc[1,1] + sc * one[0]
    '''
    new_ast = converter.visit(ast.parse(code))
    print(astunparse.unparse(new_ast))
    print(converter.names)
