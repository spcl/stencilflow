import ast
import operator
import math


class Calculator:

    def __init__(self, variable_map):
        self.variables = variable_map

    _OP_MAP = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Invert: operator.neg
    }

    _CALL_MAP = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sinh": math.sinh,
        "cosh": math.cosh
    }

    def eval_expr(self, computation_string):
        return Calc.evaluate(computation_string)


class Calc(ast.NodeVisitor):

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        return Calculator._OP_MAP[type(node.op)](left, right)

    def visit_Num(self, node):
        return node.n

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Name(self, node):
        return variables[node.id];

    def visit_Call(self, node):
        return Calculator._CALL_MAP[node.func.id](self.visit(node.args[0]))

    @classmethod
    def evaluate(cls, expression):
        tree = ast.parse(expression)
        calc = cls()
        return calc.visit(tree.body[0])


'''
    safe calculator class (instead of evaluate())
        -input:
            - map: variable name -> value
            - computation string
        - output: resulting value

    credits: https://stackoverflow.com/questions/33029168/how-to-calculate-an-equation-in-a-string-python

'''

if __name__ == "__main__":

    '''
        simple example for debugging purpose
    '''

    variables = dict()
    variables["a"] = 7
    variables["b"] = 2

    for var in variables:
        print("name: " + var + " value: " + str(variables[var]))

    computation = "(a + 5) * cos(a + b)"

    calculator = Calculator(variables)
    result = calculator.eval_expr(computation)
    print(computation + " = " + str(result))


