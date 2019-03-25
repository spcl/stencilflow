import ast
import operator
import math
from typing import Dict


class Calculator:

    def __init__(self) -> None:
        self.calc = self.Calc()

    _OP_MAP: Dict[type(ast), type(operator)] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Invert: operator.neg
    }

    _COMP_MAP: Dict[type(ast), type(operator)] = {
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Eq: operator.eq
    }

    _CALL_MAP: Dict[str, type(math)] = {
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sinh": math.sinh,
        "cosh": math.cosh
    }

    def eval_expr(self, variable_map: Dict[str, float], computation_string: str) -> float:
        """
        Given a mapping from variable names to values and a mathematical (python) expression, it evaluates the expression.
        :param variable_map: a dictionary map containing all variables of the computation_string
        :param computation_string: a python-syntax-compatible input string
        :return: the result of the expression
        """
        return self.calc.evaluate(variable_map, computation_string)

    class Calc(ast.NodeVisitor):

        def __init__(self) -> None:
            self.var_map: Dict[str, float] = None

        def visit_BinOp(self, node: ast) -> float:
            left = self.visit(node.left)
            right = self.visit(node.right)
            return Calculator._OP_MAP[type(node.op)](left, right)

        def visit_Num(self, node: ast) -> float:
            return node.n

        def visit_Expr(self, node: ast) -> float:
            return self.visit(node.value)

        def visit_IfExp(self, node: ast) -> float:  # added for ternary operations of the (python syntax: a if expr else b)
            if self.visit(node.test):  # evaluate comparison
                return self.visit(node.body)  # use left
            else:
                return self.visit(node.orelse)  # use right

        def visit_Compare(self, node: ast) -> bool:  # added for ternary operations (python syntax: a if expr else b)
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            return Calculator._COMP_MAP[type(node.ops[0])](left, right)

        def visit_Name(self, node: ast) -> float:
            return self.var_map[node.id]

        def visit_Call(self, node: ast) -> float:
            return Calculator._CALL_MAP[node.func.id](self.visit(node.args[0]))

        @classmethod
        def evaluate(cls, variable_map: Dict[str, float], expression: str) -> float:
            tree = ast.parse(expression)
            calc = cls()
            calc.var_map = variable_map
            return calc.visit(tree.body[0])


'''
    safe (in contrast to evaluate()) python expression evaluator class
        -input:
            - map: variable name -> value
            - computation string (must be python syntax, e.g. for ternary operations)
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

    computation = "cos(a + b) if (a > b) else (a + 5) * b"

    calculator = Calculator()
    result = calculator.eval_expr(variables, computation)
    print(computation + " = " + str(result))
