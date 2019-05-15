from abc import ABCMeta, abstractmethod
from typing import List, Dict
from bounded_queue import BoundedQueue
from enum import Enum
import ast


class BoundaryCondition(Enum):
    CONSTANT = 1
    COPY = 2

    @staticmethod
    def to_bc(text: str):
        if text == "const":
            return BoundaryCondition.CONSTANT
        elif text == "copy":
            return BoundaryCondition.COPY
        else:
            raise Exception("{} is not a valid boundary condition string".format(text))


class Precision(Enum):
    FLOAT32 = 1
    FLOAT64 = 2

    @staticmethod
    def to_prec(text: str):
        if text == "float32":
            return Precision.FLOAT32
        elif text == "float64":
            return Precision.FLOAT64
        else:
            raise Exception("{} is not a valid precision string".format(text))


class BaseKernelNodeClass:

    __metaclass__ = ABCMeta

    def __init__(self, name: str, data_queue, precision: Precision) -> None:
        self.name: str = name
        self.data_queue: BoundedQueue = data_queue
        self.input_paths: Dict[str, List] = dict()
        self.outputs: Dict[str, BoundedQueue] = dict()
        self.delay_buffer: Dict[str, List] = dict()
        self.precision = precision

    def generate_label(self) -> str:  # wrapper for customizations
        return self.name


class BaseOperationNodeClass:

    __metaclass__ = ABCMeta

    def __init__(self, ast_node: ast, number: int) -> None:
        self.number: int = number
        self.name: str = self.generate_name(ast_node)
        self.latency: int = -1

    @abstractmethod
    def generate_name(self, ast_node: ast) -> str:  # every subclass must implement this
        pass

    def generate_label(self) -> str:  # subclass can, if necessary, override the default implementation
        return str(self.name)
