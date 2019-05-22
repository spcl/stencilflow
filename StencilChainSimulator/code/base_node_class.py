from abc import ABCMeta, abstractmethod
from typing import List, Dict
from bounded_queue import BoundedQueue
from enum import Enum
import ast
import dace.types


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


class BaseKernelNodeClass:

    __metaclass__ = ABCMeta

    def __init__(self, name: str, data_queue, data_type: dace.types.typeclass) -> None:
        self.name: str = name
        self.data_queue: BoundedQueue = data_queue
        self.input_paths: Dict[str, List] = dict()
        self.inputs: Dict[str, BoundedQueue] = dict()
        self.outputs: Dict[str, BoundedQueue] = dict()
        self.delay_buffer: Dict[str, List] = dict()
        self.program_counter = 0
        if not isinstance(data_type, dace.types.typeclass):
            raise TypeError("Expected dace.types.typeclass, got: " + type(data_type).__name__)
        self.data_type = data_type

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
