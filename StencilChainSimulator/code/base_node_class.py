from abc import ABCMeta, abstractmethod
from typing import List, Dict
from bounded_queue import BoundedQueue
import ast


class BaseKernelNodeClass:

    __metaclass__ = ABCMeta

    def __init__(self, name: str, data_queue) -> None:
        self.name: str = name
        self.data_queue: BoundedQueue = data_queue
        self.input_paths: Dict[str, List] = dict()
        self.outputs: Dict[str, BoundedQueue] = dict()
        self.delay_buffer: Dict[str, List] = dict()

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
