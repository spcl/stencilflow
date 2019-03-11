from abc import ABCMeta, abstractmethod


class BaseKernelNodeClass:

    __metaclass__ = ABCMeta
    # Note: can enforce implementation in derived class by using @abstractmethod

    def __init__(self, name):
        self.name = name
        self.input_paths = dict()
        self.outputs = dict()
        self.delay_buffer = dict()

    def generate_label(self):  # wrapper for customizations
        return self.name


class BaseOperationNodeClass:

    __metaclass__ = ABCMeta
    # Note: can enforce implementation in derived class by using @abstractmethod

    def __init__(self, ast_node, number):
        self.number = number
        self.name = self.generate_name(ast_node)
        self.latency = -1

    @abstractmethod
    def generate_name(self, ast_node):  # every subclass must implement this
        pass

    def generate_label(self):  # subclass can, if necessary, override the default implementation
        return str(self.name)
