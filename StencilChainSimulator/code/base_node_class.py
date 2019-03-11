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

    def __init__(self):
        pass