import json
import os.path
import functools
import warnings
import operator
from functools import reduce


class Helper:

    def __init__(self):
        pass

    def deprecated(func):
        """This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used."""

        @functools.wraps(func)
        def new_func(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn("Call to deprecated function {}.".format(func.__name__),
                          category=DeprecationWarning,
                          stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return func(*args, **kwargs)

        return new_func

    @staticmethod
    def parse_json(config_path):
        """
        Read input file from disk and parse it.
        :param config_path: path to the file
        :return: parsed file
        """
        # check file exists
        if not os.path.isfile(config_path):
            raise Exception("file {} does not exists.".format(config_path))

        # open file in with-clause, to ensure proper file closing even in the event of an exception
        with open(config_path, "r") as file_handle:
            # try to parse it
            config = json.loads(file_handle.read())  # type: dict

        # return dict
        return config

    @staticmethod
    def max_dict_entry_key(dict1):
        """
        Get key of largest value entry out of the input dictionary.
        :param dict1: a dictionary with keys as names and values as buffer sizes
        :return: key of buffer entry with maximum size
        """
        # check type
        if not isinstance(dict1, dict):
            raise Exception("dict1 should be of type {}, but is of type {}".format(type(dict), type(dict1)))
        # extract max value entry
        return max(dict1, key=dict1.get)

    @staticmethod
    @deprecated  # since method is now only one line -> can be substituted
    def max_list_entry(list1):
        """
        Get largest list entry (lexicographic).
        :param list1: a list of lists [[..],[..],[..], ..]
        :return: max list entry of input list
        """
        # check type
        if not isinstance(list1, list):
            raise Exception("list1 should be of type {}, but is of type {}".format(type(list), type(list1)))
        # extract max entry
        return max(list1)

    @staticmethod
    @deprecated  # since method is now only one line -> can be substituted
    def min_list_entry(list1):
        """
        Get smallest list entry (lexicographic).
        :param list1: a list of lists [[..],[..],[..], ..]
        :return: min list entry of input list
        """
        # check type
        if not isinstance(list1, list):
            raise Exception("list1 should be of type {}, but is of type {}".format(type(list), type(list1)))
        # extract min entry
        return min(list1)

    @staticmethod
    def list_add_cwise(list1, list2):
        """
        Merge two lists by component-wise addition.
        :param list1: input list: summand
        :param list2: input list: summand
        :return: merged list
        """
        # check type
        if not isinstance(list1, list):
            raise Exception("list1 should be of type {}, but is of type {}".format(type(list), type(list1)))
        if not isinstance(list2, list):
            raise Exception("list2 should be of type {}, but is of type {}".format(type(list), type(list2)))
        # do map lambda operation over both lists
        return list(map(lambda x, y: x + y, list1, list2))

    @staticmethod
    def list_subtract_cwise(list1, list2):
        """
        Merge two lists by component-wise subtraction.
        :param list1: input list: minuend
        :param list2: input list: subtrahend
        :return: merged list
        """
        # check type
        if not isinstance(list1, list):
            raise Exception("list1 should be of type {}, but is of type {}".format(type(list), type(list1)))
        if not isinstance(list2, list):
            raise Exception("list2 should be of type {}, but is of type {}".format(type(list), type(list2)))
        # do map lambda operation over both lists
        return list(map(lambda x, y: x - y, list1, list2))

    @staticmethod
    def dim_to_abs_val(input, dimensions):
        """
        Computes scalar number out of independent dimension unit.
        :param input: vector to evaluate
        :param dimensions: vector of global array dimensions
        :return: scalar value
        """
        # dim [X, Y, Z], size [a, b, c] -> 1*c + X*(b + Y*a) = [a, b, c] * transpose([Z*Y, Z, 1])
        vec = [dimensions[2]*dimensions[1], dimensions[2], 1]
        return reduce(operator.add, map(operator.mul, input, vec))  # inner product


if __name__ == "__main__":

    example_list = [[1, 2, 2], [1, 2, 3], [3, 2, 1], [2, 3, 1]]
    print("properties of list {}:\nmin: {}\nmax: {}\n".format(example_list, min(example_list),
                                                              max(example_list)))

    example_dict = {"small": [0, 10, 10],
                    "very small": [0, 1, 0],
                    "extra large": [12, 1, 2],
                    "large": [10, 10, 10]}
    print("max value entry key of dict {} is:\n\'{}\'".format(example_dict, Helper.max_dict_entry_key(example_dict)))
