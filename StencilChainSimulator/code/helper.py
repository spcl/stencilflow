import json


class Helper:

    def __init__(self):
        pass

    @staticmethod
    def parse_json(config_path):

        # open file in with-clause, to ensure proper file closing even in the event of an exception
        with open(config_path, "r") as file_handle:
            # try to parse it
            config = json.loads(file_handle.read())  # type: dict

        # return dict
        return config

    @staticmethod
    def max_buffer(buffer):  # sort by value of the dictionary entries
        return sorted(buffer, key=buffer.get, reverse=True)[0]

    @staticmethod
    def max_list_entry(buf):
        return sorted(buf, reverse=True)[0]

    @staticmethod
    def min_list_entry(buf):
        return sorted(buf, reverse=False)[0]

    @staticmethod
    def list_add_cwise(list1, list2):
        return list(map(lambda x, y: x + y, list1, list2))

    @staticmethod
    def list_subtract_cwise(list1, list2):
        return list(map(lambda x, y: x - y, list1, list2))

    @staticmethod
    def compare_to(index_a, index_b):  # A >= B ?
        return index_a >= index_b

    @staticmethod
    def stencil_memory_index(indices, dimensions):
        if len(indices) != len(dimensions):
            raise ValueError("Dimension mismatch")
        factor = 1
        res = 0
        for i, d in zip(reversed(indices), reversed(dimensions)):
            res += i * factor
            factor *= d
        return res

    @staticmethod
    def stencil_distance(a, b):
        return abs(Helper.stencil_memory_index(a) - Helper.stencil_memory_index(b))


if __name__ == "__main__":

    dict1 = {"small": [0, 10, 10],
             "very small": [0, 1, 0],
             "extra large": [12, 1, 2],
             "large": [10, 10, 10]}
    print("max value entry key of dict {} is: {}".format(dict1, Helper.max_buffer(dict1)))
