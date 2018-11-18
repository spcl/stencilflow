import json


class Helper:

    def __init__(self):
        pass

    @staticmethod
    def parse_json(config_path):

        # open file in with-clause, to ensure proper file closing even in the event of an exception
        with open("config_path", "r") as file_handle:
            # try to parse it
            config = json.loads(file_handle.read())  # type: dict

        # return dict
        return config
