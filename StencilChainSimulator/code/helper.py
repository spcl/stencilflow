import os
import json


class Helper:

    def __init__(self):
        return

    @staticmethod
    def parse_json(config_path):

        # check file exist and readable
        if not os.access(config_path, os.R_OK):
            raise Exception("Config does not exist or is not readable.")

        # open the file read-only
        file_handle = open(config_path, "r")

        # try to parse it
        config = json.loads(file_handle.read())  # type: dict

        # close the file handle
        file_handle.close()

        # return dict
        return config
