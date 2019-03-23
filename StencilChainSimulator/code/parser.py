import os
import json
from typing import List, Dict


class Parser:

    def __init__(self, source_path: str):
        self.source_path: str = source_path
        self.program: Dict = self.parse_input()
        self.kernels = self.get_kernel_list()

    def parse_input(self) -> Dict:

        # check file exist and readable
        if not os.access(self.source_path, os.R_OK):
            raise FileNotFoundError("Input does not exist or is not readable.")

        # open the file read-only
        file_handle = open(self.source_path, "r")

        # get content
        raw_program = file_handle.read()

        # try to parse it
        program = json.loads(raw_program)  # type: dict

        # close the file handle
        file_handle.close()

        return program

    def get_kernel_list(self) -> List:

        # split program into kernels
        result = list()
        for kernel in self.program:
            result.append(kernel)
        return result


'''
    Basic version:  parse same input as we parsed in StencilFPGA
    Extension:      parse actual input from the cosmo model
'''