import collections

import dace
import dace.library

from .fpga import ExpandStencilFPGA
from .cpu import ExpandStencilCPU


@dace.library.node
class Stencil(dace.library.LibraryNode):
    """Represents applying a stencil to a full input domain."""

    implementations = {"FPGA": ExpandStencilFPGA, "CPU": ExpandStencilCPU}
    default_implementation = None

    # Definition of stencil computation
    shape = dace.properties.ListProperty(
        dace.symbolic.pystr_to_symbolic, desc="Shape of stencil dimensions")
    # Example:
    # accesses = {
    #   "a": ((True, True, True), [(0, 0, -1), (0, -1, 0), (1, 0, 0)]),
    #   "b": ((True, False, True), [(0, 1), (1, 0), (-1, 0), (0, -1)])
    # }
    accesses = dace.properties.OrderedDictProperty(
        desc=("Dictionary mapping input fields to lists of offsets "
              "and index mapping"),
        default=collections.OrderedDict())
    output_fields = dace.properties.OrderedDictProperty(
        desc="Dictionary mapping output fields to their offsets",
        default=collections.OrderedDict())
    boundary_conditions = dace.properties.OrderedDictProperty(
        desc="Boundary condition specifications for each accessed field",
        default=collections.OrderedDict())
    code = dace.properties.CodeProperty(
        desc="Stencil code using all inputs to produce all outputs",
        default="")


    def __init__(self,
                 label,
                 shape=[],
                 accesses={},
                 output_fields={},
                 boundary_conditions={},
                 code=""):
        in_connectors = accesses.keys()
        out_connectors = output_fields.keys()
        super().__init__(label, inputs=in_connectors, outputs=out_connectors)
        self.shape = shape
        self.accesses = accesses
        self.output_fields = output_fields
        self.boundary_conditions = boundary_conditions
        self.code = type(self).code.from_string(code,
                                                dace.dtypes.Language.Python)
