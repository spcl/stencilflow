import functools
from typing import List, Dict
import dace.types
import helper
from base_node_class import BaseKernelNodeClass, BaseOperationNodeClass
from bounded_queue import BoundedQueue
from calculator import Calculator
from compute_graph import ComputeGraph
from compute_graph import Name, Num, Binop, Call, Output, Subscript, Ternary, Compare
from input import Input
import numpy as np
import operator

class Kernel(BaseKernelNodeClass):
    """
        interface for FPGA-like execution (get called from the scheduler)

            - read:
                    - saturation phase: read unconditionally
                    - execution phase: read all inputs iff they are available
            - execute:
                    - saturation phase: do nothing
                    - execution phase: if input read, execute stencil using the input
            - write:
                    - saturation phase: do nothing
                    - execution phase: write result from execution to output buffers
                        --> if output buffer overflows: assumptions about size were wrong!

            - return:
                    - True  iff successful
                    - False otherwise

            - boundary condition:
                    - We know that the stencil boundary conditions in the COSMO model are functions of the local
                      neighbourhood (e.g. gradient, average, replicate value form n to n+1 (border replication),...)
                    - Idea: We implement the border replication strategy for all scenarios statically (this is enough
                      accuracy, since the latency would be most likely the same (the strategies mentioned above can be
                      implemented in parallel in hardware), and the amount of buffer space does not change as well.
                      Therefore this is a valid assumption and reduction of the problem complexity.

            - note:
                    - re-emptying of queue after reaching bound
                    - scenario:
                      suppose:  for i=1..N
                                    for j=1..M
                                        for k=1..P
                                            out[i,j,k] = in[i-1,j,k] + in[i,j,k] + in[i+1,j,k]
                                        end
                                    end
                                end
                      the internal buffer is of size: 2*P*N
                       j
                      /
                      --> i     in the case above we have to buffer 2 complete layers in i-j-direction till we can start
                      |         doing meaning full pipeline operations
                      k

                      if we reach the bound i == N, TODO: special handling?
    """

    def __init__(self, name: str, kernel_string: str, dimensions: List[int], data_type: dace.types.typeclass,
                 boundary_conditions: Dict[str, str], plot_graph: bool = False, verbose: bool = False) -> None:
        super().__init__(name, None, data_type)

        # store arguments
        self.kernel_string: str = kernel_string  # raw kernel string input
        self.dimensions: List[int] = dimensions  # type: [int, int, int] # input array dimensions [dimX, dimY, dimZ]
        self.boundary_conditions = boundary_conditions
        self.verbose = verbose

        # read static parameters from config
        self.config: Dict = helper.parse_json("kernel.config")
        self.calculator: Calculator = Calculator()

        self.all_available = False
        self.not_available = set()

        # analyse input
        self.graph: ComputeGraph = ComputeGraph()
        self.graph.generate_graph(kernel_string)
        self.graph.calculate_latency()
        self.graph.determine_inputs_outputs()
        self.graph.setup_internal_buffers()
        if plot_graph:
            self.graph.plot_graph(name + ".png")

        # init sim specific params
        self.var_map: Dict[str, float] = None  # var_map[var_name] = var_value
        self.read_success: bool = False
        self.exec_success: bool = False
        self.result: float = None
        self.outputs: Dict[str, BoundedQueue] = dict()

        # output delay queue: for simulation of calculation latency, fill it up with bubbles
        self.out_delay_queue: BoundedQueue = BoundedQueue(name="delay_output",
                                                          maxsize=self.graph.max_latency+1,
                                                          collection=[None] * (self.graph.max_latency))

        # setup internal buffer queues
        self.internal_buffer: Dict[str, BoundedQueue] = dict()
        self.setup_internal_buffers()

        self.dist_to_center: Dict = dict()
        self.set_up_dist_to_center()
        self.center_reached = False

    def set_up_dist_to_center(self):
        for item in self.graph.accesses:
            furthest = max(self.graph.accesses[item])
            self.dist_to_center[item] = helper.dim_to_abs_val(furthest, self.dimensions)


    def iter_comp_tree(self, node: BaseOperationNodeClass, index_relative_to_center=True, replace_negative_index=False) -> str:

        pred = list(self.graph.graph.pred[node])

        if isinstance(node, Binop):
            lhs = pred[0]
            rhs = pred[1]
            return "(" + self.iter_comp_tree(lhs,
                                             index_relative_to_center, replace_negative_index) + " " + node.generate_op_sym() + " " + self.iter_comp_tree(
                rhs, index_relative_to_center, replace_negative_index) + ")"
        elif isinstance(node, Call):
            return node.name + "(" + self.iter_comp_tree(pred[0], index_relative_to_center, replace_negative_index) + ")"
        elif isinstance(node, Name) or isinstance(node, Num):
            return str(node.name)
        elif isinstance(node, Subscript):
            if index_relative_to_center:
                dim_index = node.index
            else:
                dim_index = helper.list_subtract_cwise(node.index, self.graph.max_index[node.name])
            word_index = self.convert_3d_to_1d(dim_index)

            if replace_negative_index and word_index < 0:
                return node.name + "[" + "n" + str(abs(word_index)) + "]"
            else:
                return node.name + "[" + str(word_index) + "]"
        elif isinstance(node, Ternary):

            compare = [x for x in pred if type(x) == Compare][0]
            lhs = [x for x in pred if type(x) != Compare][0]
            rhs = [x for x in pred if type(x) != Compare][1]

            return "(({}) if ({}) else ({}))".format(self.iter_comp_tree(lhs, index_relative_to_center, replace_negative_index),
                                             self.iter_comp_tree(compare, index_relative_to_center, replace_negative_index),
                                             self.iter_comp_tree(rhs, index_relative_to_center, replace_negative_index))
        elif isinstance(node, Compare):
            return "{} {} {}".format(self.iter_comp_tree(pred[0], index_relative_to_center, replace_negative_index), str(node.name),
                                     self.iter_comp_tree(pred[1], index_relative_to_center, replace_negative_index))
        else:
            raise NotImplementedError("iter_comp_tree is not implemented for node type {}".format(type(node)))

    def generate_relative_access_kernel_string(self, relative_to_center=True, replace_negative_index=False) -> str:
        # format: 'res = vdc[index1] + vout[index2]'

        res = []

        # Treat named nodes
        for n in self.graph.graph.nodes:
            if isinstance(n, Name):
                res.append(n.name + " = " + self.iter_comp_tree(
                    list(self.graph.graph.pred[n])[0], relative_to_center, replace_negative_index))

        # Treat output node
        output_node = [
            n for n in self.graph.graph.nodes if isinstance(n, Output)
        ]
        if len(output_node) != 1:
            raise Exception("Expected a single output node")
        output_node = output_node[0]

        res.append("res = " + self.iter_comp_tree(
            list(self.graph.graph.pred[output_node])[0], index_relative_to_center=relative_to_center, replace_negative_index=replace_negative_index))

        return "; ".join(res)

    def reset_old_compute_state(self) -> None:
        self.var_map = dict()
        self.read_success = False
        self.exec_success = False
        self.result = None

    def convert_3d_to_1d(self, index: List[int]) -> int:
        # convert [i, j, k] to flat 1D array index using the given dimensions [dimX, dimY, dimZ]
        # index = i*dimY*dimZ + j*dimZ + k = (i*dimY + j)*dimZ + k
        if not index:
            return 0  # empty list
        return helper.dim_to_abs_val(index, self.dimensions)

    def remove_duplicate_accesses(self, inp: List) -> List:
        tuple_set = set(tuple(row) for row in inp)
        return [[x,y,z] for (x,y,z) in tuple_set]

    def setup_internal_buffers(self) -> None:

        # remove duplicate accesses
        for item in self.graph.accesses:
            self.graph.accesses[item] = self.remove_duplicate_accesses(self.graph.accesses[item])

        # slice the internal buffer into junks of accesses
        for buf_name in self.graph.buffer_size:
            self.internal_buffer[buf_name] = list()
            list.sort(self.graph.accesses[buf_name], reverse=True)

            if len(self.graph.accesses[buf_name]) == 0:
                pass
            elif len(self.graph.accesses[buf_name]) == 1:
                # this line would add an additional internal buffer for fields that only have a single access
                # TODO: check if this is always 1, also for non-[0,0,0] indices
                self.internal_buffer[buf_name].append(BoundedQueue(name=buf_name, maxsize=1, collection=[None]))
            else:
                itr = self.graph.accesses[buf_name].__iter__()
                pre = itr.__next__()
                for item in itr:
                    curr = item

                    diff = abs(helper.dim_to_abs_val(helper.list_subtract_cwise(pre, curr), self.dimensions))
                    if diff == 0: # two accesses on same field
                        pass
                    else:
                        self.internal_buffer[buf_name].append(BoundedQueue(name=buf_name, maxsize=diff, collection=[None]*diff))

                    pre = curr

    def buffer_position(self, access: BaseKernelNodeClass) -> int:
        return self.convert_3d_to_1d(self.graph.min_index[access.name]) - self.convert_3d_to_1d(access.index)

    def index_to_ijk(self, index: List[int]):
        if len(index):
            '''
            # v1:
            return "[i{},j{},k{}]".format(
                "" if index[0] == 0 else "+{}".format(index[0]),
                "" if index[1] == 0 else "+{}".format(index[1]),
                "" if index[2] == 0 else "+{}".format(index[2])
            )
            # v2:
            return "_{}_{}_{}".format(index[0], index[1], index[2])
            '''
            ind = helper.convert_3d_to_1d(self.dimensions, index)
            return "_{}".format(ind) if ind >= 0 else "_n{}".format(abs(ind))
        else:
            raise NotImplementedError("Method index_to_ijk has not been implemented for |indices|!=3, here: |indices|={}".format(len(index)))

    def buffer_number(self, node: Subscript):
        selected = [x.index for x in self.graph.inputs if x.name == node.name]
        selected_unique = self.remove_duplicate_accesses(selected)
        ordered = sorted(selected_unique, reverse=True)
        result = ordered.index(node.index)
        return result - 1
        # selected = [x for x in self.graph.inputs if x.name == node.name]
        # ordered = sorted(selected, key=lambda x:x.index, reverse=True)
        # return ordered.index(node) - 1

    def get_global_kernel_index(self) -> List[int]:
        index = self.dimensions
        number = self.program_counter
        n = len(index)
        all_dim = functools.reduce(operator.mul, index, 1) // index[0]

        output = list()

        for i in range(1, n + 1):
            output.append(number // all_dim)
            number -= output[-1] * all_dim
            if i < n:
                all_dim = all_dim // index[i]

        # print("{}: global index is: {}\n".format(self.name, output))
        return output

    def is_out_of_bound(self, index):

        for i in range(len(index)):
            if index[i] < 0 or index[i] >= self.dimensions[i]:
                return True
        return False

    def get_data(self, inp: Input, global_index: List[int], relative_index: List[int]):
        """
        returns data of current stencil access (could be real data or boundary condition)
        :param global_index: center location of current stencil
        :param relative_index: offset from center of stencil
        :return: data
        """
        """
            Boundary Condition
        """
        access_index = helper.list_add_cwise(global_index, relative_index)
        if self.is_out_of_bound(access_index):
            if self.boundary_conditions[inp.name]["type"] == "constant":
                return self.boundary_conditions[inp.name]["value"]
            elif self.boundary_conditions[inp.name]["type"] == "copy":
                raise NotImplementedError("Copy boundary conditions have not been implemented yet.")
            else:
                raise NotImplementedError("We currently do not support boundary conditions of type {}".format(
                    self.boundary_conditions[inp.name]["type"]))
        """
            Data Access
        """
        pos = self.buffer_number(inp)
        if pos == -1:  # delay buffer
            return self.inputs[inp.name]["delay_buffer"].try_peek_last()
        elif pos >= 0:
            return self.inputs[inp.name]["internal_buffer"][pos].try_peek_last()

    def test_availability(self):

        all_available = True
        self.not_available = set()

        for inp in self.graph.inputs:

            if isinstance(inp, Num):
                pass
            elif len(self.inputs[inp.name]['internal_buffer']) == 0:
                pass
            elif isinstance(inp, Subscript):
                gki = self.get_global_kernel_index()
                if self.is_out_of_bound(helper.list_add_cwise(inp.index, gki)):
                    pass
                else:
                    index = self.buffer_number(inp)
                    if index == -1:
                        if self.inputs[inp.name]['delay_buffer'].try_peek_last() is None or self.inputs[inp.name]['delay_buffer'].try_peek_last() is False:
                            all_available = False
                            self.not_available.add(inp.name)
                    elif 0 <= index < len(self.inputs[inp.name]['internal_buffer']):
                        if self.inputs[inp.name]['internal_buffer'][index].try_peek_last() is False\
                        or self.inputs[inp.name]['internal_buffer'][index].try_peek_last() is None:
                            all_available = False
                            self.not_available.add(inp.name)
                    else:
                        raise Exception("index out of bound: {}".format(index))


        return all_available

    def move_forward(self, items):
        # move all forward

        for name in items:

            if len(items[name]['internal_buffer']) == 0:
                pass
            elif len(self.inputs[name]['internal_buffer']) == 1:
                items[name]['internal_buffer'][0].dequeue()
                items[name]['internal_buffer'][0].enqueue(items[name]['delay_buffer'].dequeue())
            else:
                index = len(items[name]['internal_buffer']) - 1
                pre = items[name]['internal_buffer'][index - 1]
                next = items[name]['internal_buffer'][index]
                next.dequeue()

                while index > 0:
                    next.enqueue(pre.dequeue())
                    next = pre
                    index -= 1
                    pre = items[name]['internal_buffer'][index - 1]

                # self.inputs[name]['internal_buffer'][0].dequeue()
                items[name]['internal_buffer'][0].enqueue(items[name]['delay_buffer'].dequeue())

    def decrement_center_reached(self):

        for item in self.dist_to_center:
            if self.inputs[item]['delay_buffer'].try_peek_last() is not None:
                self.dist_to_center[item] -= 1

    def try_read(self) -> bool:

        # check if all inputs are available
        self.all_available = self.test_availability()

        """
        if self.all_available == False:
            all_available = True
            for inp in self.graph.inputs:
                if isinstance(inp, Num):
                    pass
                elif len(self.inputs[inp.name]['internal_buffer']) == 0:
                    pass
                elif self.inputs[inp.name]['internal_buffer'][len(self.inputs[inp.name]['internal_buffer'])-1].try_peek_last() is False\
                        or self.inputs[inp.name]['internal_buffer'][len(self.inputs[inp.name]['internal_buffer'])-1].try_peek_last() is None \
                        or self.inputs[inp.name]['delay_buffer'].try_peek_last() is None:
                    # check if array access location is filled with a bubble
                    all_available = False
                    break
            if all_available == True:
                self.all_available = True
        else:
            if helper.list_add_cwise(self.get_global_kernel_index(), [1,1,1]) > self.dimensions:
                self.all_available = False
        """
        # get all values and put them into the variable map
        if self.all_available:
            for inp in self.graph.inputs:
                # read inputs into var_map
                if isinstance(inp, Num):
                    self.var_map[inp.name] = float(inp.name)  # TODO: boundary check?
                elif isinstance(inp, Name):
                    # get value from internal_buffer
                    try:
                        # check for duplicate
                        if not self.var_map.__contains__(inp.name):
                            self.var_map[inp.name] = self.internal_buffer[inp.name].peek(self.buffer_position(inp))  # TODO: boundary check!
                    except Exception as ex:
                        self.diagnostics(ex)
                elif isinstance(inp, Subscript):
                    # get value from internal buffer
                    try:
                        name = inp.name + self.index_to_ijk(inp.index)
                        if not self.var_map.__contains__(name):
                            self.var_map[name] = self.get_data( inp=inp,
                                                                global_index=self.get_global_kernel_index(),
                                                                relative_index=inp.index)
                    except Exception as ex:
                        self.diagnostics(ex)

        self.read_success = self.all_available

        # test center reached
        self.decrement_center_reached()
        self.center_reached = True
        for item in self.dist_to_center:
            if self.dist_to_center[item] >= 0:
                self.center_reached = False

        if self.center_reached:
            if self.all_available:
                self.move_forward(self.inputs)
            else:
                not_avail_dict = dict()
                for item in self.not_available:
                    not_avail_dict[item] = self.inputs[item]
                self.move_forward(not_avail_dict)
        else:
            not_reached_dict = dict()
            for item in self.dist_to_center:
                if self.dist_to_center[item] >= 0:
                    not_reached_dict[item] = self.inputs[item]
            self.move_forward(not_reached_dict)

        return self.all_available

    def try_execute(self):

        # check if read has been successful
        if self.center_reached and self.read_success and 0 <= self.program_counter < functools.reduce(operator.mul, self.dimensions, 1) :
            # execute calculation
            try:
                computation = self.generate_relative_access_kernel_string(relative_to_center=True, replace_negative_index=True).replace("[", "_")\
                    .replace("]", "").replace(" ", "")
                self.result = self.calculator.eval_expr(self.var_map, computation)
                # write result to latency-simulating buffer
                self.out_delay_queue.enqueue(self.result)
                self.program_counter += 1
            except Exception as ex:
                self.diagnostics(ex)
        else:
            # write bubble to latency-simulating buffer
            self.out_delay_queue.enqueue(None)

    def try_write(self):

        data = self.out_delay_queue.dequeue()
        """
        # check if data (not a bubble) is available
        if data is not None:
            # write result to all output queues
            for outp in self.outputs:
                try:
                    self.outputs[outp]["delay_buffer"].enqueue(data)  # use delay buffer to be consistent with others, db is used to write to output data queue here
                except Exception as ex:
                    self.diagnostics(ex)
        """
        # write result to all output queues
        for outp in self.outputs:
            try:
                self.outputs[outp]["delay_buffer"].try_enqueue(data)  # use delay buffer to be consistent with others, db is used to write to output data queue here
            except Exception as ex:
                self.diagnostics(ex)

    '''
        interface for error overview reporting (gets called in case of an exception)

        - goal:
                - get an overview over the whole stencil chain state in case of an error
                    - maximal and current size of all buffers
                    - type of phase (saturation/execution)
                    - efficiency (#execution cycles / #total cycles)
    '''

    def diagnostics(self, ex) -> None:
        raise ex

    '''
        simple test kernel for debugging
    '''


if __name__ == "__main__":
    dim = [100, 100, 100]

    kernel1 = Kernel("ppgk", "res = wgtfac[i,j+1,k] * ppuv[i,j,k] + (1.0 - wgtfac[i,j,k]) * ppuv[i,j,k-1];", dim)
    print("dimensions are: {}".format(dim))
    print("Critical path latency: " + str(kernel1.graph.max_latency))
    print()

    kernel2 = Kernel("dummy", "res = (sin(a[i,j,k])-b[i,j,k]) * (a[i,j,k-1]+b[i,j-1,k-1])", [100, 100, 100])
    print("dimensions are: {}".format(dim))
    print("Kernel string conversion:")
    print(kernel2.kernel_string)
    print(kernel2.generate_relative_access_kernel_string())
    print()

    kernel3 = Kernel("dummy", "res = a[i,j,k] + a[i,j,k-1] + a[i,j-1,k] + a[i-1,j,k] + a[i,j,k]", dim)
    print("Kernel string conversion:")
    print("dimensions are: {}".format(dim))
    print(kernel3.kernel_string)
    print(kernel3.generate_relative_access_kernel_string())
    print()

    kernel4 = Kernel("dummy", "res = SUBST + a[i,j,k];SUBST = a[i,j,k] + a[i,j,k-1] + a[i,j-1,k] + a[i-1,j,k]", dim)
    print("Kernel string conversion:")
    print("dimensions are: {}".format(dim))
    print(kernel4.kernel_string)
    print(kernel4.generate_relative_access_kernel_string())
    print()

    kernel5 = Kernel("dummy", "res = a[i+1,j+1,k+1] + a[i+1,j,k] + a[i-1,j-1,k-1] + a[i+1,j+1,k] + a[i,j,k]", dim)
    print("Kernel string conversion:")
    print("dimensions are: {}".format(dim))
    print(kernel5.kernel_string)
    print(kernel5.generate_relative_access_kernel_string(relative_to_center=False))
    print()

    kernel6 = Kernel("dummy", "res = a if (a > b) else b", dim)
    print()
