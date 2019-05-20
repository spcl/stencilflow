import unittest
from bounded_queue import BoundedQueue

class BoundedQueueTest(unittest.TestCase):

    def test_init(self):
        # init
        queue = BoundedQueue(name="test",
                             maxsize=5)
        # init_queue
        collection = [1.0, 2.0, 3.0, 4.0, 5.0]
        queue.init_queue(collection)
        self.assertEqual(queue.size(), len(collection))
        self.assertEqual(queue.peek(1), collection[1])
        self.assertRaises(RuntimeError, queue.init_queue, 6*[1.0])

    def test_enq_deq(self):
        # init
        queue = BoundedQueue(name="test",
                             maxsize=1,
                             collection=[1.0])
        # dequeue
        self.assertEqual(queue.size(), 1)
        self.assertEqual(queue.dequeue(), 1.0)
        self.assertEqual(queue.size(), 0)
        self.assertTrue(queue.is_empty())
        self.assertRaises(RuntimeError, queue.dequeue)
        # enqueue
        queue.enqueue(1.0)
        self.assertTrue(queue.is_full())
        self.assertRaises(RuntimeError, queue.enqueue, 2.0)

    def test_try_enq_deq(self):
        # init
        queue = BoundedQueue(name="test",
                             maxsize=1,
                             collection=[1.0])
        # try dequeue
        self.assertEqual(queue.size(), 1)
        self.assertEqual(queue.try_dequeue(), 1.0)
        self.assertEqual(queue.size(), 0)
        self.assertTrue(queue.is_empty())
        self.assertFalse(queue.try_dequeue())
        # try enqueue
        self.assertTrue(queue.try_enqueue(1.0))
        self.assertTrue(queue.is_full())
        self.assertFalse(queue.try_enqueue(), 2.0)

    def test_peek(self):
        # init
        queue = BoundedQueue(name="test",
                             maxsize=2,
                             collection=[1.0, 2.0])
        # peek / try peek
        self.assertEqual(queue.peek(0), 1.0)
        self.assertEqual(queue.peek(1), 2.0)
        self.assertEqual(queue.try_peek_last(), 2.0)
        queue.dequeue()
        queue.dequeue()
        self.assertFalse(queue.try_peek_last())


from calculator import Calculator
from numpy import cos

class CalculatorTest(unittest.TestCase):

    def test_calc(self):
        # init vars
        variables = dict()
        variables["a"] = 7.0
        variables["b"] = 2.0
        # init calc
        computation = "cos(a + b) if (a > b) else (a + 5) * b"
        calculator = Calculator()
        # do manual calculation and compare result
        result = cos(variables["a"] + variables["b"]) if (variables["a"] > variables["b"]) else (variables["a"] + 5) * variables["b"]
        self.assertEqual(calculator.eval_expr(variables, computation), result)


#import run_program
class RunProgramTest(unittest.TestCase):

    def test(self):
        pass


import helper
import os
import numpy as np
class HelperTest(unittest.TestCase):

    def test(self):
        self.assertEqual(helper.max_dict_entry_key({"a": [1, 0, 0], "b": [0, 1, 0], "c": [0, 0, 1]}), "a")
        self.assertEqual(helper.list_add_cwise([1, 2, 3], [3, 2, 1]), [4, 4, 4])
        self.assertEqual(helper.list_subtract_cwise([1, 2, 3], [1, 2, 3]), [0, 0, 0])
        self.assertEqual(helper.dim_to_abs_val([3, 2, 1], [10, 10, 10]), 321)
        self.assertEqual(helper.convert_3d_to_1d([10, 10, 10], [3, 2, 1]), 321)
        self.assertListEqual(list(helper.load_array("testing/helper_test.csv")), [7.0, 7.0])
        self.assertListEqual(list(helper.load_array("testing/helper_test.dat")), [7.0, 7.0])

        out_data = np.array([1.0, 2.0, 3.0])
        file = "test.dat"
        helper.save_array(out_data, file)
        in_data = helper.load_array(file)
        self.assertTrue(helper.arrays_are_equal(out_data, in_data))
        os.remove(file)

        not_unique = [1.0, 2.0, 1.0]
        self.assertListEqual(sorted(helper.unique(not_unique)), [1.0, 2.0])


from compute_graph import ComputeGraph
import json
import os
class ComputeGraphTest(unittest.TestCase):

    def test(self):

        computation = "out = cos(3.14);res = A[i,j,k] if (A[i,j,k]+1 > A[i,j,k]-B[i,j,k]) else out"
        graph = ComputeGraph()
        graph.generate_graph(computation)
        graph.calculate_latency()
        with open('compute_graph.config') as json_file:
            op_latency = json.load(json_file)
        self.assertEqual(op_latency["op_latency"]["cos"] + 1, graph.max_latency)
        filename = "compute_graph_unittest.png"
        graph.plot_graph(filename)  # write graph to file
        os.remove(filename)


from kernel import Kernel
from base_node_class import DataType
class KernelTest(unittest.TestCase):

    def test(self):
        dimensions = [100, 100, 100]
        kernel = Kernel(name="dummy",
                        kernel_string="SUBST = a[i,j,k] + a[i,j,k-1] + a[i,j-1,k] + a[i-1,j,k]; res = SUBST + a[i,j,k]",
                        dimensions=dimensions,
                        data_type=DataType.FLOAT64,
                        boundary_conditions={"a": {"type": "constant", "value": 1.0}})
        self.assertEqual(kernel.generate_relative_access_kernel_string(),
                         "SUBST = (((a[0] + a[-1]) + a[-100]) + a[-10000]); res = (SUBST + a[0])")


from kernel_chain_graph import KernelChainGraph
class KernelChainGraphTest(unittest.TestCase):

    def test(self):
        pass


from optimizer import Optimizer
class OptimizerTest(unittest.TestCase):

    def test(self):
        pass


from simulator import Simulator
class SimulatorTest(unittest.TestCase):

    def test(self):
        pass


if __name__ == '__main__':
    unittest.main()
