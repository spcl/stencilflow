#!/usr/bin/env python3
import multiprocessing as mp
import os
import sys
import json
import unittest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

TEST_FOLDER = os.path.join(os.path.dirname(__file__), "stencils")

from stencilflow.bounded_queue import BoundedQueue

import dace.dtypes


class BoundedQueueTest(unittest.TestCase):
    def test_import(self):
        # init
        queue = BoundedQueue(name="test", maxsize=5)
        # init_queue
        collection = [1.0, 2.0, 3.0, 4.0, 5.0]
        queue.import_data(collection)
        # check size
        self.assertEqual(queue.size(), len(collection))
        # check if data added in the right order
        self.assertEqual(queue.try_peek_last(), collection[len(collection) - 1])
        # check exception for overfilling queue
        self.assertRaises(RuntimeError, queue.import_data, 6 * [1.0])

    def test_enq_deq(self):
        # init
        queue = BoundedQueue(name="test", maxsize=1, collection=[1.0])
        # check size
        self.assertEqual(queue.size(), 1)
        # empty queue, check element value
        self.assertEqual(queue.dequeue(), 1.0)
        # check size
        self.assertEqual(queue.size(), 0)
        # check size
        self.assertTrue(queue.is_empty())
        # check exception on underflow
        self.assertRaises(RuntimeError, queue.dequeue)
        # enqueue element
        queue.enqueue(1.0)
        # check size
        self.assertTrue(queue.is_full())
        # check exception on overflow
        self.assertRaises(RuntimeError, queue.enqueue, 2.0)

    def test_try_enq_deq(self):
        # init
        queue = BoundedQueue(name="test", maxsize=1, collection=[1.0])
        # check size
        self.assertEqual(queue.size(), 1)
        # empty queue, check element value
        self.assertEqual(queue.try_dequeue(), 1.0)
        # check size
        self.assertEqual(queue.size(), 0)
        # check size
        self.assertTrue(queue.is_empty())
        # dequeue from emtpy queue, check return value
        self.assertFalse(queue.try_dequeue())
        # enqueue, into non-full list, check return value
        self.assertTrue(queue.try_enqueue(1.0))
        # check size
        self.assertTrue(queue.is_full())
        # enqueue into full queue, check return value
        self.assertFalse(queue.try_enqueue(1.0), 2.0)

    def test_peek(self):
        # init
        queue = BoundedQueue(name="test", maxsize=2, collection=[1.0, 2.0])
        # check value at index 0
        self.assertEqual(queue.peek(0), 1.0)
        # check value at index 1
        self.assertEqual(queue.peek(1), 2.0)
        # check value at last location
        self.assertEqual(queue.try_peek_last(), 2.0)
        # empty queue
        queue.dequeue()
        queue.dequeue()
        # peek on empty queue, check return value
        self.assertFalse(queue.try_peek_last())


from stencilflow.calculator import Calculator
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
        result = cos(variables["a"] + variables["b"]) if (variables["a"] > variables["b"]) else (variables["a"] + 5) * \
                                                                                                variables["b"]
        self.assertEqual(calculator.eval_expr(variables, computation), result)


class RunProgramTest(unittest.TestCase):
    def test(self):
        pass  # not a general test case, since dace and intel fgpa opencl sdk has to be installed and configured


import stencilflow.helper as helper


class HelperTest(unittest.TestCase):
    def test(self):
        # check max_dict_entry_key
        self.assertEqual(
            helper.max_dict_entry_key({
                "a": [1, 0, 0],
                "b": [0, 1, 0],
                "c": [0, 0, 1]
            }), "a")
        # check list_add_cwise
        self.assertEqual(helper.list_add_cwise([1, 2, 3], [3, 2, 1]), [4, 4, 4])
        # check list_subtract_cwise
        self.assertEqual(helper.list_subtract_cwise([1, 2, 3], [1, 2, 3]),
                         [0, 0, 0])
        # check dim_to_abs_val
        self.assertEqual(helper.dim_to_abs_val([3, 2, 1], [10, 10, 10]), 321)
        # check convert_3d_to_1d
        self.assertEqual(
            helper.convert_3d_to_1d(dimensions=[10, 10, 10], index=[3, 2, 1]),
            321)
        # check load_array
        self.assertListEqual(
            list(
                helper.load_array({
                    "data":
                    os.path.join(os.path.dirname(__file__), "stencils",
                                 "helper_test.csv"),
                    "data_type":
                    helper.str_to_dtype("float64")
                })), [7.0, 7.0])
        self.assertListEqual(
            list(
                helper.load_array({
                    "data":
                    os.path.join(os.path.dirname(__file__), "stencils",
                                 "helper_test.dat"),
                    "data_type":
                    helper.str_to_dtype("float64")
                })), [7.0, 7.0])
        # check save_array / load_array
        out_data = np.array([1.0, 2.0, 3.0])
        file = {"data": "test.dat", "data_type": helper.str_to_dtype("float64")}
        helper.save_array(out_data, file["data"])
        in_data = helper.load_array(file)
        self.assertTrue(helper.arrays_are_equal(out_data, in_data))
        os.remove(file["data"])
        # check unique
        not_unique = [1.0, 2.0, 1.0]
        self.assertListEqual(sorted(helper.unique(not_unique)), [1.0, 2.0])


from stencilflow.log_level import LogLevel
import numpy as np

from stencilflow import run_program


def _return_result(queue, *args, **kwargs):
    ret = run_program(*args, **kwargs)
    queue.put(ret)


def _run_program(*args, **kwargs):
    # We run each kernel with multiprocessing, because the Altera environment
    # does not seem to properly tear down the environment when destroyed.
    # This way, each kernel is run in a separate process, so that it is run
    # with a clean environment.
    queue = mp.Queue()
    p = mp.Process(target=_return_result, args=(queue, ) + args, kwargs=kwargs)
    p.start()
    p.join()
    return queue.get()


class ProgramTest(unittest.TestCase):
    def test_and_simulate(self):
        test_directory = os.path.join(os.path.dirname(__file__), "stencils")
        for stencil_file in [
                "simulator", "simulator2", "simulator3", "simulator4",
                "simulator5", "simulator6", "simulator8", "simulator9",
                "simulator10", "simulator11"
        ]:
            print("Simulating and emulating program {}...".format(stencil_file))
            stencil_file = os.path.join(test_directory, stencil_file + ".json")
            _run_program(
                stencil_file,
                "emulation",
                compare_to_reference=True,
                # TODO: Simulation is broken for 2D
                run_simulation=False,
                # run_simulation=True,
                log_level=LogLevel.BASIC,
                input_directory=os.path.abspath(test_directory))

    def test_program(self):
        test_directory = os.path.join(os.path.dirname(__file__), "stencils")
        for stencil_file in [
                "varying_dimensionality",
                "jacobi2d_128x128",
                "jacobi2d_128x128_8vec",
                "jacobi3d_32x32x32_8itr",
                "jacobi3d_32x32x32_8itr_4vec",
        ]:
            print("Testing program {}...".format(stencil_file))
            stencil_file = os.path.join(test_directory, stencil_file + ".json")
            _run_program(stencil_file,
                         "emulation",
                         compare_to_reference=True,
                         run_simulation=False,
                         log_level=LogLevel.NO_LOG,
                         input_directory=os.path.abspath(test_directory))


if __name__ == '__main__':
    """
        Run all unit tests.
    """
    try:
        unittest.main()
    except SystemExit as ex:
        print('\n', flush=True)
        # Skip all teardown to avoid crashes affecting exit code
        os._exit(ex.code)
