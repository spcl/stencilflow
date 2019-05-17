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


from kernel import Kernel
class KernelTest(unittest.TestCase):

    def test(self):
        pass


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
