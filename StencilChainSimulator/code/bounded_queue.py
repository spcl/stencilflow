import collections
from typing import List


class BoundedQueue:

    def __init__(self, name: str, maxsize: int, collection: List = [], swap_out: bool) -> None:
        """
        Create new BoundedQueue with given initialization parameters.
        :param name: name of the queue
        :param maxsize: maximum number of elements the queue can hold at a time
        :param collection: initial data in queue
        """
        # check input
        assert maxsize > 0
        # save params
        self.maxsize: int = maxsize
        self.name: str = name
        # create queue
        self.queue: collection.dequeue = collections.deque(collection, maxsize)
        # init current size
        self.current_size: int = len(collection)
        # indication of where the buffer is located (slow memory or fast memory)
        self.swap_out = swap_out

    def size(self) -> int:
        """
        Get number of data items the queue currently contains.
        :return: current queue size
        """
        return self.current_size

    def is_empty(self) -> bool:
        """
        Test if queue is empty.
        :return: if queue is empty
        """
        return self.size() == 0

    def enqueue(self, item) -> None:
        """
        Add data element to queue, causes an exception if queue is full.
        :param item: data element
        :return: None
        """
        # check bound
        if self.current_size >= self.maxsize:
            raise RuntimeError("buffer {} overflow occurred".format(self.name))
        # add a new item to the left side
        self.queue.appendleft(item)
        # adjust counter
        self.current_size += 1

    def dequeue(self):
        """
        Remove and return data element from queue, causes an exception if queue is empty.
        :return: data element
        """
        # check bound
        if self.current_size > 0:
            # adjust size
            self.current_size -= 1
            # return and remove the rightmost item
            return self.queue.pop()
        else:
            raise RuntimeError("buffer {} underflow occurred".format(self.name))

    def try_enqueue(self, item) -> bool:
        """
        Add data element to queue..
        :param item: data item
        :return: True: successful, False: unsuccessful
        """
        # check bound, do not raise exception in case of an overflow
        if self.current_size >= self.maxsize:
            # report: unsuccessful
            return False
        # add a new item to the left side
        self.queue.appendleft(item)
        # adjust counter
        self.current_size += 1
        # report: successful
        return True

    def try_dequeue(self):
        """
        Remove and return data item from queue.
        :return: data item: successful, False: unsuccessful
        """
        # check bound, do not raise exception in case of an underflow
        if self.current_size > 0:
            # adjust size
            self.current_size -= 1
            # return and remove the rightmost item
            return self.queue.pop()
        else:
            # report: unsuccessful
            return False

    def peek(self, index: int):
        """
        Returns data item at position 'index' without removal, causes an exception if index > BoundedQueue.current_size
        :param index: queue position of peeking element
        :return: data item
        """
        # check bound
        if self.current_size <= index:
            raise RuntimeError("buffer {} index out of bound access occurred".format(self.name))
        else:
            return self.queue[index]


'''
Notes:
    - implementation:               Uses two stacks as underlying data structure to ensure overall complexity of O(1)
                                    for appendleft() and pop().
    - maxsize for bounded queue:    Default behaviour is to remove oldest element of queue, therefore we have to check 
                                    it and raise an exception.
    - reference:                    https://docs.python.org/3/library/collections.html#deque-objects
'''

if __name__ == "__main__":

    queue = BoundedQueue("debug", 5, [1, 2, 3, 4, 5])

    try:
        print("Enqueue element into full queue, should throw an exception.")
        queue.enqueue(6)
        print("Peek element at pos=3, value is: " + str(queue.peek(3)))
    except Exception as ex:
        print("Exception has been thrown.")
