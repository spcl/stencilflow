import collections


class BoundedQueue:

    def __init__(self, name, maxsize, collection=[]):
        # check input
        # assert maxsize > 0 TODO: check why this assertion fails..
        # save params
        self.maxsize = maxsize
        self.name = name
        # create queue
        self.queue = collections.deque(collection, maxsize)
        # init current size
        self.current_size = len(collection)

    def size(self):
        return self.current_size

    def is_empty(self):
        return self.size() == 0

    def enqueue(self, item):
        # check bound
        if self.current_size >= self.maxsize:
            raise RuntimeError("buffer {} overflow occurred".format(self.name))
        # add a new item to the left side
        self.queue.appendleft(item)
        # adjust counter
        self.current_size += 1

    def dequeue(self):
        # check bound
        if self.current_size > 0:
            # adjust size
            self.current_size -= 1
            # return and remove the rightmost item
            return self.queue.pop()
        else:
            raise RuntimeError("buffer {} underflow occurred".format(self.name))

    def try_enqueue(self, item):
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
        # check bound, do not raise exception in case of an underflow
        if self.current_size > 0:
            # adjust size
            self.current_size -= 1
            # return and remove the rightmost item
            return self.queue.pop()
        else:
            # report: unsuccessful
            return False

    def peek(self, index):
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
