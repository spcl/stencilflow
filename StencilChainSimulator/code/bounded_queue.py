import collections


class BoundedQueue:

    def __init__(self, maxsize):
        # check input
        assert maxsize > 0
        # save param
        self.maxsize = maxsize
        # create queue
        self.queue = collections.dequeue(maxsize)
        # init current size
        self.currentsize = 0

    def size(self):
        return self.currentsize

    def isempty(self):
        return self.size() == 0

    def enqueue(self, item):
        # check bound
        if self.currentsize >= self.maxsize:
            raise Exception("queue overflow")
        # add a new item to the left side
        self.queue.appendleft(item)
        # adjust counter
        self.currentsize += 1

    def dequeue(self):
        # check bound
        if self.currentsize > 0:
            # adjust size
            self.currentsize -= 1
            # return and remove the rightmost item
            return self.queue.pop()
        else:
            raise Exception("queue underflow")

    def try_enqueue(self, item):
        # check bound, do not raise exception in case of an overflow
        if self.currentsize >= self.maxsize:
            return
        # add a new item to the left side
        self.queue.appendleft(item)
        # adjust counter
        self.currentsize += 1

    def try_dequeue(self):
        # check bound, do not raise exception in case of an underflow
        if self.currentsize > 0:
            # adjust size
            self.currentsize -= 1
            # return and remove the rightmost item
            return self.queue.pop()


'''
Notes:
    - implementation:               Uses two stacks as underlying datastructure to ensure overall complexity of O(1) for appendleft() and pop().
    - maxsize for bounded queue:    Default behaviour is to remove oldest element of queue, therefore we have to check 
                                    it and raise an exception.
'''