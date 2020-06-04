import functools
from enum import Enum


@functools.total_ordering
class LogLevel(Enum):
    NO_LOG = 0
    BASIC = 1
    MODERATE = 2
    FULL = 3

    # https://stackoverflow.com/a/39269589
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
