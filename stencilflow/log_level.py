#!/usr/bin/env python3

"""
    The LogLevel class enables a fine-grained distinction of logging verbosity.
"""

__author__ = "Andreas Kuster (kustera@ethz.ch)"
__copyright__ = "BSD 3-Clause License"

import functools

from enum import Enum

@functools.total_ordering
class LogLevel(Enum):
    NO_LOG = 0
    BASIC = 1
    MODERATE = 2
    FULL = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented()
        # credits: https://stackoverflow.com/a/39269589
