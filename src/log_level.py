#!/usr/bin/env python3
# encoding: utf-8
from enum import Enum


class LogLevel(Enum):
    NO_LOG = 0
    BASIC = 1
    MODERATE = 2
    FULL = 3
