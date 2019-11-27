# adapted from https://github.com/ildoonet/pystopwatch2/blob/master/pystopwatch2/watch.py

import threading
import time

from enum import Enum
from collections import defaultdict

class _ClockState(Enum):
    PAUSE = 0
    RUN = 1

class _Clock:
    tag_default = '__default1958__'
    th_lock = threading.Lock()

    def __init__(self):
        self.prev_time = time.time()
        self.sum = 0.
        self.state = _ClockState.PAUSE

    def __str__(self):
        return 'state=%s elapsed=%.4f prev_time=%.8f' % (self.state, self.sum, self.prev_time)

    def __repr__(self):
        return self.__str__()


class StopWatch:
    stopwatch:'StopWatch' = None

    def __init__(self):
        self.clocks = defaultdict(lambda: _Clock())

    def start(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        with _Clock.th_lock:
            clock = self.clocks[tag]
            if clock.state == _ClockState.RUN:
                return
            clock.state = _ClockState.RUN
            clock.prev_time = time.time()

    def pause(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        with _Clock.th_lock:
            clock = self.clocks[tag]
            clock.state = _ClockState.PAUSE
            delta = time.time() - clock.prev_time
            clock.sum += delta
            return clock.sum

    def clear(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        del self.clocks[tag]

    def get_elapsed(self, tag=None):
        if tag is None:
            tag = _Clock.tag_default
        clock = self.clocks[tag]
        elapsed = clock.sum
        if clock.state == _ClockState.RUN:
            elapsed += time.time() - clock.prev_time

        return elapsed

    def keys(self):
        return self.clocks.keys()

    def __str__(self):
        return '\n'.join(['%s: %s' % (k, v) for k, v in self.clocks.items()])

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def set(instance:'StopWatch')->None:
        StopWatch.stopwatch = instance

    @staticmethod
    def get()->'StopWatch':
        return StopWatch.stopwatch