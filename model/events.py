###########################################################################
# MIT License
#
# Copyright (c) 2020 Matthew Sottile
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
###########################################################################
# module for code representing scheduled events
from enum import Enum
import heapq
from functools import total_ordering

# {{{ event enumeration object
@total_ordering
class Event(Enum):
    """
    Event types.  Note that order is important - in the case where
    two events occur at the same time, we tie-break on the event
    type.  The enum values with lower numbers will be prioritized over
    those with higher numbers since we use a min-heap discipline for
    the priority queue.

    See:

    https://stackoverflow.com/questions/39268052/how-to-compare-enums-in-python

    """
    GISUPDATE    = 1
    CULL_OLDAGE  = 5
    LIV_FERTILE  = 12
    LIV_BIRTH    = 14
    VACCINATE    = 20
    INFECTION    = 30
    WEAROFF      = 40
    MOVEMENT     = 50
    WORLDSTEP    = 100
    AGENTSTEP    = 110

    def __lt__(self, other):
        if self.__class__ is other.__class__:
          return self.value < other.value
        return NotImplemented
# }}}

# {{{ event exceptions
class EventOutOfBounds(Exception):
  """
  Exception to be thrown when an event is added to the event queue
  that is not within the established bounds.
  """
  pass
# }}}

# {{{ event queue
class EventQueue:
  """
  Simple wrapper around a heapq priority queue.  We establish upper and
  lower bounds on the times to detect whether an event is added that
  falls outside an allowed window.
  """
  def __init__(self, lo_time=None, hi_time=None):
    self.events = []
    self.lo_time = lo_time
    self.hi_time = hi_time

  def in_bounds(self, time):
    if self.lo_time is not None and self.lo_time > time:
      return False
    if self.hi_time is not None and self.hi_time < time:
      return False
    return True

  def add_event(self, time, event, subject=None):
    """ Add an event to the queue.  We treat the time boundaries
        differently : if an event is added that is before the lower bound, we
        raise an exception since this makes no plausible sense.  On the other
        hand, an event added after the end of the time range may be valid but
        will be silently ignored since we consider events that occur after
        the simulatuon epoch to be irrelevant. """
    if self.lo_time is not None and self.lo_time > time:
      raise EventOutOfBounds((time, event, subject))
    if self.hi_time is not None and self.hi_time < time:
      #raise EventOutOfBounds((time, event, subject))
      return

    if event == Event.WORLDSTEP:
      if (time, event, subject) not in self.events:
        heapq.heappush(self.events, (time, event, subject))
    else:
      heapq.heappush(self.events, (time, event, subject))

  def next_event(self):
    if len(self.events) > 0:
      return heapq.heappop(self.events)
    else:
      return None
# }}}
