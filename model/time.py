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
"""
Time modeling.  Mapping of current time to calendar date, step sizes.
"""
import datetime as D
import dateutil.relativedelta as RD
import numpy as np
import numpy.random as rand
import model.events as E

# {{{ Exception
class TimeOrderViolation(Exception):
  """
  When someone tries to do something in the past.
  No time travel!
  """
  pass
# }}}

def most_recent(cur, month_days):
  """
    Given the current time, and list of month/day pairs, find the most
    recent month/day pair in the current or previos year.
    """
  this_year = [D.date(year=cur.year, month=x[0], day=x[1]) for x in month_days]
  last_year = [D.date(year=cur.year-1, month=x[0], day=x[1]) for x in month_days]
  dates = [(x,cur-x) for x in this_year + last_year if (cur-x).days >= 0]
  best = dates[0]
  for i in range(1,len(dates)):
    if dates[i][1] < best[1]:
      best = dates[i]
  return best[0]

class Time:
  """ Class representing the current time, the default step size, and
      start of the simulation epoch.  Encapsulates all time and date
      calculations as well via helper methods that hide the user of the
      class from the underlying dateutil implementation.
  """
  def __init__(self, initial_date, stepsize):
    """
    Initial_date is a date object.
    Stepsize is a timedelta object.
    """
    self.initial_date = initial_date
    self.current_time = initial_date
    self.stepsize = stepsize
    self.last_timestep = None

  def current_step_duration(self):
    """ Return the current time step as the number of days from
        the current time to the last timestep.  If no last timestep,
        return the default timestep size. """
    if self.last_timestep is None:
      return self.stepsize.days
    else:
      return (self.current_time - self.last_timestep).days

  def enumerate_annual_events(self, month, day, end_date):
    """
    Generate a sequence of annual dates on a given month and day
    between the initial_date and the given end_date.
    """
    dates = []
    cur = D.date(year=self.initial_date.year, month=month, day=day)
    if cur < self.initial_date:
      cur = cur + RD.relativedelta(years=+1)
    while cur <= end_date:
      dates.append(cur)
      cur = cur + RD.relativedelta(years=+1)
    return dates

  def enumerate_month_starts(self, start_date, end_date):
    """
    Enumerate all of the month-starts after the start date up to
    the end date.  The start is not included.
    """
    dates = []
    cur = D.date(year=start_date.year, month=start_date.month, day=1)
    cur = cur + RD.relativedelta(months=+1)
    while cur <= end_date:
      dates.append(cur)
      cur = cur + RD.relativedelta(months=+1)
    return dates

  def tomorrow(self):
    """ Return the date that follows the current time. """
    now = self.current_time
    nextday = now + RD.relativedelta(days=+1)
    return nextday

  def enumerate_step_events(self, end_date):
    """
    Enumerate all events from initial_date to end_date that
    correspond to timestep events.
    """
    events = []
    cur = self.initial_date
    while cur <= end_date:
      events.append((cur, E.Event.WORLDSTEP))
      events.append((cur, E.Event.AGENTSTEP))
      cur = cur + self.stepsize
    return events

  def set_time(self, time):
    """
    Step the current time forward, raising an exception if the new
    time doesn't progress forward.
    """
    if self.current_time > time:
      raise TimeOrderViolation((time, self.current_time))

    self.current_time = time

  def day_of_epoch(self, time=None):
    """ Return the day number since the start of the epoch (day 0). """
    if time is None:
      delta = self.current_time - self.initial_date
    else:
      delta = time - self.initial_date
    return delta.days

  def step_size_days(self):
    """ Return the current step size in days. """
    return self.stepsize.days

  def steps_for_timedelta(self, period):
    """ Return the number of steps in a given time period,
        rounding up. """
    return int(np.ceil(period.days / self.stepsize.days))

  def day_of_year(self):
    """ Return the day of year for the current time. """
    return self.current_time.timetuple().tm_yday
