"""
Time modeling.  Mapping of current time to calendar date, step sizes.
"""
import datetime as D
import numpy as np
import numpy.random as rand

class Time:
  def __init__(self, initial_date, stepsize):
    """
    Initial_date is a date object.
    Stepsize is a timedelta object.
    """
    self.current_time = initial_date
    self.stepsize = stepsize

  def step_forward(self):
    """
    Step the time forward by stepsize.
    """
    self.current_time = self.current_time + self.stepsize

  def step_size_weeks(self):
    return self.stepsize.days / 7

  def steps_for_timedelta(self, td):
    x = td.days / self.stepsize.days
    return int(np.ceil(x))

  def day_of_year(self):
    return self.current_time.timetuple().tm_yday

  def weeks_since(self, d):
    dt = self.current_time - d
    return dt.days / 7

  def weekly_probability(self):
    """ how many trials to perform an action that is supposed to happen
        weekly.  for example, if dt = 3.5 days, then there is a 50% chance
        we try it once.  if dt=7 days, then definitely one trial.
        if dt = 17 days, then we have 2 definite trials, and the final
        trial has a 3/7 chance of happening. """
    trials = self.stepsize.days // 7
    p_onemore = (self.stepsize.days % 7) / 7

    if rand.rand() < p_onemore:
      trials = trials + 1

    return trials