# module for code representing scheduled events

class Event:
  def __init__(self, when):
    self.time = when

  def action(self, time):
    pass

class EventQueue:
  def __init__(self):
    self.events = []

  def add_event(self, event):
    self.events.append(event)

  def step(self, time):
    # find all events in the queue that are at or before the given time
    before = [e.time <= time.current_time for e in self.events]
    
    # put them in order
    before.sort()

    # iterate through sorted list of events invoking action with the
    # 
    for e in before:
      e.action(time)