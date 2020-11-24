class ModelState:
  def __init__(self):
    self.model_state = {}

  def add_state_variable(self, name, initial):
    self.model_state[name] = initial

  def __getitem__(self,name):
    return self.model_state[name]

