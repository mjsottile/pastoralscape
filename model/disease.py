import sys
import numpy.random as rand
from enum import Enum

# {{{ SIRV enum
class SIRV(Enum):
    """
    States for SIRV model.
    """
    S = 1
    I = 2
    R = 3
    V = 4
# }}}

# {{{ Single disease object 
class Disease:

  # {{{ constructor
  def __init__(self, name, model_params):
    """ Disease object constructor.  A disease has a name and is given
        the model parameters.
    """
    self.name = name
    self.model_params = model_params
  # }}}

  # {{{ step
  def step(self, herds, tracker, time):
    """ One step.  The step applies the disease to one or more herds in
        a collection.  
    """

    # step 1: partition animals from all herds into a collective set for 
    #         each disease state
    s = []
    i = []
    r = []
    v = []

    for herd in herds:
      for animal in herd.animals:
        dstate = animal.diseases[self.name]
        if dstate == SIRV.S:
          s.append(animal)
        elif dstate == SIRV.I:
          i.append(animal)
        elif dstate == SIRV.R:
          r.append(animal)
        elif dstate == SIRV.V:
          v.append(animal)
        else:
          sys.exit('impossible SIRV state')

    popsize = len(s)+len(i)+len(r)+len(v)

    # get params from param object

    # p_si is the only one that is proportional to the number of infected
    # individuals / the total size of the population.  the lethality, recovery,
    # and relapse to susceptability is an internal process of the individual and
    # independent of outside factors.
    if popsize > 0:
      p_si = self.model_params['disease'][self.name]['p_si'] * float(len(i)) / float(popsize)
    else:
      p_si = 0.0
    p_ir = self.model_params['disease'][self.name]['p_ir']
    p_id = self.model_params['disease'][self.name]['p_id']
    p_rs = self.model_params['disease'][self.name]['p_rs']
    p_vs = self.model_params['disease'][self.name]['p_vs']

    # probabilities are defined on weekly basis.  scale by timestep size.
    p_si = p_si * time.step_size_weeks()
    p_ir = p_ir * time.step_size_weeks()
    p_id = p_id * time.step_size_weeks()
    p_rs = p_rs * time.step_size_weeks()
    p_vs = p_vs * time.step_size_weeks()

    # step 2: model state transitions
    for animal in s:
      if rand.rand() < p_si:
        animal.set_disease_state(self.name, SIRV.I, time)
    
    for animal in i:
      p = rand.rand()
      if p < p_ir + p_id:
        # dead or recovered

        if p < p_id:
          # dead
          tracker.disease_death(self.name)
          animal.herd.remove(animal)
        else:
          # recovered
          animal.set_disease_state(self.name, SIRV.R, time)
    
    for animal in r:
      if rand.rand() < p_rs:
        animal.set_disease_state(self.name, SIRV.S, time)
    
    # only consider vs transition if there is a nonnegative p_vs.
    # set p_vs to negative value to suppress v->s transitions
    if p_vs > 0.0:
      for animal in v:
        if rand.rand() < p_vs:
          animal.set_disease_state(self.name, SIRV.S, time)

  # }}}
  
# }}}
