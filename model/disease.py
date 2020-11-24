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
import sys
import math
from enum import Enum
import numpy.random as rand

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
  """ Class representing the transition system and infection model for a single
      disease. 
  """

  # {{{ constructor
  def __init__(self, name, model_state, model_params):
    """ Disease object constructor.  A disease has a name and is given
        the model parameters.
    """
    self.name = name
    self.model_params = model_params
    self.model_state = model_state
  # }}}

  # {{{ sample_infection
  def sample_infection(self, time):
    """ Test if a single infection (S->I) spontaneously occurs from environment
        at given time.
    """
    if self.model_params['disease'][self.name]['new_infection_model'] == 'harmonic':
      day = time.day_of_year()
      b0 = self.model_params['disease'][self.name]['harmonic']['constant']
      b1 = self.model_params['disease'][self.name]['harmonic']['cos']
      b2 = self.model_params['disease'][self.name]['harmonic']['sin']
      m = self.model_params['disease'][self.name]['harmonic']['m']
      omega = 1.0 / m
      p = math.exp(b0 + b1 * math.cos(2.0*math.pi*omega*day) + b2 * math.sin(2.0*math.pi*omega*day))
      return rand.rand() < p
    elif self.model_params['disease'][self.name]['new_infection_model'] == 'uniform':
      p = self.model_params['disease'][self.name]['p_si_spontaneous']
      return rand.rand() < p
    else:
      print("Unsupported infection model: "+self.model_params['disease'][self.name]['new_infection_model'])
      exit()
    return False
  # }}}

  # {{{ step
  def step(self, herds, time):
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

    dt = time.current_step_duration()

    # scale probabilities by timestep size vs timefactor that they are defined for
    p_si = p_si * dt / self.model_params['model']['disease_timefactor']
    p_ir = p_ir * dt / self.model_params['model']['disease_timefactor']
    p_id = p_id * dt / self.model_params['model']['disease_timefactor']
    p_rs = p_rs * dt / self.model_params['model']['disease_timefactor']
    p_vs = p_vs * dt / self.model_params['model']['disease_timefactor']

    # step 2: model state transitions
    for animal in s:
      if rand.rand() < p_si:
        animal.set_disease_state(self.name, SIRV.I)
    
    for animal in i:
      p = rand.rand()
      if p < p_ir + p_id:
        # dead or recovered

        if p < p_id:
          # dead
          self.model_state.tracker.record_death(self.name, time.day_of_epoch())
          animal.herd.cull(animal)
        else:
          # recovered
          animal.set_disease_state(self.name, SIRV.R)
    
    for animal in r:
      if rand.rand() < p_rs:
        animal.set_disease_state(self.name, SIRV.S)
    
    # only consider vs transition if there is a nonnegative p_vs.
    # set p_vs to negative value to suppress v->s transitions
    if p_vs > 0.0:
      for animal in v:
        if rand.rand() < p_vs:
          animal.set_disease_state(self.name, SIRV.S)
  # }}}
# }}}
