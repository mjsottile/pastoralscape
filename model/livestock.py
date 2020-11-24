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
from enum import Enum
from functools import total_ordering
import dateutil.relativedelta as RD
import numpy as np
import numpy.random as rand
import model.events as E
import model.disease as D

# {{{ Gender enumeration
class Gender(Enum):
    """
    Gender for animals.
    """
    FEMALE = 1
    MALE = 2
# }}}

# {{{ ID generation

# current identifier : treat as private and access only through gen_id()
_NEXTID = 1

def gen_id():
    """ Generate a unique identifier for an agent. """
    global _NEXTID

    i = _NEXTID
    _NEXTID = _NEXTID + 1
    return i
# }}}

# {{{ Animal
@total_ordering
class Animal:
    """ Class representing a single animal. """
    
    # {{{ constructor 
    def __init__(self, gender, birthday, herd, model_state, params):
        """
        Create a new animal with the given gender, a herd object that
        contains the set of animals it co-exists with, and the model
        parameters.
        """
        self.birthday = birthday
        self.gender = gender
        self.params = params
        self.herd = herd
        self.model_state = model_state

        # active flag to indicate if an animal is dead but still present
        # in events that occur after its death.  when this is false the
        # animal is dead and events can be ignored
        self.active = True
        
        # breeding state variables
        self.fertile = False
        self.pregnant = False
        self.nursing = False

        # health state variables
        self.health = params['livestock']['initial_health']

        # disease state variables
        self.diseases = {}

        # ID for order generation
        self.id = gen_id()
    # }}}

    # {{{ __lt__
    # __lt__ operator necessary for including Animal objects in
    # priority queues.
    def __lt__(self, other):
        """ Implementation of < operator to provide an ordering of
            Person objects based on their unique ID. """
        return self.id < other.id
    # }}}

    # {{{ set_disease_state 
    def set_disease_state(self, disease, state):
        self.diseases[disease] = state
    # }}}

    # {{{ immunize 
    def immunize(self, disease, time):
        """ Set the state of the animal to immunized if it in the S or R state.
            If this vaccine wears off, return a date in the future for the
            wearoff event.  The caller (herd) to this function is responsible for
            adding the event since this object cannot see the event queue. """
        # if in the S or R state for the disease, transition to the V state.
        # if in the I state, too late.
        if self.diseases[disease] == D.SIRV.S or self.diseases[disease] == D.SIRV.R:
            self.diseases[disease] = D.SIRV.V

            if 'wearoff' in self.params['disease'][disease]:
                wparams = self.params['disease'][disease]['wearoff']
                wearoff_days = wparams['sigma']*rand.randn() + wparams['mu']
                return time.current_time + RD.relativedelta(days=int(wearoff_days))
        return None
    # }}}

    # {{{ age
    def age(self, time):
        period = time.current_time - self.birthday
        return period.days
    # }}}

    # {{{ handle_event
    def handle_event(self, time, event):
        # pregnant -> nursing transition, creates new animal, schedules future
        # nursing -> fertile transition
        if event == E.Event.LIV_BIRTH:
            assert(self.pregnant and not self.fertile)
            # create child
            child = self.spawn(time)
            self.herd.add(child)

            # update state flags for parent
            self.pregnant = False
            self.nursing = True

            # schedule nursing -> fertility event
            next_date = time.current_time + RD.relativedelta(days=self.params['livestock']['nursing_period'])
            self.model_state.event_queue.add_event(next_date, E.Event.LIV_FERTILE, self)

        # immature -> fertile or nursing -> fertile transition
        elif event == E.Event.LIV_FERTILE:
            assert(not self.pregnant)
            self.nursing = False
            self.fertile = True
        else:
            print(f'Unknown event for animal: {event} @ {time}')
            exit()
    # }}}

    # {{{ can_breed
    def can_breed(self):
        return (self.gender == Gender.FEMALE and self.fertile)
    # }}}

    # {{{ breed
    def breed(self, time):
        """ 
        Animal becomes pregnant. 
        """
        self.pregnant = True
        self.fertile = False
        self.nursing = False

        birth_date = time.current_time + RD.relativedelta(days=self.params['livestock']['gestation_period'])
        self.model_state.event_queue.add_event(birth_date, E.Event.LIV_BIRTH, self)
    # }}}

    # {{{ spawn
    def spawn(self, time):
        # randomly pick gender of offspring
        if rand.rand() > self.params['livestock']['bull_probability']:
            gender = Gender.FEMALE
        else:
            gender = Gender.MALE

        # create child in the same herd as the parent, inheriting the parent parameters and
        # model state.
        child = Animal(gender, time.current_time, self.herd, self.model_state, self.params)

        self.model_state.tracker.record_birth()

        # child starts susceptable to all diseases known
        for disease in self.diseases:
            child.set_disease_state(disease, D.SIRV.S)
            
        # calculate lifespan of individual.  morbid.
        lifespan = self.params['livestock']['death_sigma'] * rand.randn() + self.params['livestock']['death_mu']
        end_date = time.current_time + RD.relativedelta(days=lifespan)
        self.model_state.event_queue.add_event(end_date, E.Event.CULL_OLDAGE, child)

        # calculate maturity date of child for females
        if gender == Gender.FEMALE:
            f_date = time.current_time + RD.relativedelta(days=self.params['livestock']['maturity'])
            self.model_state.event_queue.add_event(f_date, E.Event.LIV_FERTILE, child)

        return child
    # }}}
# }}}

# {{{ Herd
class Herd:
    """
    A herd is a collection of animals.
    """
    # {{{ constructor
    def __init__(self, model_state, params):
        """ Animals. """
        self.animals = []

        self.model_state = model_state
        self.params = params
    # }}}

    # {{{ step
    def step(self, time):
        """ 
        A single herd step focuses on reproduction and culling unhealthy animals. 
        """
        # step 1: iterate through the set of animals to identify the
        #         number of males and breedable females
        males = []
        b_females = []
        for animal in self.animals:
            if animal.gender == Gender.MALE:
                males.append(animal)
            else:
                if animal.can_breed():
                    b_females.append(animal)

        # step 2: breed animals
        if len(males) > 0 and len(b_females) > 0:
            p_breed = float(len(males))/float(len(males)+len(b_females))
  
            day_of_year = time.day_of_year()
            p_date_scale = np.exp((-(day_of_year - self.params['livestock']['breed_date_mu'])**2)/(2*(self.params['livestock']['breed_date_sigma']**2)))
            for b_female in b_females:
                if b_female.health >= self.params['livestock']['min_breed_health'] and rand.rand() < p_date_scale*p_breed*self.params['livestock']['breed_pscale']*time.step_size_days():
                    b_female.breed(time)

        # step 3: cull herd due to bad health
        cull_set = []
        for animal in self.animals:
            if animal.health <= 0.0:
                self.model_state.tracker.record_death("health", time.day_of_epoch())
                cull_set.append(animal)

        for animal in cull_set:
            self.cull(animal)
    # }}}

    # {{{ population management

    # {{{ size 
    def size(self):
        return len(self.animals)
    # }}}

    # {{{ count_gender 
    def count_gender(self, gender):
        i = 0
        for animal in self.animals:
            if animal.gender == gender:
                i = i + 1
        return i
    # }}}

    # {{{ add
    def add(self, animal):
        """ Add a new animal. """
        self.animals.append(animal)
    # }}}

    # {{{ cull
    def cull(self, animal):
        """ Remove an animal from the herd that has died. """
        self.animals.remove(animal)

        # flag animal as inactive to prevent future events from being handled
        animal.active = False
    # }}}

    # {{{ immunize
    def immunize(self, disease, time):
        for animal in self.animals:
            wearoff = animal.immunize(disease, time)
            if wearoff is not None:
                self.model_state.event_queue.add_event(wearoff, E.Event.WEAROFF, (disease, animal))
    # }}}

    # }}}

    # {{{ feeding
    # {{{ food_need
    def food_need(self, time_period):
        """ Calculate the amount of food needed for the
            collection of animals over the given time period. 
        """
        return self.size() * self.params['livestock']['eat'] * time_period
    # }}}

    # {{{ feed
    def feed(self, units, time_period):
        """ Update the health of the herd based on consuming a set
            number of food units over some time period. """
        # no cows, so return
        if self.size() < 1:
            return

        # compute ratio of food available vs food needed
        frac_avail = min(1.0, units / (self.food_need(time_period)))

        h_inc = self.params['livestock']['health_fed'] * frac_avail * time_period
        h_dec = self.params['livestock']['health_starve'] * (1.0 - frac_avail) * time_period

        for animal in self.animals:
            animal.health = animal.health + h_inc - h_dec
            # clamp health to range [0,1]
            animal.health = max(0.0, min(animal.health, 1.0))
    # }}}
    # }}}
# }}}
