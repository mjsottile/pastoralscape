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
Code for representing human agents within the model.  This includes the base
person class for a generic human agent, and derived classes representing
specific actors (e.g., herdsmen, heads of household).
"""
import sys
from functools import total_ordering
import numpy as np
import geopy.distance
import model.events as E

# {{{ AgentSet
class AgentSet:
    """ An agent set is a collection of agents.  In addition to
        acting as a container, it also acts as a delegator for
        events and step actions.
    """
    def __init__(self, name, model_params):
        self.agents = []
        self.name = name
        self.model_params = model_params

    def add(self, a):
        """ Add an agent to the set. """
        self.agents.append(a)

    def size(self):
        """ Return the size of the set. """
        return len(self.agents)

    def get(self, i):
        """ Get a specific agent from the set by index. """
        return self.agents[i]

    def handle_event(self, time, event):
        """ Broadcast an event to all agents within the set. """
        for agent in self.agents:
            agent.pre_event_handler(time, event)

        for agent in self.agents:
            agent.handle_event(time, event)

        for agent in self.agents:
            agent.post_event_handler(time, event)

    def step(self, time):
        """ Step each agent in the set. """
        for agent in self.agents:
            agent.step(time)

    def record(self, time):
        """ Record state date in model_state tracker. """
        for agent in self.agents:
            agent.record(time)
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

# {{{ Person
@total_ordering
class Person:
    # {{{ constructor
    def __init__(self, model_state, model_params, individual_params):
        """ A person has:

            - A reference to the model state.
            - Model parameters (independent of the individual).
            - Individual parameters.

            Constructing a person adds them to the social network of
            the model_state and allocates a unique identifier for them.
            Their location is initialized to None.
        """

        # Cache parameter references
        self.model_params = model_params
        self.individual_params = individual_params

        # Reference to model state
        self.model_state = model_state

        # Location starts off as None.  An individual gets a concrete
        # location later when they are placed, and the location is
        # modified when they move.
        self.location = None

        # Add self to the social network.
        self.model_state.social_net.add_individual(self)

        # Adopt a unique identifier
        self.id = gen_id()
    # }}} 

    # {{{ get_world_cell
    def get_world_cell(self):
        """ Resolve the agent location to the world cell object
            where they reside. """
        if self.location is None:
            return None
        return self.model_state.world.grid[self.location][1]
    # }}}

    # {{{ get_world_cell_by_id
    def get_world_cell_by_id(self, cell_id):
        """ Return the cell object from the world corresponding
            to a specific GIS cell ID. """
        idx = self.model_state.world.id_to_index[cell_id]
        return self.model_state.world.grid[idx][1]
    # }}}

    # {{{ get_world_cell_by_index
    def get_world_cell_by_index(self, idx):
        """ Return the cell object from the world corresponding
            to a given world grid index. """
        return self.model_state.world.grid[idx][1]
    # }}}

    # {{{ get_world_cell_by_latlon
    def get_world_cell_by_latlon(self, latlon):
        """ Return the cell object from the world that is closest
            to the given latitude and longitude. """
        nearest = self.model_state.world.nearest_cell(latlon)
        return self.model_state.world.grid[nearest][1]
    # }}}

    # {{{ handle_event
    def handle_event(self, time, event):
        """ Default event handler does nothing. """
        pass
    # }}}

    # {{{
    def pre_event_handler(self, time, event):
        """ Default handler to apply to agent before handling event. """
        pass
    # }}}

    # {{{
    def post_event_handler(self, time, event):
        """ Default handler to apply to agent after handling event. """
        pass
    # }}}

    # {{{ step
    def step(self, time):
        """ Default step function does nothing. """
        pass
    # }}}

    # {{{ record
    def record(self, time):
        """ Default record does nothing. """
        pass
    # }}}

    # {{{ __lt__
    # __lt__ operator necessary for including Person objects in
    # priority queues.
    def __lt__(self, other):
        """ Implementation of < operator to provide an ordering of
            Person objects based on their unique ID. """
        return self.id < other.id
    # }}}
# }}} 

# {{{ Head of household
class HeadOfHousehold(Person):
    """ A head of household is an instance of a person object that
        manages one or more herdsmen.  A head of household is responsible
        for disease vaccination decisions, economic decisions, and some
        decisions related to the behavior of herdsmen. """
    # {{{ constructor
    def __init__(self, model_state, model_params, individual_params):
        super().__init__(model_state, model_params, individual_params)

        # per-disease ising state dictionaries
        self.decision = {}
        self.last_decision = {}

        # setup per-disease ising states
        for disease in model_params['model']['setup']['ising']:
            # decision state variables
            if np.random.rand() < model_params['model']['setup']['ising'][disease]['prob_positive']:
                self.decision[disease] = 1.0
            else:
                self.decision[disease] = -1.0
            self.last_decision[disease] = self.decision[disease]

        self.herdsmen = []

        self.params = model_params
    # }}}

    # {{{ setters
    def add_herdsman(self, a):
        """ Add a herdsman to the collection that this head of household
            manages. """
        self.herdsmen.append(a)
    # }}}

    # {{{ immunize
    def immunize_action(self, disease, time):
        """ Delegate immunication actions to the herdsmen who hold
            livestock in their herds.  Heads of household do not
            directly interact with livestock but do so through their
            herdsmen. """
        if self.params['model']['disable_vaccination'] == 0:
            for herdsman in self.herdsmen:
                herdsman.immunize(disease, time)
    # }}}

    # {{{ cycle_disease_decisions
    def cycle_disease_decisions(self, time):
        """ Move all current decisions to be previous decisions in advance of
            a new decision making phase. """
        for disease in self.decision:
            self.last_decision[disease] = self.decision[disease]
    # }}}

    # {{{ disease_decisions
    def disease_decisions(self, time):
        """ For each disease, calculate the immunication decision based on the
            Random Field Ising Model from Bouchaud.  Assumes that cycle_disease_decisions
            has been called to have most recent decision in the last_decision slot. """
        for disease in self.params['disease']:
            #   implementation of eq. 4 from bouchaud
            u = (self.individual_params['ising'][disease]['f'] + 
                 self.model_state.ising[disease]['f_public'])
        
            neighbors = self.model_state.social_net.neighbors(self)

            u = u + sum([self.model_state.social_net.weight(self, n) * n.last_decision[disease] for n in neighbors])

            # if last was negative, eq 4.
            if self.last_decision[disease] < 0:
                # eq 4 : P(s(t) = 1 | s(t-1) = -1)
                p_pos_neg = (self.model_params['ising'][disease]['mu'] / 
                                (1.0 + np.exp(-self.model_params['ising'][disease]['beta'] * u)))

                if np.random.rand() < p_pos_neg:
                    self.decision[disease] = 1.0
                else:
                    self.decision[disease] = -1.0

            # otherwise if last was positive, eq 5.
            else:
                # eq 5 : P(s(t) = -1 | s(t-1) = 1)
                p_neg_pos = (self.model_params['ising'][disease]['mu'] /
                                   (1.0 + np.exp(self.model_params['ising'][disease]['beta'] * u)))
    
                if np.random.rand() < p_neg_pos:
                    self.decision[disease] = -1.0
                else:
                    self.decision[disease] = 1.0

            self.model_state.tracker.vaccinate_decision(disease, self.decision[disease], time.day_of_epoch())

            if self.decision[disease] == 1:
                self.immunize_action(disease, time)
    # }}}

    # {{{ handle_event
    def handle_event(self, time, event):
        if event == E.Event.VACCINATE:
            self.disease_decisions(time)
    # }}}

    # {{{ 
    def pre_event_handler(self, time, event):
        if event == E.Event.VACCINATE:
            self.cycle_disease_decisions(time)
    # }}}

    # {{{ step
    def step(self, time):
        pass
    # }}}

# }}}

# {{{ Herdsman
class Herdsman(Person):

    # {{{ constructor
    """ A herdsman is an instance of a person that has additional
        properties: 
        
        - a head-of-household that they answer to,
        - a herd of livestock that they manage,
        - a home village

        They have additional decisions that they can make based on
        their opinion about whether they should move.
    """
    def __init__(self, model_state, model_params, individual_params):
        super().__init__(model_state, model_params, individual_params)
        
        # redundant: parent class sets location to none, but lint tool
        # unhappy unless here too.
        self.location = None
        self.herd = None
        self.head_of_household = None
        self.home = None
        
        # when an agent embarks on a journey from their home village
        # they enter the moving state and acquire a path that they
        # step along.
        self.moving = False
        self.current_path = None
        self.current_path_step = None
        self.next_waypoint = None
        self.latlon = None
        self.direction = None
    # }}}

    # {{{ setters
    def set_herd(self, herd):
        self.herd = herd

    def set_head_of_household(self, hoh):
        self.head_of_household = hoh

    def set_home_village(self, home):
        self.home = home
    # }}}

    # {{{ immunize
    def immunize(self, d, time):
        self.herd.immunize(d, time)
    # }}}

    # {{{ movement / location helpers
    def at_next_waypoint(self):
        """
        Check if the agent is at the next waypoint by mapping
        the current latlon coordinate to cell ID and checking
        against the next waypoint.
        """
        cell_obj = self.get_world_cell_by_latlon(self.latlon)
        return cell_obj.cell_id == self.next_waypoint

    def update_direction_vector(self):
        cur_cell = self.get_world_cell()
        way_cell = self.get_world_cell_by_id(self.next_waypoint)

        cur_latlon = (cur_cell.latitude, cur_cell.longitude)
        way_latlon = (way_cell.latitude, way_cell.longitude)
        d = geopy.distance.distance(cur_latlon, way_latlon).kilometers

        # TODO: currently we force a 1km constant agent movement speed.
        #       may want to allow for this to be a parameter.

        # avoid overstepping - the largest move we can make is to go exactly from
        # start to finish.  d < 1 would overshoot.
        if d < 1:
            d = 1.0

        self.model_state.tracker.record_distance(d)

        self.direction = ((way_latlon[0] - cur_latlon[0]) / d,
                          (way_latlon[1] - cur_latlon[1]) / d)
    
    def move_location_by_step(self):
        self.latlon = (self.latlon[0] + self.direction[0], self.latlon[1] + self.direction[1])
        cell_obj = self.get_world_cell()
        nearest_obj = self.get_world_cell_by_latlon(self.latlon)
        if cell_obj.cell_id != nearest_obj.cell_id:
            self.model_state.world.move(self, nearest_obj.location)
    # }}}

    # {{{ decide_move
    def decide_move(self, time):
        cell_obj = self.get_world_cell()

        return cell_obj.veg_capacity < self.model_params["agents"]["move_veg_threshold"]
    # }}}

    # {{{ handle_event
    def handle_event(self, time, event):
        if event == E.Event.MOVEMENT:
            self.handle_move_event(time)
    # }}}

    # {{{ handle movement events
    def handle_move_event(self, time):
        # if we are moving, perform move action, which includes
        # updating our place on the current path.
        if self.moving:
            if self.at_next_waypoint():
                # if we've returned to step 0, then we're at the beginning of the path again and have
                # looped around and movement stops.
                if self.current_path_step == 0:
                    self.moving = False
                else:
                    # get the next waypoint, update the movement vector and make a step.
                    (self.current_path_step, self.next_waypoint) = self.current_path.nextstep(self.current_path_step)
                    self.update_direction_vector()
                    self.move_location_by_step()

                    # set the next move event to tomorrow.
                    t = time.tomorrow()
                    if self.model_state.event_queue.in_bounds(t):
                        self.model_state.event_queue.add_event(t, E.Event.MOVEMENT, self)
                        self.model_state.event_queue.add_event(t, E.Event.WORLDSTEP)
            else:
                # take a step and set the next move event to tomorrow.
                self.move_location_by_step()
                t = time.tomorrow()
                if self.model_state.event_queue.in_bounds(t):
                    self.model_state.event_queue.add_event(t, E.Event.MOVEMENT, self)
                    self.model_state.event_queue.add_event(t, E.Event.WORLDSTEP)
        else:
            print("ERROR: encountered movement event for a nonmoving agent")
            sys.exit()
        self.model_state.tracker.record_occupancy(self.location, 1, "herdsman", time.day_of_epoch())
    # }}}

    # {{{ step
    def step(self, time):
        # first perform generic agent step
        super().step(time)

        # step the herd we own
        self.herd.step(time)

        # decide if it's time to move
        if (not self.moving) and self.decide_move(time):
            # get an arbitrary path originating from the home village of the agent.
            path = self.home.get_path()
            if path is not None:
                # movement takes place with respect to lat/long coordinates, not
                # cell coordinates.  so set initial position of moving agent to the
                # center of the cell they are currently in.
                cell_obj = self.get_world_cell()
                self.latlon = (cell_obj.latitude, cell_obj.longitude)

                # flag moving and remember the current path
                self.moving = True
                self.current_path = path

                # get the first step of the path and set the movement vector to point
                # from the current location to the next waypoint.
                (self.current_path_step, self.next_waypoint) = self.current_path.nextstep(0)
                self.update_direction_vector()

                # inject a movement event to move the agent tomorrow.
                t = time.tomorrow()
                self.model_state.event_queue.add_event(t, E.Event.MOVEMENT, self)
                self.model_state.event_queue.add_event(t, E.Event.WORLDSTEP)
    # }}}

    # {{{ record
    def record(self, time):
        """ Record status of herd. """
        self.model_state.tracker.record_occupancy(self.location, 1, "herdsman", time.day_of_epoch())
        self.model_state.tracker.record_herd(self.herd, time)
    # }}}

# }}} 
