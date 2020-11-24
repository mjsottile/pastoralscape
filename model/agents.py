#import model.util as U
import numpy as np
import math

# {{{ AgentSet
class AgentSet:
    def __init__(self, name, model_params):
        self.agents = []
        self.name = name
        self.model_params = model_params
        
    def add(self, a):
        self.agents.append(a)

    def size(self):
        return len(self.agents)

    def get(self, i):
        return self.agents[i]
        
    def step(self, tracker, time):
        for agent in self.agents:
            agent.step(tracker, time)
# }}}

# {{{ ID generation
next_id = 1
def gen_id():
    global next_id
    
    i = next_id
    next_id = next_id + 1
    return i
# }}}

# {{{ Person
class Person:
    # {{{ constructor
    def __init__(self, socialnet, model_state, model_params, individual_params, world):
        """ A person has:

            - A social network.
            - Model parameters (independent of the individual).
            - Individual parameters.
            - A world.
        """

        # Cache parameter references
        self.model_params = model_params
        self.individual_params = individual_params

        # Reference to model state
        self.model_state = model_state

        # Store a reference to the social network object, to be used
        # when computing updated opinions. 
        self.socialnet = socialnet

        # initialize state variables for self

        # Location starts off as None.  An individual gets a concrete
        # location later when they are placed, and the location is
        # modified when they move.
        self.location = None

        # Add self to the social network.
        self.socialnet.add_individual(self)

        # Adopt a unique identifier
        self.id = gen_id()
        
        # Store pointer to world object
        self.world = world
    # }}} 

    # {{{ get_world_cell
    def get_world_cell(self):
        if self.location is None:
            return None
        else:
            return self.world.grid[self.location] 
    # }}}

    # {{{ step
    def step(self, tracker, time):
        pass
    # }}}
# }}} 

# {{{ Head of household
class HeadOfHousehold(Person):
    # {{{ constructor
    def __init__(self, socialnet, model_state, model_params, individual_params, world):
        super().__init__(socialnet, model_state, model_params, individual_params, world)

        # per-disease ising state dictionaries
        self.decision = {}
        self.last_decision = {}

        # setup per-disease ising states
        for disease in model_params['model']['setup']['ising']:
            # decision state variables
            if np.random.rand() < model_params['model']['setup']['ising'][disease]:
                self.decision[disease] = 1.0
            else:
                self.decision[disease] = -1.0
            self.last_decision[disease] = self.decision[disease]

        self.herdsmen = []

        self.params = model_params

        self.days_since_decision = 0
    # }}}

    # {{{ setters
    def add_herdsman(self, a):
        self.herdsmen.append(a)
    # }}}

    # {{{ immunize
    def immunize_action(self, disease, time):
        if self.params['model']['disable_vaccination'] == 0:
            for herdsman in self.herdsmen:
                herdsman.immunize(disease, time)
    # }}}

    # {{{ step
    def step(self, tracker, time):
        # step 0: interact with others to update opinions
        
        ### TODO: bogus forcing code here
        if self.days_since_decision < 365:
            self.days_since_decision = self.days_since_decision + time.stepsize.days
        else:
            self.days_since_decision = self.days_since_decision - 365

            # step 1: make decisions for each disease

            for disease in self.params['disease']:
                #   implementation of eq. 4 from bouchaud
                u = (self.individual_params['ising'][disease]['f'] + 
                     self.model_state['ising'][disease]['f_public'])
            
                neighbors = self.socialnet.neighbors(self)

                u = u + sum([self.socialnet.w(self, n) * n.last_decision[disease] for n in neighbors])

                self.last_decision[disease] = self.decision[disease]

                p_pos = (self.model_params['ising'][disease]['mu'] / 
                         (1.0 + np.exp(-self.model_params['ising'][disease]['beta'] * u)))

                if np.random.rand() < p_pos:
                    self.decision[disease] = 1.0
                else:
                    self.decision[disease] = -1.0

                tracker.vaccinate_decision(disease, self.decision[disease])

                if self.decision[disease] == 1:
                    self.immunize_action(disease, time)

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
    def __init__(self, socialnet, model_state, model_params, individual_params, world):
        super().__init__(socialnet, model_state, model_params, individual_params, world)
        
        # redundant: parent class sets location to none, but lint tool
        # unhappy unless here too.
        self.location = None
        self.herd = None
        self.head_of_household = None
        self.home = None
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

    # {{{ movement decision
    def move(self, dt):
        # look out from current place in world to find:
        # - nearest cells with food within some visibility radius
        # - cells with most food that are closest 
        #     ( 1/d * veg_capacity_food_cell )
        # - cells with most food that are closest that will return is in the direction of home
        #     (d_current_cell_to_home / d_food_cell_to_home) * (1/d * veg_capacity_food_cell)

        # get neighbors of current cell within visibility radius
        # neighbors is pair ((i,j), d) where (i,j) is the coordinate in the grid,
        # and d is the distance from self.location to the cell.        
        neighbors = self.world.neighborhood(self.location, self.model_params['agents']['visibility'])
                        
        # add an element to neighbor tuple reflecting capacity weighted by distance
        neighbors = [((i,j), d, (1.0/d) * self.world.grid[i,j][1].veg_capacity) for ((i,j), d) in neighbors]
        
        # TODO: add attraction to water

        # add a final element with the 
        neighbors = [((i,j), d, wd, 
                      wd*(math.hypot(self.location[0]-self.home.location[0], self.location[1]-self.home.location[1]) / max(1.0, math.hypot(i-self.home.location[0],j-self.home.location[1])))) 
                     for ((i,j), d, wd) in neighbors]
        # neighbors = [((i,j), d, wd, 
        #               wd*(U.dist(self.location, self.home.location) / max(1.0, U.dist((i,j), self.home.location)))) 
        #              for ((i,j), d, wd) in neighbors]

        # NOTE: must move TOWARDS neighbor desired.  Might not make it if they can only move partially
        # there in DT.
        neighbors.sort(key=lambda tup: tup[3], reverse=True)
                
        target_pos = neighbors[0][0]
        
        def sgn(x):
            if x == 0:
                return 0
            else:
                return x / np.abs(x)
        
        move_vec = (sgn(target_pos[0]-self.location[0]), 
                    sgn(target_pos[1]-self.location[1]))
        # NOTE: speed not accounted for
        new_pos = (int(self.location[0] + move_vec[0]), 
                   int(self.location[1] + move_vec[1]))
        
        if (new_pos[0] == self.location[0] and new_pos[1] == self.location[1]):
            print("WARNING: timestep too small for speed - didn't move!")
        self.world.move(self, new_pos)
    # }}}

    # {{{ decide_move
    def decide_move(self, time):
        cell_obj = self.get_world_cell()[1]

        if cell_obj.veg_capacity >= self.model_params["agents"]["move_veg_threshold"]:
            return False
        else:
            return True
    # }}}

    # {{{ step
    def step(self, tracker, time):
        # first perform generic agent step
        super().step(tracker, time)

        # step the herd we own
        self.herd.step(tracker, time)

        # decide if it's time to move
        if self.decide_move(time):
            self.move(time)
        else:
            pass
            #print(str(self.id)+' :: no move')
    # }}}

# }}} 