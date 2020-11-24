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
import geopy.distance
import numpy as np
import model.agents as A

# {{{ path
class Path:
    """
    A path defines an ordered sequence of cell IDs that represent a path that
    originates and terminates at a specific cell ID.
    """
    def __init__(self, waypoints):
        self.waypoints = waypoints

    def nextstep(self, stepid):
        """
        Given a step in the list of waypoints, return the step number
        for the next step and the cellID of that step.  We assume the
        path is a cycle, so when we hit the end of the path assume that
        we've returned to the beginning.
        """
        if stepid == len(self.waypoints)-1:
            return (0, self.waypoints[0])
        return (stepid+1, self.waypoints[stepid+1])
# }}}

# {{{ grid space
class GridSpace:
    """ 
    A grid space in the world has some vegetation state as well as presence of water.
    """

    # {{{ constructor
    def __init__(self, params):
        # NDVI ranges from -1.0 to 1.0
        self.mean_ndvi = 0.0
        self.mean_ndvi_alltime = 0.0

        self.params = params
        self.location = None

        self.has_water = False

        self.veg_capacity = None
    # }}}

    # {{{ forage
    def forage(self, num_animals, dt):
        # units of livestock.eat : m^2
        # area of cell : 1km^2 = 1000*1000 m^2
        cell_area = 1e6
        food_required = num_animals * self.params['livestock']['eat'] * dt
        frac_avail = min((cell_area * self.veg_capacity)/food_required, 1.0)
        food_obtained = food_required * frac_avail
        return food_obtained
    # }}}

    # {{{ step
    def step(self, time):
        pass
    # }}}
# }}}

# {{{ Village 
class Village(GridSpace):
    """ 
    A village is a grid space that holds a set of individuals that do not move. 
    A village also has a set of paths associated with it.  Individuals who call
    the village home will follow one of these paths when they decide to move.
    """

    # {{{ __init__ constructor
    def __init__(self, params):
        super().__init__(params)
        self.fixed_individuals = []
        self.paths = []
    # }}}

    # {{{ add_path
    def add_path(self, path):
        """ Add a path defining a sequence of waypoints to this village. """
        self.paths.append(path)
    # }}}

    # {{{ get_path
    def get_path(self):
        """
        Select a path at random.  If no paths present, return None.
        """
        if len(self.paths) < 1:
            return None

        path_num = np.random.randint(len(self.paths))
        return self.paths[path_num]
    # }}}

    # {{{ add_individual
    def add_individual(self, i):
        """ Add an individual agent to the set of agents that are fixed at
            this cell.  Example: heads of household.
        """
        self.fixed_individuals.append(i)
    # }}}

    def popsize(self):
        return len(self.fixed_individuals)

    def get_individual(self, i):
        return self.fixed_individuals[i]
# }}}

# {{{ World 
class World:
    # {{{ __init__ constructor
    def __init__(self, model_state, w, h, d):
        """ 
        create a 2d world of width w and height h kilometers.  d is a default
        cell object constructor.  this may be a lambda like:
            
        d = lambda x : GridSpace(0.0, 100.0, {'eat':1.0, 'constant_growth'=0.0, 'precip_growth'=0.1}) 

        indexing of world cells is (lat,lon)
        """
        self.width = w
        self.height = h

        self.model_state = model_state

        # GIS data store
        self.gis = model_state.gis

        # grid cell is a pair:
        # - a list of agents residing there
        # - a grid cell object
        self.grid = np.empty((h, w), dtype=object)
        
        for i in range(h):
            for j in range(w):
                self.grid[i, j] = ([], d((i, j)))
                self.grid[i, j][1].location = (i, j)

        # diseases are propagated by the world, so we must track them
        self.diseases = {}

        # GIS related state variables
        self.id_to_index = {}
        self.mean_fci = None
        self.neighbor_cache = {}

        # extra fields to help with efficient nearest cell lookup
        self.first_dim_lat = False
        self.lat_boundaries = None
        self.lon_boundaries = None

        self.live_cells = None
    # }}}

    # {{{ nearest_cell
    def nearest_cell(self, latlon):
        """ Find the nearest cell to a given latitude and longitude
            by bisection search.  Two assumptions:

            - Latitude and longitude increase as their respective cell indices
              increase.  (e.g., cell[i,j] < cell[i,j] w.r.t. lat or lon).
            - For a given column or row of cells, either the latitude or longitude
              remains fixed.  This restricts us to grids aligned with the latitude/longitude
              lines.

            Bisection search for an m by n grid will require O(log n) + O(log m) time, versus
            the O(n * m) time required for brute force search through all cells.  This
            results in a substantial speedup.
        """
        (lat, lon) = latlon
        # first find lat cell coordinate
        n = len(self.lat_boundaries)
        lo = 0
        hi = n-1
        lat_idx = n//2
        while hi-lo > 1:
            if lat < self.lat_boundaries[lat_idx]:
                hi = lat_idx
            else:
                lo = lat_idx
            lat_idx = (hi-lo)//2 + lo

        # then find lon cell coordinate
        n = len(self.lon_boundaries)
        lo = 0
        hi = n-1
        lon_idx = n//2
        while hi-lo > 1:
            if lon < self.lon_boundaries[lon_idx]:
                hi = lon_idx
            else:
                lo = lon_idx
            lon_idx = (hi-lo)//2 + lo

        if self.first_dim_lat:
            return (lat_idx, lon_idx)
        else:
            return (lon_idx, lat_idx)
    # }}}

    # {{{ add_disease
    def add_disease(self, disease):
        """ Add a disease to the set that the world steps. """
        self.diseases[disease.name] = disease
    # }}}

    # {{{ set_cell
    def set_cell(self, position, cell_obj):
        """
        Set the cell object for a grid space to the given object.
        The agent set at the cell is reset to empty.
        The location attribute of the cell object is set to the
        position.
        """
        self.grid[position] = ([], cell_obj)
        self.grid[position][1].location = position
    # }}}

    # {{{ move
    def move(self, agent, position):
        """
        Move the agent a from its current location to the coordinate
        (i,j).
        """
        # where is agent now?
        old_position = agent.location

        # remove agent from current location resident set
        (residents, _) = self.grid[old_position]
        residents.remove(agent)
        if len(residents) == 0:
            self.live_cells.remove(old_position)

        # add agent to new location resident set
        (residents, _) = self.grid[position]
        residents.append(agent)
        if position not in self.live_cells:
            self.live_cells.add(position)

        # set agent location to new coordinates
        agent.location = position
    # }}}
        
    # {{{ neighborhood
    def neighborhood(self, position, r):
        """
        Find coordinates for all cells around position that are within
        the radius r.  Return a list of (coordinate, distance) pairs.
        """
        if position not in self.neighbor_cache:
            neighbors = []
            for i in np.arange(max(0, position[0]-r), min(self.height, position[0]+r)):
                for j in np.arange(max(0, position[1]-r), min(self.width, position[1]+r)):
                    # ignore (i,j) == pos
                    if (i != position[0]) or (j != position[1]):
                        d = math.hypot(position[0]-i, position[1]-j)
                        #d = U.dist(pos, (i,j))
                        if d <= r:
                            neighbors.append(((int(i), int(j)), d))
            self.neighbor_cache[position] = neighbors
        else:
            neighbors = self.neighbor_cache[position]
        return neighbors
    # }}}

    # {{{ place_agent
    def place_agent(self, agent, position):
        """ Place an agent at a position.  This is only used during initialization
            and does not update the live cell set. """
        agent.location = position
        (residents, _) = self.grid[position]
        residents.append(agent)
    # }}}

    # {{{ update_vegetation
    def update_vegetation(self, params, date):
        # load monthly GIS data
        gis_data = self.gis.get_date(date.year, date.month)

        # map GIS data onto cells, calculate mean NDVI for this time period over
        # the world.
        self.world_mean_ndvi = 0.0
        for cell_id in gis_data:
            row = gis_data[cell_id]
            self.grid[self.id_to_index[cell_id]][1].mean_ndvi = row['mean_ndvi']
            self.world_mean_ndvi += row['mean_ndvi']
            self.grid[self.id_to_index[cell_id]][1].mean_precip = row['mean_precip']
        self.world_mean_ndvi = self.world_mean_ndvi / (self.width * self.height)

        # get FCI for current month and update veg_capacity
        fci_data = self.gis.get_fci_month(date.year, date.month)
        if fci_data is not None:
            for i in np.arange(self.height):
                for j in np.arange(self.width):
                    cellobj = self.grid[i, j][1]
                    cellobj.veg_capacity = fci_data[cellobj.cell_id] / self.gis.grid_fci_averages[cellobj.cell_id]
    # }}}

    # {{{ update_gis
    def update_gis(self, params, date):
        """ Update all GIS-driven state for the given date and parameter set. """
        self.update_vegetation(params, date)
    # }}}

    def step(self, params, time):
        """ Handler for the worldstep event type.  This causes all cells that contain agents to
            step forward with respect to foraging and disease propagation. """
        # calculate the time since the last step occurred
        dt = time.current_step_duration()

        if dt==0:
            print(f'Already stepped! {time.current_time}')
            exit()
            return

        # if the live cell set has not yet been created, populate it via a traversal
        # of the world.  after this initial creation it is maintained by the move()
        # function.
        # TODO: possibly handle this in place_agent if we know that agents are ONLY
        #       ever placed in the grid by either place_agent or move calls.
        if self.live_cells is None:
            self.live_cells = set([])
            for i in np.arange(self.height):
                for j in np.arange(self.width):
                    (agents, _) = self.grid[i, j]
                    if len(agents) > 0:
                        self.live_cells.add((i, j))

        for (i, j) in self.live_cells:
            # get the agents in each cell and the cell object
            (agents, cell_obj) = self.grid[i, j]

            # collect all of the herds colocated here
            herds = []
            for agent in agents:
                if isinstance(agent, A.Herdsman):
                    herds.append(agent.herd)

            # if we have at least one herd here, eat and propagate diseases
            if len(herds) > 0:
                n_animals = sum([herd.size() for herd in herds])

                # make sure we don't just have empty herds
                if n_animals > 0:
                    # foraging
                    food_available = cell_obj.forage(n_animals, dt)
                    for herd in herds:
                        # distribute food proportionally
                        frac_food = (herd.size()/n_animals) * food_available
                        herd.feed(frac_food, dt)

                    # disease spread
                    for d in self.diseases:
                        self.diseases[d].step(herds, time)

        # Record the time now to calculate the duration of time until the
        # next step
        time.last_timestep = time.current_time

    ### NOTE: this is equivalent to the code above except in the order by which
    ###       cells with agents are visited due to the order that the set iterator
    ###       traverses the set of live cells.  Keep this in mind when comparing the
    ###       two.
    def old_step(self, params, time):
        dt = time.current_step_duration()

        if dt==0:
            print(f'Already stepped! {time.current_time}')
            exit()
            return

        # iterate over all grid cells
        for i in np.arange(self.height):
            for j in np.arange(self.width):
                # get the agents in each cell and the cell object
                (agents, cell_obj) = self.grid[i, j]

                # collect all of the herds colocated here
                herds = []
                for agent in agents:
                    if isinstance(agent, A.Herdsman):
                        herds.append(agent.herd)

                # if we have at least one herd here, eat and propagate diseases
                if len(herds) > 0:
                    # feeding
                    n_animals = sum([herd.size() for herd in herds])

                    # make sure we don't just have empty herds
                    if n_animals > 0:
                        food_available = cell_obj.forage(n_animals, dt)
                        for herd in herds:
                            # distribute food proportionally
                            frac_food = (herd.size()/n_animals) * food_available
                            herd.feed(frac_food, dt)

                        # disease spread
                        for d in self.diseases:
                            self.diseases[d].step(herds, time)

        time.last_timestep = time.current_time
                
# }}}
