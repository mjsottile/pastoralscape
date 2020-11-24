import numpy as np
import model.util as U
import model.agents as A
import model.gis as G
import datetime
import sys
import math

# {{{ grid space
class GridSpace:
    """ 
    A grid space in the world has some vegetation state as well as presence of water.
    """

    # {{{ constructor
    def __init__(self, params):
        # FCI ranges from 0 to 100
        self.fci = 0.0

        # NDVI ranges from -1.0 to 1.0
        self.mean_ndvi = 0.0
        self.mean_ndvi_alltime = 0.0

        self.params = params
        self.location = None

        self.has_water = False

        self.using_fci = True
    # }}}

    # {{{ forage
    def forage(self, nanimals, time):
        food_required = nanimals * self.params['livestock']['eat'] * time.step_size_weeks()

        if self.using_fci:
            # if above average, we get all the food we need.  otherwise, a fraction.
            food_obtained = food_required * min(self.veg_capacity, 1.0)
        else:
            print('FCI FORAGING IS ALL THAT IS IMPLEMENTED')
            sys.exit(1)
            # food_obtained = min(self.veg_capacity, food_required)
            # self.veg_

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
    """
    def __init__(self, params):
        super().__init__(params)
        self.fixed_individuals = []

    def add_individual(self, i):
        self.fixed_individuals.append(i)

    def popsize(self):
        return len(self.fixed_individuals)

    def get_individual(self, i):
        return self.fixed_individuals[i]
# }}}

# {{{ World 
class World:
    def __init__(self, gis, w, h, d):
        """ 
        create a 2d world of width w and height h kilometers.  d is a default
        cell object constructor.  this may be a lambda like:
            
        d = lambda x : GridSpace(0.0, 100.0, {'eat':1.0, 'constant_growth'=0.0, 'precip_growth'=0.1}) 

        indexing of world cells is (lat,lon)
        """
        self.width = w
        self.height = h

        # GIS data store
        self.gis = gis

        # grid cell is a pair:
        # - a list of agents residing there
        # - a grid cell object
        self.grid = np.empty((h,w), dtype=object)
        
        for i in range(h):
            for j in range(w):
                self.grid[i,j] = ([], d((i,j)))
                self.grid[i,j][1].location = (i,j)

        # diseases are propagated by the world, so we must track them
        self.diseases = {}

        # GIS related state variables
        self.last_gis_update = None
        self.id_to_index = None
        self.mean_fci = None
        self.neighbor_cache = {}

    def add_disease(self, d):
        self.diseases[d.name] = d

    def set_cell(self, pos, o):
        """
        Set the cell object for a grid space to the given object.
        The agent set at the cell is reset to empty.
        The location attribute of the cell object is set to the
        position.
        """
        self.grid[pos] = ([], o)
        self.grid[pos][1].location = pos

    def move(self, a, pos):
        """
        Move the agent a from its current location to the coordinate
        (i,j).
        """
        
        # where is agent now?
        old_pos = a.location

        # remove agent from current location resident set
        (residents,_) = self.grid[old_pos]
        residents.remove(a)

        # add agent to new location resident set
        (residents,_) = self.grid[pos]
        residents.append(a)

        # set agent location to new coordinates
        a.location = pos
        
    def neighborhood(self, pos, r):
        """
        Find coordinates for all cells around pos that are within
        the radius r.  Return a list of (coordinate, distance) pairs.
        """
        if pos not in self.neighbor_cache:
            neighbors = []
            for i in np.arange(max(0,pos[0]-r), min(self.height, pos[0]+r)):
                for j in np.arange(max(0,pos[1]-r), min(self.width, pos[1]+r)):
                    # ignore (i,j) == pos
                    if (i != pos[0]) or (j != pos[1]):
                        d = math.hypot(pos[0]-i, pos[1]-j)
                        #d = U.dist(pos, (i,j))
                        if d <= r:
                            neighbors.append(((int(i), int(j)), d))
            self.neighbor_cache[pos] = neighbors
        else:
            neighbors = self.neighbor_cache[pos]
        return neighbors

    def place_agent(self, a, pos):
        a.location = pos
        (residents,_) = self.grid[pos]
        residents.append(a)

    def veg_map(self):
        v = np.zeros((self.height, self.width))
        for i in np.arange(self.height):
            for j in np.arange(self.width):
                v[i,j] = self.grid[i,j][1].veg_capacity
        return v

    def update_vegetation(self, params, time):
        # cache FCI data since it contains all time points - no need to read
        # more than once
        if self.mean_fci is None:
            # pre-calculate mean FCI, excluding dates with null values
            self.mean_fci = self.gis.fci['FCI'][self.gis.fci['FCI'].notnull()].mean()

        # load monthly GIS data
        gis_data = self.gis.get_date(time.current_time.year, time.current_time.month)

        # map GIS data onto cells, calculate mean NDVI for this time period over
        # the world.
        self.world_mean_ndvi = 0.0
        for _,row in gis_data.iterrows():
            cell_id = row['ID']
            self.grid[self.id_to_index[cell_id]][1].mean_ndvi = row['mean_ndvi']
            self.world_mean_ndvi += row['mean_ndvi']
            self.grid[self.id_to_index[cell_id]][1].mean_precip = row['mean_precip']
        self.world_mean_ndvi = self.world_mean_ndvi / (self.width * self.height)

        # get FCI for current month
        fci_now = self.gis.fci.query(f'Year=={time.current_time.year} and Month=={time.current_time.month}')['FCI'].values[0]
        for i in np.arange(self.height):
            for j in np.arange(self.width):
                cellobj = self.grid[i,j][1]

                # for now: set veg_capacity to fci * (cell_ndvi_mean_alltime / world_ndvi_mean_now) for current month
                cellobj.veg_capacity = (fci_now / self.mean_fci) * (cellobj.mean_ndvi_alltime / self.world_mean_ndvi)

    def step(self, params, tracker, time):
        if (self.last_gis_update is None) or (time.current_time.month != self.last_gis_update.month):
            # copy the current time to the last update time
            self.last_gis_update = datetime.date(time.current_time.year, time.current_time.month, time.current_time.day)
            self.update_vegetation(params, time)
        
        occupancy = np.ndarray(shape=(self.height, self.width))

        # iterate over all grid cells
        for i in np.arange(self.height):
            for j in np.arange(self.width):
                # get the agents in each cell and the cell object
                (agents, cellobj) = self.grid[i, j]
                occupancy[i,j] = len(agents)

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
                        food_available = cellobj.forage(n_animals, time)
                        for herd in herds:
                            # distribute food proportionally
                            frac_food = (herd.size()/n_animals) * food_available
                            herd.feed(frac_food, time)

                        # disease spread
                        for d in self.diseases:
                            self.diseases[d].step(herds, tracker, time)

        tracker.record_occupancy(occupancy)
                
# }}}