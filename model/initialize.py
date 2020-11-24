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
import dateutil.relativedelta as RD
import numpy as np
import numpy.random as rand
import model.world as W
import model.agents as A
import model.disease as D
import model.livestock as L
import model.social as S
import model.gis as G
import model.events as E
import model.time as T

verbose = False

# {{{ logmsg
def logmsg(s):
  if verbose:
    print(s)
  else:
    pass
# }}}

# {{{ calculate cell boundaries given cell centers.
# boundaries used for bisection search to map continuous
# latlon to integer cell coordinates in world grid.
def boundaries(centers):
  n = len(centers)
  b = np.ndarray(shape=(n+1,))
  for i in np.arange(n-1):
    b[i+1] = (centers[i]+centers[i+1])/2.0
  b[0] = centers[0]-(b[1]-centers[0])
  b[n] = centers[n-1]+(centers[n-1]-b[n-1])
  return b
# }}}

# {{{ initialize_objects
def initialize_objects(model_params, model_state, individual_params):
  # get reference to setup subdictionary to simplify code
  setup_params = model_params['model']['setup']

  # create GIS object
  gis = G.GISData(model_params)
  model_state.gis = gis

  # create world
  cell_gen = lambda x: W.GridSpace(model_params)
  w = W.World(model_state, setup_params['width'], setup_params['height'], cell_gen)
  model_state.world = w

  # load real world coordinates.  assumes the cells are ordered in either a
  # row major or column major serial format, not random.
  coords = gis.coordinates
  cell_ids = coords['ID'].reshape((setup_params['height'], setup_params['width']))
  cell_lon = coords['Long'].reshape((setup_params['height'], setup_params['width']))
  cell_lat = coords['Lat'].reshape((setup_params['height'], setup_params['width']))
  id_to_index = {}
  for i in range(setup_params['height']):
    for j in range(setup_params['width']):
      id_to_index[cell_ids[i,j]] = (i,j)
      (_,cell_obj) = w.grid[i,j]
      cell_obj.cell_id = cell_ids[i,j]
      cell_obj.longitude = cell_lon[i,j]
      cell_obj.latitude = cell_lat[i,j]

  # determine whether the first dimension of the grid index corresponds
  # to latitude or longitude by checking the first element of the first
  # and second columns.
  w.first_dim_lat = False
  if cell_lat[0,0] == cell_lat[0,1]:
    w.first_dim_lat = True

  if not w.first_dim_lat:
    w.lat_boundaries = boundaries(cell_lat[0,:])
    w.lon_boundaries = boundaries(cell_lon[:,0])
  else:
    w.lat_boundaries = boundaries(cell_lat[:,0])
    w.lon_boundaries = boundaries(cell_lon[0,:])

  # store ID->index map in world for later use when loading monthly data
  w.id_to_index = id_to_index

  # read static location data
  village_table = gis.villages
  water_bodies = gis.waterbodies

  # create n villages with a desired separation
  positions = []
  for i in village_table['ID']:
    positions.append((i, w.id_to_index[i]))
    
  villages = []
  for v_id,pos in positions:
    v = W.Village(model_params)

    # if we have paths originating at this village ID, add them
    # to the village.
    if v_id in gis.paths:
      for path in gis.paths[v_id]:
        p = W.Path(path)
        v.add_path(p)

    logmsg(f"Creating village at {pos}")

    # need to copy gridspace attributes before replacing the object.
    (_, cell_obj) = w.grid[pos]

    v.latitude = cell_obj.latitude
    v.longitude = cell_obj.longitude
    v.cell_id = cell_obj.cell_id
    w.set_cell(pos, v)
    villages.append(v)

  # setup water features
  positions = []
  for i in water_bodies['ID']:
    positions.append(w.id_to_index[i])
    #print(w.id_to_index[i])

  for pos in positions:
    (_,cellobj) = w.grid[pos]
    cellobj.has_water = True

  # create social network : two people per household - the HoH and the herdsman
  model_state.social_net = S.SocialNetwork(setup_params['n_hoh']*2)

  # create heads of household
  v = 0
  aset_hoh = A.AgentSet("Heads of Household", model_params)
  aset_herdsmen = A.AgentSet("Herdsmen", model_params)

  # round robin allocation to villages
  for _ in range(setup_params['n_hoh']):
    # create head of household
    hoh = A.HeadOfHousehold(model_state, model_params, individual_params)
    aset_hoh.add(hoh)
    villages[v].add_individual(hoh)
    logmsg(f'HeadOfHousehold {hoh.id} at village {v} {villages[v].location}')

    # create herdsman who works for head of household
    herdsman = A.Herdsman(model_state, model_params, individual_params)
    herdsman.set_head_of_household(hoh)
    hoh.add_herdsman(herdsman)
    herdsman.set_home_village(villages[v])
    w.place_agent(herdsman, villages[v].location)

    # create their herd
    herd = L.Herd(model_state, model_params)
    herdsman.set_herd(herd)

    aset_herdsmen.add(herdsman)
    logmsg(f'Herdsman {herdsman.id} at village {v} {villages[v].location} with HoH {hoh.id}')

    v = (v + 1) % len(villages)

  # connect heads of household together
  for i in range(aset_hoh.size()):
    for j in range(i,aset_hoh.size()):
      model_state.social_net.add_relationship(aset_hoh.agents[i], aset_hoh.agents[j], 1.0, 1.0)

  ##
  ## create animals
  ##

  # first, calculate most recent immunization date prior to model start
  most_recent_vaccination = T.most_recent(setup_params['start_date'], model_params['agents']['vaccination_schedule'])
  
  for _ in range(setup_params['n_animals']):
    # pick a herdsman
    owner = aset_herdsmen.get(rand.randint(aset_herdsmen.size()))

    # determine gender
    if rand.rand() < setup_params['pct_bull']:
      g = L.Gender.MALE
    else:
      g = L.Gender.FEMALE
    
    ## set age of animal
    # calculate lifespan of animal
    lifespan = model_params['livestock']['death_sigma'] * rand.randn() + model_params['livestock']['death_mu']

    # calculate length of acceptable age range
    lrange = lifespan - (setup_params['min_age'] + setup_params['min_remain'])

    # age is some random place uniformly distributed in that age range
    age = rand.rand() * lrange + setup_params['min_age']

    # end date relative to age at start of model epoch
    end_date = setup_params['start_date'] + RD.relativedelta(days=lifespan-age)

    # calculate birth date based on age at start of epoch
    birthdate = setup_params['start_date'] - RD.relativedelta(days=age)

    # create animal
    a = L.Animal(g, birthdate, owner.herd, model_state, model_params)

    # schedule end
    model_state.event_queue.add_event(end_date, E.Event.CULL_OLDAGE, a)

    # if animal mature, set fertile flag.  otherwise schedule maturity event
    if g == L.Gender.FEMALE:
      if age >= model_params['livestock']['maturity']:
        a.fertile = True
      else:
        f_date = setup_params['start_date'] + RD.relativedelta(days=model_params['livestock']['maturity']-age)
        model_state.event_queue.add_event(f_date, E.Event.LIV_FERTILE, a)

    # set disease state
    for disease in setup_params['pct_vaccinated']:
      if rand.rand() < setup_params['pct_vaccinated'][disease]:
        #
        # if animal may be vaccinated, sample the date when the pre-existing vaccination
        # would have worn off.  if it is after the start of the model, set the animal state
        # to V and schedule the wearoff event.  otherwise, assume it has already worn off
        # and set to the S state.
        #
        if 'wearoff' in model_params['disease'][disease]:
          wparams = model_params['disease'][disease]['wearoff']
          wearoff_days = wparams['sigma']*rand.randn() * wparams['mu']
          wearoff_date = most_recent_vaccination + RD.relativedelta(days=int(wearoff_days))
          if wearoff_date > setup_params['start_date']:
            model_state.event_queue.add_event(wearoff_date, E.Event.WEAROFF, (disease, a))
            a.diseases[disease] = D.SIRV.V
          else:
            a.diseases[disease] = D.SIRV.S
        else:
          #
          # if no wearoff period, just set to V
          #
          a.diseases[disease] = D.SIRV.V
      else:
        a.diseases[disease] = D.SIRV.S

    logmsg(f'Animal {g} {a.birthday} => {owner.id} : {a.diseases}')

    # add animal to herd
    owner.herd.add(a)

  ##
  ## create diseases
  ##
  diseases = {}
  for d in model_params['disease']:
    diseases[d] = D.Disease(d, model_state, model_params)
    w.add_disease(diseases[d])

  model_state.hoh_set = aset_hoh
  model_state.herdsmen_set = aset_herdsmen
  model_state.diseases = diseases

  return (aset_hoh, aset_herdsmen, diseases)
# }}}
