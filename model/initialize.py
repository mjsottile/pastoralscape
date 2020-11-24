import numpy as np
import numpy.random as rand
import model.world as W
import model.agents as A
import model.disease as D
import model.livestock as L
import model.social as S
import model.gis as G

verbose = False

def logmsg(s):
  if verbose:
    print(s)
  else:
    pass

def generate_n_points(n,w,h,sep):
    done = False
    while not done:
        xs = rand.randint(0,w,size=n)
        ys = rand.randint(0,h,size=n)
        done = True
        for i in range(n-1):
            for j in range(i+1,n):
                d = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                if d < sep:
                    done = False
    return [(xs[i],ys[i]) for i in range(n)]

def initialize_objects(model_params, model_state, individual_params):
  # get reference to setup subdictionary to simplify code
  setup_params = model_params['model']['setup']
  
  # seed the generator
  rand.seed(setup_params['seed'])

  # create GIS object
  gis = G.GISData(model_params)

  # create world
  cell_gen = lambda x: W.GridSpace(model_params)
  w = W.World(gis, setup_params['width'], setup_params['height'], cell_gen)

  # load real world coordinates
  coords = gis.coordinates
  cell_ids = coords['ID'].values.reshape((setup_params['height'], setup_params['width']))
  cell_lon = coords['Long'].values.reshape((setup_params['height'], setup_params['width']))
  cell_lat = coords['Lat'].values.reshape((setup_params['height'], setup_params['width']))
  id_to_index = {}
  for i in range(setup_params['height']):
    for j in range(setup_params['width']):
      id_to_index[cell_ids[i,j]] = (i,j)
      (_,cell_obj) = w.grid[i,j]
      cell_obj.cell_id = cell_ids[i,j]
      cell_obj.longitude = cell_lon[i,j]
      cell_obj.latitude = cell_lat[i,j]

  # store ID->index map in world for later use when loading monthly data
  w.id_to_index = id_to_index

  # read static location data
  village_table = gis.villages
  water_bodies = gis.waterbodies

  # create n villages with a desired separation
  positions = []
  for i in village_table['ID']:
    positions.append(w.id_to_index[i])
    
  villages = []
  # positions = generate_n_points(setup_params['n_villages'], setup_params['width'],
  #                               setup_params['height'], setup_params['village_sep'])
  for pos in positions:
    v = W.Village(model_params)
    logmsg(f"Creating village at {pos}")
    #print(pos)
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

  # create social network
  social_net = S.SocialNetwork(setup_params['n_hoh']+setup_params['n_herdsmen'])

  # create heads of household
  v = 0
  aset_hoh = A.AgentSet("Heads of Household", model_params)

  # round robin allocation to villages
  for _ in range(setup_params['n_hoh']):
    hoh = A.HeadOfHousehold(social_net, model_state, model_params, individual_params, w)
    aset_hoh.add(hoh)
    villages[v].add_individual(hoh)
    logmsg(f'HeadOfHousehold {hoh.id} at village {v} {villages[v].location}')
    v = (v + 1) % len(villages)

  # connect heads of household together
  for i in range(aset_hoh.size()):
    for j in range(i,aset_hoh.size()):
      social_net.add_relationship(aset_hoh.agents[i], aset_hoh.agents[j], 1.0, 1.0)

  # round robin allocation of herdsmen to heads of household, with random selection
  # inside villages
  v = 0
  aset_herdsmen = A.AgentSet("Herdsmen", model_params)

  for _ in range(setup_params['n_herdsmen']):
    # generate a person
    person = A.Herdsman(social_net, model_state, model_params, individual_params, w)

    # find a head of household in village v
    hoh = villages[v].get_individual(rand.randint(villages[v].popsize()))

    # connect the two
    person.set_head_of_household(hoh)
    hoh.add_herdsman(person)

    # create an empty herd
    herd = L.Herd(model_params)
    person.set_herd(herd)

    # set the village
    person.set_home_village(villages[v])
    w.place_agent(person, villages[v].location)

    aset_herdsmen.add(person)

    logmsg(f'Herdsman {person.id} at village {v} {villages[v].location} with HoH {hoh.id}')

    v = (v + 1) % len(villages)

  ##
  ## create animals
  ##
  for _ in range(setup_params['n_animals']):
    # pick a herdsman
    owner = aset_herdsmen.get(rand.randint(aset_herdsmen.size()))

    # determine gender
    if rand.rand() < setup_params['pct_bull']:
      g = L.Gender.MALE
    else:
      g = L.Gender.FEMALE
    
    # create animal
    a = L.Animal(g, owner.herd, model_params)

    # set age of animal
    a.age = max(setup_params['min_age'],
                setup_params['age_sigma']*rand.randn() + (model_params['livestock']['death_mu'] / 2))

    # set disease state
    for disease in setup_params['pct_vaccinated']:
      if rand.rand() < setup_params['pct_vaccinated'][disease]:
        a.diseases[disease] = D.SIRV.V
      else:
        a.diseases[disease] = D.SIRV.S

    logmsg(f'Animal {g} {a.age} => {owner.id} : {a.diseases}')

    # add animal to herd
    owner.herd.add(a)

  ##
  ## create diseases
  ##
  diseases = {}
  for d in model_params['disease']:
    diseases[d] = D.Disease(d, model_params)
    w.add_disease(diseases[d])

  return (w, villages, gis, aset_hoh, aset_herdsmen, diseases)
