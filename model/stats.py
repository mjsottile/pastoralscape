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
import os.path
import numpy as np
import matplotlib.pyplot as plt
import h5py
import model.disease as D

class IncompatibleParameters(Exception):
  pass

class Tracker:
  """ 
  Tracker class to record results of model as it runs.  This class uses a
  DataArchive class to record the results of a run to disk.
  """
  def __init__(self, model_state, initial_animal_count):
    self.model_state = model_state
    self.occupants = []
    self.occupant_totals = np.zeros((model_state.world.height, model_state.world.width))
    self.vaccine_decisions = {}
    self.vaccinated = {}
    self.deaths = {}
    self.herdsize = []
    self.avg_health = []
    self.avg_ages = []
    self.disease_breakdown = []
    self.total_animals = initial_animal_count
    self.total_distance = 0.0

  def record_birth(self):
    self.total_animals += 1

  def record_distance(self, d):
    self.total_distance += d

  def record_occupancy(self, location, num, agent_type, time):
    """
    Record the number of agents of a given agent type that
    are present in grid cell location=(i,j) at the given time.
    """
    self.occupants.append((time, num, agent_type, location))
    self.occupant_totals[location] += num

  def vaccinate_decision(self, disease, decision, time):
    """
    Record the decision value for a given disease at some time.
    """
    if disease in self.vaccine_decisions:
      self.vaccine_decisions[disease].append((decision, time))
    else:
      self.vaccine_decisions[disease] = [(decision, time)]

  def record_death(self, cause, time):
    """
    Recurd a death at a specific time for a given cause.
    """
#    print(f"death: {cause} at {time}")
    if cause in self.deaths:
      self.deaths[cause].append(time)
    else:
      self.deaths[cause] = [time]

  def record_herd(self, herd, time):
    day_of_epoch = time.day_of_epoch()
    self.herdsize.append((herd.size(), day_of_epoch))
    healths = [a.health for a in herd.animals]
    ages = [a.age(time) for a in herd.animals]

    for disease in self.model_state.diseases:
      count = sum([1 for a in herd.animals if a.diseases[disease] == D.SIRV.V])
      if disease in self.vaccinated:
        self.vaccinated[disease].append((count, herd.size(), day_of_epoch))
      else:
        self.vaccinated[disease] = [(count, herd.size(), day_of_epoch)]
    self.avg_health.append((np.average(np.array(healths)), day_of_epoch))
    self.avg_ages.append((np.average(np.array(ages)), day_of_epoch))

  def check_redundant_run(self, param_string, seed, filename):
    """ Check if we are trying to do a run for a seed that has already
        been run.  If the parameters do not match, fatal error since we have
        chosen an incompatible output file that already has data from a different
        parameter set. """
    if os.path.isfile(filename):
      f = h5py.File(filename, 'r')

      # check if param string matches
      f_pstr = f['params']['yaml'][()]
      if param_string != f_pstr:
        print("FATAL ERROR: parameter string does not match output file.")
        f.close()
        raise IncompatibleParameters

      if str(seed) in f:
        return True
    else:
      return False

  def to_archive(self, param_string, seed, filename):
    """ Emit the stats object to an archive file.  The seed is required to distinguish
        runs within an ensemble from the same base parameter set. """
    if os.path.isfile(filename):
      f = h5py.File(filename, 'r+')

      # check if param string matches
      f_pstr = f['params']['yaml'][()]
      if param_string != f_pstr:
        print("FATAL ERROR: parameter string does not match output file.")
        f.close()
        exit()

      seed_group = f.create_group(str(seed))
    else:
      f = h5py.File(filename, "w")

      # archive parameters and seed
      grp = f.create_group('params')
      grp.create_dataset('yaml', data=param_string)
      seed_group = f.create_group(str(seed))

    # record scalar counts for animals and distance
    print(f'd={self.total_distance}  n={self.total_animals}')
    seed_group.create_dataset('total_distance', data=self.total_distance)
    seed_group.create_dataset('total_animals', data=self.total_animals)

    # store GIS data to translate cell IDs to latlon
    grp = seed_group.create_group('gis')
    dset_id = grp.create_dataset('id', (self.model_state.world.height, self.model_state.world.width), dtype='i')
    dset_lat = grp.create_dataset('latitude', (self.model_state.world.height, self.model_state.world.width), dtype='f')
    dset_lon = grp.create_dataset('longitude', (self.model_state.world.height, self.model_state.world.width), dtype='f')

    for i in range(self.model_state.world.height):
      for j in range(self.model_state.world.width):
        (_, cell_obj) = self.model_state.world.grid[i, j]
        dset_id[i, j] = cell_obj.cell_id
        dset_lat[i, j] = cell_obj.latitude
        dset_lon[i, j] = cell_obj.longitude

    # store cell occupancy statistics
    grp = seed_group.create_group('world')
    grp.create_dataset('occupancy', data=self.occupant_totals)

    ## store time series data
    # vaccine decisions
    grp = seed_group.create_group('vaccination')
    for disease in self.vaccine_decisions:
      grp.create_dataset(disease, data=np.fliplr(np.array(self.vaccine_decisions[disease])))

    # vaccinated counts
    grp = seed_group.create_group('vaccinated')
    for disease in self.vaccinated:
      grp.create_dataset(disease, data=np.fliplr(np.array(self.vaccinated[disease])))

    # livestock
    grp = seed_group.create_group('livestock')
    grp.create_dataset('herdsize', data=np.fliplr(np.array(self.herdsize)))
    grp.create_dataset('avg_health', data=np.fliplr(np.array(self.avg_health)))
    grp.create_dataset('avg_age', data=np.fliplr(np.array(self.avg_ages)))

    # deaths
    grp = seed_group.create_group('deaths')
    for cause in self.deaths:
      dset_cause = grp.create_dataset(cause, (len(self.deaths[cause]), 1))
      for (i, t) in enumerate(self.deaths[cause]):
        dset_cause[i, 0] = t

    f.close()
