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
import warnings
import argparse
import yaml
#import cProfile
try:
  from yaml import CLoader as Loader
except ImportError:
  from yaml import Loader
import datetime as date
import numpy.random as rand
import model.initialize as I
import model.time as T
import model.disease as D
import model.stats as S
import model.events as E
import model.state as ST

## command line
parser = argparse.ArgumentParser(description="PastoralScape")
parser.add_argument('-p', '--params', help='Params file.', required=True)
parser.add_argument('-o', '--output', help='Output HDF5 file.', required=True)
args = parser.parse_args()

def main():
  # load parameter file from disk as YAML file to dictionary
  with open(args.params) as f:
    paramfile_string = f.read()
    model_params = yaml.load(paramfile_string, Loader=Loader)

  for seed in range(model_params['model']['seed_min'], model_params['model']['seed_max']):
    model_params['model']['setup']['seed'] = seed
    print(f'running seed={seed}')

    # create the model state: this is data that applies to many objects
    # and may change, so we pass a reference to it.
    model_state = ST.ModelState(model_params)

    # default parameters for individuals.
    default_individual_params = {'ising': {}}
    for disease, dparams in model_params['model']['setup']['ising'].items():
      default_individual_params['ising'][disease] = {'f': dparams['f_initial']}

    # create time tracking object
    # NOTE: all time steps are based on dt in days.
    t_start = model_params['model']['setup']['start_date']
    t_step = date.timedelta(days=model_params['model']['setup']['time_delta_days'])
    t_end = model_params['model']['setup']['end_date']
    time = T.Time(t_start, t_step)

    # initialize the event queue with known lo/hi time boundaries
    eq = E.EventQueue(lo_time=t_start, hi_time=t_end)
    model_state.event_queue = eq

    # initialize model objects.  note: event queue must exist and be in the model
    # state at this point so newly created animals can have their demise scheduled
    (hoh, hmen, diseases) = I.initialize_objects(model_params, model_state, default_individual_params)

    # pre-load GIS data for time period
    model_state.gis.load(time.current_time.year, time.current_time.month, 
                         t_end.year, t_end.month)

    # create a tracker to record model data over the run
    model_state.tracker = S.Tracker(model_state, model_params['model']['setup']['n_animals'])
    if model_state.tracker.check_redundant_run(paramfile_string, seed, args.output):
      print(f"redundant seed: skipping")
      continue

    # distribute mean_ndvi to all cells
    for id in model_state.gis.mean_ndvi_alltime:
      idx = model_state.world.id_to_index[id]
      cellobj = model_state.world.grid[idx][1]
      cellobj.mean_ndvi_alltime = model_state.gis.mean_ndvi_alltime[id]

    ###### Create initial events
    
    ## load up all of the timestep events
    events = time.enumerate_step_events(t_end)
    for (event_time, event_type) in events:
      eq.add_event(event_time, event_type)
      eq.add_event(event_time, E.Event.INFECTION)

    ## set up monthly GIS updates
    update_times = time.enumerate_month_starts(t_start, t_end)
    
    # add start date - it is excluded from the enumeration in case
    # it doesn't fall at the start of a month.
    update_times.append(t_start)
    for event_time in update_times:
      eq.add_event(event_time, E.Event.GISUPDATE)

    ## set up periodic vaccinations
    for month_day in model_params['agents']['vaccination_schedule']:
      vaccine_times = time.enumerate_annual_events(month_day[0], month_day[1], t_end)
      for event_time in vaccine_times:
        eq.add_event(event_time, E.Event.VACCINATE)

    ###### Main loop
    current_event = eq.next_event()
    while current_event is not None:
      #print(time.current_time)
      (event_time, event_type, subject) = current_event

      # move time forward in the time tracker
      time.set_time(event_time)

      if event_type == E.Event.GISUPDATE:
        # update GIS data
        model_state.world.update_gis(model_params, event_time)

      elif event_type == E.Event.MOVEMENT:
        # handle a movement event for one agent
        subject.handle_event(time, event_type)

      elif event_type == E.Event.LIV_BIRTH:
        subject.handle_event(time, event_type)

      elif event_type == E.Event.LIV_FERTILE:
        subject.handle_event(time, event_type)

      elif event_type == E.Event.WORLDSTEP:
        ## TODO: should only step world and herd health.  not breeding and decisions
        # step the world forward
        model_state.world.step(model_params, time)

      elif event_type == E.Event.AGENTSTEP:
        # step the heads of household forward
        hoh.step(time)

        # step the herdsmen forward
        hmen.step(time)

        # record statistics about the agents and the world
        hmen.record(time)
        hoh.record(time)

      elif event_type == E.Event.VACCINATE:
        # head of household disease decisions
        hoh.handle_event(time, event_type)

      elif event_type == E.Event.CULL_OLDAGE:
        # event corresponding to a single animal expiring due to old age.
        if subject.active:
          # Congratulations little cow, disease and malnutrition didn't get you.
          model_state.tracker.record_death("age", time.day_of_epoch())
          subject.herd.cull(subject)

      elif event_type == E.Event.WEAROFF:
        # vaccination wearoff, V -> S transition.  only consider animals that 
        # are still active in case the animal left the simulation for some other 
        # cause before now.
        (disease, animal) = subject
        if animal.active:
          animal.set_disease_state(disease, D.SIRV.S)

      elif event_type == E.Event.INFECTION:
        # TODO: currently only allows one infection per event.  may explore allowing
        #       more than one infection
        # TODO: currently does not use GIS data (e.g., water sources) to add spatial
        #       factors in likelihood of infection.  fine for now, but may add later.

        # randomly infect an animal
        for d in diseases:
          # sample the disease to see if an infection event occurs right now.
          infect = diseases[d].sample_infection(time)
          if infect:
            # if an infection event occurs, pick a herd at random and
            # an animal in the herd at random.
            herd = hmen.get(rand.randint(hmen.size())).herd
            if herd.size() > 0:
              animal = herd.animals[rand.randint(herd.size())]
              animal.set_disease_state(d, D.SIRV.I)

      else:
        print("Unsupported event: "+str(current_event))
        sys.exit()

      # pop next event
      current_event = eq.next_event()

    model_state.tracker.to_archive(paramfile_string, seed, args.output)

# run block of code and catch warnings
with warnings.catch_warnings():
  # ignore all caught warnings
  warnings.filterwarnings("ignore")
  main()
  #cProfile.run('main()',sort='tottime')
