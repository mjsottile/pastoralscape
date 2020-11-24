import json
import model.initialize as I
import model.time as T
import datetime as date
import numpy.random as rand
import model.disease as D
import model.stats as S
import matplotlib.pyplot as plt
#import cProfile
import warnings
import argparse
import numpy as np

## command line
parser = argparse.ArgumentParser(description="Livestock model driver")
parser.add_argument('-p','--params',help='Params file.',required=True)
parser.add_argument('-o','--outputdir',help='Directory to dump output files.',required=True)
args = parser.parse_args()

def main():
  # load parameter file from disk as JSON file to dictionary
  with open(args.params) as f:
    model_params = json.load(f)

  first_iter = True

  total_animal_history = []
  decision_history = {'rvf':[], 'cbpp':[]}

  for seed in range(model_params['model']['seed_min'], model_params['model']['seed_max']):
    model_params['model']['setup']['seed'] = seed

    # create the model state: this is data that applies to many objects
    # and may change, so we pass a reference to it.
    model_state = {'ising': {'rvf': {'f_public': 1.0},
                            'cbpp': {'f_public': 1.0}}}

    # default parameters for individuals.
    default_individual_params = {'ising': {'rvf': {'f': 0.1},
                                          'cbpp': {'f': 0.1}}}

    (w,v,gis,hoh,hmen,diseases) = I.initialize_objects(model_params, model_state, default_individual_params)

    tracker = S.StatTracker(hmen)

    # NOTE: all time steps are based on dt in weeks.
    time = T.Time(date.date.fromisoformat(model_params['model']['setup']['start_date']), 
                  date.timedelta(days=model_params['model']['setup']['time_delta_days']))

    end_date = date.date.fromisoformat(model_params['model']['setup']['end_date'])

    # pre-load GIS data for time period
    gis.load(time.current_time.year, time.current_time.month, 
            end_date.year, end_date.month)

    # distribute mean_ndvi to all cells
    for id in gis.mean_ndvi_alltime:
      idx = w.id_to_index[id]
      cellobj = w.grid[idx][1]
      cellobj.mean_ndvi_alltime = gis.mean_ndvi_alltime[id]

    # main time stepping loop
    while time.current_time < end_date:
      #print(time.current_time)
      # step the world forward
      w.step(model_params, tracker, time)

      # step the heads of household forward
      hoh.step(tracker, time)

      # step the herdsmen forward
      hmen.step(tracker, time)

      # randomly infect an animal
      for d in diseases:
        herd = hmen.get(rand.randint(hmen.size())).herd
        if herd.size() > 0:
          animal = herd.animals[rand.randint(herd.size())]
          animal.set_disease_state(d, D.SIRV.I, time)

      # step time forward
      time.step_forward()

      tracker.step()

    total_animal_history.append(tracker.total_animals)
    decision_history['rvf'].append(tracker.f_history['rvf'])
    decision_history['cbpp'].append(tracker.f_history['cbpp'])
    tracker.print_stats(first_iter)
    first_iter = False

  tot_file = args.outputdir+'/total_animals.csv'
  dec_file_rvf = args.outputdir+'/rvf_decisions.csv'
  dec_file_cbpp = args.outputdir+'/cbpp_decisions.csv'

  np.savetxt(tot_file, np.array(total_animal_history), delimiter=",")
  np.savetxt(dec_file_rvf, np.array(decision_history['rvf']), delimiter=",")
  np.savetxt(dec_file_cbpp, np.array(decision_history['cbpp']), delimiter=",")

# run block of code and catch warnings
with warnings.catch_warnings():
  # ignore all caught warnings
  warnings.filterwarnings("ignore")
  main()

#cProfile.run('main()')
