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
import argparse
import h5py
import numpy as np
import yaml
from yaml import Loader

## command line
parser = argparse.ArgumentParser(description="PastoralScape Reporter")
parser.add_argument('-i', '--input', help='Input HDF5 file.', required=True)
parser.add_argument('-c', '--header', help='Print row of column headers.', action='store_true', required=False, default=False)
parser.add_argument('-p', '--params', help='Parameters from config file to dump.  Colon separated dot paths (e.g.: ising.rvf.mu:ising.cbpp.mu).', required=True)
args = parser.parse_args()

f = h5py.File(args.input, 'r')
model_params = yaml.load(f['params']['yaml'][()], Loader=Loader)
num_hoh = model_params['model']['setup']['n_hoh']
num_herds = num_hoh

seeds = list(f.keys())
seeds.remove('params')
seed_scale = 1.0 / len(seeds)

columns = [
    'mean_tot_animals',
    'mean_births',
    'tot_distance',
    'mean_dist_herd',
    'd_age_mean',
    'd_age_std',
    'd_health_mean',
    'd_health_std',
    'd_rvf_mean',
    'd_rvf_std',
    'd_cbpp_mean',
    'd_cbpp_std',
    'v_rvf_pos',
    'v_rvf_neg',
    'v_cbpp_pos',
    'v_cbpp_neg'
]

output = {}

params_include = args.params.split(':')
for param in params_include:
    d = model_params
    for part in param.split('.'):
        d = d[part]
    key = param.replace('.','_')
    output[key] = d
    columns.insert(0,key)

#######################################################################
## movement and animal count
tot_dist = 0.0
tot_animal = 0.0
for seed in seeds:
    tot_dist += f[seed]['total_distance'][()] * seed_scale
    tot_animal += f[seed]['total_animals'][()] * seed_scale

output['mean_tot_animals'] = tot_animal
output['mean_births'] = tot_animal - model_params['model']['setup']['n_animals']
output['tot_distance'] = tot_dist
output['mean_dist_herd'] = tot_dist / num_herds

#######################################################################
## death counts
def report_deaths(f, seed, maxday=4018):
    causes = list(f[seed]['deaths'].keys())
    
    cumulative = {}
    counts = {}
    for cause in causes:
        counts[cause] = np.zeros((1 + maxday//7,))
    
    for cause in causes:
        dset=f[seed]['deaths'][cause]
        (v, c) = np.unique(dset[:].transpose()[0], return_counts=True)
        counts[cause][v.astype(int)//7] = c
        cumulative[cause] = np.cumsum(counts[cause])
        
    return counts,cumulative

death_stats = {}

for seed in seeds:
    counts, cumulative = report_deaths(f, seed)
    for cause in counts:
        if cause not in death_stats:
            death_stats[cause] = []
        death_stats[cause].append(np.sum(counts[cause]))

causes = ['age', 'health', 'rvf', 'cbpp']
for cause in causes:
    if cause in death_stats:
        output[f'd_{cause}_mean'] = np.mean(death_stats[cause])
        output[f'd_{cause}_std'] = np.std(death_stats[cause])
    else:
        output[f'd_{cause}_mean'] = 0.0
        output[f'd_{cause}_std'] = 0.0

#######################################################################
## vaccination decision results
vacc_dec = {}

num_dec = 0

for seed in seeds:
    for disease in f[seed]['vaccination'].keys():
        dset=f[seed]['vaccination'][disease][:].astype(int)
        dnew=np.reshape(dset, (dset.shape[0]//num_herds,num_herds,2))
        num_dec = dnew.shape[0]
        if disease not in vacc_dec:
            vacc_dec[disease] = 0

        mean_d = np.mean([np.sum(dnew[i][:,1]) for i in range(dnew.shape[0])])
        vacc_dec[disease] += mean_d * seed_scale

for disease in vacc_dec:
    d = vacc_dec[disease]
    neg = (num_herds - d) / 2
    pos = num_herds - neg
    output[f'v_{disease}_pos'] = pos
    output[f'v_{disease}_neg'] = neg

if args.header:
    print(','.join(columns))
print(','.join([str(output[c]) for c in columns]))
