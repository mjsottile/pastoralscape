import argparse
import yaml
try:
  from yaml import CLoader as Loader
except ImportError:
  from yaml import Loader

## command line
parser = argparse.ArgumentParser(description="PastoralScape param generator")
parser.add_argument('-p', '--params', help='Params file.', required=True)
args = parser.parse_args()

def main():
    # load parameter file from disk as YAML file to dictionary
    with open(args.params) as f:
        paramfile_string = f.read()
        model_params = yaml.load(paramfile_string, Loader=Loader)

        for mont in range(1,13):
          model_params['agents']['vaccination_schedule'] = [[mont,1]]
          stream = open(f'params_{mont}.yaml', 'w')
          yaml.dump(model_params, stream) 
        
    
main()
