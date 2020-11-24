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

        runscript = ""
        
        for mu in range(0,21):
          for d in model_params['ising']:
            model_params['ising'][d]['mu'] = max(0.0, min(1.0, mu * 0.05))
            stream = open(f'params_{mu}.yaml', 'w')
            yaml.dump(model_params, stream) 
          runscript += f"python3 ../../driver.py -p params_{mu}.yaml -o output_{mu}.yaml\n"
        print(runscript)
        
    
main()
