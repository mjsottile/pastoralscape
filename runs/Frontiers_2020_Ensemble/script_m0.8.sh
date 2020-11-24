#!/bin/sh

python3 ../driver.py -p params_b0.0_m0.8.json -o output_b0.0_m0.8 > output_b0.0_m0.8/stats.csv
python3 ../driver.py -p params_b0.5_m0.8.json -o output_b0.5_m0.8 > output_b0.5_m0.8/stats.csv
python3 ../driver.py -p params_b1.0_m0.8.json -o output_b1.0_m0.8 > output_b1.0_m0.8/stats.csv
