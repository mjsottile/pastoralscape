ls *yaml | parallel --jobs 4 --bar python3 ../../../driver.py -p {} -o output_{.}.h5
