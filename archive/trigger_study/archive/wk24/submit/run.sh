#!/bin/bash

##### WN informaton
echo -e "Hostname : portal1@etp.kit.edu"
echo -e "user: dwong"
echo -e "spawndir: /work/dwong/"

##### setup environment
conda activate base

mkdir test
cd test
##### start workload (results stored in output.txt)
python text.py

##### copy output from WN
cp -r test /work/dwong/training_samples/test/

