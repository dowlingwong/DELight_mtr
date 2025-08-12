#!/bin/bash

##### WN informaton
echo -e "Hostname :$(hostname)"
echo -e "user: $(whoami)"
echo -e "spawndir: $(pwd)"

##### setup environment
source /cvmfs/belle.cern.ch/tools/b2setup release-09-00-01 

##### start workload (results stored in output.txt)
basf2 mc_example.py -o ouput.txt

##### copy output 
## is done by htcondor

