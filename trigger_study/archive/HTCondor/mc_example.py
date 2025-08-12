#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Descriptor: mixed_BBbar

#############################################################
# Steering file for official MC production of early phase 3
# 'mixed' BBbar samples without beam backgrounds (BGx0).
#
# August 2019 - Belle II Collaboration
#############################################################

import basf2 as b2
import generators as ge
import simulation as si
import reconstruction as re
from ROOT import Belle2
import glob as glob
import mdst

b2.set_random_seed(12345)

#: number of events to generate, can be overriden with -n
num_events = 10
#: output filename, can be overriden with -o
output_filename = "mdst.root"

# create path
main = b2.create_path()

# specify number of events to be generated
main.add_module("EventInfoSetter", expList=1003, runList=0, evtNumList=num_events)

# generate BBbar events
ge.add_evtgen_generator(main, finalstate='mixed')

# detector simulation
si.add_simulation(main)

# reconstruction
#RJS
re.add_reconstruction(main)

# Finally add mdst output
mdst.add_mdst_output(main, filename=output_filename)

# process events and print call statistics
b2.process(main)
print(b2.statistics)

