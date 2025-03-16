import numpy as np
import pandas as pd
import RQ
import sys
import os
import yaml
import hashlib
from base64 import b32encode
import configparser

def get_hash(config, length=10):
    config_str = str(config)
    hash_obj = hashlib.sha1(config_str.encode())
    config_hash = b32encode(hash_obj.digest()).decode('ascii').lower()[:length]
    return config_hash

#########################
# Temporary data import #
#########################

# from DAQRead import *
# raw = DAQReader('../../greta/utils/spectrum_for_1p3eV.raw')
# raw.read_pulses()
# shape_data = raw.data['data'].shape
# traces = np.reshape(raw.data['data'], (shape_data[0], 1, shape_data[1]))
traces = np.vstack(pd.read_pickle('spectrum_for_1p3eV.pkl')['data'])
time = np.vstack(pd.read_pickle('spectrum_for_1p3eV.pkl')['TimeStamp'])
shape_data = traces.shape
traces = np.reshape(traces, (shape_data[0], 1, shape_data[1]))
channels = np.reshape(np.tile([1, -1], shape_data[0]), (shape_data[0], 2))
indeces = np.reshape(np.repeat(np.arange(shape_data[0]), 2), (shape_data[0],2))

#########################

run = '000000'

# Load configuration file containing RQs to be calculated
def load_config(config_file='./config.yaml'):
    # Reading yaml analysis file
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config

config = load_config()

# Produce summed traces (assuming traces to be aligned)
channel_weights = np.loadtxt(config['channel_weights']['path'])
sum_traces = np.sum(traces * channel_weights[np.newaxis,:,np.newaxis], axis=1)
shape_data = sum_traces.shape
sum_traces = np.reshape(sum_traces, (shape_data[0], 1, shape_data[1]))
traces = np.concatenate([traces, sum_traces], axis=1)

# Load RQ calculators and define the corresponding data type
rq_calcs = {}
data_type = [('time', np.int32), ('channel', np.int32),
             ('trace_index', np.int32)]
config_rq = {}
rq_to_be_calculated = []
for c in config:
    if not 'function' in config[c]:
        continue
    rq_calcs.append(getattr(RQ, config[c]['function'])(config[c]['config']))
    data_type += rq_calcs[-1].data_type
    config_rq[config[c]['function']] = rq_calcs[-1].get_config()
    config_rq[config[c]['function']]['version'] = rq_calcs[-1].__version__
    # rq_to_be_calculated.append(rq_calcs[-1].__name__)

# Create an array for RQs and other infos and fill it with DAQ-related infos
rqs = np.zeros(traces.shape[:-1], dtype=data_type)
rqs['time'] = time
rqs['channel'] = channels
rqs['trace_index'] = indeces

# Fill array with RQs
rq_unavail = list(rq_calcs.keys())
while len(rq_unavail) > 0:
    for rq_name in rq_unavail:
        print(rq_name)
        if any(elem in rq_unavail for elem in rq_calcs[rq_name].dependencies):
            continue
        rq_calcs[rq_name].apply(traces, rqs)
        rq_unavail.remove(rq_name)

# Save file
directory_output = './output'
if not os.path.exists(directory_output):
    os.makedirs(directory_output)
    print(f"Directory {directory_output} created")

# Create a unique hash based on the configuration files
# and save data in files using these unique hashes
config_hash = get_hash(config)
np.savez(os.path.join(directory_output, f'{run}_traces_{config_hash}.npz'),
         data=traces, config=config)

# Save data in files using unique hash
config_hash = get_hash(config_rq)
np.savez(os.path.join(directory_output, f'{run}_rqs_{config_hash}.npz'),
         data=rqs, config=config_rq)
