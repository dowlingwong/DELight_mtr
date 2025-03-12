import numpy as np
import pandas as pd
import RQ
import sys
import os
import yaml
import hashlib
from base64 import b32encode
import configparser
from DAQRead import *

def get_hash(config, length=10):

    config_str = str(config)
    hash_obj = hashlib.sha1(config_str.encode())
    config_hash = b32encode(hash_obj.digest()).decode('ascii').lower()[:length]
    return config_hash

def load(path):

    raw_format = path.split('.')[-1]
    loading_function = {'pkl': _load_pickle,
                        'raw': _load_raw}
    return loading_function[raw_format](path)

def _load_pickle(path):

    traces = np.vstack(pd.read_pickle(path)['data'])
    time = np.vstack(pd.read_pickle(path)['TimeStamp'])
    shape_data = traces.shape
    traces = np.reshape(traces, (shape_data[0], 1, shape_data[1]))
    channels = np.reshape(np.tile([1, -1], shape_data[0]), (shape_data[0], 2))
    indeces = np.reshape(np.repeat(np.arange(shape_data[0]), 2), (shape_data[0],2))
    return traces, time, channels, indeces

def _load_raw(path):

    raw = DAQReader(path)
    tr_tmp = []
    time = []
    for i in range(100):
        tr_tmp.append(raw.read_pulse_i(i))
        time.append(raw.TimeStamp)
    traces = np.vstack(tr_tmp)
    shape_data = traces.shape
    traces = np.reshape(traces, (shape_data[0], 1, shape_data[1]))
    channels = np.reshape(np.tile([1, -1], shape_data[0]), (shape_data[0], 2))
    time = np.reshape(np.repeat(time, 2), (shape_data[0], 2))
    indeces = np.reshape(np.repeat(np.arange(shape_data[0]), 2), (shape_data[0],2))
    return traces, time, channels, indeces

if len(sys.argv) < 2:
    raise ValueError('Raw file path is required')

raw_input = sys.argv[1]
traces, time, channels, indeces = load(raw_input)

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
    class_name = config[c]['function']
    rq_calcs[class_name] = getattr(RQ, class_name)(config[c]['config'])
    data_type += rq_calcs[class_name].data_type
    config_rq[class_name] = rq_calcs[class_name].get_config()
    config_rq[class_name]['version'] = rq_calcs[class_name].__version__

# Create an array for RQs and other infos and fill it with DAQ-related infos
rqs = np.zeros(traces.shape[:-1], dtype=data_type)
rqs['time'] = time
rqs['channel'] = channels
rqs['trace_index'] = indeces

unprocessed = list(rq_calcs.keys())

while len(unprocessed) > 0:
    for rq_key, rq_item in rq_calcs.items():
        if any(elem in unprocessed for elem in rq_item.dependencies):
            continue
        else:
            print(f'{rq_key} processed')
        rq_item.apply(traces, rqs)
        unprocessed.remove(rq_key)



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
