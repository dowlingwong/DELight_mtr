# from .RQ import *
import importlib
import numpy as np
import RQ
import sys
import yaml

trace_dummy = np.random.normal(0, 1, (10, 2, 1000))

config_file = './config.yaml'
def load_config(config_file):
    # Reading yaml analysis file
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config

config = load_config(config_file)

rq_calcs = []
data_type = []
for c in config:
    rq_calcs.append(getattr(RQ, config[c]['function'])())
    data_type += rq_calcs[-1].data_type

print(data_type)
rqs = np.zeros((*trace_dummy.shape[:-1], len(data_type)), dtype=data_type)
# for rq_c in rq_calcs:
#     rq_c.apply(trace_dummy, rqs)
# print(rqs)
