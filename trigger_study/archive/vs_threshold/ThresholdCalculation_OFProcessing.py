import yaml
from TraceSimulator import TraceSimulator
from TraceSimulator import NoiseGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
sys.path.insert(0,'/home/ws/fm7040/utilities/')
plt.style.use('/home/ws/fm7040/utilities/delight.mplstyle')
import utility as ut
import pickle
import time
from scipy.optimize import curve_fit
from scipy import interpolate
import json
import optimum_filter as OF
from scipy.fft import *
import argparse
import ast

def read_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def run_OF_processing(n_traces=50,save_tag=''):
    
    config = read_yaml_to_dict('/home/ws/fm7040/software/TraceSimulator/tutorials/config_test.yaml')
    ts = TraceSimulator(config)
    ng = NoiseGenerator(config)
    
    lamcal_position = np.genfromtxt('/home/ws/fm7040/software/analysis/Utilities/position_MMC_V2.dat')
    templates = np.load('/home/ws/fm7040/helix/helix/plugins/event_rqs/signalformation_templates_V2-1.npy')


    def distance(p1,p2):
        return np.sqrt(sum((p1-p2)**2))

    def find_closest_LAMCALs(idx):
        indices = []
        for ii in range(37):
            d = distance(lamcal_position[idx+19][1:4],lamcal_position[ts.n_LAMCAL-ts.n_vacuum_LAMCAL+ii][1:4])
            if (d < 46):
                indices.append(ii)
        return indices

    def find_closest_LAMCALs_exclude(idx):
        indices = []
        for ii in range(37):
            d = distance(lamcal_position[idx+19][1:4],lamcal_position[ts.n_LAMCAL-ts.n_vacuum_LAMCAL+ii][1:4])
            if (d < 46)&(d>0):
                indices.append(ii)
        return indices

    closest_lamcals = [find_closest_LAMCALs(idx) for idx in range(37)]
    closest_lamcals_exclude = [find_closest_LAMCALs_exclude(idx) for idx in range(37)]
    idx_closest  = [11, 12, 17, 19, 24, 25]
    
    E_lin = np.random.uniform(low=0,high=100,size=n_traces)
    
    # free position, energy distributed randomly between 5 and 200eV
    trace_events_nonoise_pos, (x_pos, y_pos, z_pos) = ts.generate(E_lin,phonon_only=True,type_recoil='NR', quantize=False, no_noise=True)
    
    # drop the sumberged channels 
    trace_events_nonoise_pos  = [arr[19:, :].tolist() for arr in trace_events_nonoise_pos]
    
    trace_LEE_nonoise = np.vstack([templates[0]*E_lin[ii] for ii in range(n_traces)])


    needed_power = {0: 1.7541796759980222, 1: 4.900298149655333, 2: 14.548386416160952, 3: 54.332578659784794} 
    
    trace_noise = []
    for iN in range(len(needed_power)):
        ng.set_noise_power(needed_power[iN])    
        trace_noise.append([[ng.generate_noise(ts.trace_samples).tolist() for ii in range(37)] for jj in range(n_traces)])
    
    trace_info = {'E_eV':E_lin.tolist(),
                  'traces_nonoise_pos':trace_events_nonoise_pos,
                  'traces_noise':trace_noise,
                  'x_pos':x_pos.tolist(),
                  'y_pos':y_pos.tolist(),
                  'z_pos':z_pos.tolist()}
    
    #with open('/kalinka/scratch/fm7040/DELightTraces/Traces_'+save_tag+'.json','w') as f:
    #    json.dump(trace_info,f)    


    OFamp = {iN:{} for iN in range(len(needed_power))} 
    for iN in range(len(needed_power)):
        
        config['noise_power'] = needed_power[iN]
        ng = NoiseGenerator(config)
        
        # make OF for the QP and UV templates with the noise 
        psd = ng.spectrum(rfftfreq(ts.trace_samples, d=1./ng.sampling_frequency))*needed_power[iN]*ng._normalize[ng.noise_type](rfftfreq(ts.trace_samples, d=1./ng.sampling_frequency))

        of_qp = OF.OptimumFilter(templates[1],psd,ng.sampling_frequency)
        of_uv = OF.OptimumFilter(templates[0],psd,ng.sampling_frequency)     
        
        # event traces with fixed position
        trace_events_pos = np.array(trace_events_nonoise_pos)+np.array(trace_noise[iN])
        OFamp[iN]['fit_qp_QP_pos'] = of_qp.fit(trace_events_pos)
        OFamp[iN]['fit_uv_QP_pos'] = of_uv.fit(trace_events_pos)   
        
        
        traces_max = [trace_events_pos[ii,np.argmax(OFamp[iN]['fit_qp_QP_pos'][0][ii])] for ii in range(n_traces)]
       
        ## find the LAMCAL that gives the maximum, then take the closest LAMCALs and sum the total trace
        traces_summed = [np.sum(trace_events_pos[ii,closest_lamcals[[np.argmax(OFamp[iN]['fit_qp_QP_pos'][0],axis=1)][0][ii]]],axis=0) for ii in range(n_traces)]
        traces_summed_exclude = [np.sum(trace_events_pos[ii,closest_lamcals_exclude[[np.argmax(OFamp[iN]['fit_qp_QP_pos'][0],axis=1)][0][ii]]],axis=0) for ii in range(n_traces)]        
        
        OFamp[iN]['fit_qp_QP_pos_max'] = of_qp.fit(np.array(traces_max))
        OFamp[iN]['fit_uv_QP_pos_max'] = of_uv.fit(np.array(traces_max))        
        OFamp[iN]['fit_qp_QP_pos_max0'] = of_qp.fit_with_no_shift(np.array(traces_max))
        OFamp[iN]['fit_uv_QP_pos_max0'] = of_uv.fit_with_no_shift(np.array(traces_max))
        
        # fit the noise as well for good measure
        OFamp[iN]['fit_qp_NOISE'] = of_qp.fit(np.array(trace_noise[iN]))
        OFamp[iN]['fit_uv_NOISE'] = of_uv.fit(np.array(trace_noise[iN]))
        OFamp[iN]['fit_qp_NOISE0'] = of_qp.fit_with_no_shift(np.array(trace_noise[iN]))
        OFamp[iN]['fit_uv_NOISE0'] = of_uv.fit_with_no_shift(np.array(trace_noise[iN]))
        
        
        OFamp[iN]['E_lin'] = E_lin
        
        #for key in OFamp[iN].keys():
        #    OFamp[iN][key] = OFamp[iN][key].tolist()
        
        np.savez('/kalinka/storage/darkmatter/DELight/efascione/DELightTraces/Threshold_OFamp_Noise-'+str(iN)+'_'+save_tag+'.npz', **OFamp[iN])    
    #with open('/kalinka/scratch/fm7040/DELightTraces/OFamp_'+save_tag+'.json','w') as f:
    #    json.dump(OFamp,f)   
    return

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_traces", type=int, help="Number of traces")
    parser.add_argument("save_tag", type=str, help="File name output tag")

    args = parser.parse_args()

    run_OF_processing(n_traces=args.n_traces,save_tag=args.save_tag)               