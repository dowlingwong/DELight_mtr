import json
import os
import shutil
from datetime import datetime, timedelta
import timeit
import pickle

import yaml
from TraceSimulator import TraceSimulator
from TraceSimulator import DummyTraceSimulator

def read_yaml_to_dict(file_path):
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict

import numpy as np
import scipy
import strax
from bson import json_util
from tqdm import tqdm
import lz4.frame as lz4
import pandas as pd
import uproot

import matplotlib.pyplot as plt

import helix as hx
from helix import units

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
def get_pink_psd(trace_length, sampling_dt, noise_std):
    """
    Returns folded PSD corresponding to 1/f pink noise, empirically scaled to make the standard deviation of the noise
    produced from this PSD to be close to the provided noise_std value

    :param trace_length: length of noise traces in samples
    :param sampling_dt: sampling time in ns
    :param noise_std: noise standard deviation
    :return: a tuple of (f, psd), where f is the array of frequencies in Hz, and psd is the array of PSD components
    in A^2/Hz. Can be plotted with plt.loglog(f, psd)
    """
    f = scipy.fft.rfftfreq(trace_length, d=sampling_dt / units.s)
    f[0] = f[1]
    # empirical approximate scaling to match the standard deviation of the resulting noise.
    # This is totally crazy, it's a random expression that scales the PSD in a way that the standard deviation of the
    # noise produced from this PSD somewhat matches the requested noise_std
    # TODO: derive the actual scaling that would properly work and make sense instead of this monster
    scaling = 1 / (1 + (np.log10(trace_length) - 4.5) / 10) ** 2 / 10.3
    psd = scaling*(noise_std**2) / f
    psd[-1] = psd[-1] / 2
    # Could use 0 for the DC component of the PSD, but it messes up the plotting in log scale.
    # The value doesn't matter anyway, we are not generating noise DC components. Using the smallest positive float
    psd[0] = np.nextafter(1, 2)
    f[0] = 0
    return f, psd


@export
def generate_tracesimulator_data(run, duration, directory='tracesimulator_data', record_length=hx.DEFAULT_RECORD_LENGTH,
                      sampling_dt=hx.DEFAULT_SAMPLING_DT, template_length=hx.DEFAULT_TEMPLATE_LENGTH,
                      channel_map=hx.DEFAULT_CHANNEL_MAP, noise_std=3, event_rate=3, event_energy=20, recoil_type='ER',
                      event_position='random',geant4input=False, overwrite=False, helix_data_dir='test_helix_data', baseline_step=0,debug=False):
    """
    Generates and saves toy data with multiple channels of vacuum and submerged types, physics events consisting of UV
    and QP signals, as well as background lone hits and muon saturated events. CAUTION: it's slow!
    Noise is uncorrelated pink noise, with a correlated 5 kHz feature in all channels. Channels have different baselines

    :param run: run id
    :param duration: run duration in seconds. Caution: the function is slow, don't ask to generate days of data
    :param directory: output directory, where a directory with run_id name will be created
    :param record_length: length of records in each file in time samples
    :param sampling_dt: sampling time in ns
    :param template_length: length of UV and QP templates
    :param channel_map: dictionary of channel types and channel number ranges
    :param noise_std: standard deviation of the noise
    :param event_rate: rate of physics events in Hz
    :param event_energy: energy of events in eV, can be a single value, a range, spectrum, or geant4 simulation input
    :param event_position: position of events in the detector as (x,y,z), 
    :param recoil_type: event recoil type, 'ER', 'NR', or 'random'
    :param geant4input: default False, otherwise path to geant4 simulation root file - will ignore energy, recoil type, and event position if provided 
    :param overwrite: a boolean specifying whether the function should overwrite data, if it already exists. If False,
    a RuntimeError is raise when a directory with the same run id exists
    :param helix_data_dir: a directory to save the run metadata. Should be the same as the helix output directory.
    :param baseline_step: add a baseline to each channel, equal to baseline_step * channel_index
    """
    
    # check recoil type input
    if recoil_type not in ['ER','NR','random']:
        raise ValueError("Invalid recoil_type, must be 'ER', 'NR', or 'random'")
    
    # check event energy input #TODO
    
    config = read_yaml_to_dict('config.yaml')
    config['sampling_frequency'] = 1 / (sampling_dt / units.s)
    config['trace_samples'] = template_length
    config["noise_std"] = 0
    
    if debug:
        ts = DummyTraceSimulator(config)
    else:
        ts = TraceSimulator(config)

    run_dir = os.path.join(directory, run)
    if os.path.exists(run_dir):
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise RuntimeError(f'Directory {run_dir} already exists.')

    os.makedirs(run_dir)

    record_length_s = record_length * sampling_dt / units.s

    n_records = int(duration / record_length_s)
    channels = hx.Channels(channel_map)
    n_channels = len(channels)
    n_submerged_channels = channels.counts[hx.ChannelType.SUBMERGED]
    n_vacuum_channels = channels.counts[hx.ChannelType.VACUUM]
    batch_size = 5  # number of records per batch
    sampling_frequency = config['sampling_frequency']

    _, psd = get_pink_psd(record_length * batch_size, sampling_dt, noise_std)

    correlated_freq = 5e3  # Hz
    correlated_noise_amp = noise_std/5

    # adding different baselines to each channel
    baseline = baseline_step * np.arange(n_channels)

    # generating records in batches, i.e. we generate a longer record than necessary and then split it into records
    # of required length.
    # This allows us to have events on the record edges
    n_batches = int(np.ceil(n_records / batch_size))
    correlated_phases = np.random.uniform(0, 2 * np.pi, n_batches)

    batch_length_s = batch_size * record_length_s
    n_events = np.random.poisson(event_rate*batch_length_s, n_batches)

    max_adc = 10000
    ev_to_adc = 10
    
    # prepare simulation truth data
    truthdata = {'time_truth':n_batches*[0],'E_er_truth':n_batches*[0],'E_nr_truth':n_batches*[0],'XYZ_truth':n_batches*[0]}
    if geant4input:
        with uproot.open(geant4input) as file:
            simevents = pd.DataFrame(file["Events"].arrays(file["Events"].keys(), library="np", array_cache=None))

    for i in tqdm(range(n_batches)):
        # generating noise and adding a correlated feature to it
        print("Generating noise")
        waveform = hx.generate_noise(n_channels, psd, sampling_frequency)
        waveform += np.sin(2 * np.pi * correlated_freq * np.arange(record_length * batch_size) / sampling_frequency +
                           correlated_phases[i]) * correlated_noise_amp + baseline[:, np.newaxis]
        print("Finished noise generation, beginning event simulation")

        # generate random event times
        event_times = np.random.randint(0, batch_size * record_length - template_length,
                                        size=n_events[i]) 
        
        # set, generate, or load event energies
        if geant4input:
            energies_er = np.asarray(simevents['E_er'][sum(n_events[:i]):sum(n_events[:i+1])])
            energies_nr = np.asarray(simevents['E_nr'][sum(n_events[:i]):sum(n_events[:i+1])])
            positions = np.transpose([np.asarray(simevents['x'][sum(n_events[:i]):sum(n_events[:i+1])]),np.asarray(simevents['y'][sum(n_events[:i]):sum(n_events[:i+1])]),np.asarray(simevents['z'][sum(n_events[:i]):sum(n_events[:i+1])])])
        else: 
            if np.isscalar(event_energy):
                energies = event_energy*np.ones(n_events[i])
            elif len(event_energy)==2:
                energies = np.random.uniform(event_energy[0], event_energy[1], size=n_events[i])
            elif len(event_energy)>2:
                if len(event_energy)<sum(n_events):
                    raise ValueError("provided list of energies not long enough for requested time and event rate")
                else:
                    energies = event_energy[sum(n_events[:i]),sum(n_events[:i+1])]
            else:
                raise ValueError("event_energies has the wrong format")
                
            # deciding which events are ER events if 'random' option selected TODO
            if recoil_type == 'random':
                is_er = np.random.randint(0, 2, size=n_events[i]).astype(bool)
                energies_er = is_er*energies
                energies_nr = (1-is_er)*energies
            elif recoil_type == 'ER':
                energies_er = energies
                energies_nr = np.zeros(np.shape(energies))
            else:
                energies_nr = energies
                energies_er = np.zeros(np.shape(energies))
            if event_position!='random':
                if np.size(event_position)==3:
                    positions = n_events[i]*[event_position]
                elif np.size(event_position)/3>sum(n_events):
                    positions = event_position[sum(n_events[:i]),sum(n_events[:i+1])]
            else:
                positions = n_events[i]*[[0,0,-1700]]
            
                
        # adding physics events to the traces
        # TODO currently can't do both NR and ER at the same position if not loading from geant4 simulation
        x, y, z = n_events[i]*[0], n_events[i]*[0], n_events[i]*[0]
        for j in range(n_events[i]):
            # Add ER part
            if energies_er[j]>0:
                if (not geant4input) & (event_position=='random'):
                    if debug:
                        trace_er = ts.generate(energies_er[j],0,0,-1700,type_recoil='ER') # for debugging only
                    else:
                        trace_er, (x[j], y[j], z[j]) = ts.generate(energies_er[j],type_recoil='ER')
                        positions[j] = [x[j], y[j], z[j]]
                else:
                    trace_er = ts.generate(energies_er[j],positions[j][0],positions[j][1],positions[j][2],type_recoil='ER')                    
                for ich, ch_type in enumerate(channels.types):
                    waveform[ich, event_times[j]:event_times[j] + template_length] += trace_er[0][ich]                    
            # Add NR part
            if energies_nr[j]>0:
                if (not geant4input) & (event_position=='random'):
                    if debug:                 
                        trace_nr = ts.generate(energies_nr[j],0,0,-1700,type_recoil='NR') # for debugging only
                    else:
                        trace_nr, (x[j], y[j], z[j]) = ts.generate(energies_nr[j],type_recoil='NR')
                        positions[j] = [x[j], y[j], z[j]]
                else:
                    trace_nr = ts.generate(energies_nr[j],positions[j][0],positions[j][1],positions[j][2],type_recoil='NR')
                for ich, ch_type in enumerate(channels.types):
                    waveform[ich, event_times[j]:event_times[j] + template_length] += trace_nr[0][ich]
            # Check something is happening at least
            if (energies_er[j]<=0)&(energies_nr[j]<=0):
                print("Both ER and NR energies are 0...")
            
                    
        # fill truth data    
        truthdata['time_truth'][i] = event_times 
        truthdata['E_er_truth'][i] = energies_er
        truthdata['E_nr_truth'][i] = energies_nr
        truthdata['XYZ_truth'][i] = positions
        
        # clipping the traces to simulate saturation
        waveform = np.clip(waveform*ev_to_adc, -max_adc, max_adc)

        # splitting the data into record_length records
        for j in range(batch_size):
            i_record = i * batch_size + j
            if i_record == n_records:
                break
            fn = f'{directory}/{run}/{run}-{i_record:05d}'
            # saving the data as a flattened array
            with open(fn, mode='wb') as f:
                data = np.ascontiguousarray(waveform[:, j * record_length:(j + 1) * record_length], dtype=np.int16)
                f.write(lz4.compress(data))

    # saving run metadata
    if not os.path.exists(helix_data_dir):
        os.makedirs(helix_data_dir)

    metadata_path = os.path.join(helix_data_dir, "%s-metadata.json" % run)
    truthdata_path = os.path.join(helix_data_dir, "%s-truthdata.pkl" % run)
    start = datetime.now().replace(microsecond=0)
    end = start + timedelta(seconds=duration)
    metadata = {'start': start, 'end': end}
    print(truthdata)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, default=json_util.default)
        
    with open(truthdata_path, 'wb') as f:
        pickle.dump(truthdata, f)    
