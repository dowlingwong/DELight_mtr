import json
import os
import shutil
from datetime import datetime, timedelta

import numpy as np
import scipy
import strax
from bson import json_util
from tqdm import tqdm
import lz4.frame as lz4

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
def generate_toy_data(run, duration, directory='toy_data', record_length=hx.DEFAULT_RECORD_LENGTH,
                      sampling_dt=hx.DEFAULT_SAMPLING_DT, template_length=hx.DEFAULT_TEMPLATE_LENGTH,
                      channel_map=hx.DEFAULT_CHANNEL_MAP, noise_std=3, event_rate=3, lone_hit_rate=2,
                      muon_rate=0.1, overwrite=False, helix_data_dir='test_helix_data', baseline_step=0):
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
    :param lone_hit_rate: rate of lone background hits in Hz (events with a UV signal in only one channel)
    :param muon_rate: rate of muon events
    :param overwrite: a boolean specifying whether the function should overwrite data, if it already exists. If False,
    a RuntimeError is raise when a directory with the same run id exists
    :param helix_data_dir: a directory to save the run metadata. Should be the same as the helix output directory.
    :param baseline_step: add a baseline to each channel, equal to baseline_step * channel_index
    """

    # Both UV and QP templates are simple double-exponential templates, with QP template having a slower rise time
    uv_template = hx.get_analytical_template(length=template_length, sampling_dt=sampling_dt)
    qp_template = hx.get_analytical_template(2 * units.ms, length=template_length, sampling_dt=sampling_dt)
    # a muon template is a long template with 30 ms fall time
    muon_template_length = template_length*8
    muon_template = hx.get_analytical_template(fall_time=30 * units.ms, length=muon_template_length)

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
    sampling_frequency = 1 / (sampling_dt / units.s)

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
    n_lone_hits = np.random.poisson(lone_hit_rate*batch_length_s, n_batches)
    n_muons = np.random.poisson(muon_rate*batch_length_s, n_batches)

    # mean and max time intervals between UV and QP signals
    mean_qp_shift = int(1 * units.ms / sampling_dt)
    max_qp_shift = 2 * mean_qp_shift

    uv_photon_energy = 15
    max_adc = 10000
    ev_to_adc = 10

    for i in tqdm(range(n_batches)):
        # generating noise and adding a correlated feature to it
        waveform = hx.generate_noise(n_channels, psd, sampling_frequency)
        waveform += np.sin(2 * np.pi * correlated_freq * np.arange(record_length * batch_size) / sampling_frequency +
                           correlated_phases[i]) * correlated_noise_amp + baseline[:, np.newaxis]

        # generate random event times
        event_times = np.random.randint(0, batch_size * record_length - template_length - max_qp_shift,
                                        size=n_events[i])
        # generate event energies uniformly between 20 and 5000 units
        energies = np.random.uniform(20, 5000, size=n_events[i])
        # deciding which events are ER events
        is_er = np.random.randint(0, 2, size=n_events[i]).astype(bool)
        # for NR events, 10% of energy goes to UV production (in these toy data, not in reality)
        n_photons_lambdas = 0.1 * energies / uv_photon_energy / n_channels
        # for ER events, 30%
        n_photons_lambdas[is_er] = n_photons_lambdas[is_er] * 3
        # generating UV photon numbers in each channel from Poisson distribution, multiply by 15 eV per photon
        uv_energies = np.random.poisson(n_photons_lambdas, size=(n_channels, n_events[i])) * 15
        # 70% of energy goes to QP signals for NR events
        # all the energy is split randomly between the channels, and smeared with a gaussian sigma=10% of channel energy
        qp_energies = 0.7 * energies / n_vacuum_channels * np.random.normal(1, 0.1,
                                                                            size=(n_vacuum_channels, n_events[i]))
        # 47% (70%/1.5) for ER events
        qp_energies[:, is_er] = qp_energies[:, is_er] / 1.5

        # random time shifts between UV and QP signals. Std is 10% of the mean shift value
        qp_shifts = np.random.normal(mean_qp_shift, mean_qp_shift / 10,
                                     size=(n_vacuum_channels, n_events[i])).astype(int)

        # generating lone hits
        lone_hit_channels = np.random.randint(0, n_channels, size=n_lone_hits[i])
        lone_hit_energies = np.random.uniform(0, 100, size=n_lone_hits[i])
        lone_hit_times = np.random.randint(0, batch_size * record_length - template_length, size=n_lone_hits[i])

        # generating muon events
        muon_energies = np.random.uniform(5000, 50000, size=(n_channels, n_muons[i]))
        muon_times = np.random.randint(0, batch_size * record_length - muon_template_length, size=n_muons[i])

        # adding physics events and muons to the traces
        for ich, ch_type in enumerate(channels.types):
            for j in range(n_events[i]):
                if uv_energies[ich, j] > 0:
                    waveform[ich, event_times[j]:event_times[j] + template_length] += uv_template * uv_energies[ich, j]
                if ch_type == hx.ChannelType.VACUUM:
                    i_vac_ch = ich - n_submerged_channels
                    waveform[ich, event_times[j] + qp_shifts[i_vac_ch, j]:event_times[j] + qp_shifts[i_vac_ch, j] + template_length] +=\
                        qp_template * qp_energies[i_vac_ch, j]
            for j in range(n_muons[i]):
                waveform[ich, muon_times[j]:muon_times[j] + muon_template_length] +=\
                    muon_template * muon_energies[ich, j]

        # adding lone hits to the traces
        for j in range(n_lone_hits[i]):
            ich = lone_hit_channels[j]
            waveform[ich, lone_hit_times[j]:lone_hit_times[j] + template_length] += uv_template * lone_hit_energies[j]

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
    start = datetime.now().replace(microsecond=0)
    end = start + timedelta(seconds=duration)
    metadata = {'start': start, 'end': end}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, default=json_util.default)
