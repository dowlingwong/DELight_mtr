
import strax
import numpy as np
import os
import json
from bson import json_util

import pandas as pd
import numba
from helix import units
import helix as hx

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()

@export
def load_data_csv(path):
    """
    Loads the new CSV-based data file into a numpy structured array.

    :param path: Path to the CSV data file.
    :return: Numpy structured array with raw data.
    """
    df = pd.read_csv(path)
    return df.to_numpy(dtype=np.int16).flatten()

@numba.njit(nogil=True, cache=True)
def fill_raw_records(raw_records, daq_record, record_length, channels):
    """
    Takes one block of data with all the channels concatenated together, splits into channel records and write them
    to the provided raw_records structured array. Accelerated by numba JIT compiler.

    :param raw_records: structured array of length len(channels) with empty 'channel' and 'data' fields to fill out.
    :param daq_record: flattened 1d array containing raw ADC data corresponding to one data block, i.e. 1 record of
    length record_length from each channel concatenated together. length = record_length * len(channels)
    :param record_length: length of one record in time samples.
    :param channels: array of channel names, sorted in the same order as the records within the flattened daq_record.
    :return: filled_out raw_records array with.
    """
    for i, ch in enumerate(channels):
        raw_records['channel'][i] = ch
        raw_records['data'][i] = daq_record[i * record_length:(i + 1) * record_length]
    return raw_records



@export
class MMCRecords(strax.Plugin):
    __version__ = '0.0.0'
    provides = 'raw_records'
    data_kind = 'raw_records'
    rechunk_on_save = False  # making sure that one chunk correspond to one raw file
    depends_on = tuple()  # This tells strax that we are not taking any strax-formatted data as input

    data_directory = strax.Config(
        default='toy_data', track=False, type=str,
        help='Path to the directory containing run directories with the raw data'
    )
    sampling_dt = strax.Config(
        default=hx.DEFAULT_SAMPLING_DT, track=False, type=int,
        help='Sampling length in ns for MMC readout'
    )
    record_length = strax.Config(
        default=hx.DEFAULT_RECORD_LENGTH, track=False, type=int,
        help='Number of samples per record'
    )

    channel_map = strax.Config(
        default=hx.DEFAULT_CHANNEL_MAP, track=False, type=dict,
        help='Dictionary mapping channel types (SUBMERGED, VACUUM) to the corresponding range of channel numbers'
    )
    run_metadata_directory = strax.Config(
        default='', track=False, type=str,
        help='Path to the directory containing run json metadata'
    )

    def infer_dtype(self):
        """
        Returns the raw_record dtype with the records of length corresponding to the record_length setting

        :return: numpy dtype for raw_records structured array
        """
        return hx.get_raw_records_dtype(self.record_length)
    
    def get_daq_file_path(self, chunk_i):
        """
        Returns path to the data file corresponding to the run_id (inherited strax.Plugin argument, set by strax
        internally) and the provided chunk_i

        :param chunk_i: requested chunk id
        :return: path to the data file
        """
        # In this plugin we decided that one chunk should correspond to one data file. In principle, this is not a
        # strict requirement. One can write a plugin where 1 chunk correspond to multiple files, or one file contains
        # multiple chunks.
        return os.path.join(self.data_directory, self.run_id, f'{self.run_id}-{chunk_i:05d}')

    def setup(self):
        """
        Plugin initialization. Uses the config to construct a channel map and to build the output dtype.
        """
        self.channels = hx.Channels(self.channel_map)  # converting channel_map setting to an object of Channels class
        self.channel_numbers = self.channels.numbers
        self.n_channels = len(self.channels)
        self.inferred_dtype = self.infer_dtype()
        self.run_start = self.get_run_start()  # read the run start from the run metadata
        self.record_length_ns = self.record_length * self.sampling_dt

    def compute(self, chunk_i=None):
        """
        Main method. Processes the entire dataset in one chunk.

        :param chunk_i: Ignored, as we are processing the entire dataset as one chunk.
        :return: strax.Chunk with the entire dataset.
        """
        # Set the start and end times for the entire dataset
        t_start = self.run_start
        t_end = t_start + self.total_duration_ns  # Adjust this based on the total duration of the data

        # Load the full dataset
        path = self.get_daq_file_path()  # Adjust this to load the complete dataset
        raw_records = np.empty(self.n_channels, dtype=self.inferred_dtype)

        # Fill the raw_records array with the full dataset
        raw_records = fill_raw_records(
            raw_records,
            load_data_csv(path),
            self.record_length,
            self.channel_numbers,
        )

        # Fill the remaining fields
        raw_records['time'] = t_start
        raw_records['length'] = self.record_length
        raw_records['dt'] = self.sampling_dt
        raw_records['block_id'] = 0  # Single block for the whole dataset
        raw_records['channel_type'] = self.channels.numbers_to_types(raw_records['channel'])

        # Return a single chunk for the entire dataset
        return self.chunk(start=t_start, end=t_end, data=raw_records)

    
    
    def get_run_start(self):
        """
        Returns start of the run, if it is provided in the run metadata. Otherwise, return 0.

        :return: start of the run as a unix timestamp in ns
        """
        # TODO: Currently we call it from the setup() method. I am wondering whether this is right. If I process
        #  multiple runs, would strax call the setup method again to recalculate the run start?

        run_metadata_dir = self.run_metadata_directory
        if not run_metadata_dir:
            return 0  # if not metadata provided, set the run start to 0
        path = os.path.join(run_metadata_dir, f"{self.run_id}-metadata.json")
        with open(path, mode="r") as f:
            run_metadata = json.loads(f.read(), object_hook=json_util.object_hook)

        if 'start' in run_metadata:
            # units.s actually converts s to ns. Maybe it should be renamed accordingly...
            return int(run_metadata['start'].timestamp()) * units.s
        else:
            return 0




    def is_ready(self, chunk_i):
        """
        A requirement for plugins with depends_on=() is to override this function (strax.Plugin.is_ready()).

        Here we return true if the data file with the requested chunk exists. To use this plugin for online processing,
        implement this method, together with the source_finished() method, as e.g.
        in straxen.plugins.raw_records.daq_reader:
        https://github.com/XENONnT/straxen/blob/master/straxen/plugins/raw_records/daqreader.py#L209

        :param chunk_i: chunk id
        :return: True if file corresponding to the requested chunk exists. False otherwise.
        """
        return os.path.exists(self.get_daq_file_path(chunk_i))

    def source_finished(self):
        """
        Another requirement for plugins with depends_on=() is to override this function (strax.Plugin.source_finished())
        Here we simply return True. To use this plugin for online processing, implement this method, together with the
        is_ready() method, as e.g. in straxen.plugins.raw_records.daq_reader:
        https://github.com/XENONnT/straxen/blob/master/straxen/plugins/raw_records/daqreader.py#L209

        :return: True
        """
        return True

