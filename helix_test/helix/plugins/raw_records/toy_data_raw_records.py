import strax
import numpy as np
import json
from bson import json_util
import os
from helix import units
import helix as hx
import lz4.frame as lz4
import numba
from datetime import datetime

# methods and classes marked with the @export decorator are added to the __all__ namespace to make them importable via
# the star-notation ('from .module_name import *')
export, __all__ = strax.exporter()


@export
def load_toy_data(path):
    """
    Converts lz4-compressed data file to a numpy array

    :param path: path to the data file
    :return: 1d int16 numpy array with the raw data
    """
    with open(path, mode='rb') as f:
        return np.frombuffer(lz4.decompress(f.read()), dtype=np.int16)


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
class ToyDataRawRecords(strax.Plugin):
    """
    A plugin that reads raw data from disc and converts it to the strax format. It does not perform any other processing
    steps.

    The raw data must be organized in folders corresponding to run ids, located at the directory configured by the
    'daq_data_directory' setting. Each run_id folder contains files with lz4-compressed contiguous C-order flattened
    arrays of digitized data from all the channels configured by the 'channel_map' setting. The length of the data in
    each file must correspond to the 'record_length' setting. The sampling time is configured by the 'sampling_dt'
    setting. In addition, strax-formatted run metadata containing run's start and end times could be provided in a
    directory configured by the 'run_metadata_directory' setting.

    This plugin is the bottom-most plugin in the plugin dependency tree. Its 'depends_on' property is an empty tuple,
    therefore strax does not pass any strax-formatted data to it. Instead, strax passes an incrementing integer - the
    chunk id - to the plugin's compute() method, which defines which raw data corresponds to this chunk_id, reads the
    data, and converts it to the strax data format (namely, structured numpy array with mandatory 'time', 'dt' and
    'length' fields, and other fields as specified in helix.dtypes).
    """
    # TODO:
    #  an event might be right at the edge of a file. We need to either deal with resplitting the records
    #  (this would require either keeping state or opening two files simultaneously)
    #  or add artificial_dead_time like in straxen.plugins.DAQReader

    provides = 'raw_records'
    data_kind = 'raw_records'
    depends_on = tuple()
    rechunk_on_save = False  # making sure that one chunk correspond to one raw file
    compressor = 'lz4'
    __version__ = '0.0.0'

    record_length = strax.Config(
        default=hx.DEFAULT_RECORD_LENGTH, track=False, type=int,
        help='Number of samples per raw_record'
    )
    sampling_dt = strax.Config(
        default=hx.DEFAULT_SAMPLING_DT, track=False, type=int,
        help='Sampling length in ns'
    )
    daq_data_directory = strax.Config(
        default='toy_data', track=False, type=str,
        help='Path to the directory containing run directories with the raw data'
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
        return os.path.join(self.daq_data_directory, self.run_id, f'{self.run_id}-{chunk_i:05d}')

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

    def compute(self, chunk_i):
        """
        Main method. Takes chunk id as an input. Returns chunks filled with data.

        :param chunk_i: chunk id
        :return: strax.Chunk with the data corresponding to the requested chunk id
        """
        t_start = self.run_start + chunk_i * self.record_length_ns
        t_end = t_start + self.record_length_ns

        path = self.get_daq_file_path(chunk_i)
        # creating an empty structured array for the output
        raw_records = np.empty(self.n_channels, dtype=self.inferred_dtype)
        # filling the channel and data fields
        raw_records = fill_raw_records(raw_records, load_toy_data(path),
                                       self.record_length, self.channel_numbers)
        # filling the remaining fields (the fields are defined in the helix.dtypes file)
        raw_records['time'] = t_start
        raw_records['length'] = self.record_length
        raw_records['dt'] = self.sampling_dt
        # We decided in this plugin that one chunk of data contains one block of data.
        # Chunk is the internal strax term for handling data. One chunk correspond to one strax-file of data.
        # Record is the data from one channel with the length defined by the record_length setting.
        # Block is a collection of records from all the channels in the time frame corresponding to one record.
        # That is, one block contains N records, 1 record per channel, where N is the number of channels.
        raw_records['block_id'] = chunk_i
        raw_records['channel_type'] = self.channels.numbers_to_types(raw_records['channel'])

        # normally, strax plugins can return structured arrays as the output (i.e. raw_records). But here we define
        # a chunk around the array, to ensure that one chunk correspond to one input file.
        # Otherwise, strax would combine multiple files into one chunk.
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
