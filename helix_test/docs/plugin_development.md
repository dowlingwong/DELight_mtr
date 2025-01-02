# How to develop plugins

Your first source of information for plugin development is [strax documentation](https://strax.readthedocs.io/en/latest/index.html).

Additionally, there is a strax tutorial "straxferno": [git](https://github.com/XENONnT/straxferno), [youtube](https://www.youtube.com/playlist?list=PLKTnVg_X0GC9zOcX7v-eVtwwr_GiGP1Oe).

[straxen](https://github.com/XENONnT/straxen/tree/master/straxen/plugins) is a great place to look for plugin examples.

## Minimal plugin example

Below is a minimal plugin example. This plugin calculates integrals of records

```python
import strax
import numpy as np

class Integrals(strax.Plugin):

    __version__ = '0.0.0' 

    # Which input is this plugin based on
    depends_on = 'records'

    # Which data type does the plugin provide
    provides = 'integrals'

    # The numpy-dtype of the output. The time_dt_fields are mandatory
    dtype = strax.time_dt_fields + [
        (('Record integral', 'integral'), np.int64),
    ]

    # A plugin must implement the compute method.
    # As an input, it takes the data_kinds the plugin depends on
    def compute(self, records):
        result = np.zeros(len(records), dtype=self.dtype)

        # Strax always needs time fields (described in strax.time_dt_fields)
        result['time'] = records['time']      # record start time as UNIX timestamp in ns
        result['dt'] = records['dt']          # sampling time in ns
        result['length'] = records['length']  # length of record in time samples

        # Calculate the integrals
        result['integral'] = np.sum(records['data'], axis=1)

        # returning a np array of self.dtype
        return result
```


Normally, we want plugins to be configurable. Let's add an option to our plugin to skip N samples at the beginning of each record before integration. It can be achieved with the use of strax.Config class. It allows users to pass settings to the plugins via the `strax.Context.set_config` method

```python
import strax
import numpy as np

class Integrals(strax.Plugin):

    __version__ = '0.0.0' 
    depends_on = 'records'
    provides = 'integrals'

    # A configuration option, which we can use in the computation (self.compute)
    skip_n_samples = strax.Config(default=0,
                                  type=int,
                                  help='Specifies how many samples to skip in the beginning '
                                       'of each record before integration')

    dtype = strax.time_dt_fields + [
        (('Record integral', 'integral'), np.int64),
    ]

    def compute(self, records):
        result = np.zeros(len(records), dtype=self.dtype)

        # Strax always needs time fields (described in strax.time_dt_fields)
        result['time'] = records['time']      # record start time as UNIX timestamp in ns
        result['dt'] = records['dt']          # sampling time in ns
        result['length'] = records['length']  # length of record in time samples

        # Calculate the integrals. Use the config to skip n samples
        result['integral'] = np.sum(records['data'][:, self.skip_n_samples:], axis=1)

        # returning a np array of self.dtype
        return result
```

## Data-reading plugin

The `records` data in the example above is already in the strax format. However, your experimental data comes in some non-strax format. The very first plugin you want to develop is a plugin that reads your data and converts it to the strax format. Here is a minimal example of a data-reading plugin:

```python
import strax
import numpy as np
import os


class MMCRecords(strax.Plugin):
    __version__ = '0.0.0'
    provides = 'records'
    depends_on = tuple()  # This tells strax that we are not taking any strax-formatted data as input

    data_directory = strax.Config(
        default='mmc_data', type=str,
        help='Path to the directory containing run directories with the raw data'
    )
    sampling_dt = strax.Config(
        default=4000, type=int,
        help='Sampling length in ns'
    )
    record_length = strax.Config(
        default=100_000, type=int,
        help='Number of samples per record'
    )

    # Plugins must either have a dtype property or an infer_dtype method. 
    # Here dtype depends on one of the config options, record_length, therefore we use infer_dtype method
    def infer_dtype(self):
        # dtype is a list of the following tuples:
        # ((field_description, field_name), field_type, shape)
        # If field contains one value per entry, shape is ommited
        # In our case, each record's 'data' field is an array of length 'record_length'
        dtype = strax.time_dt_fields + [
            (('ADC data', 'data'), np.int16, self.record_length), 
        ]
        return dtype

    def load_data(self, chunk_i):
        # implement your data reading logic here.
        # for example:
        return np.load(f'{self.data_directory}/{self.run_id}/{chunk_i}.npy')

    def determine_record_start_times(self, chunk_i):
        # you need to determine the times of each record.
        # for example, if it is saved in some metadata, retrieve it from there
        times = ...
        return times

    # when the depends_on property is an empty tuple, strax passes an incrmenting chunk number to the compute method.
    # The compute method defines which data to assign to each chunk number.
    def compute(self, chunk_i):
        # retrieve data and possibly metadata to determine start times of each record
        data = self.load_data(chunk_i)
        start_times = self.determine_record_start_times(chunk_i)

        # fill out the results
        records = np.zeros(len(data), dtype=self.infer_dtype())
        # mandatory time fields
        records['time'] = start_times
        records['length'] = self.record_length
        records['dt'] = self.sampling_dt
        # the fields you defined in infer_dtype
        records['data'] = data
        
        return records

    def is_ready(self, chunk_i):
        # this function tells strax when to stop incrementing the chunk_i and calling the compute method.
        # for offline processing, you can simply check if files corresponding to the requested chunk_i exist
        return os.path.exists(f'{self.data_directory}/{self.run_id}/{chunk_i}.npy')

    def source_finished(self):
        # I am not 100% sure, but I think strax requires this function for data-reading plugins. Just return True.
        # If you are doing online processing, this function should tell whether the data can be read
        return True
```

See [straxen.DAQReader](https://github.com/XENONnT/straxen/blob/master/straxen/plugins/raw_records/daqreader.py) or [helix.ToyDataRawRecords](https://gitlab.etp.kit.edu/delight/helix/-/blob/main/helix/plugins/raw_records/toy_data_raw_records.py?ref_type=heads) for data-reading plugin implementation examples.

## Tips

- When developing plugins it is convineint to set their `__version__` to `None`. This would tell strax to hash the
plugin's source code rather than its version. This allows to make changes to the plugin's code and run the plugin on test
data that was already processed by this plugin before. If version is None, strax would calculate the code hash, see that
the code has changed, and reprocess the data. If, however, some version number is specified (e.g., `0.0.0`), strax would
not reprocess the test data, assuming that nothing has changed since it was processed the last time.
