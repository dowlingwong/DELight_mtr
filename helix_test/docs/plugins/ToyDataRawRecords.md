# ToyDataRawRecords plugin

### depends_on: `()`
### provides: `raw_records`
### data_kind: `raw_records`

## Options
    
<table>
    <thead align=left>
        <tr><th>option</th><th>type</th><th>default</th><th>description</th></tr>
    </thead>
    <tr>
        <td><tt>record_length</tt></td><td>int</td><td>1250000</td><td>Number of samples per raw_record </td>
    </tr>
    <tr>
        <td><tt>sampling_dt</tt></td><td>int</td><td>4000</td><td>Sampling length in ns </td>
    </tr>
    <tr>
        <td><tt>daq_data_directory</tt></td><td>str</td><td>'toy_data'</td><td>Path to the directory containing run directories with the raw data </td>
    </tr>
    <tr>
        <td><tt>channel_map</tt></td><td>dict</td><td>{ChannelType.SUBMERGED: (1, 35), ChannelType.VACUUM: (36, 50)}</td><td>Dictionary mapping channel types (SUBMERGED, VACUUM) to the corresponding range of channel numbers </td>
    </tr>
    <tr>
        <td><tt>run_metadata_directory</tt></td><td>str</td><td>''</td><td>Path to the directory containing run json metadata </td>
    </tr>
</table>


## Output dtype

<table>
    <thead align=left>
        <tr><th>field</th><th>type</th><th>shape</th><th>description</th></tr>
    </thead>
    <tr>
        <td><tt>time</tt></td><td>int64</td><td>-</td><td>Start time since unix epoch [ns]</td>
    </tr>
    <tr>
        <td><tt>length</tt></td><td>int64</td><td>-</td><td>Length of the interval in samples</td>
    </tr>
    <tr>
        <td><tt>dt</tt></td><td>int16</td><td>-</td><td>Width of one sample [ns]</td>
    </tr>
    <tr>
        <td><tt>channel</tt></td><td>int16</td><td>-</td><td>Channel number</td>
    </tr>
    <tr>
        <td><tt>channel_type</tt></td><td>int16</td><td>-</td><td>Channel type</td>
    </tr>
    <tr>
        <td><tt>block_id</tt></td><td>int32</td><td>-</td><td>Id of the block of records in the run</td>
    </tr>
    <tr>
        <td><tt>data</tt></td><td>int16</td><td>(record_length)</td><td>Record data in raw ADC counts</td>
    </tr>
</table>

## Description

A plugin that reads raw data from disc and converts it to the strax format. It does not perform any other processing
steps.

The raw data must be organized in folders corresponding to run ids, located at the directory configured by the
`daq_data_directory` setting. Each run_id folder contains files with lz4-compressed contiguous C-order flattened
arrays of digitized data from all the channels configured by the `channel_map` setting. The length of the data in
each file must correspond to the `record_length` setting. The sampling time is configured by the `sampling_dt`
setting. In addition, strax-formatted run metadata containing run's start and end times could be provided in a
directory configured by the `run_metadata_directory` setting.

This plugin is the bottom-most plugin in the plugin dependency tree. Its `depends_on` property is an empty tuple,
therefore strax does not pass any strax-formatted data to it. Instead, strax passes an incrementing integer - the
chunk id - to the plugin's `compute()` method, which defines which raw data corresponds to this chunk_id, reads the
data, and converts it to the strax data format (namely, structured numpy array with mandatory `time`, `dt` and
`length` fields, and other fields as specified in `helix/dtypes.py`).