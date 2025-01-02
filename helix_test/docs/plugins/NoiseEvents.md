# NoiseEvents plugin

### depends_on: `('raw_records', 'qp_triggers', 'uv_triggers')`
### provides: `noise_events`
### data_kind: `noise_events`

## Options
    
<table>
    <thead align=left>
        <tr><th>option</th><th>type</th><th>default</th><th>description</th></tr>
    </thead>
    <tr>
        <td><tt>channel_map</tt></td><td>dict</td><td>{ChannelType.SUBMERGED: (1, 35), ChannelType.VACUUM: (36, 50)}</td><td>Dictionary mapping channel types (SUBMERGED, VACUUM) to the corresponding range of channel numbers </td>
    </tr>
    <tr>
        <td><tt>of_length</tt></td><td>int</td><td>16384</td><td>Trace interval length (in samples) for the Optimum Filter fits </td>
    </tr>
    <tr>
        <td><tt>pre_trigger_noise_trace_veto</tt></td><td>int</td><td>200</td><td>Minimum number of samples between the end of a noise trace and a trigger </td>
    </tr>
    <tr>
        <td><tt>post_trigger_noise_trace_veto</tt></td><td>int</td><td>50000</td><td>Minimum number of samples between a trigger and the start of a noise trace </td>
    </tr>
    <tr>
        <td><tt>n_noise_events_per_record</tt></td><td>int</td><td>2</td><td>Number of noise events per record </td>
    </tr>
    <tr>
        <td><tt>allow_noise_events_overlaps</tt></td><td>bool</td><td>False</td><td>If true, noise traces can overlap with each other </td>
    </tr>
    <tr>
        <td><tt>noise_events_random_seed</tt></td><td>int</td><td>None</td><td>Seed for random number generator for noise events. If None, random seed is used </td>
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
        <td><tt>block_id</tt></td><td>int32</td><td>-</td><td>Id of the block of records in the run</td>
    </tr>
    <tr>
        <td><tt>block_time</tt></td><td>int64</td><td>-</td><td>Start time of the block since unix epoch [ns]</td>
    </tr>
    <tr>
        <td><tt>event_id</tt></td><td>int16</td><td>-</td><td>Event number in the record block</td>
    </tr>
    <tr>
        <td><tt>start_loc</tt></td><td>int32</td><td>-</td><td>Sample index in the record where the trace starts</td>
    </tr>
    <tr>
        <td><tt>channel_data</tt></td><td>float64</td><td>(n_channels, event_length)</td><td>Trace data in individual channels</td>
    </tr>
    <tr>
        <td><tt>channels</tt></td><td>int16</td><td>(n_channels)</td><td>Channel numbers</td>
    </tr>
    <tr>
        <td><tt>data</tt></td><td>float64</td><td>(n_summed_channels, event_length)</td><td>Summed traces of summed_channel_types</td>
    </tr>
    <tr>
        <td><tt>summed_channel_types</tt></td><td>int16</td><td>(n_summed_channels)</td><td>Types of the summed traces</td>
    </tr>
    <tr>
        <td><tt>summed_channel_masks</tt></td><td>bool_</td><td>-</td><td>Mask of channels that were summed up to produce the summed traces</td>
    </tr>
</table>

`n_summed_channels = 3`, corresponding to the sum of submerged channels, sum of vacuum channels, and sum of all channels

## Description

Plugin to find noise samples of length `of_length`. Takes raw records and UV and QP triggers, and finds intervals in the
records without any triggers in them. 