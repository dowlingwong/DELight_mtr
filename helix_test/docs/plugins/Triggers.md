# Triggers plugin

### depends_on: `raw_records`
### provides: `triggers`
### data_kind: `triggers`

## Options
    
<table>
    <thead align=left>
        <tr><th>option</th><th>type</th><th>default</th><th>description</th></tr>
    </thead>
    <tr>
        <td><tt>filter_kernel</tt></td><td>list</td><td>None</td><td>Trigger filter kernel </td>
    </tr>
    <tr>
        <td><tt>trigger_threshold</tt></td><td>float</td><td>100.0</td><td>Trigger threshold in filtered trace units </td>
    </tr>
    <tr>
        <td><tt>deactivation_threshold_coefficient</tt></td><td>float</td><td>1.0</td><td>Coefficient that the trigger threshold is multiplied by to produce the deactivation threshold. A hit is defined by the interval from where the waveform crosses the activation threshold, to where it crosses the deactivation threshold in the opposite direction </td>
    </tr>
    <tr>
        <td><tt>trigger_holdoff</tt></td><td>int</td><td>0</td><td>Time in samples after each trigger when triggering is not allowed </td>
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
        <td><tt>start_loc</tt></td><td>int32</td><td>-</td><td>Sample index in the record where the filtered record crossed the threshold</td>
    </tr>
    <tr>
        <td><tt>loc</tt></td><td>int32</td><td>-</td><td>Sample index in the record where the filtered record reached the maximum</td>
    </tr>
    <tr>
        <td><tt>amplitude</tt></td><td>float32</td><td>-</td><td>Max amplitude of the filtered record withing the trigger</td>
    </tr>
    <tr>
        <td><tt>deactivation_crossed</tt></td><td>bool_</td><td>-</td><td>Whether or not the deactivation threshold was crossed</td>
    </tr>
</table>

## Description

Base plugin for triggering that takes raw records as input, convolves them with a configurable filter kernel,
and applies a double-threshold triggering with configurable activation and deactivation thresholds. A trigger holdoff of
configurable time is applied after each trigger. During this time, no other triggers are issued.
