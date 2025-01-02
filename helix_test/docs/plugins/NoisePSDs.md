# NoisePSDs plugin

### depends_on: `noise_events`
### provides: `noise_psds`
### data_kind: `noise_psds`

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
        <td><tt>noise_psd_duration</tt></td><td>int</td><td>1800 * units.s</td><td>Time intervals for which noise PSDs are calculated in ns </td>
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
        <td><tt>endtime</tt></td><td>int64</td><td>-</td><td>Exclusive end time since unix epoch [ns]</td>
    </tr>
    <tr>
        <td><tt>n_events</tt></td><td>int32</td><td>-</td><td>Number of noise events used to calculate the PSDs</td>
    </tr>
    <tr>
        <td><tt>psds</tt></td><td>float64</td><td>(n_channels, psd_length)</td><td>Noise PSDs in ADC^2/Hz</td>
    </tr>
    <tr>
        <td><tt>csds</tt></td><td>float64</td><td>(n_csds, psd_length)</td><td>Noise CSDs in ADC^2/Hz</td>
    </tr>
    <tr>
        <td><tt>frequencies</tt></td><td>float64</td><td>(psd_length)</td><td>PSD frequencies in Hz</td>
    </tr>
    <tr>
        <td><tt>channels</tt></td><td>int16</td><td>(n_channels)</td><td>Channel numbers</td>
    </tr>
    <tr>
        <td><tt>summed_channel_psds</tt></td><td>float64</td><td>-</td><td>Noise PSDs of summed channels in ADC^2/Hz</td>
    </tr>
    <tr>
        <td><tt>summed_channel_types</tt></td><td>int16</td><td>(n_summed_channels)</td><td>Types of the summed channels</td>
    </tr>
</table>

## Description

Plugin to calculate noise Power Spectrum and Cross-Spectrum Densities (PSDs and CSDs) from the noise events. `of_length`
must correspond to the length of noise events and is included into the plugin config to allow for the dtype building at
the plugin setup. The PSDs are recalculated for every `noise_psd_duration` time periods of data.