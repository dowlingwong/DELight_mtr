# FitResults plugin

### depends_on: `('noise_psds', 'events')`
### provides: `fit_results`
### data_kind: `events`

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
        <td><tt>templates_path</tt></td><td>str</td><td>'plugins/event_rqs/default_templates.npy'</td><td>Path to the OF template file. The file should contain an np array with shape (2, L),  with UV and QP templates of length L=of_length </td>
    </tr>
    <tr>
        <td><tt>allowed_fit_shifts</tt></td><td>tuple</td><td>(-200, 800)</td><td>Tuple of allowed left and right time shifts in the OF fits. In samples. Left shifts are negative. </td>
    </tr>
    <tr>
        <td><tt>allowed_two_template_deltas</tt></td><td>tuple</td><td>(150, 350)</td><td>Tuple of allowed relative time shifts of UV and QP templates. In samples </td>
    </tr>
    <tr>
        <td><tt>fit_summed_triggered_channels</tt></td><td>tuple</td><td>True</td><td>If true, fit_results plugin will fit summed triggered channels. This is slow. </td>
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
        <td><tt>event_id</tt></td><td>int16</td><td>-</td><td>Event number in the record block</td>
    </tr>
    <tr>
        <td><tt>channels</tt></td><td>int16</td><td>(n_channels)</td><td>Channel numbers</td>
    </tr>
    <tr>
        <td><tt>submerged_channel_uv_amplitude</tt></td><td>float64</td><td>(n_submerged_channels)</td><td>Amplitudes of OF UV fits in individual submerged channels</td>
    </tr>
    <tr>
        <td><tt>submerged_channel_fit_chi2</tt></td><td>float64</td><td>(n_submerged_channels)</td><td>Chi-squared values of OF UV fits in individual submerged channels</td>
    </tr>
    <tr>
        <td><tt>submerged_channel_uv_offset</tt></td><td>int32</td><td>(n_submerged_channels)</td><td>Time offsets of UV template in the OF fits in individual submerged channels in samples</td>
    </tr>
    <tr>
        <td><tt>submerged_sum_uv_amplitude</tt></td><td>float64</td><td>-</td><td>OF UV fit amplitude in the sum of submerged channels</td>
    </tr>
    <tr>
        <td><tt>submerged_sum_fit_chi2</tt></td><td>float64</td><td>-</td><td>OF UV fit chi-squared value in sum of submerged channels</td>
    </tr>
    <tr>
        <td><tt>submerged_sum_uv_offset</tt></td><td>int32</td><td>-</td><td>OF UV fit template offset in the sum of submerged channels in samples</td>
    </tr>
    <tr>
        <td><tt>submerged_triggered_uv_amplitude</tt></td><td>float64</td><td>-</td><td>OF UV fit amplitude in the sum of triggered submerged channels</td>
    </tr>
    <tr>
        <td><tt>submerged_triggered_fit_chi2</tt></td><td>float64</td><td>-</td><td>OF UV fit chi-squared value in the sum of triggered submerged channels</td>
    </tr>
    <tr>
        <td><tt>submerged_triggered_uv_offset</tt></td><td>int32</td><td>-</td><td>OF UV fit template offset in the sum of triggered submerged channels in samples</td>
    </tr>
    <tr>
        <td><tt>submerged_triggered_channel_masks</tt></td><td>bool_</td><td>(n_channels)</td><td>Mask of triggered submerged channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_channel_uv_amplitude</tt></td><td>float64</td><td>(n_vacuum_channels)</td><td>UV amplitudes of 2-template OF fits in individual vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_channel_qp_amplitude</tt></td><td>float64</td><td>(n_vacuum_channels)</td><td>QP amplitudes of 2-template OF fits in individual vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_channel_fit_chi2</tt></td><td>float64</td><td>(n_vacuum_channels)</td><td>Chi-squared values of 2-template OF fits in individual vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_channel_uv_offset</tt></td><td>int32</td><td>(n_vacuum_channels)</td><td>UV template time shifts in 2-template OF fits in individual vacuum channels in samples</td>
    </tr>
    <tr>
        <td><tt>vacuum_channel_qp_offset</tt></td><td>int32</td><td>(n_vacuum_channels)</td><td>QP template time shifts in 2-template OF fits in individual vacuum channels in samples</td>
    </tr>
    <tr>
        <td><tt>vacuum_sum_uv_amplitude</tt></td><td>float64</td><td>-</td><td>UV amplitude of 2-template OF fits in the sum of vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_sum_qp_amplitude</tt></td><td>float64</td><td>-</td><td>QP amplitude of 2-template OF fits in the sum of vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_sum_fit_chi2</tt></td><td>float64</td><td>-</td><td>2-template OF fit chi-squared value in the sum of vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_sum_uv_offset</tt></td><td>int32</td><td>-</td><td>UV template time shift in the 2-template OF fit in the sum of vacuum channels in samples</td>
    </tr>
    <tr>
        <td><tt>vacuum_sum_qp_offset</tt></td><td>int32</td><td>-</td><td>QP template time shift in the 2-template OF fit in the sum of vacuum channels in samples</td>
    </tr>
    <tr>
        <td><tt>vacuum_triggered_uv_amplitude</tt></td><td>float64</td><td>-</td><td>UV amplitude of 2-template OF fits in the sum of triggered vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_triggered_qp_amplitude</tt></td><td>float64</td><td>-</td><td>QP amplitude of 2-template OF fits in the sum of triggered vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_triggered_fit_chi2</tt></td><td>float64</td><td>-</td><td>2-template OF fit chi-squared value in the sum of triggered vacuum channels</td>
    </tr>
    <tr>
        <td><tt>vacuum_triggered_uv_offset</tt></td><td>int32</td><td>-</td><td>UV template time shift in the 2-template OF fit in the sum of triggered vacuum channels in samples</td>
    </tr>
    <tr>
        <td><tt>vacuum_triggered_qp_offset</tt></td><td>int32</td><td>-</td><td>QP template time shift in the 2-template OF fit in the sum of triggered vacuum channels in samples</td>
    </tr>
    <tr>
        <td><tt>vacuum_triggered_channel_masks</tt></td><td>bool_</td><td>(n_channels)</td><td>Mask of triggered vacuum channels</td>
    </tr>
    <tr>
        <td><tt>triggered_uv_amplitude</tt></td><td>float64</td><td>-</td><td>UV amplitude of 2-template OF fits in the sum of all triggered channels</td>
    </tr>
    <tr>
        <td><tt>triggered_qp_amplitude</tt></td><td>float64</td><td>-</td><td>QP amplitude of 2-template OF fits in the sum of all triggered channels</td>
    </tr>
    <tr>
        <td><tt>triggered_fit_chi2</tt></td><td>float64</td><td>-</td><td>2-template OF fit chi-squared value in the sum of all triggered channels</td>
    </tr>
    <tr>
        <td><tt>triggered_uv_offset</tt></td><td>int32</td><td>-</td><td>UV template time shift in the 2-template OF fit in the sum of all triggered channels in samples</td>
    </tr>
    <tr>
        <td><tt>triggered_qp_offset</tt></td><td>int32</td><td>-</td><td>QP template time shift in the 2-template OF fit in the sum of all triggered channels in samples</td>
    </tr>
    <tr>
        <td><tt>triggered_channel_masks</tt></td><td>bool_</td><td>(n_channels)</td><td>Mask of all triggered channels</td>
    </tr>
    <tr>
        <td><tt>sum_uv_amplitude</tt></td><td>float64</td><td>-</td><td>UV amplitude of 2-template OF fits in the sum of all channels</td>
    </tr>
    <tr>
        <td><tt>sum_qp_amplitude</tt></td><td>float64</td><td>-</td><td>QP amplitude of 2-template OF fits in the sum of all channels</td>
    </tr>
    <tr>
        <td><tt>sum_fit_chi2</tt></td><td>float64</td><td>-</td><td>2-template OF fit chi-squared value in the sum of all channels</td>
    </tr>
    <tr>
        <td><tt>sum_uv_offset</tt></td><td>int32</td><td>-</td><td>UV template time shift in the 2-template OF fit in the sum of all channels in samples</td>
    </tr>
    <tr>
        <td><tt>sum_qp_offset</tt></td><td>int32</td><td>-</td><td>QP template time shift in the 2-template OF fit in the sum of all channels in samples</td>
    </tr>
</table>

## Description

Plugin that applies Optimum Filter (OF) fits to events. Two-template OF is applied to the vacuum channels, with templates
corresponding to UV and QP signals. Regular one UV template OF is applied to the submerged channels.

Currently, the plugin does not take advantage of the fact that events are longer than the templates. The fits are
performed in a fixed window of `of_length` starting at the sample `-allowed_fit_shifts[0]`. The time shifts in the fits
are actually performed by rolling the templates around their edges.
