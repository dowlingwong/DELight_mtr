# Quantites contained in fit_result attribute of context
| Field Description | Field Name | Data Type | Shape |
|------------------|-----------|-----------|-------|
| Start time since unix epoch [ns] | time | `<i8` | - |
| Length of the interval in samples | length | `<i8` | - |
| Width of one sample [ns] | dt | `<i2` | - |
| Id of the block of records in the run | block_id | `<i4` | - |
| Event number in the record block | event_id | `<i2` | - |
| Channel numbers | channels | `<i2` | (50) |
| Amplitudes of OF UV fits in individual submerged channels | submerged_channel_uv_amplitude | `<f8` | (35) |
| Chi-squared values of OF UV fits in individual submerged channels | submerged_channel_fit_chi2 | `<f8` | (35) |
| Time offsets of UV template in the OF fits in individual submerged channels in samples | submerged_channel_uv_offset | `<i4` | (35) |
| OF UV fit amplitude in the sum of submerged channels | submerged_sum_uv_amplitude | `<f8` | - |
| OF UV fit chi-squared value in sum of submerged channels | submerged_sum_fit_chi2 | `<f8` | - |
| OF UV fit template offset in the sum of submerged channels in samples | submerged_sum_uv_offset | `<i4` | - |
| OF UV fit amplitude in the sum of triggered submerged channels | submerged_triggered_uv_amplitude | `<f8` | - |
| OF UV fit chi-squared value in the sum of triggered submerged channels | submerged_triggered_fit_chi2 | `<f8` | - |
| OF UV fit template offset in the sum of triggered submerged channels in samples | submerged_triggered_uv_offset | `<i4` | - |
| Mask of triggered submerged channels | submerged_triggered_channel_masks | `u1` | (50) |
| UV amplitudes of 2-template OF fits in individual vacuum channels | vacuum_channel_uv_amplitude | `<f8` | (15) |
| QP amplitudes of 2-template OF fits in individual vacuum channels | vacuum_channel_qp_amplitude | `<f8` | (15) |
| Chi-squared values of 2-template OF fits in individual vacuum channels | vacuum_channel_fit_chi2 | `<f8` | (15) |
| UV template time shifts in 2-template OF fits in individual vacuum channels in samples | vacuum_channel_uv_offset | `<i4` | (15) |
| QP template time shifts in 2-template OF fits in individual vacuum channels in samples | vacuum_channel_qp_offset | `<i4` | (15) |
| UV amplitude of 2-template OF fits in the sum of vacuum channels | vacuum_sum_uv_amplitude | `<f8` | - |
| QP amplitude of 2-template OF fits in the sum of vacuum channels | vacuum_sum_qp_amplitude | `<f8` | - |
| 2-template OF fit chi-squared value in the sum of vacuum channels | vacuum_sum_fit_chi2 | `<f8` | - |
| UV template time shift in the 2-template OF fit in the sum of vacuum channels in samples | vacuum_sum_uv_offset | `<i4` | - |
| QP template time shift in the 2-template OF fit in the sum of vacuum channels in samples | vacuum_sum_qp_offset | `<i4` | - |
| UV amplitude of 2-template OF fits in the sum of triggered vacuum channels | vacuum_triggered_uv_amplitude | `<f8` | - |
| QP amplitude of 2-template OF fits in the sum of triggered vacuum channels | vacuum_triggered_qp_amplitude | `<f8` | - |
| 2-template OF fit chi-squared value in the sum of triggered vacuum channels | vacuum_triggered_fit_chi2 | `<f8` | - |
| UV template time shift in the 2-template OF fit in the sum of triggered vacuum channels in samples | vacuum_triggered_uv_offset | `<i4` | - |
| QP template time shift in the 2-template OF fit in the sum of triggered vacuum channels in samples | vacuum_triggered_qp_offset | `<i4` | - |
| Mask of triggered vacuum channels | vacuum_triggered_channel_masks | `u1` | (50) |
| UV amplitude of 2-template OF fits in the sum of all triggered channels | triggered_uv_amplitude | `<f8` | - |
| QP amplitude of 2-template OF fits in the sum of all triggered channels | triggered_qp_amplitude | `<f8` | - |
| 2-template OF fit chi-squared value in the sum of all triggered channels | triggered_fit_chi2 | `<f8` | - |
| UV template time shift in the 2-template OF fit in the sum of all triggered channels in samples | triggered_uv_offset | `<i4` | - |
| QP template time shift in the 2-template OF fit in the sum of all triggered channels in samples | triggered_qp_offset | `<i4` | - |
| Mask of all triggered channels | triggered_channel_masks | `u1` | (50) |
| UV amplitude of 2-template OF fits in the sum of all channels | sum_uv_amplitude | `<f8` | - |
| QP amplitude of 2-template OF fits in the sum of all channels | sum_qp_amplitude | `<f8` | - |
| 2-template OF fit chi-squared value in the sum of all channels | sum_fit_chi2 | `<f8` | - |
| UV template time shift in the 2-template OF fit in the sum of all channels in samples | sum_uv_offset | `<i4` | - |
| QP template time shift in the 2-template OF fit in the sum of all channels in samples | sum_qp_offset | `<i4` | - |

# Headers in rqs
| Column Name         |
|---------------------|
| time               |
| channel            |
| trace_index        |
| temperature        |
| mean_baseline      |
| std_baseline       |
| mean               |
| std                |
| A                  |
| rise_time          |
| TF_ampl            |
| TF_chi2            |
| TF_baseline        |
| baseline_slope     |
| baseline_offset    |
| time_shift         |
| OF_ampl_0         |
| OF_chi2_0         |
| OF_time_0         |
| OF_ampl_1         |
| OF_chi2_1         |
| OF_time_1         |
