import numpy as np
from trigger_study.wk19.OptimumFilter import *

sampling_frequency = 3906250
vac_template = np.load("/home/dwong/DELight_mtr/trigger_study/archive/wk15/templates/vac_ch_template.npy")

noise_psd = np.load("/home/dwong/DELight_mtr/templates/noise_psd_from_MMC.npy")
noise_psd_sum = 37 * noise_psd

vac_of = OptimumFilter(vac_template, noise_psd, sampling_frequency)
noise_trace32 = np.load("/home/dwong/DELight_mtr/trigger_study/wk18/pt_noise.npy")
ampl_noise, chisq_noise = vac_of.sliding_fit(noise_trace32)
np.save("ampl_pt_noise.npy", ampl_noise)
np.save("chisq_pt_noise.npy", chisq_noise)