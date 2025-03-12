import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from utils.plot_utils.plot import watermark, infotext, Plotter, AnyObjectHandler

class WaveformPlotter():
    
    def __init__(self,rqs,data_path):
        # initialize class with run id to load waveform data file
        # rqs object and run_id need to correspond to the same dataset
        # I don't like having these as separate parameters, it opens possibility
        # of a mismatch.
        # this will likely be substituted with a loader function to load raw data (?)
        # this loads the traces that are ouputed from Processor.py so that total channel
        # is there. But this can be changed.
        
        self.rqs = rqs['data']
        
        try:
            self.traces = np.load(data_path)['data']
        except Exception as e: print(e)
        
        # sampling rate is harded coded - needs to be changed
        self.sampling_rate = 2.5599999275982555e-07 # s
        
        
    
    def plot_waveforms(self,trace_index,channels=-1,xunits='bins',baseline_subtracted=False,template_fit_OF=False,num_cols=1,save_path=None):
        '''
        function to plot a waveforms
        
        initial plotter function - needs working on. Especially to decide how waveform data is loaded.
        
        Params:
          trace_index : trace index or indicies to plot. Can be int or list of ints.
          channels : which channels to plot. waveforms for each channel plotted on same plot. -1 means total channel. Can be int or list of ints or "all" to indicate plot all channels.
          xunits : whether to plot waveforms in units of bins or time (ms). Default is bins.
          baseline_subtracted : whether or not subtract the mean baselines from the traces for each channel and each trace index. Default is False.
          template_fit_OF : whether or not to also plot the fitted OF template for each waveform. Default is False. NOTE: template file is hardcoded but it should be variable. NOTE: template only uses OF amplitude but not OF time offset. This should be added. 
          num_cols : int, number of columns to display
          save_path : to save the figure, provide the location to save.
        
        '''
        
        if isinstance(channels,int):
            channels = [channels]
        if isinstance(trace_index,int):
            trace_index = [trace_index]
        
        if template_fit_OF:
            template = np.load('./data/template.npy')
        if xunits == 'bins':
            xlabel_str = 'time bins'
        elif xunits == 'time':
            xlabel_str = r"time $t$ (ms)"
        
        plt.style.use('./utils/plot_utils/style/delight.mplstyle')
        num_plots = len(trace_index)
        num_rows = int(np.ceil(num_plots / num_cols))

        figwidth = num_cols * 8
        figheight = figwidth * num_rows / num_cols * 6 / 8
        figsize = (figwidth, figheight)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        
        for i, t_id in enumerate(trace_index):
            row = i // num_cols
            col = i % num_cols  
            if num_plots == 1:
                ax = axes
            elif num_rows == 1:
                ax = axes[col]
            elif num_cols == 1:
                ax = axes[row]
            else:
                ax = axes[row,col]
                          

            ch_inds = self.rqs[np.where(self.rqs['trace_index']==t_id)]['channel']
            y_data = self.traces[np.where(self.rqs['trace_index']==t_id)]
            
            # if baseline_subtracted, subtract the baseline
            if baseline_subtracted:
                mean_baselines = self.rqs[np.where(self.rqs['trace_index']==t_id)]['mean_baseline']
            # if adding the fitted OF templates
            if template_fit_OF:
                OF_amps = self.rqs[np.where(self.rqs['trace_index']==t_id)]['OF_ampl']
                # ignoring time shifting for now!!
                
                if not baseline_subtracted:
                    # in this case, we need to mean baseline rq
                    mean_baselines = self.rqs[np.where(self.rqs['trace_index']==t_id)]['mean_baseline']
                
            config = {  'ylabel': 'ADC counts (a.u.)',
                        'xlabel': xlabel_str,
                        'infotext': infotext(t_id,baseline_subtracted=baseline_subtracted),
                        'xlabel_shift': True,
                        'ax': ax
                    }
            # if channels is all, all channels will be plotted
            if channels == 'all':
                use_channels = ch_inds
            else:
                use_channels = channels
            for ch in use_channels:
                ch_ind = np.where(ch_inds == ch)[0]
                y_data_ch = y_data[ch_ind[0]]
                if baseline_subtracted:
                    y_data_ch += -1*mean_baselines[ch_ind[0]]
                    
                if template_fit_OF:
                    y_data_template = template*OF_amps[ch_ind[0]]
                    if not baseline_subtracted:
                        y_data_template += mean_baselines[ch_ind[0]]
                if ch == -1:
                    ch_str = 'tot'
                else:
                    ch_str = str(ch)
                if xunits == 'time':
                    x_data = np.arange(len(y_data_ch))*self.sampling_rate*1000
                    ax.plot(x_data,y_data_ch,'-',label='trace CH_'+ch_str)
                    if template_fit_OF:
                        ax.plot(x_data,y_data_template,':',label='OF fit CH_'+ch_str,color='k')
                else:
                    ax.plot(y_data_ch,'-',label='trace CH_'+ch_str)
                    if template_fit_OF:
                        ax.plot(y_data_template,':',label='OF fit CH_'+ch_str,color='k')
            Plotter(config).pretty()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
        
        
        
        return
        
        
        