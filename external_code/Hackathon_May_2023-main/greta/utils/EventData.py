import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import pickle

from scipy.optimize import curve_fit, minimize
from scipy.ndimage.filters import gaussian_filter1d
from scipy import signal

from iminuit import Minuit
from iminuit import cost
from jacobi import propagate

import warnings

from .DAQRead import *
from .plot import watermark, infotext, Plotter, AnyObjectHandler

warnings.filterwarnings("ignore")

class EventData: 
    """
    Class to process and analyze raw MMC detector data.
    """
    
    def __init__(self, filename, reload=False):

        """
        Initializes an EventData object.

        Parameters:
        - filename (str): The path to the raw data file.
        - reload (bool): Flag to reload the data or load from pre-loaded pickle file instead.

        Returns:
        - None
        """
        self.filename = filename
        self.reload = reload
        self.raw, self.scopesettings = self.load_raw()
        self.data = self.raw.data
        self.processed_df = self.process_data()

    # =========================
    # Load MMC binary data
    # =========================    

    def load_raw(self):
        """
        Loads the raw data from a binary file or pre-loaded pickle file instead.

        Returns:
        - raw (DataFrame): The raw data.
        - scopesettings (dict): The settings of the scope.

        Raises:
        - FileNotFoundError: If the file is not found.
        """
        if self.reload:
            raw = DAQReader(f"data/{self.filename}.raw")
            raw.read_pulses(True)
            df = pd.DataFrame.from_records(raw.data.tolist(), columns=['Text', 
                                                                       'TraceNo', 
                                                                       'TimeStamp', 
                                                                       'Sec100', 
                                                                       'Temperature', 
                                                                       'xref', 
                                                                       'yref', 
                                                                       'data'])
            df = df.set_index('TraceNo')
            df = df.drop(columns = ['Text', 'Sec100'])

            df.to_pickle(f"data/{self.filename}.pkl")
            scopesettings = vars(raw.ScopeSettings)
            pickle.dump(scopesettings, open(f"data/{self.filename}_settings.p", "wb"))

        else:
            df = pd.read_pickle(f"data/{self.filename}.pkl")
            scopesettings = pickle.load(open(f"data/{self.filename}_settings.p", "rb"))
            
        return df, scopesettings
    
    # ===============================================
    # Process data & calculate basic pulse attributes
    # ===============================================
    
    def _get_rising_edge_arg(self, event):
        """calculates the index of the rising edge"""
        rising_arg = np.argmax(np.diff(event, n=2)>5*np.std(np.diff(event, n=2)))
        if rising_arg == 0:
            rising_arg = np.argmax(np.diff(event, n=2)>4*np.std(np.diff(event, n=2)))
        return rising_arg
    
    def _get_time(self, length, rising_arg):
        """calculates the shifted time in an event, such that time=0 at the beginning of the rising edge"""
        time = (np.arange(length)-rising_arg) * self.scopesettings['xfac'] #in s
        return time
    
    def _get_baseline(self, event, rising_arg):
        """calculates the baseline mean and standard deviation"""
        baseline_mean = np.mean(event[:rising_arg])
        baseline_std = np.std(event[:rising_arg])
        return baseline_mean, baseline_std
    
    def _get_rising_edge_time(self, event, baseline_mean, pulse_height, time, low=0.2, high=0.8):
        """calculates the rising time between low and high percentage of the pulse amplitude"""
        tlow_arg = np.argmax(event[:np.argmax(event)] - baseline_mean > 0.2*pulse_height)
        thigh_arg = np.argmax(event[:np.argmax(event)] - baseline_mean > 0.8*pulse_height)
        t_rising = (time[thigh_arg] - time[tlow_arg]) * 1000 #in ms
        return t_rising
    
    def _get_peak(self, event, baseline_mean, n_MAfilter=10):
        """calculates the peak position and height"""
        peak_arg = np.argmax(event)
        pulse_height = np.mean(event[peak_arg:peak_arg+n_MAfilter]) - baseline_mean
        return pulse_height, peak_arg        
        
    def process_data(self):
        """
        Process the data and calculate basic pulse attributes.

        Returns:
        - processed_df (DataFrame): DataFrame containing the processed data.
        """
        processed = []
        self.time = pd.DataFrame(index=self.data.index, columns=['time'])
        for evID in self.data.index: 
            event = self.data.loc[evID]
                      
            rising_arg = self._get_rising_edge_arg(event)
            time = self._get_time(len(event), rising_arg)
            self.time.loc[evID].time = time
                       
            baseline_mean, baseline_std = self._get_baseline(event, rising_arg)
            pulse_height, peak_arg = self._get_peak(event, baseline_mean)

            t_rising = self._get_rising_edge_time(event, baseline_mean, pulse_height, time)
                
            processed.append({'pulse_height': pulse_height,
                              'peak_arg': peak_arg,
                              'rising_arg': rising_arg,
                              'baseline_mean': baseline_mean, 
                              'baseline_std': baseline_std,
                              't_rising': t_rising})
        return pd.DataFrame(processed, index=self.data.index)

    # ===============================================
    # Query data
    # ===============================================   

    def query_data(self, df=None, **kwargs):
        """
        Query the data based on specified criteria.

        Args:
            df (DataFrame, optional): Dataframe to query. Defaults to None.
            **kwargs: Keyword arguments representing the query criteria.

        Returns:
            tuple: A tuple containing the filtered dataframe and corresponding data.
        
        Example: 
            ev_data.query_data(**{'pulse_height': (50000,80000), 'baseline_mean': (40000,55000)})

        """

        if df is None:
            df = self.processed_df
        
        query = np.ones(len(df), dtype=bool)
        data_index = pd.Series([True]*len(self.data), index=self.data.index)

        for column, (low, high) in kwargs.items():
            query &= (df[column] >= low) & (df[column] <= high)

        data_index = data_index & query 
        return df.loc[query], self.data[data_index]
    
    
    # ===============================================
    # Plot traces
    # ===============================================

    def plot_trace(self, evID, ax=None):
        """
        Plot a single trace.

        Args:
            evID: ID of the trace to plot.
            ax: Axes object to plot on. Defaults to None.

        """
        
        data = self.data.loc[evID]
        time = self.time.loc[evID].time
       
        config = {  'ylabel': 'ADC counts (a.u.)',
                    'xlabel': r"time $t$ (ms)",
                    'infotext': infotext(evID),
                    'ax': ax                 
                }
        Plotter(config).plot(time*1000, data, label='trace') # time in ms

                 
    
    def plot_traces(self, IDs, num_cols=1, infos=None, save_path=None):
        """
        Plot multiple traces.

        Args:
            IDs: List of trace IDs to plot.
            num_cols: Number of columns in the subplot grid. Defaults to 1.
            infos: Information about the traces. Defaults to None.
            save_path: Path to save the plot. Defaults to None.

        """    
        plt.style.use('./style/delight.mplstyle')
        num_plots = len(IDs)
        num_rows = int(np.ceil(num_plots / num_cols))

        figwidth = num_cols * 8
        figheight = figwidth * num_rows / num_cols * 6 / 8
        figsize = (figwidth, figheight)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
              
        for i, evID in enumerate(tqdm(IDs)):
              row = i // num_cols
              col = i % num_cols  
              self.plot_trace(evID, axes[row,col])
              if infos is not None:
                axes[row,col].set_title(rf'$\chi^2_\nu=$ {infos[evID]:.1e}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    # ===============================================
    # Plot correlations
    # ===============================================

    def plot_correlations(self,df):
        """
        Plot correlations between pulse attributes/ RQs in a DataFrame.

        Args:
            df (DataFrame): DataFrame containing the data.

        """
        
        # Create scatter plot matrix
        plt.style.use('./style/delight.mplstyle')
    
        # Create scatter plot matrix
        fig, ax = plt.subplots(nrows=len(df.columns), ncols=len(df.columns),
                            figsize=(20,20), sharex='col', sharey='row')

        # Loop through each variable and create scatter plot with histograms
        for i, col1 in enumerate(df.columns):
            for j, col2 in enumerate(df.columns):

                if i!=j:
        #             # scatter correlation plots
        #             ax[i,j].scatter(df[col2], df[col1], alpha=0.5)
                    # 2D histogram
                    ax[i,j].hist2d(df[col2], df[col1], bins=40, cmap='viridis')

                else:
                    # Add histogram on diagonal and rescale
                    n, bins, patches = ax[i,j].hist(df[col1], bins=30)
                    hist_scale = n.max()
                    axis_scale = abs(df[col1].max())-abs(df[col1].min())
                    scaling_factor = axis_scale / hist_scale * 0.9

                    for patch in patches:
                        patch.set_height(patch.get_height()*scaling_factor+df[col1].min())

                ax[i,j].tick_params(axis='both', which='major', labelsize=10)
                ax[i,j].yaxis.offsetText.set_fontsize(10)
                ax[i,j].xaxis.offsetText.set_fontsize(10)
                ax[i,j].set_ylim(df[col1].min(), df[col1].max())

                # Set axis labels for bottom row and leftmost column
                if i == len(df.columns) - 1:
                    ax[i,j].set_xlabel(col2, fontsize=26, ha='center')

                if j == 0:
                    ax[i,j].set_ylabel(col1, fontsize=26, ha='center')

        # Plotter({'ax':ax[i,j]}).pretty()

        plt.tight_layout()
        plt.show()    
    
    
    # ===============================================
    # Template Fit
    # ===============================================

    # Build Template
          
    def _select_Kalpha1_samples(self, bins=180, tolerance=0.02, batch_size=200, deviation_factor=2):
        """
        Select samples from Kalpha1 line to have clean and representative samples of traces for the template building.

        Args:
            bins (int): Number of bins for histogram.
            tolerance (float): Tolerance for selecting samples.
            batch_size (int): Size of each batch.
            deviation_factor (int): Factor to filter traces with PIT and other defects.

        Returns:
            pandas.Series: Kalpha1 samples.
        """
        pulse_heights = self.processed_df.pulse_height
        counts, bins, _ = plt.hist(pulse_heights, bins=bins, label='data')
        bin_centers = (bins[1:]+bins[:-1])/2
        #manually chosen frame for Kalpha1 line between 100 and 125
        peak_ind = signal.find_peaks_cwt(counts, np.arange(100,125, 0.5))   
        
        Kalpha1_signal_height = bins[peak_ind][0]
        lower = Kalpha1_signal_height - tolerance * Kalpha1_signal_height
        upper = Kalpha1_signal_height + tolerance * Kalpha1_signal_height
        Kalpha1_samples,_ = self.query_data(**{'pulse_height': (lower, upper)})

        config = {  'ylabel': 'Entries (a.u.)',
            'xlabel': 'signal height (a.u.)',
            'xlabel_shift':True, 
            'yscale' : 'log'                    
            }
        
        Plotter(config).plot(bins[peak_ind], 
                             counts[peak_ind], 
                             marker='x', 
                             color='red', 
                             ls='', 
                             label='peak')
        plt.show()

        return Kalpha1_samples
    
    def _template_sample_filter(self, samples, batch_size=50, deviation_factor=2):
        """
        Filter chosen samples by self-cleaning.

        Args:
            samples (pandas.Series): Template samples.
            batch_size (int): Size of each batch.
            deviation_factor (int): Factor to filter traces with PIT and other defects.

        Returns:
            pandas.Series: Filtered template samples.
        """
        filtered_traces = pd.Series([])

        # Loop over the data and read in traces in batches
        for i in range(0, len(samples), batch_size):
            # Extract a batch of traces
            batch = samples[i:i+batch_size]
                    
            # Filter traces with PIT and other defects
            diffs = np.vstack(np.diff(batch, axis=0))
            diffs = np.concatenate((diffs, [batch.iloc[-1] - batch.iloc[0]]))
            quadratic_diffs = np.sum(diffs**2, axis=1)
            median_diff = np.median(quadratic_diffs)
            
            good_traces = batch[quadratic_diffs < deviation_factor*median_diff]
            filtered_traces = pd.concat([filtered_traces, good_traces])
        return filtered_traces
    
    
    def build_template(self, bins=180, tolerance=0.02, batch_size=200, deviation_factor=2, filename=None):
        """
        Build a template using the selected samples.

        Args:
            bins (int): Number of bins for histogram.
            tolerance (float): Tolerance for selecting samples.
            batch_size (int): Size of each batch.
            deviation_factor (int): Factor to filter traces with PIT and other defects.
            filename (str): Name of the file to save the template.

        Returns:
            pandas.Series: Template.
        """

        samples = self._select_Kalpha1_samples(bins, tolerance, batch_size, deviation_factor)
        self.filtered_traces = self._template_sample_filter(self.data[samples.index], 
                                                       deviation_factor=deviation_factor)
        self.num_filtered_traces = len(self.filtered_traces)
        print(f'{self.num_filtered_traces} traces from peak used for template building')
        self.template = self.filtered_traces.mean()

        rising_arg = self._get_rising_edge_arg(self.template)
        time = self._get_time(len(self.template), rising_arg)
        
        if filename is not None:
            np.save(filename, self.template)
            print(f'template saved to {filename}')

        return self.template

    def plot_template(self, xmin=None, xmax=None, ymin=None, ymax=None):
        """
        Plot the event template and the selected samples for the template building in a single plot.

        Args:
            xmin (float): Minimum x-axis limit.
            xmax (float): Maximum x-axis limit.
            ymin (float): Minimum y-axis limit.
            ymax (float): Maximum y-axis limit.
        """

        for i, trace in enumerate(self.filtered_traces):
            evID = self.filtered_traces.index[i]
            time = self.time.loc[evID].time
            cmap = plt.cm.get_cmap('viridis')
            plt.plot(time*1000, trace, 
                        color=cmap(i/self.num_filtered_traces), 
                        linestyle='-', 
                        marker='', 
                        alpha=0.5)
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
        line_template = plt.plot(time*1000, self.template, label='template', marker='', color='red') #ms
        plt.legend([line_template, object], 
                    ['template', 'traces'],
                    handler_map={object: AnyObjectHandler('viridis', self.num_filtered_traces)})

        config = {'infotext': 'pulse template', 
                  'xlabel': r"time $t$ (ms)"}
        Plotter(config).pretty()                   
    
    # Fit Templates

    def chi_square_reduced(self, params, trace, template, baseline_std):
        """
        Calculate the reduced chi-square value for fitting.

        Args:
            params (list): Fit parameters.
            trace (array): Event trace data.
            template (array): Event template data.
            baseline_std (float): Baseline standard deviation.

        Returns:
            float: Reduced chi-square value.
        """
        A, O = params
        dof = (len(trace)-len(params)) * baseline_std
        return 1/dof * np.sum((trace - A*template - O)**2) 

    def fit_template(self, trace, template, baseline_std, A_guess=1.0, O_guess=0.0):
        """
        Fit the event template to the trace data for a single trace.

        Args:
            trace (array): Event trace data.
            template (array): Event template data.
            baseline_std (float): Baseline standard deviation.
            A_guess (float): Initial guess for amplitude parameter. Default is 1.0.
            O_guess (float): Initial guess for offset parameter. Default is 0.0.

        Returns:
            OptimizeResult: Result of the template fitting.
        """
        return minimize(self.chi_square_reduced, [A_guess, O_guess], args=(trace, template, baseline_std))


    def fit_templates(self, template, traces=None):
        """
        Fit the event template to multiple traces.

        Args:
            template (array): Event template data.
            traces (DataFrame): Traces data. Default is None.

        Returns:
            DataFrame: Processed data with chi-square values.
        """
        if traces is None:
            traces = self.data
        
        if 'chi2r_templ' in self.processed_df.columns:
            self.processed_df = self.processed_df.drop(columns = ['chi2r_templ'])
            
        for i, trace in enumerate(tqdm(traces)):
            baseline_std = self.processed_df.loc[self.processed_df.index[i], 'baseline_std']
            result = self.fit_template(trace, template, baseline_std)
            self.processed_df.loc[self.processed_df.index[i], 'chi2r_templ'] = result.fun
        config = {'xlabel':r'$\chi^2_\nu$ (au)',
                  'xlabel_shift': True,
                  'yscale':'log',
                  'infotext': infotext(ntraces=len(traces))}
        chi2r_templ = self.processed_df.chi2r_templ
        Plotter(config).hist(chi2r_templ, bins=80)

    # ===============================================
    # Clean up dataset
    # ===============================================
    def clean_dataset(self, mask=None):
        """
        Clean up the dataset based on specified filters.

        Args:
            mask (dict): Filter mask. Default is None.

        Returns:
            DataFrame: Cleaned dataset.
            DataFrame: Cleaned data.
        """
        # TODO: add proper mask editing
        if mask is None:
            mask = {'chi2r_templ': (0, 60),
                    'pulse_height': (0.96e4,1.2e4),
                    'baseline_mean': (0,200),
                    't_rising': (2.0e-2,3.0e-2)   
                    }
        df = self.processed_df    
        clean_df, clean_data = self.query_data(df, **mask)
        removed = (len(df)-len(clean_df))/len(df) * 100
        print(f'{removed:.2f} % of original dataset removed by filters')

        return clean_df, clean_data


    
    # ===============================================
    # Energy Calibration
    # ===============================================
    
    def cauchy_scaled(self, x, x0, gamma, amplitude):
        """
        Calculate the Cauchy-scaled function, which is used for the calculation of the theoretical line shapes of the Fe source.

        Args:
            x (array): x-values.
            x0 (float): Center position.
            gamma (float): Width parameter.
            amplitude (float): Amplitude parameter.

        Returns:
            array: Cauchy-scaled values.
        """
        return amplitude / (1 + ((x-x0)/gamma)**2)
    
    def calc_Klines(self, x, energy, rel_int, linewidth, xscale=1., xoff=0.):
        """
        Calculate the K-lines using the cauchy distributions.

        Args:
            x (array): x-values.
            energy (array): Energy values.
            rel_int (array): Relative intensity values.
            linewidth (array): Line width values.
            xscale (float): x-scale factor. Default is 1.0.
            xoff (float): x-offset value. Default is 0.0.

        Returns:
            tuple: K-lines array, K-lines total array.
        """
        energy = energy * xscale + xoff
        linewidth = linewidth * xscale

        Klines = []
        for i in range(len(energy)):
            Kline = self.cauchy_scaled(x, energy[i], linewidth[i]/2, rel_int[i])
            Klines.append(Kline)
        Klines_total = np.sum(Klines, axis=0)

        return Klines, Klines_total
    
    def plot_Klines(self, x, Klines, Klines_total, labels=[r'$K_{ij}$',r'$K_{\alpha 1/2}$']):
        """
        Plot the K-lines (total and sub-lines).

        Args:
            x (array): x-values.
            Klines (array): K-lines data.
            Klines_total (array): K-lines total data.
            labels (list): Labels for the K-lines. Default is ["$K_{\\alpha}$", "$K_{\\beta}$"].
            colors (list): Colors for the K-lines. Default is ["red", "blue"].
        """
        plt.style.use('./style/delight.mplstyle')
        
        spectrals=[]
        for Kline in Klines:
            specs = plt.plot(x, Kline, '--')
            spectrals.append(specs[0])         
        
        Ktotal = plt.plot(x, Klines_total, 'k-')

        plt.legend( [tuple(spectrals), tuple(Ktotal)], 
                    labels,
                    handler_map={tuple: AnyObjectHandler()})

        Plotter({'xlabel':'energy (eV)', 'infotext': 'literature'}).pretty()
        plt.show()
    
    def Klines_literature(self, x, ampl1=1, ampl2=1, xscale=1., xoff=0.):
        """
        Calculate the literature values of K alpha and K beta lines.

        Args:
            x (array): x-values.
            ampl1 (float): Amplitude 1. Default is 1.
            ampl2 (float): Amplitude 2. Default is 1.
            xscale (float): Scaling factor for x-values. Default is 1.
            xoff (float): Offset for x-values. Default is 0.

        Returns:
            tuple: K_lit (literature values), Kalpha_lines (K alpha lines), Kbeta_lines (K beta lines).
        """
        # define literature values of K alpha and K beta lines
        energy_Ka1 = np.array([5898.853, 5897.867, 5894.829, 5896.532, 5899.417, 5902.712])
        rel_int_Ka1 = np.array([0.790, 0.264, 0.068, 0.096, 0.007, 0.0106])
        linewidth_Ka1 = np.array([1.715, 2.043, 4.499, 2.663, 0.969, 1.5528])

        energy_Ka2 = np.array([5887.743, 5886.495])
        rel_int_Ka2 = np.array([0.372, 0.100])
        linewidth_Ka2 = np.array([2.361, 4.216])

        energy_Kb = np.array([6490.89, 6486.31, 6477.73, 6490.06, 6488.83])
        linewidth_Kb = np.array([1.83, 9.40, 13.22, 1.81, 2.81, ])
        rel_int_Kb = np.array([0.608, 0.109, 0.077, 0.397, 0.176])

        energy_Ka = np.concatenate((energy_Ka1, energy_Ka2))
        rel_int_Ka = np.concatenate((rel_int_Ka1, rel_int_Ka2))
        linewidth_Ka = np.concatenate((linewidth_Ka1, linewidth_Ka2))

        Kalpha_lines, Kalpha_total = self.calc_Klines(x, energy_Ka, rel_int_Ka, linewidth_Ka, xscale, xoff)
        Kbeta_lines, Kbeta_total = self.calc_Klines(x, energy_Kb, rel_int_Kb, linewidth_Kb, xscale, xoff)

        K_lit = ampl1 * Kalpha_total + ampl2 * Kbeta_total

        return K_lit, Kalpha_lines, Kbeta_lines

    def energy_fitter(self, df):
        """
        Fit the theoretical energy distributions to the pulse height of the data. This is not working yet. The minimizer does not converge.

        Args:
            df (DataFrame): Event data.

        Returns:
            Minuit: Fitted parameters.
        """

        #TODO: get running, does not converge
        x = df['pulse_height']
        counts, bins = np.histogram(x, bins=3000)
        
        def model(x, ampla, amplb, xscale, xoff, sigma):
            Klines = self.Klines_literature(x, ampla, amplb, xscale, xoff)[0]
            conv = gaussian_filter1d(Klines, abs(sigma))
            return conv
        
        par_guess = 0.08, 0.01, 1.655, 651, 4.3
        c = cost.BinnedNLL(counts, bins, model)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        m = Minuit(c, *par_guess)

        m.limits["ampla", "amplb"] = (0,0.4)
        m.limits["xscale"] = (1.6, 1.7)
        m.limits["xoff"] = (0,1000)
        m.limits["sigma"] = (1, 6)

        # m.interactive()
        m.migrad()
        return m.params
    
    def energy_calculator(self, df):
        """
        Calculate the energy of the traces according to the fitter. The params xscale and xoff of the fit are used
        to calculate the energy by rescaling of the pulse heights of the data

        Args:
            df (DataFrame): Event data.
        """
        #TODO: replace by fitter results
        # ampla, amplb, xscale, xoff, sigma = energy_fitter(df)              #fitter does not converge
        ampla, amplb, xscale, xoff, sigma = 430, 48, 1.655, 651, 4.1
        
        self.processed_df['energy'] = (self.processed_df['pulse_height']-xoff)/xscale
        print(f'energy = (pulse heights - {xoff}) * {xscale}')

        f, (ax, ax2) = plt.subplots(1, 2, sharey=True, facecolor='w', figsize=(14, 6))
        energy = (df['pulse_height']-xoff)/xscale
        counts, bins, _ = ax.hist(energy, bins=3000, edgecolor='#006477')
        bin_centers = 0.5*(bins[1:]+bins[:-1])

        lit = self.Klines_literature(bin_centers, ampla, amplb)[0]
        lit = gaussian_filter1d(lit, sigma)
        Plotter({'ax':ax,
                'title':None,
                'xlabel':None,
                'watermark_kwargs':{'shift':0.185}}).plot(bin_centers, lit)
        ax.set_xlim(5850, 5929)

        ax2.hist(energy, bins=3000, edgecolor='#006477', label='data')
        ax2.plot(bin_centers, lit, '-', label='fit')
        Plotter({'ax':ax2,
                'show_watermark':False,
                'ylabel':None,
                'xlabel': 'energy(eV)'}).pretty()
        ax2.set_xlim(6461, 6510)


        chisquare = np.sum((counts-lit)**2)
        dofs = 3*len(counts)
        plt.plot([],[],' ', label=rf'$\chi^2/$dof = {chisquare:.1f}/{dofs} = {chisquare/dofs:1f}')
        plt.plot([],[],' ', label=rf'$A_\alpha$ = {ampla} a.u.')
        plt.plot([],[],' ', label=rf'$A_\beta$ = {amplb} a.u.')
        plt.plot([],[],' ', label=r'$\Delta E_{FWHM}$'+rf'= {sigma} eV')
        ax2.legend()

        # hide the spines between ax and ax2
        ax.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        ax.tick_params(axis='y', which='both', left=True, right=False)
        ax2.tick_params(axis='y', which='both', left=False, right=False)

        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((1-d, 1+d), (-d, +d), '-', **kwargs)
        ax.plot((1-d, 1+d), (1-d, 1+d), '-', **kwargs)

        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1-d, 1+d),'-', **kwargs)
        ax2.plot((-d, +d), (-d, +d),'-', **kwargs)

        plt.subplots_adjust(wspace=0.05)
        plt.show()


    # ===============================================
    # Exponential decay Fit (deprecated)
    # ===============================================
    
    def fit_decay(self, evID):
        
        data = self.data.loc[evID]
        baseline = self.processed_df.baseline_mean[evID]
        
        def pulse_func(t, tau1, A1, tau2, A2):
            return A1 * (np.exp(-t/tau1)) + A2 * (np.exp(-t/tau2))
        
        time = (np.arange(len(data))-self.processed_df.rising_arg[evID]) * self.scopesettings['xfac'] #in s
        arg = self.processed_df.peak_arg[evID]

        pulse = data[arg:]

        init_vals = [0.010, 2000, 0.004, 8000]
        
        try:
            popt, pcov = curve_fit(pulse_func, time[arg:], pulse, p0=init_vals)
            pulse_fit = pulse_func(time[arg:],*popt)
            chi2 = np.sum((pulse_fit - pulse)**2 / pulse_fit)
            dof = len(pulse)-len(init_vals)
            chi2_reduced = chi2/dof
        except RuntimeError:
            chi2 = 1e10
            chi2_reduced = 1e8
            popt = [0,0,0,0]
        except TypeError:
            chi2 = 1e10
            chi2_reduced = 1e8
            popt = [0,0,0,0]
        
        return chi2, chi2_reduced, popt
    
    def fit_decays(self, IDs=None):
        
        if IDs==None:
            IDs = self.data.index
        self.processed_df = self.processed_df.drop(columns=['tau1', 'A1', 'tau2', 'A2'], errors='ignore')
        fits = []
        
        for evID in tqdm(IDs): 
            chi2, chi2_reduced, popt = self.fit_decay(evID)
            tau1, A1, tau2, A2 = popt
            fits.append({'chi2': chi2, 'chi2_reduced': chi2_reduced, 'tau1': tau1, 'A1':A1, 'tau2':tau2, 'A2':A2})
            
        self.fits = pd.DataFrame(fits, index=self.data.index[np.array(IDs)-1])
        self.processed_df = pd.concat([self.processed_df, self.fits], axis=1)





# ===============================================
# Event Generator
# ===============================================        

                
class event_generator():
    """
    Class for generating fake events.
    """
    def __init__(self, scopesettings, template, data_len=32768, rising_arg=4035):
        """
        Initialize the event generator.

        Args:
            scopesettings (dict): Scope settings.
            template (array): Event template data.
            data_len (int): Length of the event data. Default is 32768.
            rising_arg (int): Rising argument. Default is 4035.
        """

        self.data_len = data_len
        self.rising_arg = rising_arg
        self.scopesettings = scopesettings
        self.time = self._get_time()
        self.template = template

    def _get_time(self):
        """
        Calculate the time for the x-axis.

        Returns:
            array: time.
        """
        time = (np.arange(self.data_len)-self.rising_arg) * self.scopesettings['xfac'] #in s
        return time

    def baseline(self, baseline_mean=133.4, baseline_std=60.45):
        """
        Generate the baseline.

        Args:
            baseline_mean (float): Mean value of the baseline. Default is 133.4.
            baseline_std (float): Standard deviation of the baseline. Default is 60.45.

        Returns:
            array: Generated baseline data.
        """
        fake_baseline = np.random.normal(baseline_mean, baseline_std, self.data_len//9+1)
        fake_baseline = np.repeat(fake_baseline, 9)
               
        return fake_baseline[:self.data_len]

    def event(self, sigma=3):
        """
        Generate an event.

        Args:
            sigma (float): Sigma value for scaling the pulse. Default is 3.

        Returns:
            array: Generated event data.
        """
        baseline = self.baseline()
        pulse = self.template
        pulse_amp = np.max(pulse)
        scale_fac = pulse_amp / np.std(baseline)
        pulse_scaled = pulse / scale_fac * sigma
        signal = baseline + pulse_scaled
        
        return signal
    

                