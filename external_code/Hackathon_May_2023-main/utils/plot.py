import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

def infotext(evID='', timestamp='', slopemax='', ntraces='', ntest='', pzmin='', chi2=''):
    
    text = []
    
    if evID:
        text.append(r'Trace ID ' + str(evID))    
    if timestamp:
        text.append(rf'$t_0 = $ {timestamp:.2f} ms ?')
    if ntraces:
        text.append(r'$N_{traces}=$' + str(ntraces))
    if pzmin:
        text.append(r'$p_z^{min}=$' + str(pzmin) + ' GeV/c')
    if slopemax:
        text.append(r'$s^{max}=$' + str(slopemax))
    if chi2:
        text.append(r'$\chi^2_\nu=$' + str(chi2))
        
    infotext = ', '.join(text)
    return infotext  

def watermark(ax=None, t=None,logo="DELight", px=0.033, py=0.9, fontsize=20, information=None, information2=None, t_color='#666666', alpha_logo=0.95, shift=0.155, bstyle='italic', scale = 1.4, *args, **kwargs):
    """
    Args:
        t:
        logo:
        px:
        py:
        fontsize:
        alpha:
        shift:
        *args:
        **kwargs:
    Returns:
    """
    if ax is None:
        ax = plt.gca()

    if t is None:
        import datetime
        t = " %d (R&D data)" % datetime.date.today().year
    
    scaletype = ax.get_yscale()
    if scaletype == 'log':   
        bottomylim, topylim = ax.get_ylim()
        ax.set_ylim(top=bottomylim+(topylim-bottomylim)**scale)
    else:
        bottomylim, topylim = ax.get_ylim()
        ax.set_ylim(top=bottomylim+(topylim-bottomylim)*scale)
    
    ax.text(px, py, logo, ha='left',
             transform=ax.transAxes,
             fontsize=fontsize,
             style=bstyle,
             alpha=alpha_logo,
             weight='bold',
             *args, **kwargs,
             # fontproperties=font,
             # bbox={'facecolor':'#377eb7', 'alpha':0.1, 'pad':10}
             )
    ax.text(px + shift, py, t, ha='left',
             transform=ax.transAxes,
             fontsize=fontsize,
             #          style='italic',
             color=t_color,  *args, **kwargs
             #          fontproperties=font,
             # bbox={'facecolor':'#377eb7', 'alpha':0.1, 'pad':10}
             )
    
    ax.text(px, py-0.08, information, transform=ax.transAxes, fontsize=16)
    ax.text(px, py-0.16, information2, transform=ax.transAxes, fontsize=16)
    


class Plotter:
    """
    A class for creating DELight plots.

    Parameters
    ----------
    data : array_like
        The data to plot.
    config : dict, optional
        A dictionary containing plot configuration information.
        Valid keys and their default values are:
            plot_kwargs : dict, default={'linestyle': '-', 'label': ''}
                A dictionary of keyword arguments for the plot command.
            xlabel : str, default=None
                The label for the x-axis.
            ylabel : str, default=None
                The label for the y-axis.
            title : str, default=None
                The title for the plot.
            legend_kwargs : dict, default={'loc': 'best', 'frameon': True,
                                            'framealpha': 1, 'facecolor': 'white',
                                            'edgecolor': 'white', 'bbox_to_anchor': [1., 1.]}
                A dictionary of keyword arguments for the legend command.

    Attributes
    ----------
    fig : matplotlib Figure
        The Figure object for the plot.
    ax : matplotlib Axes
        The Axes object for the plot.

    Examples
    --------
    Create a DELight plot:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from plot import Plotter

    >>> # Generate some example data
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y1 = np.sin(x)
    >>> y2 = np.cos(x)

    >>> # Create a DELight plot
    >>> config = {
    ...     'plot_kwargs': {'linestyle': '-', 'label': 'sin(x)'},
    ...     'xlabel': 'x',
    ...     'ylabel': 'y',
    ...     'title': 'Plot of sin(x) and cos(x)',
    ...     'legend_kwargs': {'loc': 'upper right', 'frameon': True, 'framealpha': 1,
    ...                        'facecolor': 'white', 'edgecolor': 'white', 'bbox_to_anchor': [1., 1.]}
    ... }
    >>> p = Plotter(config=config)
    >>> p.ylabel = 'new ylabel'
    >>> p.plot([x, y2])
    >>> p.show()

    """

    def __init__(self, config={}):
        
        # Load config and set defaults for optional config variables
        self.ax = config.get('ax', None)
        self.figsize = config.get('figsize', (8,6))
        self.xlabel = config.get('xlabel', r"time $t$ (ms)")
        self.ylabel = config.get('ylabel', "ADC counts (a.u.)")
        self.title = config.get('title', "spectrum_for_1p3eV")
        self.infotext = config.get('infotext', '')
        self.yscale = config.get('yscale', 'linear')

        # self.plot_kwargs = {}
        # self.hist_kwargs= {'histtype': 'stepfilled', 'facecolor':'#006477'}
        self.legend_kwargs = {
            'loc': 'upper right',
            'frameon': True,
            'framealpha': 1,
            'facecolor': 'white',
            'edgecolor': 'white',
            'bbox_to_anchor': [1., 0.9]
        }
        self.xlabel_kwargs = {}
        self.ylabel_kwargs = {}

        # Update default values with user-provided config
        if config:
            # if 'plot_kwargs' in config:
            #     self.plot_kwargs.update(config['plot_kwargs'])
            # if 'hist_kwargs' in config:
            #     self.hist_kwargs.update(config['hist_kwargs'])
            if 'legend_kwargs' in config:
                self.legend_kwargs.update(config['legend_kwargs'])
            if ('xlabel_shift' in config) and (config['xlabel_shift'] ==True):
                self.xlabel_kwargs.update({'x':0.9, 'ha':'right'})

        if self.ax is None:
            try:
                self.ax = plt.gca()
            except:
                fig = plt.figure(figsize=self.figsize)
                self.ax = plt.gca()
                
    def plot(self, xdata, *ydata, **kwargs):
        plt.style.use('styles/delight.mplstyle')
        if ('linestyle' not in kwargs.keys()) and ('ls' not in kwargs.keys()):
            kwargs['linestyle'] = '-'
        self.ax.plot(xdata, *ydata,'-', **kwargs)
        self.pretty()

    def hist(self, data, **kwargs):
        plt.style.use('styles/delight_hist.mplstyle')
        if 'histtype' not in kwargs.keys():
            kwargs['histtype'] = 'stepfilled'
        if 'facecolor' not in kwargs.keys():
            kwargs['facecolor'] = '#006477'

        hist = plt.hist(data, **kwargs)
        binwidth = np.mean(np.diff(hist[1]))
        self.ylabel = f'Entries / ({binwidth:.2e})'       
        self.pretty()
        return hist
        
    def pretty(self):
        self.ax.set_yscale(self.yscale)
        watermark(ax=self.ax, information=self.infotext)
        self.ax.set_xlabel(self.xlabel, **self.xlabel_kwargs)
        self.ax.set_ylabel(self.ylabel, **self.ylabel_kwargs)
        self.ax.set_title(self.title, loc='right', color='#666666')
        if self.ax.get_legend() is None:
            handles, labels = self.ax.get_legend_handles_labels()
            if handles and labels:
                self.ax.legend(**self.legend_kwargs)

class AnyObjectHandler(HandlerBase):
    
    def __init__(self, cmap=None, num_lines=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = plt.cm.get_cmap(cmap)
        self.num_lines = num_lines
        
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        
        lines = []
        if orig_handle == object:
            for i in range(self.num_lines):
                frac = i/self.num_lines
                s = plt.Line2D([x0,y0+width], 
                               [frac*height,frac*height], 
                               color=self.cmap(frac))
                lines.append(s)

        else:
            if type(orig_handle) is tuple:
                for i, handle in enumerate(orig_handle):
                    frac = i/len(orig_handle)
                    s = plt.Line2D([x0,y0+width], 
                                [frac*height,frac*height], 
                                    color=handle.get_c(),
                                    ls=handle.get_ls(),
                                    marker=handle.get_marker())
                    lines.append(s)
            else:
                line = plt.Line2D([x0,y0+width], 
                                [0.5*height, 0.5*height],
                                color=orig_handle[0].get_c(),
                                ls=orig_handle[0].get_ls(),
                                marker=orig_handle[0].get_marker())
                lines.append(line)
        return lines
            
