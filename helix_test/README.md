# Helix

Helix is a [strax](https://github.com/AxFoundation/strax)-based data processing software for
[DELight](https://delight.kit.edu/) (superfluid Helium Dark Matter search experiment) and Magnetic Micro-Calorimeter
(MMC) R&D projects.

Helix contains a set of strax plugins for offline (and in the future, possibly online) processing of data coming from
MMCs. The main goal of the processing is to locate energy deposition events in the data streams (triggering), reconstruct
their energy, and provide any additional information needed for further analysis of the data (e.g., position reconstruction,
event classification, etc). Helix provides tools for
noise characterization (Power and Cross-Spectral Densities) and vectorized Optimum Filter energy reconstruction.

Helix should eventually become a self-sufficient interface between the DELight data and the analyzers, capable of locating
requested data, processing it in an efficient and highly configurable way suitable for different types of analyses,
and providing tools necessary for analyzing these data. See [straxen](https://github.com/XENONnT/straxen) (XENONnT
strax-based framework) for inspiration.  

## Installation
To install Helix locally, clone the repo and install it with pip:
1. `git clone git@gitlab.etp.kit.edu:delight/helix.git` or `git clone https://gitlab.etp.kit.edu/delight/helix.git`
2. `pip install -e ./helix`

The `-e` option installs Helix in an editable mode, meaning that all the local changes applied to the source code take
immediate effect without the need of reinstalling the package every time a change to the code is made.

To test the installation, open and run `helix/test_notebook.ipynb`. If you get a plot with channel fits at the end, the
installation is successful. Please, clear all the cell outputs after running the notebook. Do not push cell outputs to
the repo.

Helix is tested for python versions 3.10 and 3.11.

## Usage
For a more complete usage instructions refer to the [strax documentation](https://strax.readthedocs.io/en/latest/).
Currently, Helix only contains a collection of strax plugins. All the interactions between user and plugins happen via
the strax interface, namely via a strax `Context` object. Here is an example of how a user can interact with Helix:

- Create a strax `Context` object, provide a `storage` to it (a directory to store strax outputs and, optionally, run metadata), 
and register required Helix plugins.
- To pass configuration options to registered plugins, pass a dictionary of options to the `context.set_config()` method.
- With the plugins registered and configured as necessary, run the processing or retrieve the data processed earlier via 
the `context.get_array(run_id, data_type)` method, where `run_id` is the run identifier telling your data-reading plugin
which data to read, and `data_type` is a string specifying the type of data you need (must match the `provides` argument
of one of the registered plugins).

Example:
```python
import helix as hx
import strax as sx

run = 'run10'
output_dir = 'test_helix_data'

# creating context, registering storage and plugins
context = sx.Context(storage=[sx.DataDirectory(output_dir, provide_run_metadata=True), ],
                     register=[hx.ToyDataRawRecords,
                               hx.QPTriggers, hx.UVTriggers,
                               hx.Events, hx.NoiseEvents,
                               hx.NoisePSDs, hx.FitResults])    # all the plugins required for getting fit_results

# creating a dictionary of plugins' options that we want to modify. 
config = {'run_metadata_directory': output_dir,      # for the hx.ToyDataRawRecords plugin
          'noise_events_random_seed': 1}             # for the hx.NoiseEvents plugin

# passing the settings to the plugins. Strax finds which plugins take these options automatically
context.set_config(config)

# running the processing (or retrieving the data, if it was processed earlier)
fit_results = context.get_array(run, 'fit_results')  # hx.FitResults plugin provides this data type 

# fit_results is a structured numpy array of events containing fields described in the FitResults documentation
# one can work with it as a dictionary on numpy arrays, or as a numpy array of dictionaries. Works both ways. E.g.
# fit_results['sum_uv_amplitude'][:100]  # amplitudes of UV signals in the sum of all channels in the first 100 events
# fit_results[:100]['sum_uv_amplitude']  # same
```

## Plugins
- [Helix plugins](docs/plugin_tree.md)
- [How to develop plugins](docs/plugin_development.md)

## Other features
- [Optimum Filter](docs/optimum_filter.md)
- [PSDs](docs/psds.md)
- [Channel map](docs/channel_map.md)

## Miscellaneous
- [Helix and strax lingo](docs/lingo.md)

## Contributing
To contribute to Helix, follow git workflow. Create a new feature branch from the develop branch. Develop your feature.
Test it. Merge back to develop when the feature is ready. Merge regularly, do not diverge from develop too much.
