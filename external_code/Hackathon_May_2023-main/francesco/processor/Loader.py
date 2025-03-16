import os
import numpy as np

def listify(x):
    if not hasattr(x, '__len__') or isinstance(x, str):
        return [x]
    else:
        return x


class Loader():
    """
    Loader class to load RQ values and traces.

    :param output_dir: directory containing the output (default: ./output)
    """

    def __init__(self, output_dir='./output'):

        self.output_dir = output_dir
        assert os.path.exists(output_dir), f'output directory {output_dir} does not exist'

    def _load_single_run(self, run_number, data='rqs'):

        run_id = f'{run_number:06d}' if isinstance(run_number, int) else run_number
        out_file = [l for l in os.listdir(self.output_dir) if l.startswith(f'{run_id}_{data}')][0]
        output = np.load(os.path.join(self.output_dir, out_file), allow_pickle=True)
        return output['data'], output['config']

    def _load_multiple_runs(self, run_list, data='rqs'):

        print('To be implemented')
        return -1, -1

    def __call__(self, run_list, data='rqs'):
        """
        Calling the loader loads the specified runs returning RQ values and the configuration.

        :param run_list: run to be loaded as run number, run ID or list of them
        :param data: type of data to be loaded ['rqs', 'traces']
        """

        run_list = listify(run_list)
        if len(run_list) == 1:
            return self._load_single_run(run_list[0], data=data)
        else:
            return self._load_multiple_runs(run_list, data=data)
