from abc import ABC, abstractmethod

class RQCalculator(ABC):
    """
    Abstract base class for the calculation of single set of reduced quantities
    from raw traces.
    """

    dependencies = []

    @abstractmethod
    def apply(self, traces, rqs):
        pass

    @abstractmethod
    def load_config(self, config_file):
        pass

    def get_config(self):
        if self.config is None:
            return {}
        else:
            return self.config
