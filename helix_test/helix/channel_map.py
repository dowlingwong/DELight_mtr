from enum import IntEnum
import numpy as np

# define importable classes
__all__ = ['ChannelNotInMap', 'ChannelType', 'Channels']


class ChannelNotInMap(Exception):
    """
    Custom exception for the Channels class
    """
    pass


class ChannelType(IntEnum):
    """
    Channel types, including various types of summs of channels
    """
    SUBMERGED = 0
    VACUUM = 1
    SUBMERGED_SUMMED = 10  # sum of submerged channels
    VACUUM_SUMMED = 11     # sum of vacuum channels
    SUMMED = 12            # sum of all channels
    TRIGGERED_SUBMERGED_SUMMED = 20  # sum of submerged channels that have triggers
    TRIGGERED_VACUUM_SUMMED = 21     # sum of vacuum channels that have triggers
    TRIGGERED_SUMMED = 22            # sum of all channels that have triggers


class Channels:
    """
    Class for handling channel maps. Allows channels to have numbers (i.e. names) different from its ordinal index.
    E.g. one can define channels from 10 to 30 to be submerged channels, while channels 100 to 120 to be vacuum channels
    """
    def __init__(self, channel_ranges):
        """
        Create a channel map object from a dictionary of channel types and ranges
        :param channel_ranges: dictionary of {channel_type: channel_range), where channel_range is a tuple of min and
        max channel number corresponding to this channel type.
        E.g., {hx.ChannelType.SUBMERGED: (1, 35), hx.ChannelType.VACUUM: (36, 50)} means that all the channels with
        numbers from 1 to 35 (inclusive) are submerged channels, while from 36 to 50 - vacuum channels.
        """
        self._ranges = channel_ranges
        range_keys = np.sort(list(self.ranges.keys()))
        self._numbers = np.concatenate([np.arange(self.ranges[key][0], self.ranges[key][1] + 1) for key in range_keys])
        self._types = self._types = self._fill_types(self.numbers)
        self._counts = {name: right - left + 1 for name, (left, right) in list(self.ranges.items())}
        self._order = self.numbers_to_indices(np.sort(self._numbers))
        self._is_unsorted = np.all(np.diff(self._order) > 0)

    @property
    def ranges(self):
        """
        Dictionary of channel types and corresponding to them channel number ranges, with inclusive boundaries.
        """
        return self._ranges

    @property
    def numbers(self):
        """
        Array of channel numbers, where by numbers we mean names, or labels, of integer type.
        """
        # TODO: perhaps numbers is not a very fitting name. "names" or "labels" might be better. But we have to convey
        #  that these names or labels are not arbitrary strings, but integers.
        return self._numbers

    @property
    def types(self):
        """
        Array of channel types (hx.ChannelType) corresponding to each channel
        """
        return self._types

    @property
    def counts(self):
        """
        Dictionary of channel types and channel counts of that type
        """
        return self._counts

    @property
    def order(self):
        """
        Array of indices corresponding to each channels, when channels are sorted by their numbers
        """
        return self._order

    @property
    def is_unsorted(self):
        """
        A boolean showing whether the channels in the map are sorted by their numbers
        """
        return self._is_unsorted

    # functions allowing applying len function to the Channels object, as well as indexing and iterating
    # i.e., the following expressions are possible:
    # channels = hx.Channels(hx.DEFAULT_CHANNEL_MAP)
    # len(channels)
    # channels[10]
    # for ch in channels:...
    def __len__(self):
        return len(self.numbers)

    def __getitem__(self, item):
        return self._numbers[item]

    def __iter__(self):
        return iter(self._numbers)

    def numbers_to_indices(self, channel_numbers):
        """
        Returns ordinal indices of channels corresponding to the provided channel numbers.
        This is useful if you need to find a specific channel (or channels) in an array containing some channel data.
        Such array would be indexed by an ordinal channel index, rather than by the channel number.
        For example, while in the default channel map, channel numbers start from 1, meaning that channel 1 is stored in
        the 0-th element of arrays containing channel data (e.g., events['channel_data'])
        :param channel_numbers: array of channel numbers or a single channel number
        :return: channel indices corresponding to the provided channel numbers. Either an array or a single value,
        depending on the channel_numbers
        """
        if isinstance(channel_numbers, (list, tuple)):
            channel_numbers = np.array(channel_numbers)

        one_input = False
        if not isinstance(channel_numbers, np.ndarray):
            one_input = True
            channel_numbers = np.array([channel_numbers])

        is_in = np.isin(channel_numbers, self.numbers)
        if not np.all(is_in):
            raise ChannelNotInMap(f'Channel {channel_numbers[np.argmin(is_in)]} is not in channel map')

        indices = np.argmax(self.numbers[:, None] == channel_numbers, axis=0)
        return indices[0] if one_input else indices

    def numbers_to_types(self, channel_numbers):
        """
        Returns channel types corresponding to the provided channels
        :param channel_numbers: array of channel numbers, or a single channel number
        :return: channel types (hx.ChannelType) corresponding to the provided channels. Either an array or a single
        value, depending on the channel_numbers
        """
        indices = self.numbers_to_indices(channel_numbers)
        return self.types[indices]

    def _fill_types(self, channel_numbers):
        if isinstance(channel_numbers, (np.ndarray, list, tuple)):
            channel_types = np.empty_like(channel_numbers, dtype=ChannelType)
            assigned = np.zeros_like(channel_numbers, dtype=bool)
            for channel_type, channel_range in self.ranges.items():
                mask = (channel_range[0] <= channel_numbers) & (channel_numbers <= channel_range[1])
                channel_types[mask] = channel_type
                assigned = assigned | mask
            if sum(assigned) < len(channel_numbers):
                raise ChannelNotInMap(f'Channels {channel_numbers[~assigned]} are not in channel map')
            return channel_types
        else:
            for channel_type, channel_range in self.ranges.items():
                if channel_range[0] <= channel_numbers <= channel_range[1]:
                    return channel_type
            raise ChannelNotInMap(f'Channel {channel_numbers} is not in channel map')


    def numbers_of_type(self, channel_type):
        """
        Returns an array of channel numbers corresponding to the provided channel type
        :param channel_type: hx.ChannelType
        :return: array of channel numbers corresponding to the provided channel type
        """
        return np.arange(self.ranges[channel_type][0], self.ranges[channel_type][1] + 1)

    def indices_of_type(self, channel_type):
        """
        Returns an array of channel indices corresponding to the provided channel type
        :param channel_type: hx.ChannelType
        :return: array of channel indices corresponding to the provided channel type
        """
        return self.numbers_to_indices(self.numbers_of_type(channel_type))
