# Channel map

## class Channels

Class for handling channel maps. Allows channels to have numbers (i.e. names) different from their ordinal indices.
E.g., one can define channels from 10 to 30 to be submerged channels, while channels 100 to 120 to be vacuum channels.
However, all the data would still be stored in a packed array of 40 channels. An object of the Channels class provides
tools for getting channel ordinal indices from their numbers (it might makes sense to change the term "numbers" to
"names" or "labels" to avoid confusion between "number" and "count". Although, currently the channel labels must be
integer numbers, hence the term "numbers").

## IntEnum ChannelType

Along with the actual channel types (VACUUM and SUBMERGED), this enum provides types for various sums of channels. See 
the reference below.

## Reference

```python
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

    @property
    def ranges(self):
        """
        Dictionary of channel types and corresponding to them channel number ranges, with inclusive boundaries.
        """

    @property
    def numbers(self):
        """
        Array of channel numbers, where by numbers we mean names, or labels, of integer type.
        """
        # TODO: perhaps numbers is not a very fitting name. "names" or "labels" might be better. But we have to convey
        #  that these names or labels are not arbitrary strings, but integers.

    @property
    def types(self):
        """
        Array of channel types (hx.ChannelType) corresponding to each channel
        """

    @property
    def counts(self):
        """
        Dictionary of channel types and channel counts of that type
        """

    @property
    def order(self):
        """
        Array of indices corresponding to each channels, when channels are sorted by their numbers
        """

    @property
    def is_unsorted(self):
        """
        A boolean showing whether the channels in the map are sorted by their numbers
        """

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

    def numbers_to_types(self, channel_numbers):
        """
        Returns channel types corresponding to the provided channels
        :param channel_numbers: array of channel numbers, or a single channel number
        :return: channel types (hx.ChannelType) corresponding to the provided channels. Either an array or a single
        value, depending on the channel_numbers
        """

    def numbers_of_type(self, channel_type):
        """
        Returns an array of channel numbers corresponding to the provided channel type
        :param channel_type: hx.ChannelType
        :return: array of channel numbers corresponding to the provided channel type
        """

    def indices_of_type(self, channel_type):
        """
        Returns an array of channel indices corresponding to the provided channel type
        :param channel_type: hx.ChannelType
        :return: array of channel indices corresponding to the provided channel type
        """
```