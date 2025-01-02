# Helix and strax lingo

Incomplete list of terms used in strax and Helix:

- **data type** - user-defined (or rather, Helix-developers-defined) handles for types of strax-formatted data. Data
types is what needs to be specified in the `provides` and `depends_on` properties of each Plugin. Examples of Helix
data types are `raw_records`, `qp_triggers`, `events`, `noise_events`, `fit_results`.

- **data kind** - some of the data types describe the same physical "entities", providing different set of fields for
them. For example, `events` and `fit_results` describe the same temporal "entity" - events. Each data type belongs to
a certain data kind. In most cases, a data type belongs to its own data kind (for example, data kind of the data type
`qp_trigger` is `qp_trigger`; data kind of the data type `events` is `events`). However, the data kind of the data type
`fit_results` is also `events`, because `fit_results` do not describe a different temporal entities in the stream of
data. They describe events. One row of `fit_results` correspond to one event. Data types of the same data kind can be
merged together. Plugins' `compute` method takes data kinds as an input, not data types. If a plugin depends on two
different data types of the same data kind, the data will be merged into one input array before being passed to the
`compute` method.

- **dtype** - numpy term for the types of rows in numpy structured arrays. In the most simply case in numpy, dtype is
just an int or a float. But in numpy structured arrays, dtypes are tuples tuples of field names, their np types, and
their shapes, if the one row of the structured array contains an array in that field rather than just one value.
Each data type has a corresponding dtype. Data type is the handle for strax to understand what kind of data is requested,
while dtype is a tuple of fields and numpy types to build the structured array to hold the data of that data type.

- **record** - one trace of fixed length from one channel.

- **block** - all records from all the channels corresponding to a time interval equal to the record length.

- **trace** - array of data, waveform. Of any length.

- **chunk** - a piece of strax-formatted data saved by strax into one file. Strax splits data into chunks trying to make
the file sizes to be close to 200 MB (defined by the `Plugin.chunk_target_size_mb` property). This means that one chunk
of "heavy" data (like `raw_records`) could correspond to just a second of data, while a chunk of high level processed
data (like `event_basics`) could correspond to an hour of data. Plugins' `compute` method gets one chunk of data at a time.

- **hit** - trace interval starting at the points where it crosses the activation threshold and finishing at the point
where it crosses the deactivation thresholds.
