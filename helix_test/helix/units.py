# In strax, all the time variables are integers in ns.
# time = 3 * units.s would mean 3 seconds, represented as 3_000_000_000 int
# to convert strax times to desired units do time / units.s

# perhaps it would be better to define functions like s_to_ns, ns_to_s, and so on...

ns = 1
us = 1_000
ms = 1_000_000
s = 1_000_000_000
