"""Online spectrum from a Bliss channel -- optional (Bliss exists for Linux only).

At the beamline the MCA acquisition publishes the spectrum it is accumulating
on a Bliss channel named ``McaAcq_channel_<...>``. Typed into the spectrum-path
box instead of a file name, that name is read from Bliss and folded exactly like
the counts of an ``.mca`` file, with the current Calibration.dat. Show spectrum,
Show model and Fit then work on the spectrum as it is at that moment.

``bliss`` is NOT a dependency. If it cannot be imported, nothing happens at
all: ``AVAILABLE`` is False and a ``McaAcq_channel_...`` name is an ordinary
path that does not exist. Note that importing bliss monkey-patches the whole
process with gevent (``patch_all(thread=False)``) -- the original program
lived with that too.

Bliss is gevent-based, so a channel is read on the main (GUI) thread only. The
path check calls ``refresh`` there before any worker thread starts; the
show-model and fitting threads then get the counts it kept, from ``counts``.
"""
import os
import threading
import uuid

import numpy as np

CHANNEL_PREFIX = 'McaAcq_channel_'

try:
    from bliss.config.channels import Channel
    AVAILABLE = True
    # The ID14 beacon server, as in the original program; an environment that
    # sets BEACON_HOST itself keeps its own. (Bliss reads it on connecting.)
    os.environ.setdefault('BEACON_HOST', 'id14:25000')
except Exception:
    Channel = None
    AVAILABLE = False

# Counts of the last read of each channel, as a (1, channels) array: the layout
# of a one-block .mca file
_last_counts = {}


def is_channel(path):
    """True when ``path`` is a Bliss channel name and Bliss is available."""
    return AVAILABLE and str(path).startswith(CHANNEL_PREFIX)


def abspath(path):
    """``os.path.abspath``, except that a channel name is left as it is."""
    return path if is_channel(path) else os.path.abspath(path)


def refresh(name):
    """Read channel ``name`` now, keep its counts and return them.

    Main thread only. Raises when the channel cannot be read -- e.g. when it has
    no value because the name is wrong or the acquisition is not running.
    """
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Bliss channels can be read on the main thread only")
    # Kept from the original program: reading a throw-away channel first runs
    # gevent's loop, which delivers the updates still pending, so a Channel
    # that Bliss keeps cached for ``name`` answers with the current value.
    Channel(uuid.uuid4().hex).value
    value = Channel(name).value
    if value is None:
        raise RuntimeError("the channel has no value -- check the name and "
                           "that the acquisition is running")
    _last_counts[name] = np.array([value], dtype=float)
    return _last_counts[name]


def counts(name):
    """The counts kept by the last ``refresh(name)``, as a (1, channels) array.

    If the channel was never read, the main thread reads it now; a worker
    thread gets None.
    """
    if name in _last_counts:
        return _last_counts[name]
    if threading.current_thread() is threading.main_thread():
        return refresh(name)
    return None
