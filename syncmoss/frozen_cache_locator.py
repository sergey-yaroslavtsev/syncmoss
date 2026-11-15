"""numba's ``NUMBA_CACHE_DIR`` for the frozen app: jit_cache.configure names
FrozenCacheLocator in ``NUMBA_CACHE_LOCATOR_CLASSES`` there, and numba imports
this module by that name (the bundle specs list it as a hidden import).

A frozen module has no source file: PyInstaller keeps its bytecode in the
executable, under a file name relative to the bundle (``syncmoss/models.py``).
numba's own locator for ``NUMBA_CACHE_DIR`` requires the file, and its per-user
one -- the one a frozen app got -- makes the relative name absolute against the
working directory: a folder of compiled functions for every folder the program
was started from (Finder starts it from /). This one keeps them in
``NUMBA_CACHE_DIR``, under the name's own folder. They are dated by the
executable, numba's rule for a frozen app: a new build compiles them again.
"""
import os
import sys

from numba.core import config
from numba.core.caching import UserProvidedCacheLocator


class FrozenCacheLocator(UserProvidedCacheLocator):
    """``NUMBA_CACHE_DIR`` for a function of the frozen app."""

    @classmethod
    def from_function(cls, py_func, py_file):
        """The locator of *py_func*, or None -- not frozen, no folder set, or
        it cannot be written -- for the next locator to take it."""
        if not (getattr(sys, 'frozen', False) and config.CACHE_DIR) or os.path.isabs(py_file):
            return None
        locator = cls(py_func, py_file)
        try:
            locator.ensure_cache_path()
        except OSError:
            return None
        return locator

    @classmethod
    def get_suitable_cache_subpath(cls, py_file):
        """The folder of the module's name, as it is: ``syncmoss``."""
        return os.path.dirname(py_file)
