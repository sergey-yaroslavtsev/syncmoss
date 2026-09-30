"""License texts of the third-party software packed into a PyInstaller bundle.

BSD, Apache-2.0 and the LGPL ask for their license TEXT to go with binary
copies, not just a name and a link. So every distribution that has code in the
bundle contributes the license files its wheel ships, and they are collected
under ``licenses/<distribution>/``. The list is derived from the Analysis, not
kept by hand, so a new or dropped dependency cannot make it stale.

Two texts the build environment cannot provide are added from ``licenses/``
next to this file:

* LGPL-3.0 and GPL-3.0 (verbatim from gnu.org). PySide6/shiboken6 (Qt) are used
  under ``LGPL-3.0-only``, which must be accompanied by both texts -- but their
  wheels ship only a Qt commercial-license notice.

and the Python runtime's own LICENSE.txt, taken from the interpreter doing the
build (the one being bundled).

Used by Windows.spec and MacOS.spec: ``a.datas += license_datas(a)``.
"""
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, distribution, packages_distributions

HERE = os.path.dirname(os.path.abspath(__file__))
_LICENSE_NAME = ('LICEN', 'COPYING', 'NOTICE', 'AUTHORS')


def bundled_distributions(analysis):
    """Names of the installed distributions with code in ``analysis``."""
    top_level = {name.split('.')[0] for name, _, _ in analysis.pure}
    top_level |= {re.split(r'[\\/]', dest)[0] for dest, _, _ in analysis.binaries + analysis.datas}
    owners = packages_distributions()
    names = {dist for name in top_level for dist in owners.get(name, [])}
    names.discard('syncmoss')  # our own license is the top-level LICENSE
    return sorted(names, key=str.lower)


def _dist_license_files(name):
    """(destination, source) of the license files a distribution ships."""
    try:
        dist = distribution(name)
    except PackageNotFoundError:
        return []
    found = []
    for file in dist.files or []:
        if not file.parts[0].endswith('.dist-info') or len(file.parts) < 2:
            continue
        inside = file.parts[1:]
        if inside[0] == 'licenses':  # metadata 2.4 keeps them all there
            inside = inside[1:]
        elif not inside[-1].upper().startswith(_LICENSE_NAME):
            continue
        found.append((os.path.join('licenses', name, *inside), str(dist.locate_file(file))))
    return found


def license_datas(analysis):
    """TOC entries (destination, source, 'DATA') for every license text."""
    pairs = []
    for name in bundled_distributions(analysis):
        files = _dist_license_files(name)
        if not files:
            print(f"WARNING: {name} ships no license file")
        pairs += files

    for text in ('LGPL-3.0.txt', 'GPL-3.0.txt'):
        pairs.append((os.path.join('licenses', text), os.path.join(HERE, 'licenses', text)))

    # python.org: LICENSE.txt in the install folder (Windows) or the standard
    # library (macOS, Linux); conda: LICENSE_PYTHON.txt in the environment
    for candidate in (os.path.join(sys.base_prefix, 'LICENSE.txt'),
                      os.path.join(os.path.dirname(os.__file__), 'LICENSE.txt'),
                      os.path.join(sys.base_prefix, 'LICENSE_PYTHON.txt')):
        if os.path.isfile(candidate):
            pairs.append((os.path.join('licenses', 'Python', 'LICENSE.txt'), candidate))
            break
    else:
        print("WARNING: the Python runtime's LICENSE.txt was not found")

    return [(dest, src, 'DATA') for dest, src in pairs]
