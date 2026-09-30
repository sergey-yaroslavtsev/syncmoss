# SYNCmoss

Mössbauer Spectroscopy Analysis Software

## References

Older version can be find
https://gitlab.esrf.fr/yaroslav/syncmoss

Related article:
Yaroslavtsev S., J. Synchrotron Rad., 2023	
doi.org/10.1107/S1600577523001686

## To run on Windows and MacOS

go to releases:
https://github.com/sergey-yaroslavtsev/syncmoss/releases

download Windows related archieve, extract it, and run `.exe`

MacOS is not well tested. 
download MacOS related `.dmg`, install it, (pay attention to the comment in release)

## To run on Linux

```bash
# create venv
python -m venv syncmoss
# activate it
source syncmoss/bin/activate
# install latest release syncmoss
pip install syncmoss
# or install directly from the repo
pip install git+https://github.com/sergey-yaroslavtsev/syncmoss.git
# run it (and try to enjoy it) 
syncmoss
```

## Main features

* User-friendly graphical interface (even better now)
* Can extract instrumental function from spectrum of standard absorber
* Sequence (batch) fitting
* Simultaneous fitting
* Full Hamiltonian model for the single crystal, mosaic textured and powder cases (one model, three order parameters)
* 2-state relaxation model
* Many-state superparamagnetic relaxation model
* Anharmonic spin modulation (ASM) model
* MDGD model (instead of xVBF https://doi.org/10.1016/j.nimb.2025.165669)
* SCDW: spin- and charge-density-wave model
* Multi-dimensional distributions with correlations (not reconstruction but functional)
* Distribution reconstruction with regularization (on testing)
* Expressions (which could be linked to parameters)
* Online (along with experiment) fitting
* Parallel calculations of full-transmission integral
* Library with import/export - create your own and share it with others, ~50 reference phases included
* Interactive spectrum image


*Thick samples and proper handling of polarization*
* Every model builds a 2x2 cross-section matrix,
  so absorber thickness, polarization and multi-line interference are treated
  together
* Works in both source modes: polarized SMS
  (with a settable linear polarization degree)
  and unpolarized CMS
* `Layer` marker: stack layers of different orientation or composition, each with
  its own transmission matrix
* Magnetic texture: uniaxial order `A` on every anisotropic component plus the
  magnetic polar order `Am` on the Faraday-active ones - a random powder,
  a textured foil and a single crystal are the same model at different values

## Bug report and Feature request

Both are very welcome — SYNCmoss gets better mainly from user feedback.

* write to **sergey.yaroslavtsev@esrf.fr** or open an issue: https://github.com/sergey-yaroslavtsev/syncmoss/issues

When something goes wrong the program opens a *report a bug* window containing
the error message.

### Reporting a bug

Please describe the problem and provide all related material:

* **what you did** — which button or action created the problem, and what you
  expected to happen instead;
* **the error message** — copy it from the *report a bug* window (or from the
  terminal);
* **the spectrum file(s)** and **the model** (`.mdl`) you were working with —
  without them most problems cannot be reproduced;
* the SYNCmoss version and your operating system;
* a screenshot, if the problem is about what is drawn or displayed.

### Requesting a feature

Please describe what exactly is needed: the physics or the workflow behind it,
where in the program it should appear, and — if it exists elsewhere — a
reference, an article or an example file showing the expected result.

## Author

Yaroslavtsev Sergey (ESRF | ADA and ID14)

## License

This software is licensed under the MIT License. See [LICENSE](LICENSE) for details.

Copyright (c) European Synchrotron Radiation Facility (ESRF)

## Third-Party Software

This software uses third-party libraries that are distributed under their own licenses:

- **PySide6**: LGPL v3.0 (dynamically linked — does not affect MIT licensing of this project)
- **NumPy**: BSD 3-Clause License
- **SciPy**: BSD 3-Clause License  
- **Matplotlib**: Matplotlib License (BSD-compatible)

For complete third-party license information, see [NOTICE.txt](NOTICE.txt).
