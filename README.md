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

*Both kinds of Mössbauer source*
* SMS — synchrotron Mössbauer source (polarized, settable polarization degree)
  and CMS — conventional radioactive source (unpolarized)
* Instrumental function found from a spectrum of a standard absorber (single
  line, α-Fe, or any model, also jointly over several spectra); for SMS also a
  simulated ⁵⁷FeBO₃ source; for CMS the source line width and the cosine
  smearing of the source–aperture geometry
* Each `.dat` file can carry its own instrumental function

*Thick samples and proper handling of polarization*
* Full transmission integral, calculated in parallel on all cores
* Every model builds a 2x2 cross-section matrix, so absorber thickness,
  polarization and multi-line interference are treated together
* `Layer` marker: stack layers of different orientation or composition, each with
  its own transmission matrix
* Magnetic texture: uniaxial order `A` on every anisotropic component plus the
  magnetic polar order `Am` on the Faraday-active ones - a random powder,
  a textured foil and a single crystal are the same model at different values

*Models*
* Singlet, doublet, sextet
* Full Hamiltonian for the single crystal, mosaic textured and powder cases (one
  model, three order parameters)
* 2-state relaxation and many-state superparamagnetic relaxation
* Anharmonic spin modulation (ASM)
* MDGD (instead of xVBF https://doi.org/10.1016/j.nimb.2025.165669)
* SCDW: spin- and charge-density-wave model
* A measured spectrum used as a component
* Multi-dimensional functional distributions of any parameter, with correlations
* Distribution reconstruction with regularization (on testing)
* Thickness distribution of the sample
* Expressions and links between parameters
* Library with import/export - create your own and share it with others, ~50
  reference phases included

*Fitting*
* Maximum-likelihood fits for Poisson counts, with the exact derivatives of
  the model
* Errors and correlation matrix; fit-quality statistics (χ², deviance,
  misfit) telling a model problem from counting noise
* Several spectra: sequence (batch) fitting, simultaneous fitting with one model
  (shared and per-spectrum parameters) or with a model built spectrum by
  spectrum
* Components of many spectra (a series or a map): the spectral components
  found in the data and their amount in every spectrum
* Exclusion regions: velocity ranges left out of the fit
* Simulation without a spectrum

*Spectra*
* RAW (`.mca`, `.cmca`, `.ws5`, `.w98`, `.moe`, `.m1`, `.mcs`) and folded `.dat`
  files; online fitting along with the experiment (Bliss channel)
* Velocity calibration with α-Fe (sinusoidal or linear drive), RAW → `.dat`
* Each point modelled as the average over its channel's velocity range
* Sum spectra, subtract a model, create a spectrum from a model

*Interface*
* Interactive plot: zoom, components, distributions, a result window per
  spectrum after a simultaneous fit
* Results saved as text tables, figures (PNG, SVG) and a ready-to-load model
* Light and dark themes, built-in help and description of every model

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
- **Numba**: BSD 2-Clause License
- **llvmlite**: BSD 2-Clause License (contains LLVM: Apache License 2.0 with LLVM Exceptions)
- **Bliss** (optional, not included): LGPL v3.0 — imported only if installed
  separately (Linux beamline installations), to read spectra online from a Bliss
  channel; never part of the bundles

For complete third-party license information, see [NOTICE.txt](NOTICE.txt). The
Windows and macOS bundles also carry the full license text of every third-party
package they contain, in a `licenses` folder; in the program, **Supp → License**
shows all of it.
