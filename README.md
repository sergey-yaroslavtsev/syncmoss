# SYNCmoss

Mössbauer Spectroscopy Analysis Software

## References

Older version can be find
https://gitlab.esrf.fr/yaroslav/syncmoss

Related article:
Yaroslavtsev S., J. Synchrotron Rad., 2023	
doi.org/10.1107/S1600577523001686

## To run on Windows

go to releases:
https://github.com/sergey-yaroslavtsev/syncmoss/releases

download Windows related archieve, extract it, and run `.exe`

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
* Full Hamiltonian model for single crystal case
* 2-state relaxation model
* Many-state superparamagnetic relaxation model
* Anharmonic spin modulation (ASM) model
* MDGD model (instead of xVBF https://doi.org/10.1016/j.nimb.2025.165669)
* Multi-dimensional distributions with correlations
* Expressions (which could be linked to parameters)
* Online (along with experiment) fitting
* Parallel calculations of full-transmission integral

## Author

Yaroslavtsev Sergey (ESRF ID14)

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
