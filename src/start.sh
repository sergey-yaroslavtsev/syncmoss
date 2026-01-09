#!/bin/bash

function main() {
if [ -e ./syncmoss_env/bin/activate ]
then
  source syncmoss_env/bin/activate
  python3 prog_raw.py
else
  python3 -m venv syncmoss_env
  source syncmoss_env/bin/activate
  pip install kivy
  pip install plyer
  pip install numba
  pip install matplotlib
  pip install scipy
  python3 prog_raw.py
fi
}

main