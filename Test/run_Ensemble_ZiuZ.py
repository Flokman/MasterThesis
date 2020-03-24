#!/usr/bin/env python
import subprocess
import datetime
with open(("Ensemble_output_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".txt"), "w+") as output:
    subprocess.call(["python", "./use_ensemble.py"], stdout=output);