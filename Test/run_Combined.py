#!/usr/bin/env python
import subprocess
import datetime
with open(("Combined_output_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".txt"), "w+") as output:
    subprocess.call(["python", "./use_Combined.py"], stdout=output);