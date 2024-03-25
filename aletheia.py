#!/usr/bin/env python3 

import os
import sys
import glob
import json
import time
import scipy
import numpy
import pandas
import pickle
import shutil
import random
import imageio
import tempfile
import subprocess

import numpy as np

from scipy import misc
from imageio import imread

import aletheialib.options as options
import click


@click.group()
def main():
    pass


main.add_command(options.auto.auto)
main.add_command(options.auto.dci)
main.add_command(options.brute_force.brute_force)
main.add_command(options.calibration.launch)
main.add_command(options.embsim.embsim)
main.add_command(options.feaext.feaext)
main.add_command(options.ml.ml)
main.add_command(options.structural.structural)
main.add_command(options.tools.tools)

if __name__ == "__main__":
    main()
