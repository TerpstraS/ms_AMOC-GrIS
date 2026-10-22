#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Created By: Sjoerd Terpstra
# Created Date: 04/02/2025
# ---------------------------------------------------------------------------
"""config.py
Project configuration file, containing global variables and settings.

Necessary packages:
"""
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
import os

# Load environment variables from .env file if it exists
load_dotenv()

### Global variables

# Base directory of the project (where the data and models are stored)
DIR_BASE = os.path.join("/nethome", "terps020")

# Project folder
DIR_PROJECT = os.path.join(DIR_BASE, "analysis", "ms_AMOC-GrIS")

# Working directory of the project (doesn't work well on Lorenz)
DIR_CWD = os.getcwd()

# Directory containing the different models with output
DIR_MODEL = os.path.join(DIR_BASE, "models", "climber-x-fork", "climber-x-grl")

DIR_OUTPUT = os.path.join(DIR_MODEL, "output-no-exist")
DIR_INPUT = os.path.join(DIR_MODEL, "input")

# Data directory (of the project folder)
DIR_DATA = os.path.join(DIR_PROJECT, "data")
DIR_DATA_EXT = os.path.join(DIR_DATA, "external")
os.makedirs(DIR_DATA, exist_ok=True)
os.makedirs(DIR_DATA_EXT, exist_ok=True)

# Figures directory
DIR_FIG = os.path.join(DIR_PROJECT, "reports", "figures")
os.makedirs(DIR_FIG, exist_ok=True)

DIR_SRC = os.path.join(DIR_PROJECT, "src")


# Importing plotting module to load the plotting settings.
# Must be done after setting all the directory paths
import src.plotting as plot
