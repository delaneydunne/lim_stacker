# load some base packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load in the stacking package
sys.path.insert(0, '/home/deedunne/Documents/COMAP/stacking')
import lim_stacker as st


""" INPUT FILES """
# list of the paths to each map file
# (note they're in a weird order due to a defunct naming convention -- Field 1
#  is co2, Field 2 is co7, and Field 3 is co6)
mapfiles = ['/home/deedunne/Documents/COMAP/data/yr1data/co2_map_summer.h5',
            '/home/deedunne/Documents/COMAP/data/yr1data/co7_map_summer.h5',
            '/home/deedunne/Documents/COMAP/data/yr1data/co6_map_summer.h5']

# path to the catalog file
# needs to be a zipped numpy file with keys 'ra', 'dec', and 'z'
catfiles = '/home/deedunne/Documents/COMAP/HETDEX/4.0.0/cuthetdex_sn4.7.npz'

""" PARAMETERS """
# load the parameters into a custom python object. this empty function
# call will load the default parameters, which should all be good. To explore what
# the defaults are, check out 'param_defaults.py'
params = st.parameters()
# set up where you'd like the stack results to be output to
params.savepath = 'hx_s1_240608'
params.make_output_pathnames()


""" SETUP """
# load in and set up the maps and catalog
maplist, catlist = st.setup(mapfiles, catfiles, params)

""" RUN """
# run the stack
stackcube = st.stacker(maplist, catlist, params)
