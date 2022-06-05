from __future__ import absolute_import, print_function
from .tools import *
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import h5py
import glob
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.io import fits


""" COMBINE ALREADY-MADE SIMS WITH DATA """
def load_sim(file):
    """
    loads in a mock CO luminosity cube and stores as an object like the real
    map
    """

    simmap = empty_table()

    with np.load(file) as simfile:
        # sims output uK, data in K. stack functions all deal w K so convert
        simmap.rawmap = simfile['map_cube'].T / 1e6
        simmap.freq = simfile['map_frequencies']
        simmap.ra = simfile['map_pixel_ra']
        simmap.dec = simfile['map_pixel_dec']

    # **anything else here?

    return simmap

def beam_smooth_sim(simmap, fwhm=4.5):
    """
    Smooth the simulated data with a Gaussian 2D kernel to approximate the wider COMAP beam
    fwhm should be the beam fwhm in arcmin
    """

    std = fwhm / (2*np.sqrt(2*np.log(2)))
    pixwidth = (simmap.ra[1] - simmap.ra[0])*60

    std_pix = std / pixwidth

    beamkernel = Gaussian2DKernel(std_pix)

    # loop over the frequency axis and convolve each frame
    smoothsimlist = []
    for i in range(len(simmap.freq)):
        smoothsimlist.append(convolve(simmap.rawmap[i,:,:], beamkernel))

    simmap.map = np.stack(smoothsimlist, axis=0)

    return simmap

def dump_map(comap, filename):
    """
    save a map class as a hdf5 file
    this will be more bare-bones than the actual datafiles -- just including the
    datasets needed for stacking
    """

    # *** automatic output file naming

    # undo the coordinate shift so it doesn't happen twice when it's reloaded
    comap.freq = comap.freq + comap.fstep / 2
    comap.ra = comap.ra + comap.xstep / 2
    comap.dec = comap.dec + comap.ystep / 2

    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('map_coadd', data = comap.simdatmap, dtype='float64')
        dset = f.create_dataset('rms_coadd', data = comap.rms, dtype='float64')
        dset = f.create_dataset('freq', data = comap.freq, dtype='float64')
        dset = f.create_dataset('x', data = comap.ra, dtype='float64')
        dset = f.create_dataset('y', data = comap.dec, dtype='float64')

        patchcent = (comap.fieldcent.ra.deg, comap.fieldcent.dec.deg)
        dset = f.create_dataset('patch_center', data = patchcent, dtype='float64')

    return 0

def scalesim(datfiles, simfiles, outfiles, scale=1, beamfwhm=4.5, save=True,
             rmsscale=True):
    """
    Wrapper to load files and add simulated data. Scale can be arraylike or a single value
    ***warn properly
    """

    # *** fix
    if len(datfiles) != len(simfiles):
        print('different number of files')
        return 0

    # if an array of scale values passed, only run io once and then add and dump
    #  for each scale value
    if isinstance(scale, (list, tuple, np.ndarray)):
        for i in range(len(datfiles)):

            datmap = load_map(datfiles[i])

            simlummap = load_sim(simfiles[i])
            simlummap = beam_smooth_sim(simlummap, fwhm=beamfwhm)

            datmap.simmap = np.array(simlummap.map)

            if rmsscale:
                maprms = rootmeansquare(datmap.map)

                rawsn = np.nanmax(simlummap.map) / maprms

                scale = scale / rawsn

            for j in range(len(scale)):
                simdatmap = np.array(datmap.map + scale[j] * simlummap.map)

                # subtract the mean
                meanval = np.nanmean(simdatmap)
                datmap.simdatmap = simdatmap - meanval

                if rmsscale:
                    # rename the output files to have the correct scale in them
                    outfiles[i] = outfiles[i].split('_sn')[0] + '_sn'+str(j)+'.h5'
                else:
                    # rename the output files to have the correct scale in them
                    outfiles[i] = outfiles[i].split('_scale')[0] + '_scale'+str(j)+'.h5'

                if save:
                    dump_map(datmap, outfiles[i])

    # otherwise, just run through everything once
    else:
        for i in range(len(datfiles)):

            datmap = load_map(datfiles[i])

            simlummap = load_sim(simfiles[i])
            simlummap = beam_smooth_sim(simlummap, fwhm=beamfwhm)

            datmap.simmap = np.array(simlummap.map)

            if rmsscale:
                maprms = rootmeansquare(datmap.map)

                rawsn = np.nanmax(simlummap.map) / maprms

                scale = scale / rawsn

            simdatmap = np.array(datmap.map + scale * simlummap.map)

            # subtract off the mean (done in the actual COMAP pipeline)
            meanval = np.nanmean(simdatmap)
            datmap.simdatmap = simdatmap - meanval

            if save:
                dump_map(datmap, outfiles[i])

    return datmap
