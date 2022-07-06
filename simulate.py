from __future__ import absolute_import, print_function
from .tools import *
from .stack import *
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
import os
import pandas as pd
import h5py
import glob
from astropy.convolution import convolve, Gaussian2DKernel
from astropy.convolution import convolve_fft, Gaussian1DKernel, Kernel
from astropy.io import fits
import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


""" FUNCTIONS FOR LOADING SIMS IN """
def load_raw_sim(file):
    """
    loads in a mock CO luminosity cube and stores as an object like the real
    map
    """

    rawsimmap = empty_table()

    with np.load(file, allow_pickle=True) as simfile:
        # sims output uK, data in K. stack functions all deal w K so convert
        rawsimmap.rawmap = simfile['map_cube'] / 1e6
        # these are bin CENTERS also
        rawsimmap.freq = simfile['map_frequencies']
        rawsimmap.ra = simfile['map_pixel_ra']
        rawsimmap.dec = simfile['map_pixel_dec']

    # just go ahead and flip the frequency axis here:
    #  rearrange so frequency axis is first in the map
    rawsimmap.rawmap = np.swapaxes(rawsimmap.rawmap, 0, -1)
    rawsimmap.freq = np.flip(rawsimmap.freq)
    rawsimmap.rawmap = np.flip(rawsimmap.rawmap, axis=0)

    # get the pixel 'bin' edges for catalogue matching, etc
    rawsimmap.fdiff = rawsimmap.freq[1] - rawsimmap.freq[0]
    rawsimmap.radiff = rawsimmap.ra[1] - rawsimmap.ra[0]
    rawsimmap.decdiff = rawsimmap.dec[1] - rawsimmap.dec[0]

    rawsimmap.freqbe = np.append(rawsimmap.freq - rawsimmap.fdiff/2, rawsimmap.freq[-1] + rawsimmap.fdiff/2)
    rawsimmap.rabe = np.append(rawsimmap.ra - rawsimmap.radiff/2, rawsimmap.ra[-1] + rawsimmap.radiff/2)
    rawsimmap.decbe = np.append(rawsimmap.dec - rawsimmap.decdiff/2, rawsimmap.dec[-1] + rawsimmap.decdiff/2)


    # **anything else here?

    return rawsimmap

def load_raw_catalogue(catfile, pixel_values=None):

    # load from npz file
    with np.load(catfile, allow_pickle=True) as rawcat:
        # store in a table object to access different values as attributes
        catobj = empty_table()
        catobj.ra = rawcat['ra']
        catobj.dec = rawcat['dec']
        if 'z' in rawcat.files:
            catobj.z = rawcat['z']
        else:
            catobj.z = rawcat['redshift']
        catobj.Lco = rawcat['Lco']
        catobj.M = rawcat['M']
        catobj.freq = nuem_to_nuobs(115.27, catobj.z) #***

    # if pixel values is not none, a simulated map object should be passed instead
    # when this is the case find the pixel values in the simulated map that correspond to the
    # positions of the catalogue objects
    if pixel_values:
        # think this just has to be brute-force #***
        pixfreq = []
        pixra = []
        pixdec = []
        for i in range(len(catobj.freq)):
            objpixfreq = np.max(np.where(pixel_values.freqbe < catobj.freq[i])[0])
            objpixra = np.max(np.where(pixel_values.rabe < catobj.ra[i])[0])
            objpixdec = np.max(np.where(pixel_values.decbe < catobj.dec[i])[0])

            pixfreq.append(objpixfreq)
            pixra.append(objpixra)
            pixdec.append(objpixdec)

        catobj.chan = np.array(pixfreq)
        catobj.x = np.array(pixra)
        catobj.y = np.array(pixdec)

    # if not already sorted, sort so the most luminous halo is the first one
    if np.argmax(catobj.M) != 0:
        sortidx = np.flip(np.argsort(catobj.M))
        catobj.ra = catobj.ra[sortidx]
        catobj.dec = catobj.dec[sortidx]
        catobj.z = catobj.z[sortidx]
        catobj.Lco = catobj.Lco[sortidx]
        catobj.M = catobj.M[sortidx]
        catobj.freq = catobj.freq[sortidx]

        if pixel_values:
            catobj.chan = catobj.chan[sortidx]
            catobj.x = catobj.x[sortidx]
            catobj.y = catobj.y[sortidx]



    return catobj

"""MATCH WCS OF SIMS TO THE REAL MAP COORDINATES"""
def simcoords_to_mapcoords(simobj, mapobj, simcat):

    # ra/dec and all their permutations
    simobj.ra = simobj.ra + mapobj.ra[0]
    simobj.rabe = simobj.rabe + mapobj.rabe[0]
    simobj.dec = simobj.dec + mapobj.dec[0]
    simobj.decbe = simobj.decbe + mapobj.decbe[0]

    # same for the catalogue
    simcat.ra = simcat.ra + mapobj.ra[0]
    simcat.dec = simcat.dec + mapobj.dec[0]

    return simobj, mapobj, simcat



""" FUNCTIONS FOR SAVING SIMS"""
def dump_map(comap, filename):
    """
    save a map class as a hdf5 file
    this will be more bare-bones than the actual datafiles -- just including the
    datasets needed for stacking
    """

    # undo the coordinate shift so it doesn't happen twice when it's reloaded
    outfreq = comap.freq + comap.fstep / 2
    outra = comap.ra + comap.xstep / 2
    outdec = comap.dec + comap.ystep / 2

    with h5py.File(filename, 'w') as f:
        dset = f.create_dataset('map_coadd', data = comap.sim, dtype='float64')
        dset = f.create_dataset('rms_coadd', data = comap.rms, dtype='float64')
        dset = f.create_dataset('freq', data = outfreq, dtype='float64')
        dset = f.create_dataset('x', data = outra, dtype='float64')
        dset = f.create_dataset('y', data = outdec, dtype='float64')

        patchcent = (comap.fieldcent.ra.deg, comap.fieldcent.dec.deg)
        dset = f.create_dataset('patch_center', data = patchcent, dtype='float64')

        # store the simulation-only and data-only maps too for posterity
        dset = f.create_dataset('sim_only', data = comap.simonly, dtype='float64')
        dset = f.create_dataset('dat_only', data = comap.map, dtype='float64')

    return 0

def dump_cat(cat, filename):
    """
    save the simulated catalogue (with wcs corrected to match data) to file in the
    correct format for run_stack to just read it in like a real catalogue
    """

    np.savez(filename, z=cat.z, ra=cat.ra, dec=cat.dec)

    return 0


"""SIGNAL-INJECTION WRAPPERS"""
def sim_inject_field(datfile, simfile, catfile, outfile=None, scale=1.):
    """
    wrapper function -- loads in an actual COMAP map and a simulated halo luminosity LIM
    cube, matches the wcs between the two, and injects the simulated signal into the
    actual map, scaling by 'scale'. Will then save the new map and the associated halo
    catalogue (also wcs-matched) to file.
    Works on a SINGLE COMAP field
    """

    # load the actual data in (this acts as noise)
    datmap = load_map(datfile)

    # load the simulation in (preserving its wcs for now)
    simmap = load_raw_sim(simfile)
    simcat = load_raw_catalogue(catfile, pixel_values=simmap)

    # match wcs
    simmap, datmap, simcat = simcoords_to_mapcoords(simmap, datmap, simcat)

    # inject the signal into the map, beating the noise down by the scale factor
    injected_map = datmap.map / scale + simmap.rawmap

    # new map object to store all this info in
    injmap = datmap.copy()
    injmap.sim = injected_map
    injmap.simonly = simmap.rawmap

    # save the injected simulation like it's a normal COMAP .h5 product
    ## make a file name if none is given for both the map and the catalogue
    if not outfile:
        datname = datfile.split('/')[-1][:-3]
        simname = simfile.split('seed_')[-1][:-3]
        outfile = 'simcube_' + datname + '_sim_' + simname

        if scale != 1.:
            outfile += '_scale{:.1f}.h5'.format(scale)
        else:
            outfile += '.h5'

    ## save the map
    dump_map(injmap, outfile)

    return injmap, simcat

def sim_inject(datfiles, simfiles, catfiles, outputdir=None, scale=1., trim=None):
    """
    takes input for three different COMAP fields with associated simulations and
    does the injection, matching wcs in the simulation and catalogue to the actual
    data wcs
    --------
    scale:  DIVIDES the input data map (i.e. the noise) by this factor to simulate
            beating the noise down
    trim:   number of catalogue objects to keep (default is all of them). can be
            either a list of integers (one per field) or a single integer (same
            number kept in each field)
    """

    # output path management stuff - save all files to a passed directory
    if not outputdir:
        outputdir = './output/'
    else:
        if outputdir[-1] != '/': outputdir += '/'

    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    outfiles = []
    simnames = []
    fieldnames = []
    for i in range(3):
        datname = datfiles[i].split('/')[-1][:-3]
        fieldnames.append('co'+datname.split('co')[-1][0])
        simname = simfiles[i].split('seed_')[-1][:-4]
        simnames.append(simname[:5])
        outfile = outputdir + datname + '_sim_' + simname
        if scale != 1.:
            outfile += '_scale{:.1f}.h5'.format(scale)
        else:
            outfile += '.h5'
        outfiles.append(outfile)

    # come up with a file name for the combined catalogue
    allfieldcatfile = outputdir+'combined_cat_fields_'+'-'.join(fieldnames)+'_seeds_'+'-'.join(simnames)
    if trim:
        if isinstance(trim, (list, tuple, np.ndarray)):
            allfieldcatfile += '_trim_'+'-'.join(trim)
        else:
            allfieldcatfile += '_trim_'+trim
    allfieldcatfile +=  '.npz'

    # individual injections for each field
    fieldcats = []
    for i in range(3):
        fieldmap, fieldcat = sim_inject_field(datfiles[i], simfiles[i], catfiles[i],
                                              outfile=outfiles[i], scale=scale)

        fieldcats.append(fieldcat)

    # combine all the output catalogues together into a single file for ease
    allfieldcat = empty_table()
    if isinstance(trim, (list, tuple, np.ndarray)):
        allfieldcat.z = np.concatenate([cat.z[:trim[i]] for i,cat in enumerate(fieldcats)])
        allfieldcat.ra = np.concatenate([cat.ra[:trim[i]] for i,cat in enumerate(fieldcats)])
        allfieldcat.dec = np.concatenate([cat.dec[:trim[i]] for i,cat in enumerate(fieldcats)])

    elif trim:
        allfieldcat.z = np.concatenate([cat.z[:trim] for cat in fieldcats])
        allfieldcat.ra = np.concatenate([cat.ra[:trim] for cat in fieldcats])
        allfieldcat.dec = np.concatenate([cat.dec[:trim] for cat in fieldcats])

    else:
        allfieldcat.z = np.concatenate([cat.z for cat in fieldcats])
        allfieldcat.ra = np.concatenate([cat.ra for cat in fieldcats])
        allfieldcat.dec = np.concatenate([cat.dec for cat in fieldcats])

    dump_cat(allfieldcat, allfieldcatfile)

    return


""" DISTRIBUTION-AWARE STACKS """
def bin_field_sim_catalogue(actidxs, galcat, simcat, params):
    """
    bin the simulated galaxy catalogue to match the real one in redshift
    """
    nperbin, zedges = field_zbin_stack_output(actidxs, 0, galcat, params)
    simz = simcat.z
    coords = simcat.coords
    idx = simcat.idx

    bincatlist = []
    for i in range(params.nzbins):
        binhaloidx = np.where(np.logical_and(simz > zedges[i], simz < zedges[i+1]))[0]

        bincat = empty_table()

        bincat.z = simz[binhaloidx]
        bincat.coords = coords[binhaloidx]
        bincat.idx = idx[binhaloidx]
        bincat.goalnobj = nperbin[i]
        bincat.nobj = len(binhaloidx)
        bincatlist.append(bincat)

        if bincat.goalnobj > bincat.nobj:
            print("Too few objects in simulated catalogue!")
            return None

    return nperbin, bincatlist

def bin_get_sim_cutouts(comap, galcat, params, field=None):
    """
    wrapper to return all cutouts for a single bin
    """

    cutoutlist = []
    ngood = 0
    for i in range(galcat.nobj):
        cutout = single_cutout(i, galcat, comap, params)

        # if it passed all the tests, keep it
        if cutout:
            if field:
                cutout.field = field

            cutoutlist.append(cutout)

            ngood += 1

            if ngood == galcat.goalnobj:
                return cutoutlist


        if params.verbose:
            if i % 100 == 0:
                print('   done {} of {} cutouts in this field'.format(i, galcat.nobj))

    return None

def field_get_sim_cutouts(actidxs, comap, galcat, simcat, params, field=None, verbose=False):
    """
    return the appropriate number of simulated cutouts, binned in redshift to match galidxs
    """

    nperbin, bincatlist = bin_field_sim_catalogue(actidxs, galcat, simcat, params)

    cutoutlist = []
    for i in range(params.nzbins):
        bincat = bincatlist[i]

        if verbose:
            print("  bin {} needs {} cutouts".format(i+1, bincat.goalnobj))

        gbinlist = bin_get_sim_cutouts(comap, bincat, params)

        if gbinlist:
            cutoutlist = cutoutlist + gbinlist
        else:
            print("Didn't get enough gals in {} bin".format(i))
            break
    return cutoutlist

def sim_stacker(actcatidx, maplist, galcatlist, simcatlist, params):
    """
    wrapper to perform a stack on random locations binned to match
    the numbers of the stack in actcatidx
    """

    fields = [1,2,3]
    fieldlens = [len(actcatidx[0]), len(actcatidx[1]), len(actcatidx[2])]

    allcutouts = []
    for i in range(len(maplist)):
        if params.verbose:
            print(fields[i])
            print("need {} total cutouts".format(fieldlens[i]))
        fieldcutouts = field_get_sim_cutouts(actcatidx[i], maplist[i],
                                              galcatlist[i], simcatlist[i], params,
                                              field=fields[i], verbose=params.verbose)
        allcutouts = allcutouts + fieldcutouts

        # unzip all your cutout objects
    Tvals = []
    rmsvals = []
    catidxs = []
    if params.spacestackwidth:
        spacestack = []
        spacerms = []
    if params.freqstackwidth:
        freqstack = []
        freqrms = []
    for cut in allcutouts:
        Tvals.append(cut.T)
        rmsvals.append(cut.rms)
        catidxs.append(cut.catidx)

        if params.spacestackwidth:
            spacestack.append(cut.spacestack)
            spacerms.append(cut.spacestackrms)

        if params.freqstackwidth:
            freqstack.append(cut.freqstack)
            freqrms.append(cut.freqstackrms)

    # put everything into numpy arrays for ease
    Tvals = np.array(Tvals)
    rmsvals = np.array(rmsvals)
    catidxs = np.array(catidxs)
    if params.spacestackwidth:
        spacestack = np.array(spacestack)
        spacerms = np.array(spacerms)
    if params.freqstackwidth:
        freqstack = np.array(freqstack)
        freqrms = np.array(freqrms)


    # split indices up by field
    fieldcatidx = []
    previdx = 0
    for catlen in fieldlens:
        fieldcatidx.append(catidxs[previdx:catlen+previdx])
        previdx += catlen

    # overall stack for T value
    stacktemp, stackrms = weightmean(Tvals, rmsvals)
    stackvals = {'T':stacktemp, 'rms':stackrms}

    # overall spatial stack
    if params.spacestackwidth:
        stackim, imrms = weightmean(spacestack, spacerms, axis=0)
    else:
        stackspec, imrms = None, None

    # overall frequency stack
    if params.freqstackwidth:
        stackspec, specrms = weightmean(freqstack, freqrms, axis=0)
    else:
        stackim, imrms = None, None

    if params.saveplots:
        # make the directory to store the plots
        os.makedirs(params.savepath, exist_ok=True)

    if params.plotspace:
        spatial_plotter(stackim, params)

    if params.plotfreq:
        spectral_plotter(stackspec, params)

    if params.plotspace and params.plotfreq:
        combined_plotter(stackim, stackspec, params)

    return stackvals, stackim, stackspec, fieldcatidx
