from __future__ import absolute_import, print_function
from .tools import *
from .stack import *

import os
import numpy as np

from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.coordinates import SkyCoord

from scipy.optimize import curve_fit
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")


def bin_get_rand_cutouts(ncutouts, binzlims, comap, galcat, params, field=None):
    """
    wrapper to return ncutout randomly located cutouts in a single field +
    a single redshift bin
    """

    fac = 10.

    randz = np.random.uniform(binzlims[0], binzlims[1], size=int(ncutouts*fac))
    randra = np.random.uniform(comap.xlims[0], comap.xlims[1], size=int(ncutouts*fac))
    randdec = np.random.uniform(comap.ylims[0], comap.ylims[1], size=int(ncutouts*fac))
    randcoords = SkyCoord(randra*u.deg, randdec*u.deg)
    randidx = np.arange(ncutouts*fac)

    randcat = empty_table()
    randcat.coords = randcoords
    randcat.z = randz
    randcat.idx = randidx

    cutoutlist = []
    ngoodcuts = 0
    for i in range(len(randz)):
        cutout = single_cutout(i, randcat, comap, params)

        # if it passed the tests, keep it
        if cutout:
            if field:
                cutout.field = field
            cutoutlist.append(cutout)
            ngoodcuts += 1

            if ngoodcuts == ncutouts:
                return randcat, cutoutlist

    print(ngoodcuts)
    return None, None

def field_get_rand_cutouts(galidxs, comap, galcat, params, field=None, verbose=False):
    """
    return ncutout random cutouts, binned in redshift to match galidxs
    """

    nperbin, binedges = field_zbin_stack_output(galidxs, comap, galcat, params)

    bigrandcat = []
    cutoutlist = []
    for i in range(params.nzbins):
        nbin = nperbin[i]
        if verbose:
            print("  bin {} needs {} cutouts".format(i+1, nbin))
        binedge = binedges[i:i+2]

        randcat, binlist = bin_get_rand_cutouts(nbin, binedge, comap, galcat, params)

        if binlist:
            bigrandcat = np.append(bigrandcat, randcat)
            cutoutlist = cutoutlist + binlist
        else:
            print("Didn't get enough gals in {}:{} bin".format(binedges[0], binedges[1]))
            break
    return cutoutlist

def random_stacker_setup(maplist, galcatlist, params):
    # values using the actual galaxy catalogue
    # set all the extras to none to make this as efficient as possible
    saveplots = params.saveplots
    plotspace = params.plotspace
    plotfreq = params.plotfreq
    spacestackwidth = params.spacestackwidth
    freqstackwidth = params.freqstackwidth

    params.saveplots = False
    params.plotspace = False
    params.plotfreq = False
    outvals, actim, actspec, actcatidx = stacker(maplist, galcatlist, params)

    params.saveplots = saveplots
    params.plotspace = plotspace
    params.plotfreq = plotfreq
    params.spacestackwidth = spacestackwidth
    params.freqstackwidth = freqstackwidth

    return actcatidx

def random_stacker(actcatidx, maplist, galcatlist, params, verbose=False):
    """
    wrapper to perform a stack on random locations binned to match
    the numbers of the stack in actcatidx
    """

    fields = [1,2,3]
    fieldlens = [len(actcatidx[0]), len(actcatidx[1]), len(actcatidx[2])]

    allcutouts = []
    for i in range(len(maplist)):
        if verbose:
            print(fields[i])
            print("need {} total cutouts".format(fieldlens[i]))
        fieldcutouts = field_get_rand_cutouts(actcatidx[i], maplist[i],
                                              galcatlist[i], params,
                                              field=fields[i], verbose=verbose)
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

    return stacktemp, stackrms, stackim, stackspec, fieldcatidx


def n_random_stacks(nstacks, actidxlist, maplist, galcatlist, params, verbose=True):
    """
    wrapper to perform n different stacks on random locations to match the original
    catalogue
    """

    stackTlist = []
    stackrmslist = []

    for n in range(nstacks):
        if verbose:
            if n % 10 == 0:
                print('iteration {}'.format(n))

        stackT, stackrms, _, _, _ = random_stacker(actidxlist, maplist, galcatlist, params)
        stackTlist.append(stackT)
        stackrmslist.append(stackrms)

        if params.itersave:
            if n % params.itersavestep == 0:
                np.savez(params.itersavefile, T=stackTlist, rms=stackrmslist)

    return stackTlist, stackrmslist

def histoverplot(bootfile, stackdict, nbins=30, p0=(1000, 0, 2), rethist=False):
    """
    Function to plot the output of a bootstrap run as a histogram
    """

    # put T values in uK
    bootstrap = np.load(bootfile)['T'] * 1e6
    actT = stackdict['T'] * 1e6
    actrms = stackdict['rms'] * 1e6

    npoints = len(bootstrap)

    counts, binedges = np.histogram(bootstrap, bins=nbins)

    bincent = (binedges[1:] - binedges[:-1]) / 2 + binedges[:-1]

    xarr = np.linspace(np.min(bincent), np.max(bincent))
    opt, cov = curve_fit(gauss, bincent, counts, p0=p0)

    fig,ax = plt.subplots(1)

    ax.hist(bootstrap, bins=nbins, color='indigo')

    ax.plot(xarr, gauss(xarr, *opt), color='darkorange')

    ax.axvline(opt[1], color='0.3', ls=':', label="From Bootstrap")
    rect = Rectangle((opt[1] - opt[2], -1), 2*opt[2], 1200, color='0.3', alpha=0.4)
    ax.add_patch(rect)

    ax.axvline(actT, color='k', ls='--', label="From Map RMS")
    rect = Rectangle((actT-actrms, -1), 2*actrms,
                      1200, color='k', alpha=0.4)
    ax.add_patch(rect)

    ax.legend()


    ax.set_xlabel(r'$T_b$ ($\mu K$)')
    ax.set_ylabel('Counts')

    p_og = norm.cdf(x=opt[1], loc=stackdict['T'], scale=opt[2])

    if rethist:
        return p_og, npoints, counts, bincent

    return p_og, npoints
