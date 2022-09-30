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

def offset_and_stack(maplist, catlist, params, offrng):
    # randomly offset each field's catalogue
    offcatlist = []
    for j in range(len(catlist)):

        # make a catalogue of random offsets that shouldn't overlap with flux from the actual object
        # 2* as big to make sure there are enough objects included to hit goalnumcutouts
        randcatsize = (3,2*catlist[j].nobj)
        randoffs = offrng.uniform(2,10,randcatsize) * np.sign(offrng.uniform(-1,1,randcatsize))

        offcat = catlist[j].copy()

        raoff = np.concatenate((catlist[j].ra(), catlist[j].ra())) + maplist[j].xstep*randoffs[0,:]
        decoff = np.concatenate((catlist[j].dec(), catlist[j].dec())) + maplist[j].ystep*randoffs[1,:]
        freqoff = np.concatenate((catlist[j].freq, catlist[j].freq)) + maplist[j].fstep*randoffs[2,:]
        zoff = freq_to_z(params.centfreq, freqoff)

        offcat.coords = SkyCoord(raoff*u.deg, decoff*u.deg)
        offcat.freq = freqoff
        offcat.z = zoff
        offcat.nobj = 2*catlist[j].nobj

        offcatlist.append(offcat)

    # run the actual stack
    outdict,_,_,_,_,_ = stacker(maplist, offcatlist, params)

    return np.array([outdict['T'], outdict['rms']])


def offset_bootstrap(niter, maplist, catlist, params):

    if params.verbose:
        print('starting actual stack for reference')

    # initialize the output file
    with open(params.itersavefile, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['T', 'rms'])

    # run the actual stack purely to see how many cutouts you're going to need for each bootstrap
    actcutlist, actim, actspec, actcatidx, actcube, actcuberms = st.stacker(maplist, catlist, params)

    # set the goal numbers of cutouts
    params.goalnumcutouts = ([len(catidx) for catidx in actcatidx])

    # set up an rng for the offsets
    offrng = np.random.default_rng(params.bootstrapseed)

    # play with the output that's printed so you don't get every cutout for every stack
    if params.verbose:
        params.bootverbose = True
        params.verbose = False
        print('starting '+str(niter)+' random stack runs')
    else:
        params.bootverbose = False

    outarrs = []
    for i in range(niter):

        # randomly offset each field's catalogue and stack it
        outarr = offset_and_stack(maplist, catlist, params, offrng)
        outarrs.append(outarr)

        if params.itersave:
            if i % params.itersavestep == 0:
                with open(params.itersavefile, 'a') as csvfile:
                    w = csv.writer(csvfile)
                    w.writerow(outarr)

        if params.bootverbose:
            if i % params.itersavestep == 0:
                print('    done run '+str(i)+' of '+str(niter))

        # just in case
        plt.close('all')

    # save the final output
    outarrs = np.stack(outarrs)
    np.savez(params.nitersavefile, T=outarrs[:,0], rms=outarrs[:,1])

    return outarrs

def bin_get_rand_cutouts(ncutouts, binzlims, comap, galcat, params, field=None, seed=None):
    """
    wrapper to return ncutout randomly located cutouts in a single field +
    a single redshift bin
    """

    fac = 10.

    rng = np.random.default_rng(seed)

    randz = rng.uniform(binzlims[0], binzlims[1], size=int(ncutouts*fac))
    randra = rng.uniform(comap.xlims[0], comap.xlims[1], size=int(ncutouts*fac))
    randdec = rng.uniform(comap.ylims[0], comap.ylims[1], size=int(ncutouts*fac))
    randcoords = SkyCoord(randra*u.deg, randdec*u.deg)
    randidx = np.arange(ncutouts*fac)

    randcat = empty_table()
    randcat.coords = randcoords
    randcat.z = randz
    randcat.idx = randidx

    cutoutlist = []
    ngoodcuts = 0
    ti = 0
    for i in range(len(randz)):
        cutout = single_cutout(i, randcat, comap, params)

        # if it passed the tests, keep it
        # if it passed all the tests, keep it
        if cutout:

            if params.cubelet:
                if ti == 0:
                    cubestack = cutout.cubestack
                    cuberms = cutout.cubestackrms
                    # delete the 3d arrays
                    cutout.__delattr__('cubestack')
                    cutout.__delattr__('cubestackrms')
                    ti = 1
                else:
                    scstack = np.stack((cubestack, cutout.cubestack))
                    scrms = np.stack((cuberms, cutout.cubestackrms))
                    cubestack, cuberms = weightmean(scstack, scrms, axis=0)
                    # delete the 3d arrays
                    cutout.__delattr__('cubestack')
                    cutout.__delattr__('cubestackrms')

            cutoutlist.append(cutout)
            ngoodcuts += 1

            if ngoodcuts == ncutouts:
                return randcat, cutoutlist

    print(ngoodcuts)
    return None, None

def field_get_rand_cutouts(galidxs, comap, galcat, params, field=None, verbose=False, seed=None):
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

        randcat, binlist = bin_get_rand_cutouts(nbin, binedge, comap, galcat, params, seed=seed)

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
    if params.cubelet:
        outvals, actim, actspec, actcatidx, cube, cuberms = stacker(maplist, galcatlist, params)
    else:
        outvals, actim, actspec, actcatidx = stacker(maplist, galcatlist, params)

    params.saveplots = saveplots
    params.plotspace = plotspace
    params.plotfreq = plotfreq
    params.spacestackwidth = spacestackwidth
    params.freqstackwidth = freqstackwidth

    return actcatidx

def random_stacker(actcatidx, maplist, galcatlist, params, verbose=False, seed=None):
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
                                              field=fields[i], verbose=verbose,
                                              seed=seed)
        allcutouts = allcutouts + fieldcutouts

        # unzip all your cutout objects
    Tvals = []
    rmsvals = []
    catidxs = []
    if params.plotspace:
        spacestack = []
        spacerms = []
    if params.plotfreq:
        freqstack = []
        freqrms = []
    for cut in allcutouts:
        Tvals.append(cut.T)
        rmsvals.append(cut.rms)
        catidxs.append(cut.catidx)

        if params.plotspace:
            spacestack.append(cut.spacestack)
            spacerms.append(cut.spacestackrms)

        if params.plotfreq:
            freqstack.append(cut.freqstack)
            freqrms.append(cut.freqstackrms)

    # put everything into numpy arrays for ease
    Tvals = np.array(Tvals)
    rmsvals = np.array(rmsvals)
    catidxs = np.array(catidxs)
    if params.plotspace:
        spacestack = np.array(spacestack)
        spacerms = np.array(spacerms)
    if params.plotfreq:
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
    if params.plotspace:
        stackim, imrms = weightmean(spacestack, spacerms, axis=0)
    else:
        stackspec, imrms = None, None

    # overall frequency stack
    if params.plotfreq:
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

def field_random_stacker(actcatidx, mapinst, galcatinst, params, verbose=False, seed=None):
    if verbose:

        print("need {} total cutouts".format(len(actcatidx)))
    allcutouts = field_get_rand_cutouts(actcatidx, mapinst,
                                          galcatinst, params,
                                          verbose=verbose,
                                          seed=seed)

    Tvals = []
    rmsvals = []
    catidxs = []
    if params.plotspace:
        spacestack = []
        spacerms = []
    if params.plotfreq:
        freqstack = []
        freqrms = []
    for cut in allcutouts:
        Tvals.append(cut.T)
        rmsvals.append(cut.rms)
        catidxs.append(cut.catidx)

        if params.plotspace:
            spacestack.append(cut.spacestack)
            spacerms.append(cut.spacestackrms)

        if params.plotfreq:
            freqstack.append(cut.freqstack)
            freqrms.append(cut.freqstackrms)

    # put everything into numpy arrays for ease
    Tvals = np.array(Tvals)
    rmsvals = np.array(rmsvals)
    catidxs = np.array(catidxs)
    if params.plotspace:
        spacestack = np.array(spacestack)
        spacerms = np.array(spacerms)
    if params.plotfreq:
        freqstack = np.array(freqstack)
        freqrms = np.array(freqrms)

    return stacktemp, stackrms


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

        stackT, stackrms, _, _, _ = random_stacker(actidxlist, maplist, galcatlist, params, seed=n*10)
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

    fig,ax = plt.subplots(1, tight_layout=True)

    ax.hist(bootstrap, bins=nbins, color='indigo')

    ax.plot(xarr, gauss(xarr, *opt), color='darkorange')

    rect = Rectangle((opt[1] - opt[2], -1), 2*opt[2], 1500, color='0.1', alpha=0.5)
    ax.add_patch(rect)
    ax.axvline(opt[1], color='0.1', ls=':', label="From Bootstrap")

    rect = Rectangle((actT-actrms, -1), 2*actrms,
                      1500, color='0.5', alpha=0.5)
    ax.add_patch(rect)
    ax.axvline(actT, color='0.5', ls='--', label="From Map RMS")

    ax.set_ylim((0., np.max(counts)*1.05))

    ax.legend()


    ax.set_xlabel(r'$T_b$ ($\mu K$)')
    ax.set_ylabel('Counts')

    p_og = norm.cdf(x=opt[1], loc=stackdict['T'], scale=opt[2])

    if rethist:
        return p_og, npoints, counts, bincent

    return p_og, npoints
