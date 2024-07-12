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

# ignore divide by zero warnings
np.seterr(divide='ignore', invalid='ignore')

def field_offset_and_stack(mapinst, catinst, params, offrng, method=None):

    # pick the function to create a random catalog
    if not method or method == 'offset':
        randomize = cat_rand_offset
    elif method == 'offset_freq':
        randomize = cat_rand_offset_freq 
    elif method == 'offset_space':
        randomize = cat_rand_offset_space
    elif method == 'shuffle':
        randomize = cat_rand_offset_shuffle
    elif method == 'uniform':
        randomize = cat_rand_offset_random

    # randomly offset the catalogue
    offcat = randomize(mapinst, catinst, params, offrng)

    outcube = field_stacker(mapinst, offcat, params)

    return np.array([outcube.linelum, outcube.dlinelum])


def offset_and_stack(maplist, catlist, params, offrng, method=None):

    # pick the function to create a random catalog
    if not method or method == 'offset':
        randomize = cat_rand_offset
    elif method == 'offset_freq':
        randomize = cat_rand_offset_freq 
    elif method == 'offset_space':
        randomize = cat_rand_offset_space
    elif method == 'shuffle':
        randomize = cat_rand_offset_shuffle
    elif method == 'sensitivity':
        randomize = cat_rand_offset_sensmap
    elif method == 'uniform':
        randomize = cat_rand_offset_random

    offcatlist = []
    for j in range(len(catlist)):
        if params.goalnumcutouts[j] == 0:
            offcatlist.append(catlist[j])
            continue
        # randomly offset each field's catalogue
        offcat = randomize(maplist[j], catlist[j], params, offrng)
        offcatlist.append(offcat)

    # run the actual stack
    outcube = stacker(maplist, offcatlist, params)

    return np.array([outcube.linelum, outcube.dlinelum])

def cat_rand_offset(mapinst, catinst, params, offrng=None):

    # set up the rng (use the one passed, or failing that the one in params, or
    # failing that define a new one)
    if not offrng:
        try:
            offrng = params.bootstraprng
        except AttributeError:
            offrng = np.random.default_rng(params.bootstrapseed)
            paramsbootstraprng = offrng
            print('Defining new bootstrap rng using passed seed '+str(params.bootstrapseed))

    # make a catalogue of random offsets that shouldn't overlap with flux from the actual object
    # 2* as big to make sure there are enough objects included to hit goalnumcutouts
    randcatsize = (3,2*catinst.nobj)
    randoffs = offrng.uniform(1,5,randcatsize) * np.sign(offrng.uniform(-1,1,randcatsize))

    offcat = catinst.copy()

    raoff = np.concatenate((catinst.ra(), catinst.ra())) + mapinst.xstep*randoffs[0,:]
    decoff = np.concatenate((catinst.dec(), catinst.dec())) + mapinst.ystep*randoffs[1,:]
    freqoff = np.concatenate((catinst.freq, catinst.freq)) + mapinst.fstep*randoffs[2,:]
    zoff = freq_to_z(params.centfreq, freqoff)

    offcat.coords = SkyCoord(raoff*u.deg, decoff*u.deg)
    offcat.freq = freqoff
    offcat.z = zoff
    offcat.nobj = 2*catinst.nobj
    # for indexing -- use ra to add to the artificial index so fields are distinct
    offcat.catfileidx = np.arange(len(zoff)) + int(raoff[0]*1e6)
    offcat.idx = offcat.catfileidx

    return offcat

def cat_rand_offset_freq(mapinst, catinst, params, offrng=None):
    # set up the rng (use the one passed, or failing that the one in params, or
    # failing that define a new one)
    if not offrng:
        try:
            offrng = params.bootstraprng
        except AttributeError:
            offrng = np.random.default_rng(params.bootstrapseed)
            params.bootstraprng = offrng
            print('Defining new bootstrap rng using passed seed '+str(params.bootstrapseed))

    # make a catalogue of random offsets that shouldn't overlap with flux from the actual object
    # 2* as big to make sure there are enough objects included to hit goalnumcutouts
    randcatsize = (2*catinst.nobj)
    randoffs = offrng.uniform(1,5,randcatsize) * np.sign(offrng.uniform(-1,1,randcatsize))

    offcat = catinst.copy()

    raoff = np.concatenate((catinst.ra(), catinst.ra()))
    decoff = np.concatenate((catinst.dec(), catinst.dec()))
    freqoff = np.concatenate((catinst.freq, catinst.freq)) + mapinst.fstep*randoffs
    zoff = freq_to_z(params.centfreq, freqoff)

    offcat.coords = SkyCoord(raoff*u.deg, decoff*u.deg)
    offcat.freq = freqoff
    offcat.z = zoff
    offcat.nobj = 2*catinst.nobj
    # for indexing -- use ra to add to the artificial index so fields are distinct
    offcat.catfileidx = np.arange(len(freqoff)) + int(raoff[0]*1e6)
    offcat.idx = offcat.catfileidx

    return offcat


def cat_rand_offset_space(mapinst, catinst, params, offrng=None):
    # set up the rng (use the one passed, or failing that the one in params, or
    # failing that define a new one)
    if not offrng:
        try:
            offrng = params.bootstraprng
        except AttributeError:
            offrng = np.random.default_rng(params.bootstrapseed)
            params.bootstraprng = offrng
            print('Defining new bootstrap rng using passed seed '+str(params.bootstrapseed))

    # make a catalogue of random offsets that shouldn't overlap with flux from the actual object
    # 2* as big to make sure there are enough objects included to hit goalnumcutouts
    randcatsize = (2*catinst.nobj)
    raoffs = offrng.uniform(1,5,randcatsize) * np.sign(offrng.uniform(-1,1,randcatsize))
    decoffs = offrng.uniform(1,5,randcatsize) * np.sign(offrng.uniform(-1,1,randcatsize))

    offcat = catinst.copy()

    raoff = np.concatenate((catinst.ra(), catinst.ra())) + mapinst.xstep*randoffs
    decoff = np.concatenate((catinst.dec(), catinst.dec())) + mapinst.ystep*randoffs
    freqoff = np.concatenate((catinst.freq, catinst.freq))
    zoff = freq_to_z(params.centfreq, freqoff)

    offcat.coords = SkyCoord(raoff*u.deg, decoff*u.deg)
    offcat.freq = freqoff
    offcat.z = zoff
    offcat.nobj = 2*catinst.nobj
    # for indexing -- use ra to add to the artificial index so fields are distinct
    offcat.catfileidx = np.arange(len(freqoff)) + int(raoff[0]*1e6)
    offcat.idx = offcat.catfileidx

    return offcat


def cat_rand_offset_shuffle(mapinst, catinst, params, offrng=None):
    """
    generates a random catalog with the same spatial and spectral distribution as the input one
    by reassigning each spectral coordinate to a new spatial pair
    """
    if not offrng:
        try:
            offrng = params.bootstraprng 
        except AttributeError:
            offrng = np.random.default_rng(params.bootstrapseed)
            params.bootstraprng = offrng 
            print("Defining new bootstrap rng using passed seed "+str(params.bootstrapseed))

    randcatsize = (2*catinst.nobj)
    randcatidx = offrng.permutation(randcatsize)

    offcat = catinst.copy()
    raoff = np.concatenate((catinst.ra(), catinst.ra()))
    decoff = np.concatenate((catinst.dec(), catinst.dec()))
    zoff = np.concatenate((catinst.z, catinst.z))

    zshuff = zoff[randcatidx]
    offcat.z = zshuff 
    offcat.freq = nuem_to_nuobs(115.27, zshuff)
    offcat.coords = SkyCoord(raoff*u.deg, decoff*u.deg)
    offcat.nobj = 2*catinst.nobj
    # for indexing -- use ra to add to the artificial index so the fields are distinct
    offcat.catfileidx = np.arange(len(zshuff)) + int(raoff[0]*1e6)
    offcat.idx = offcat.catfileidx

    return offcat


def cat_rand_offset_sensmap(mapinst, catinst, params, offrng=None, senspath=None):
    """
    generates a random catalog following the same spatial and spectral distribution as the actual
    hetdex map. requires a passed sensitivity map, but pulls redshift distribution from the input catalog
    """
    # generate an rng if needed
    if not offrng:
        try:
            offrng = params.bootstraprng
        except AttributeError:
            offrng = np.random.default_rng(params.bootstrapseed)
            params.bootstraprng = offrng 
            print("Defining new bootstrap rng using passed seed "+str(params.bootstrapseed))

    # load in the sensitivity map if it hasn't already done
    try:
        _,_,_ = params.field_1_sensmap
    except AttributeError:
        if senspath:
            params.create_sensmap_bootstrap(senspath)
        else:
            print("Don't have generated sensitivity arrays, need to pass senspath")
            return

    # figure out which field you're in
    fieldra = int(np.round(mapinst.fieldcent.ra.deg))
    if fieldra == 25:
        fieldra, fielddec, fieldsens = params.field_1_sensmap
    elif fieldra == 170:
        fieldra, fielddec, fieldsens = params.field_2_sensmap
    elif fieldra == 226:
        fieldra, fielddec, fieldsens = params.field_2_sensmap
        
    # redshift axis
    zbins, zprobs = params.redshift_sensmap 
    zstep = zbins[1] - zbins[0]

    randcatsize = (2*catinst.nobj)

    # steps in RA and Dec
    dra = fieldra[1] - fieldra[0]
    ddec = fielddec[1] - fielddec[0]

    # array contains rms noise so probabilities should be inverted
    fieldsens = 1/fieldsens

    # pull a random location in the hetdex sensitivity map, weighted by the sensitivity
    hxsensflat = fieldsens.flatten() / np.nansum(fieldsens)
    # cast nans to zero
    hxsensflat[np.where(np.isnan(hxsensflat))] = 0.
    # choose a random index and index the 2d array
    sampidx = offrng.choice(a=hxsensflat.size, p=hxsensflat, size=randcatsize)
    adjidx = np.unravel_index(sampidx, fieldsens.shape)

    dx = offrng.uniform(-dra/2, dra/2, size=randcatsize)
    dy = offrng.uniform(-ddec/2, ddec/2, size=randcatsize)

    ra = fieldra[adjidx[1]] + dx
    dec = fielddec[adjidx[0]] + dy

    # redshifts
    zidx = offrng.choice(a=zbins.size, p=zprobs, size=randcatsize)
    zvals = zbins[zidx] + offrng.uniform(-zstep/2, zstep/2, size=randcatsize)

    # read into new catalog object
    offcat = catinst.copy()
    offcat.coords = SkyCoord(ra*u.deg, dec*u.deg)
    offcat.z = zvals
    offcat.nobj = 2*catinst.nobj 
    # for indexing -- use ra to add to the artificial index so fields are distinct
    offcat.catfileidx = np.arange(randcatsize) + int(ra[0]*1e6)
    offcat.idx = offcat.catfileidx

    return offcat

def cat_rand_offset_random(mapinst, catinst, params, offrng=None):
    """
    generates a completely uniform random catalog (ie no correlation with the input catalog
    other than the size)
    """
    if not offrng:
        try:
            offrng = params.bootstraprng 
        except AttributeError:
            offrng = np.random.default_rng(params.bootstrapseed)
            params.bootstraprng = offrng 
            print("Defining new bootstrap rng using passed seed "+str(params.bootstrapseed))

    randcatsize = (2*catinst.nobj)
    randcatidx = offrng.permutation(randcatsize)

    ralims = minmax(mapinst.ra)
    declims = minmax(mapinst.dec)
    zlims = minmax(freq_to_z(115.27, mapinst.freq))

    offcat = catinst.copy()
    zshuff = offrng.uniform(zlims[0], zlims[1], randcatsize)
    raoff = offrng.uniform(ralims[0], ralims[1], randcatsize)
    decoff = offrng.uniform(declims[0], declims[1], randcatsize)
    offcat.z = zshuff 
    offcat.freq = nuem_to_nuobs(115.27, zshuff)
    offcat.coords = SkyCoord(raoff*u.deg, decoff*u.deg)
    offcat.nobj = 2*catinst.nobj
    # for indexing -- use ra to add to the artificial index so the fields are distinct
    offcat.catfileidx = np.arange(len(zshuff)) + int(raoff[0]*1e6)
    offcat.idx = offcat.catfileidx

    return offcat







def offset_bootstrap(niter, maplist, catlist, params):

    if params.verbose:
        print('starting actual stack for reference')

    # initialize the output file
    with open(params.itersavefile, 'w') as csvfile:
        w = csv.writer(csvfile)
        w.writerow(['T', 'rms'])

    # run the actual stack purely to see how many cutouts you're going to need for each bootstrap
    actcutlist, actim, actspec, actcatidx, actcube, actcuberms = stacker(maplist, catlist, params)

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

def histoverplot(bootfile, stackdict, nbins=30, p0=(1000, 0, 2), rethist=False,
                 writefit=None):
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

    fig,ax = plt.subplots(1, tight_layout=True, figsize=(5,4))

    ax.hist(bootstrap, bins=nbins, color='indigo')

    yext = ax.get_ylim()

    ax.plot(xarr, gauss(xarr, *opt), color='darkorange')

    rect = Rectangle((opt[1] - opt[2], -1), 2*opt[2], yext[1]*2, color='0.1', alpha=0.5)
    ax.add_patch(rect)
    ax.axvline(opt[1], color='0.1', ls=':', label="Bootstrap")

    rect = Rectangle((actT-actrms, -1), 2*actrms,
                      yext[1], color='0.5', alpha=0.5)
    ax.add_patch(rect)
    ax.axvline(actT, color='0.5', ls='--', label="Stack RMS")

    ax.set_ylim((0., np.max(counts)*1.05))

    ax.legend(fontsize='large')


    ax.set_xlabel(r'$T_b$ ($\mu K$)', fontsize='large')
    ax.set_ylabel('Counts', fontsize='large')

    p_og = norm.cdf(x=opt[1], loc=stackdict['T'], scale=opt[2])

    # save the output as a csv
    if writefit:
        with open(writefit, 'w') as file:
            w = csv.writer(file)
            w.writerow(['amplitude', 'mean', 'std'])
            w.writerow(opt)

    if rethist:
        return p_og, npoints, counts, bincent

    return p_og, npoints, opt


def linelumhistoverplot(bootfile, stackdict, nbins=30, p0=(1000, 0, 2), rethist=False,
                        writefit=None, plotmaprms=False):
    """
    Function to plot the output of a bootstrap run as a histogram
    """

    # put T values in uK
    bootstrap = np.load(bootfile)['T']/1e10
    actT = stackdict['linelum']/1e10
    actrms = stackdict['dlinelum']/1e10

    npoints = len(bootstrap)

    counts, binedges = np.histogram(bootstrap, bins=nbins)

    bincent = (binedges[1:] - binedges[:-1]) / 2 + binedges[:-1]

    xarr = np.linspace(np.min(bincent), np.max(bincent))
    opt, cov = curve_fit(gauss, bincent, counts, p0=p0)

    fig,ax = plt.subplots(1, tight_layout=True, figsize=(5,4))

    ax.hist(bootstrap, bins=nbins, color='indigo')

    yext = ax.get_ylim()

    ax.plot(xarr, gauss(xarr, *opt), color='darkorange')

    rect = Rectangle((opt[1] - opt[2], -1), 2*opt[2], yext[1]*2, color='0.1', alpha=0.5, zorder=10)
    ax.add_patch(rect)
    ax.axvline(opt[1], color='0.5', ls=':', label="Bootstrap")

    if plotmaprms:
        rect = Rectangle((actT-actrms, -1), 2*actrms,
                        yext[1], color='0.5', alpha=0.5)
        ax.add_patch(rect)
        ax.axvline(actT, color='0.5', ls='--', label="Stack RMS")

    ax.set_ylim((0., np.max(counts)*1.05))

    if plotmaprms:
        ax.legend(fontsize='large')


    ax.set_xlabel(r"$L'_{CO} \times 10^{10}$ (K km/s pc$^2$)", fontsize='large')
    ax.set_ylabel('Counts', fontsize='large')

    try:
        p_og = norm.cdf(x=opt[1], loc=stackdict['T'], scale=opt[2])
    except KeyError:
        p_og = norm.cdf(x=opt[1], loc=stackdict['linelum'], scale=opt[2])

    # save the output as a csv
    if writefit:
        with open(writefit, 'w') as file:
            w = csv.writer(file)
            w.writerow(['amplitude', 'mean', 'std'])
            w.writerow(opt*np.array([1, 1e10, 1e10]))

    if rethist:
        return p_og, npoints, counts, bincent

    return p_og, npoints, opt
