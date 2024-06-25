from __future__ import absolute_import, print_function
from .tools import *
from .plottools import *
from .cubefilters import *
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import astropy.constants as const
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel
from astropy.modeling import models, fitting
import warnings
import csv
import pandas as pd

from spectral_cube import SpectralCube
from spectral_cube.utils import SpectralCubeWarning
from radio_beam import Beam
from reproject import reproject_adaptive

warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", category=SpectralCubeWarning, append=True)


# standard COMAP cosmology
cosmo = FlatLambdaCDM(H0=70*u.km / (u.Mpc*u.s), Om0=0.286, Ob0=0.047)

""" CUBELET OBJECT TO HOLD STACK RESULTS """
class cubelet():

    def __init__(self, input, params):
        """
        can pass either a path to load data from or a cutout object
        """

        if type(input) == str:
            self.from_files(input, params)
        else:
            self.from_cutout(input, params)


    def from_cutout(self, cutout, params):
        # housekeeping info
        self.unit = 'K' #params.plotunits ***** this is terrible
        self.ncutouts = 1
        self.catidx = [cutout.catidx]
        self.nuobs_mean = [cutout.freq]
        self.z_mean = [cutout.z]

        # read in aperture/cubelet sizes
        self.xwidth = params.xwidth
        self.ywidth = params.ywidth
        self.freqwidth = params.freqwidth

        # read in full cubelet values
        cubeshape = cutout.cubestack.shape
        self.cubexwidth = cubeshape[2]
        self.cubeywidth = cubeshape[1]
        self.cubefreqwidth = cubeshape[0]

        # do math about it
        xoff = params.xwidth // 2
        foff = params.freqwidth // 2
        self.centpix = (params.freqstackwidth, params.spacestackwidth, params.spacestackwidth)
        self.apminpix = (params.freqstackwidth-foff, params.spacestackwidth-xoff, params.spacestackwidth-xoff)
        self.apmaxpix = (params.freqstackwidth+foff+1, params.spacestackwidth+xoff+1, params.spacestackwidth+xoff+1)

        # find the width of the channel in GHz (different if physical spacing)
        # also the width of each pixel
        try:
            chanwidth = cutout.fstep 
            xwidtharcmin = cutout.xstep * 60
        except AttributeError:
            chanwidth = params.chanwidth
            xwidtharcmin = 2

        # set up frequency/angular arrays
        if params.freqwidth % 2 == 0:
            self.freqarr = np.arange(params.freqstackwidth * 2)*chanwidth - (params.freqstackwidth-0.5)*chanwidth
        else:
            self.freqarr = np.arange(params.freqstackwidth * 2 + 1)*chanwidth - (params.freqstackwidth)*chanwidth

        if params.xwidth % 2 == 0:
            self.xarr = np.arange(params.spacestackwidth * 2)*xwidtharcmin - (params.spacestackwidth-0.5)*xwidtharcmin
        else:
            self.xarr = np.arange(params.spacestackwidth * 2 + 1)*xwidtharcmin - (params.spacestackwidth)*xwidtharcmin

        # read in the cutout values
        self.cube = cutout.cubestack
        self.cuberms = cutout.cubestackrms
        self.linelum = cutout.linelum
        self.dlinelum = cutout.dlinelum
        self.rhoh2 = cutout.rhoh2
        self.drhoh2 = cutout.drhoh2

    
    def from_files(self, path, params):

        # paths to specific data files
        cubefile = path + '/stacked_3d_cubelet.npz'
        valuefile = path + '/output_values.csv'
        idxfile = path + '/included_cat_indices.npz'

        # params info
        self.unit = params.plotunits

        # read in aperture/cubelet sizes
        self.xwidth = params.xwidth
        self.ywidth = params.ywidth
        self.freqwidth = params.freqwidth

        # load in cubelet
        with np.load(cubefile) as f:
            cubevals = f['T']
            rmsvals = f['rms']

        self.cube = cubevals
        self.cuberms = rmsvals 

        # metainfo about cubelet
        cubeshape = cubevals.shape
        self.cubexwidth = cubeshape[2]
        self.cubeywidth = cubeshape[1]
        self.cubefreqwidth = cubeshape[0]
        xoff = params.xwidth // 2
        foff = params.freqwidth // 2
        self.centpix = (params.freqstackwidth, params.spacestackwidth, params.spacestackwidth)
        self.apminpix = (params.freqstackwidth-foff, params.spacestackwidth-xoff, params.spacestackwidth-xoff)
        self.apmaxpix = (params.freqstackwidth+foff+1, params.spacestackwidth+xoff+1, params.spacestackwidth+xoff+1)

        # set up frequency/angular arrays
        if params.freqwidth % 2 == 0:
            self.freqarr = np.arange(params.freqstackwidth * 2)*params.chanwidth - (params.freqstackwidth-0.5)*params.chanwidth
        else:
            self.freqarr = np.arange(params.freqstackwidth * 2 + 1)*params.chanwidth - (params.freqstackwidth)*params.chanwidth

        if params.xwidth % 2 == 0:
            self.xarr = np.arange(params.spacestackwidth * 2)*2 - (params.spacestackwidth-0.5)*2
        else:
            self.xarr = np.arange(params.spacestackwidth * 2 + 1)*2 - (params.spacestackwidth)*2


        # load in output values
        outvals = pd.read_csv(valuefile)

        self.linelum = outvals.linelum[0]
        self.dlinelum = outvals.dlinelum[0]
        self.rhoh2 = outvals.rhoh2[0]
        self.drhoh2 = outvals.drhoh2[0]

        self.ncutouts = outvals.nobj[0]
        try:
            self.nuobs_mean = outvals['nuobs_mean ()'][0]
        except KeyError:
            self.nuobs_mean = outvals.nuobs_mean[0]
        try:
            self.z_mean = outvals['z_mean ()'][0]
        except KeyError:
            self.z_mean = outvals.z_mean[0]
            
        # fix values if they've been stored weird
        if type(self.z_mean) == str:
            self.z_mean = float(self.z_mean[1:-1])
        if type(self.nuobs_mean) == str:
            self.nuobs_mean = float(self.nuobs_mean[1:-1])

        # load in catalog indices
        with np.load(idxfile) as f:
            indices = f['arr_0']

        self.catidx = indices 



    def stackin(self, cutout):
        # add a single cutout into the stacked cubelet

        # stack together the 3D cubelets
        cubevals = np.stack((self.cube, cutout.cubestack))
        rmsvals = np.stack((self.cuberms, cutout.cubestackrms))

        self.cube, self.cuberms = weightmean(cubevals, rmsvals, axis=0)

        # stack together the single values
        self.linelum, self.dlinelum = weightmean(np.array((self.linelum, cutout.linelum)),
                                                 np.array((self.dlinelum, cutout.dlinelum)))
        self.rhoh2, self.drhoh2 = weightmean(np.array((self.rhoh2, cutout.rhoh2)),
                                             np.array((self.drhoh2, cutout.drhoh2)))

        # housekeeping
        self.catidx.append(cutout.catidx)
        nuobs_mean = (self.nuobs_mean*self.ncutouts + cutout.freq)/(self.ncutouts+1)
        self.nuobs_mean = nuobs_mean
        z_mean = (self.z_mean*self.ncutouts + cutout.z)/(self.ncutouts+1)
        self.z_mean = z_mean
        self.ncutouts += 1

        # get rid of the cutout entirely
        del(cutout)


    def stackin_cubelet(self, cubelet):
        # merge two stacked cubelets
        if not cubelet:
            del(cubelet)
            return
        
        # stack together the 3d cubelets
        cubevals = np.stack((self.cube, cubelet.cube))
        rmsvals = np.stack((self.cuberms, cubelet.cuberms))

        self.cube, self.cuberms = weightmean(cubevals, rmsvals, axis=0)

        # stack together the single values
        self.linelum, self.dlinelum = weightmean(np.array((self.linelum, cubelet.linelum)),
                                                 np.array((self.dlinelum, cubelet.dlinelum)))
        self.rhoh2, self.drhoh2 = weightmean(np.array((self.rhoh2, cubelet.rhoh2)),
                                             np.array((self.drhoh2, cubelet.drhoh2)))
        
        # housekeeping **** check averaging (also why is cubelet nuobs_mean a list?)
        self.catidx = np.concatenate((self.catidx, cubelet.catidx))
        nuobs_mean = (self.nuobs_mean*self.ncutouts + cubelet.nuobs_mean[0]*cubelet.ncutouts) / (self.ncutouts + cubelet.ncutouts)
        self.nuobs_mean = nuobs_mean 
        z_mean = (self.z_mean*self.ncutouts + cubelet.z_mean[0]*cubelet.ncutouts) / (self.ncutouts + cubelet.ncutouts)
        self.z_mean = z_mean
        self.ncutouts = self.ncutouts + cubelet.ncutouts

        del(cubelet)
        return
    
    def upgrade(self, factor):
        """ 
        oversample only the spatial axes by a factor of factor
        currently doesn't interpolate at all, just repeats
        """
        
        bigcube = np.repeat(self.cubestack, factor, axis=1).repeat(factor, axis=2)
        bigrms = np.repeat(self.cubestackrms, factor, axis=1).repeat(factor, axis=2)
        
        self.cubestack = bigcube
        self.cubestackrms = bigrms
        
        # store how much it's oversampled by
        self.upgradefactor = factor
        
        return    
    
    def to_flux(self, params, velocity_integrate=True):
        """ converts from temperature units to flux units. won't do anything if the unit
            isn't already in K"""
        
        if self.unit != 'K':
            print('need units to be K, and current units are '+self.unit)
            return
        
        # correct for primary beam response
        self.cube /= 0.72
        self.cuberms /= 0.72

        # actual COMAP beam
        try:
            beam_fwhm = (params.goalbeamscale / cosmo.kpc_proper_per_arcmin(self.z_mean)).to(u.arcmin)
            self.beamscale = beam_fwhm
        except AttributeError:
            beam_fwhm = params.beamwidth * u.arcmin
        sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = sigma_x
        omega_B = (2 * np.pi * sigma_x * sigma_y).to(u.sr)

        # voxel solid angle
        try:
            pixsize = params.xstep*u.deg

        except AttributeError:
            pixsize = 2*u.arcmin
        omega_B = (pixsize**2).to(u.sr)

        # central frequency of each individual spectral channel
        freqbc = self.nuobs_mean[0] + self.freqarr
        fstep = freqbc[1] - freqbc[0]
        freqvals = np.tile(freqbc, (self.cube.shape[2], self.cube.shape[1], 1)).T * u.GHz

        # calculate fluxes in Jy
        Svals = rayleigh_jeans(self.cube*u.K, freqvals, omega_B)
        Srmss = rayleigh_jeans(self.cuberms*u.K, freqvals, omega_B)

        if velocity_integrate:
            # multiply by the channel width in km/s
            delnus = (fstep* u.GHz / freqvals * const.c).to(u.km/u.s)

            Snu_Delnu = Svals * delnus
            dSnu_Delnu = Srmss * delnus

            self.cube = Snu_Delnu.value
            self.cuberms = dSnu_Delnu.value
            self.unit = 'flux'
        else:
            self.cube = Svals.value
            self.cuberms = Srmss.value
            self.unit = 'Jy'

        return
    
    def to_linelum(self, params):

        if self.unit == 'K':
            self.to_flux(params)

        elif self.unit != 'flux':
            print('need flux or temperature units')
            return 
        
        # put into the appropriate astropy units
        self.cube = self.cube * u.Jy * u.km/u.s
        self.cuberms = self.cuberms * u.Jy * u.km/u.s

        freqbc = self.nuobs_mean[0] + self.freqarr

        nuobs = np.tile(freqbc, (self.cube.shape[2], self.cube.shape[1], 1)).T * u.GHz

        # find redshift from nuobs:
        zval = freq_to_z(params.centfreq*u.GHz, nuobs) #*****************

        # luminosity distance in Mpc
        DLs = params.cosmo.luminosity_distance(zval)

        # line luminosity
        linelum = const.c**2 / (2*const.k_B) * self.cube * DLs**2 / (nuobs**2 * (1+zval)**3)
        dlinelum = const.c**2 / (2*const.k_B) * self.cuberms * DLs**2 / (nuobs**2 * (1+zval)**3)

        # put this in per-pixel units for map purposes
        try:
            beam_fwhm = self.beamscale[0] 
        except AttributeError:
            beam_fwhm = 4.5 * u.arcmin
            
        sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
        sigma_y = sigma_x
        omega_B = (2 * np.pi * sigma_x * sigma_y).to(u.sr)
        
        xstep = self.xarr[1] - self.xarr[0]
        omega_pix = ((xstep*u.arcmin)**2).to(u.sr)

        linelum = linelum 
        dlinelum = dlinelum

        # fix units
        linelum = linelum.to(u.K*u.km/u.s*u.pc**2)
        dlinelum = dlinelum.to(u.K*u.km/u.s*u.pc**2)

        # store in object
        self.cube = linelum.value
        self.cuberms = dlinelum.value
        self.unit = 'linelum'

        return


    def get_spectrum(self, in_place=False, summed=False):

        apspec = self.cube[:,self.apminpix[1]:self.apmaxpix[1], self.apminpix[2]:self.apmaxpix[2]]
        dapspec = self.cuberms[:,self.apminpix[1]:self.apmaxpix[1], self.apminpix[2]:self.apmaxpix[2]]

        if summed:
            spec = np.nansum(apspec, axis=(1,2))
            dspec = np.sqrt(np.nansum(dapspec**2, axis=(1,2)))
        else:
            spec, dspec = weightmean(apspec, dapspec, axis=(1,2))
            spec = spec * self.xwidth * self.ywidth 
            dspec = dspec * self.xwidth * self.ywidth 

        if in_place:
            self.spectrum = spec
            self.spectrumrms = dspec

        return spec, dspec

    def get_image(self, in_place=False):

        apim = self.cube[self.apminpix[0]:self.apmaxpix[0],:,:]
        dapim = self.cuberms[self.apminpix[0]:self.apmaxpix[0],:,:]

        im = np.nansum(apim, axis=0)
        dim = np.sqrt(np.nansum(dapim**2, axis=0))

        if in_place:
            self.image = im
            self.imagerms = dim

        return im, dim

    def get_aperture(self, in_place=False, summed=False):

        ap = self.cube[self.apminpix[0]:self.apmaxpix[0]:,self.apminpix[1]:self.apmaxpix[1], self.apminpix[2]:self.apmaxpix[2]]
        dap = self.cuberms[self.apminpix[0]:self.apmaxpix[0],self.apminpix[1]:self.apmaxpix[1], self.apminpix[2]:self.apmaxpix[2]]

        if summed:
            spec = np.nansum(ap, axis=(1,2))
            dspec = np.sqrt(np.nansum(dap**2, axis=(1,2)))
        else:
            spec, dspec = weightmean(ap, dap, axis=(1,2))
            # correct for adjusted solid angle
            spec = spec * self.xwidth * self.ywidth 
            dspec = dspec * self.xwidth * self.ywidth

        val = np.nansum(spec)
        dval = np.sqrt(np.nansum(dspec**2))

        if in_place:
            self.aperture_value = val
            self.aperture_rms = dval

        return val, dval

    def get_output_dict(self, in_place=False):

        llum, dllum = self.get_aperture()
        self.linelum = llum 
        self.dlinelum = dllum

        outdict = {'linelum':self.linelum,
                   'dlinelum':self.dlinelum,
                   'rhoh2':self.rhoh2,
                   'drhoh2':self.drhoh2,
                   'nuobs_mean':self.nuobs_mean,
                   'z_mean':self.z_mean,
                   'nobj':self.ncutouts}

        if in_place:
            self.outdict = outdict

        return outdict
    
    def index_by_field(self, catlist):
        """
        wrangle the index list into three ones (one per field) so you can work with the catalog objects that
        actually made it into the stack
        """

        # store field-separated index list
        fieldidxlist = []
        for cat in catlist:
            fieldidxlist.append(np.where(np.in1d(cat.idx, self.catidx))[0]) # ** do np.isin at some point

        self.fieldcatidx = fieldidxlist 
        
        # find the number of objects in each field
        self.fieldncutouts = [len(idx) for idx in fieldidxlist]

        return


    def copy(self):
        """
        creates a deep copy of the object (ie won't overwrite original)
        """
        return copy.deepcopy(self)


    def make_plots(self, comap, galcat, params, field=None):

        if field:
            fieldstr = '/field'+str(field)
        else:
            fieldstr = ''

        # only one field in the cubelet version:
        if not isinstance(comap, list):

            if params.saveplots:
                field_catalogue_overplotter(galcat, comap, self.catidx, params, fieldstr=fieldstr)

            if params.plotspace and params.plotfreq:

                im, dim = self.get_image()
                spec, dspec = self.get_spectrum()

                try:
                    comment = params.plotcomment
                    if field:
                        comment.append('Field {} Only'.format(field))
                    else:
                        comment.append('Single-field stack')
                except AttributeError:
                    comment = ['Single-field stack']

        else:
            if params.saveplots:
                catalogue_overplotter(galcat, comap, self.catidx, params)

            if params.plotspace and params.plotfreq:
                im, dim = self.get_image()
                spec, dspec = self.get_spectrum()

                try:
                    comment = params.plotcomment
                    comment.append('Multi-field stack')
                except AttributeError:
                    comment = ['Multi-field stack']


        outdict = self.get_output_dict()

        combined_plotter(self.cube, self.cuberms, params, stackim=im, stackrms=dim,
                            stackspec=spec, cmap='PiYG_r',
                            stackresult=outdict, comment=comment, fieldstr=fieldstr)

        return

    def save_cubelet(self, params, fieldstr=None):

        if not fieldstr:
            fieldstr = ''

        # save the output values
        ovalfile = params.datasavepath + fieldstr + '/output_values.csv'
        # strip the values of their units before saving them (otherwise really annoying
        # to read out on the other end)
        outdict = self.get_output_dict()
        outputvals_nu = dict_saver(outdict, ovalfile)

        idxfile = params.datasavepath + fieldstr + '/included_cat_indices.npz'
        np.savez(idxfile, self.catidx)

        cubefile = params.datasavepath + fieldstr + '/stacked_3d_cubelet.npz'
        np.savez(cubefile, T=self.cube, rms=self.cuberms)

        return
    
""" CUTOUT FILTERS """
def upgrade(cubelet, factor, conserve_flux=False):
    """ 
    oversample only the spatial axes of a cutout object by a factor of factor
    currently doesn't interpolate at all, just repeats
    if conserve_flux is true, will divide out area difference 
    otherwise assumes surface brightness values, so value in each pixel won't change even after upgrade
    """

    bigcube = np.repeat(cubelet.cubestack, factor, axis=1).repeat(factor, axis=2)
    bigrms = np.repeat(cubelet.cubestackrms, factor, axis=1).repeat(factor, axis=2)

    if conserve_flux:
        bigcube /= factor**2
        bigrms /= factor**2
    
    cubelet.cubestack = bigcube 
    cubelet.cubestackrms = bigrms 
    
    # store how much it's oversampled by
    cubelet.upgradefactor = factor
    
    return 

def physical_spacing_setup(mapinst, params):
    """
    setup function so calculations regarding scale to do physical spacing don't have to be redone
    every single time. stores the outputs in params
    
    """
    
    # redshift range to be considering
    fexts = np.array([mapinst.freq[0], mapinst.freq[-1]])
    zexts = freq_to_z(params.centfreq, fexts)
    
    # kpc/arcmin in the extrema
    worstchanscale = params.cosmo.kpc_proper_per_arcmin(zexts[1])
    
    # worst beam scale in mpc
    worstbeamscale = (worstchanscale * params.beamwidth*u.arcmin).to(u.Mpc)
    
    # worst velocity resolution
    goaldv = const.c.to(u.km/u.s) * (zexts[0] + 1) * params.chanwidth / 115.27
    
    # naxis for the x and y axes
    oldnaxis2 = params.spacestackwidth*2 + 1
    
    worstchansize = (worstchanscale * mapinst.xstep*u.deg).to(u.Mpc)
    resampfac = (params.goalres / worstchansize).value
    worstoutcdelt2 = mapinst.xstep * resampfac
    params.xstep = worstoutcdelt2 
    mapinst.psxstep = worstoutcdelt2
    
    worstsize = oldnaxis2 * mapinst.xstep / worstoutcdelt2
    worstsize = int(np.round(worstsize))
    
    # naxis for the spectral axis
    oldnaxis1 = params.freqstackwidth*2 + 1
    
    params.goalbeamscale = worstbeamscale
    params.goalxsize = oldnaxis2
    params.goaldv = goaldv
    params.goalfsize = oldnaxis1  


def physical_spacing(cutout, mapinst, params, oversamp_factor=5, do_spectral=True, conserve_flux=False):
    """
    conserves surface brightness, and not flux -- any map going through here needs to be in 
    surface brightness units (ie K)
    if not this will raise a warning (or have a flag for that?****)
    questions:
    - should the inital oversampling be an interpolation? currently just using np.repeat
    - requires a beamwidth as a FWHM from params
    - what's the best fit standard to have constant-sized spaxels? currently using plate carree
    - supress warnings (annoying)
    - calculate (set?) the goal spatial range / number of pixels
    - calculate (set?) the goal spectral range / number of pixels
    - i think ra and dec are now correct but i'm still not convinced it's lined up properly
    - gotta deal with rms also
    - what to save in cutout output
    - catch the other warning
    - how should the aperture size change
    - ***proper vs comoving sizes (flag to change)
    """

    # warn about units based on if you're conserving flux
    if conserve_flux:
        if mapinst.unit == 'K':
            print('Map is in surface brighness units {} -- unit conversion will be done improperly'.format(mapinst.unit))
    else:
        if mapinst.unit == 'linelum':
            print('Map is in Line Luminosity units -- unit conversion will be done improperly')
        elif mapinst.unit == 'flux':
            print('Map is in flux units -- unit conversion will be done improperly')

    # test to make sure the prep function has been run
    # ** maybe force this to run anyways in field stack? in case map parameters change in a jupyter notebook session or smth
    try:
        _ = params.goalbeamscale
    except AttributeError:
        physical_spacing_setup(mapinst, params)
    
    # oversample for the beam convolution
    outcutout = cutout.copy()
    upgrade(outcutout, oversamp_factor)
    
    
    ### set up the input cube
    # WCS object w the oversampling taken into account
    xstep = mapinst.xstep / oversamp_factor
    xpixcent = params.spacestackwidth * oversamp_factor
    fpixval = (mapinst.freq[outcutout.freqidx[0]+params.freqwidth//2]+mapinst.fstep / 2)*1e9
    xpixval = mapinst.ra[outcutout.xidx[0]+params.xwidth//2] + mapinst.xstep / 2
    ypixval = mapinst.dec[outcutout.yidx[0]+params.ywidth//2] + mapinst.ystep / 2
    
    inwcsdict = {"CTYPE1": 'FREQ', 'CDELT1': mapinst.fstep * 1e9, 'CRPIX1': params.freqstackwidth+1, 
             'CRVAL1': fpixval,
             "CTYPE3": 'RA---CAR', 'CUNIT3': 'deg', 'CDELT3': xstep, 'CRPIX3': xpixcent, 'CRVAL3': xpixval,
             "CTYPE2": 'DEC--CAR', 'CUNIT2': 'deg', 'CDELT2': xstep, 'CRPIX2': xpixcent, 'CRVAL2': ypixval,
             "ZSOURCE": outcutout.z}
    inwcs = wcs.WCS(inwcsdict)
    
    # input cube
    cube = SpectralCube(data=outcutout.cubestack.T, wcs=inwcs)
    cuberms = SpectralCube(data=outcutout.cubestackrms.T, wcs=inwcs)
    # beam for the input cube
    cube_beam = Beam(params.beamwidth * u.arcmin)
    cube = cube.with_beam(cube_beam)
    cuberms = cuberms.with_beam(cube_beam)
    # give the cube an empty mask so an annoying error doesn't pop up
    blankmask = ~np.isnan(cube)
    cube = cube.with_mask(blankmask)
    cuberms = cuberms.with_mask(blankmask)
    
    ### reconvolve to the uniform beam scale
    # physical scale conversion in this channel
    chanscale = params.cosmo.kpc_proper_per_arcmin(outcutout.z)
    # beam size in this channel to give constant beam resolution in mpc
    goalchanbeam = (params.goalbeamscale / chanscale).to(u.arcmin)
    # as a beam object -- this is the common beam to convolve to
    goal_beam = Beam(goalchanbeam)
    # do the convolution
    rccube = cube.convolve_to(goal_beam)
    rcrms = cuberms.convolve_to(goal_beam)
    
    
    ### spatial reprojection
    #  set up the goal wcs -- new cdelt
    chansize = (chanscale * xstep*u.deg).to(u.Mpc)
    resampfac = (params.goalres / chansize).value
    outcdelt2 = xstep * resampfac
    # set up the goal wcs -- new crpix
    mapcent = params.goalxsize // 2
    fmapcent = params.freqstackwidth
    outwcsdict = {"CTYPE1": 'FREQ', 'CDELT1': mapinst.fstep * 1e9, 'CRPIX1': fmapcent, 
                  'CRVAL1': (outcutout.freq-mapinst.fstep)*1e9,
                  "CTYPE3": 'RA---CAR', 'CUNIT3': 'deg', 'CDELT3': outcdelt2, 
                  'CRPIX3': mapcent, 'CRVAL3': outcutout.x,
                  "CTYPE2": 'DEC--CAR', 'CUNIT2': 'deg', 'CDELT2': outcdelt2, 
                  'CRPIX2': mapcent, 'CRVAL2': outcutout.y}
    outwcs = wcs.WCS(outwcsdict)
    spacehdr = outwcs.to_header()
    spacehdr['NAXIS'] = 3
    spacehdr['NAXIS1'] = rccube.shape[0]
    spacehdr['NAXIS2'] = params.goalxsize
    spacehdr['NAXIS3'] = params.goalxsize
    # actual reprojection
    shape_out = tuple([spacehdr['NAXIS{0}'.format(i + 1)] for i in range(spacehdr['NAXIS'])])#[::-1])
    
    rpmaplist = []
    rprmslist = []
    for chan in range(shape_out[0]):
        # apply the mask by hand before doing the by-hand reprojection (spectral cube casts nans to zero,
        # which are then filled in by reproject_adaptive instead of being treated properly as nans)
        chandata = rccube._data[chan]
        chanrms = rcrms._data[chan]
        chanmask = np.where(~blankmask[chan])
        chandata[chanmask] = np.nan
        chanrms[chanmask] = np.nan
        rpchan,_ = reproject_adaptive((chandata.T, wcs.WCS(rccube.header).celestial),
                                    wcs.WCS(spacehdr).celestial,
                                    shape_out=(shape_out[1],shape_out[2]),
                                    kernel='gaussian', boundary_mode='strict',
                                    conserve_flux=conserve_flux)
        rpmaplist.append(rpchan)
        rprmschan,_ = reproject_adaptive((chanrms.T, wcs.WCS(rcrms.header).celestial),
                                         wcs.WCS(spacehdr).celestial,
                                         shape_out=(shape_out[1],shape_out[2]),
                                         kernel='gaussian', boundary_mode='strict',
                                         conserve_flux=conserve_flux)
        rprmslist.append(rprmschan)
    newcube = np.array(rpmaplist)
    newrms = np.array(rprmslist)
    
    xycube = rccube._new_cube_with(data=newcube.T, wcs=wcs.WCS(spacehdr), meta=rccube.meta)
    xyrms = rcrms._new_cube_with(data=newrms.T, wcs=wcs.WCS(spacehdr), meta=rcrms.meta)

    # peel the mask off because it'll cause a bug
    xycube = xycube.unmasked_copy()
    xyrms = xyrms.unmasked_copy()
    
    ### spectral reprojection
    if do_spectral:
        # put the cube in radio velocity units centered around the object
        vxycube = xycube.with_spectral_unit(u.km/u.s, velocity_convention='radio', 
                                            rest_value=115.27*u.GHz/(1+outcutout.z))
        vxyrms = xyrms.with_spectral_unit(u.km/u.s, velocity_convention='radio',
                                          rest_value=115.27*u.GHz/(1+outcutout.z))
        # lay out the output spectral axis *** pass this in?
        goalspecax = (np.arange(params.goalfsize) - params.goalfsize//2) * params.goaldv 
        # add a nothing mask back into the cube
        blankmask = ~np.isnan(vxycube)
        vxycube = vxycube.with_mask(blankmask)
        vxyrms = vxyrms.with_mask(blankmask)
        # do the spectral reprojection
        xyzcube = vxycube.spectral_interpolate(goalspecax, fill_value = np.nan)
        xyzrms = vxyrms.spectral_interpolate(goalspecax, fill_value = np.nan)
    else:
        xyzcube = xycube
        xyzrms = xyrms
    
    # *** put this back into a cubelet object?
    outcutout.cubestack = xyzcube._data
    #*** what else do i want to save here?
    # coordinate extents
    outcutout.raext = xyzcube.latitude_extrema
    outcutout.decext = xyzcube.longitude_extrema
    outcutout.velext = xyzcube.spectral_extrema # *** probably want this one back in frequency units
    outcutout.xstep = xyzcube.wcs.wcs.cdelt[0]
    outcutout.ystep = xyzcube.wcs.wcs.cdelt[1]
    outcutout.vstep = xyzcube.wcs.wcs.cdelt[2]/1e3
    txyzcube = xyzcube.with_spectral_unit(u.GHz, velocity_convention='radio', rest_value=115.27*u.GHz/(1+outcutout.z))
    outcutout.fstep = -np.diff(txyzcube.spectral_axis)[0].value

    # *** just transforming the RMS the same way as the regular cube for now
    # ** does spectral_cube have a way to do this nicely?
    outcutout.cubestackrms = xyzrms._data
    # outcutout.cubestackrms = np.ones(outcutout.cubestack.shape)
    return outcutout

""" CUTOUT-SPECIFIC FUNCTIONS """
def single_cutout(idx, galcat, comap, params):

    """ can i make this prettier """
    # find gal in each axis, test to make sure it falls into field
    ## freq
    zval = galcat.z[idx]
    nuobs = params.centfreq / (1 + zval)
    if nuobs < np.min(comap.freq) or nuobs > np.max(comap.freq + comap.fstep):
        return None
    freqidx = np.max(np.where(comap.freq < nuobs))
    if np.abs(nuobs - comap.freq[freqidx]) < comap.fstep / 2:
        fdiff = -1
    else:
        fdiff = 1

    # if the map has been rescaled, these arrays will be 2d 
    if len(comap.ra.shape) == 2:

        x = galcat.coords[idx].ra.deg
        if x < np.min(comap.ra) or x > np.max(comap.ra) + np.max(comap.xstep):
            return None
        xidx = np.max(np.where(comap.ra[freqidx] < x))
        if np.abs(x - comap.ra[freqidx, xidx]) < comap.xstep[freqidx] / 2:
            xdiff = -1
        else:
            xdiff = 1

        y = galcat.coords[idx].dec.deg
        if y < np.min(comap.dec) or y > np.max(comap.dec) + np.max(comap.ystep):
            return None
        yidx = np.max(np.where(comap.dec[freqidx] < y))
        if np.abs(y - comap.dec[freqidx, yidx]) < comap.ystep[freqidx] / 2:
            ydiff = -1
        else:
            ydiff = 1
            
    else:
        
        x = galcat.coords[idx].ra.deg
        if x < np.min(comap.ra) or x > np.max(comap.ra + comap.xstep):
            return None
        xidx = np.max(np.where(comap.ra < x))
        if np.abs(x - comap.ra[xidx]) < comap.xstep / 2:
            xdiff = -1
        else:
            xdiff = 1

        y = galcat.coords[idx].dec.deg
        if y < np.min(comap.dec) or y > np.max(comap.dec + comap.ystep):
            return None
        yidx = np.max(np.where(comap.dec < y))
        if np.abs(y - comap.dec[yidx]) < comap.ystep / 2:
            ydiff = -1
        else:
            ydiff = 1

    # start setting up cutout object if it passes all these tests
    cutout = empty_table()

    # center values of the gal (store for future reference)
    cutout.catidx = galcat.catfileidx[idx]
    cutout.catidx = galcat.catfileidx[idx]
    cutout.z = zval
    cutout.coords = galcat.coords[idx]
    cutout.freq = nuobs
    cutout.x = x
    cutout.y = y

    """ set up indices """
    # index the actual aperture to be stacked from the cutout
    # indices for freq axis
    dfreq = params.freqwidth // 2
    if params.freqwidth % 2 == 0:
        if fdiff < 0:
            freqcutidx = (freqidx - dfreq, freqidx + dfreq)
        else:
            freqcutidx = (freqidx - dfreq + 1, freqidx + dfreq + 1)

    else:
        freqcutidx = (freqidx - dfreq, freqidx + dfreq + 1)
    cutout.freqidx = freqcutidx

    # indices for x axis
    dx = params.xwidth // 2
    if params.xwidth  % 2 == 0:
        if xdiff < 0:
            xcutidx = (xidx - dx, xidx + dx)
        else:
            xcutidx = (xidx - dx + 1, xidx + dx + 1)
    else:
        xcutidx = (xidx - dx, xidx + dx + 1)
    cutout.xidx = xcutidx

    # indices for y axis
    dy = params.ywidth // 2
    if params.ywidth  % 2 == 0:
        if ydiff < 0:
            ycutidx = (yidx - dy, yidx + dy)
        else:
            ycutidx = (yidx - dy + 1, yidx + dy + 1)
    else:
        ycutidx = (yidx - dy, yidx + dy + 1)
    cutout.yidx = ycutidx

    # more checks -- make sure it's not going off the center of the map
    if freqcutidx[0] < 0 or xcutidx[0] < 0 or ycutidx[0] < 0:
        return None
    freqlen, xylen = len(comap.freq), len(comap.x)
    if freqcutidx[1] > freqlen or xcutidx[1] > xylen or ycutidx[1] > xylen:
        return None

    """bigger cutouts for plotting"""
    # same process as above, just wider
    df = params.freqstackwidth
    if params.freqwidth % 2 == 0:
        if fdiff < 0:
            freqcutidx = (freqidx - df, freqidx + df)
        else:
            freqcutidx = (freqidx - df + 1, freqidx + df + 1)

    else:
        freqcutidx = (freqidx - df, freqidx + df + 1)
    cutout.freqfreqidx = freqcutidx

    dxy = params.spacestackwidth
    # x-axis
    if params.xwidth  % 2 == 0:
        if xdiff < 0:
            xcutidx = (xidx - dxy, xidx + dxy)
        else:
            xcutidx = (xidx - dxy + 1, xidx + dxy + 1)
    else:
        xcutidx = (xidx - dxy, xidx + dxy + 1)
    cutout.spacexidx = xcutidx

    # y-axis
    if params.ywidth  % 2 == 0:
        if ydiff < 0:
            ycutidx = (yidx - dxy, yidx + dxy)
        else:
            ycutidx = (yidx - dxy + 1, yidx + dxy + 1)
    else:
        ycutidx = (yidx - dxy, yidx + dxy + 1)
    cutout.spaceyidx = ycutidx

    # pad edges of map with nans so you don't have to worry about going off the edge
    # *** OPTIMIZE THIS
    padmap = np.pad(comap.map, ((df,df), (dxy,dxy), (dxy,dxy)), 'constant', constant_values=np.nan)
    padrms = np.pad(comap.rms, ((df,df), (dxy,dxy), (dxy,dxy)), 'constant', constant_values=np.nan)

    padfreqidx = (cutout.freqfreqidx[0] + df, cutout.freqfreqidx[1] + df)
    padxidx = (cutout.spacexidx[0] + dxy, cutout.spacexidx[1] + dxy)
    padyidx = (cutout.spaceyidx[0] + dxy, cutout.spaceyidx[1] + dxy)

    # pull the actual values to stack
    cpixval = padmap[padfreqidx[0]:padfreqidx[1],
                     padyidx[0]:padyidx[1],
                     padxidx[0]:padxidx[1]]
    crmsval = padrms[padfreqidx[0]:padfreqidx[1],
                     padyidx[0]:padyidx[1],
                     padxidx[0]:padxidx[1]]

    # rotate randomly
    if params.rotate:
        cutout.rotangle = params.rng.integers(4) + 1
        cpixval = np.rot90(cpixval, cutout.rotangle, axes=(1,2))
        crmsval = np.rot90(crmsval, cutout.rotangle, axes=(1,2))

    cutout.cubestack = cpixval
    cutout.cubestackrms = crmsval


    # pull the actual values to stack
    pixval = comap.map[cutout.freqidx[0]:cutout.freqidx[1],
                       cutout.yidx[0]:cutout.yidx[1],
                       cutout.xidx[0]:cutout.xidx[1]]
    rmsval = comap.rms[cutout.freqidx[0]:cutout.freqidx[1],
                       cutout.yidx[0]:cutout.yidx[1],
                       cutout.xidx[0]:cutout.xidx[1]]

    # check how many center aperture pixels are masked
    if np.sum(np.isnan(pixval).flatten()) > (params.freqwidth*params.xwidth**2)/2:
        return None

    # less than half of EACH SPECTRAL CHANNEL masked
    for i in range(pixval.shape[0]):
        if np.sum(np.isnan(pixval[i,:,:]).flatten()) > params.xwidth**2 / 2:
            return None

    """ more advanced stacks """
    # subtract global spectral mean
    if params.specmeanfilter:
        cutout = remove_cutout_spectral_mean(cutout, params)

    # check if the cutout failed the tests in these functions
    if not cutout:
        return None

    # subtract the per-channel means
    if params.chanmeanfilter:
        cutout = remove_cutout_chanmean(cutout, params)

    # check if the cutout failed the tests in these functions
    if not cutout:
        return None

    # subtract the low-order modes
    if params.lowmodefilter:
        cutout = remove_cutout_lowmodes(cutout, params)

    # check if the cutout failed the tests in these functions
    if not cutout:
        return None
    
    # put the cutout into line luminosity units
    # if comap.unit != 'linelum':
    #     print('putting cutout into line lum')
    #     nuobsarr = np.tile(nuobs,[31,31,1]).T
    #     cutmap, cutmaprms = line_luminosity(cutout.cubestack, cutout.cubestackrms, nuobsarr, params, summed=True)
    #     cutout.cubestack = cutmap
    #     cutout.cubestackrms = cutmaprms

    # physical space the cutout
    if params.physicalspace:
        cutout = physical_spacing(cutout, comap, params, oversamp_factor=params.pspacefac)

    # *** is this still doing anything?
    if params.obsunits:
        observer_units_weightedsum(pixval, rmsval, cutout, params)

    
    # try:
    #     if params.physicalspace:
    #         cutout = physical_spacing(cutout, comap, params, oversamp_factor=params.pspacefac)
    # except AttributeError:
    #     print('params.physicalspace not set: defaulting to false')
    #     params.physicalspace = False

    return cutout


""" ACTUAL STACKING """

def field_stack(comap, galcat, params, field=None, goalnobj=None):
    """
    wrapper to stack up a single field, using the cubelet object
    assumes comap is already in the desired units
    """

    # set up for rotating each cutout randomly if that's set to happen
    if params.rotate:
        params.rng = np.random.default_rng(params.rotseed)

    ti = 0
    # if we're keeping track of the number of cutouts
    if goalnobj:
        field_nobj = 0

    for i in range(galcat.nobj):
        cutout = single_cutout(i, galcat, comap, params)

        # if it passed all the tests, keep it
        if cutout:
            if field:
                cutout.field = field

            # stack as you go
            if ti == 0:
                stackinst = cubelet(cutout, params)
                if stackinst.unit != 'linelum':
                    stackinst.to_linelum(params)
                ti = 1
            else:
                stackinst_new = cubelet(cutout, params)
                if stackinst_new.unit != 'linelum':
                    stackinst_new.to_linelum(params)
                stackinst.stackin_cubelet(stackinst_new)

            if goalnobj:
                field_nobj += 1

                if field_nobj == goalnobj:
                    if params.verbose:
                        print("Hit goal number of {} cutouts".format(goalnobj))
                        break

        if params.verbose:
            if i % 100 == 0:
                print('   done {} of {} cutouts in this field'.format(i, galcat.nobj))

    try:
        stackinst.make_plots(comap, galcat, params, field=field)
    except UnboundLocalError:
        print('No values to stack in this field')
        # return None

    if field:
        fieldstr = '/field'+str(field)
    else:
        fieldstr = ''

    try:
        if stackinst:
            stackinst.save_cubelet(params, fieldstr)

        return stackinst
    except UnboundLocalError: 
        return None

def stacker(maplist, catlist, params, trim_cat=True):
    """
    wrapper to perform a full stack on all available values in the catalogue.
    """

    # set up: all the housekeeping stuff
    fields = [1,2,3]

    if trim_cat:
        print('trimming catalog')
        # trim the catalogs down to match the actual signal in the maps
        for i in range(3):
            goodidx = np.where(~np.isnan(np.nanmean(maplist[i].map, axis=0)))
            raminidx, ramaxidx = np.min(goodidx[1]), np.max(goodidx[1])+1
            decminidx, decmaxidx = np.min(goodidx[0]), np.max(goodidx[0])+1

            ramin, ramax = maplist[i].ra[[raminidx, ramaxidx]]
            decmin, decmax = maplist[i].dec[[decminidx, decmaxidx]]

            catidxra = np.logical_and(catlist[i].ra() > ramin, catlist[i].ra() < ramax)
            catidxdec = np.logical_and(catlist[i].dec() > decmin, catlist[i].dec() < decmax)
            catidx = np.where(np.logical_and(catidxra, catidxdec))[0]
            
            catlist[i].subset(catidx)


    # for simulations -- if the stacker should stop after a certain number
    # of cutouts. set this up to be robust against per-field or total vals
    if params.goalnumcutouts:
        if isinstance(params.goalnumcutouts, (int, float)):
            numcutoutlist = [params.goalnumcutouts // len(maplist),
                             params.goalnumcutouts // len(maplist),
                             params.goalnumcutouts // len(maplist)]
        else:
            numcutoutlist = params.goalnumcutouts
    else:
        numcutoutlist = [None, None, None]

    # change units of the map
    # if maplist[0].unit != 'linelum':
    #     if params.verbose:
    #         print('Units are '+maplist[0].unit+'. Changing to linelum')
    #     for map in maplist:
    #         map.to_flux()
    #         map.to_linelum(params)
    

    cubelist = []
    for i in range(len(maplist)):
        if numcutoutlist[i] == 0:
                print('No cutouts required in Field {}'.format(fields[i]))
                cubelist.append(None)
                continue
        
        if params.verbose:
            print('Starting field {}'.format(i+1))
        cube = field_stack(maplist[i], catlist[i], params, field=fields[i], goalnobj=numcutoutlist[i])
        cubelist.append(cube)

    if params.verbose:
            print('Field {} complete'.format(fields[i]))

    # combine everything together into one stack
    stackedcube = cubelist[0]
    stackedcube.stackin_cubelet(cubelist[1])
    stackedcube.stackin_cubelet(cubelist[2])

    llum, dllum = stackedcube.get_aperture()


    # make plots, save stuff
    if params.plotspace:
        stackedcube.make_plots(maplist, catlist, params)
    stackedcube.save_cubelet(params)

    # rearrange the index list by field
    stackedcube.index_by_field(catlist)

    return stackedcube


""" OBSERVER UNIT FUNCTIONS """

def line_luminosity(flux, rms, nuobs, params, summed=True):
    """
    Function to calculate the (specifically CO) line luminosity of a line emitter from
    its flux. from Solomon et al. 1997 (https://iopscience.iop.org/article/10.1086/303765/pdf)
    -------
    INPUTS:
    -------
    flux:   brightness of the source in Jy (should be a quantity)
    nuobs:  central observed frequency in GHz (unitless float)
    params: lim_stacker params object (only used for central frequency)
    --------
    OUTPUTS:
    --------
    linelum: L'_CO in K km/s pc^2
    """

    dnuobs = params.chanwidth * u.GHz * (np.arange(len(flux)) - len(flux)//2)
    nuobs = nuobs*u.GHz + dnuobs

    if not summed:
        nuobs = np.tile(nuobs, (flux.shape[2], flux.shape[1], 1)).T

    # find redshift from nuobs:
    zval = freq_to_z(params.centfreq*u.GHz, nuobs)

    # luminosity distance in Mpc
    DLs = cosmo.luminosity_distance(zval)

    # line luminosity
    linelum = const.c**2 / (2*const.k_B) * flux * DLs**2 / (nuobs**2 * (1+zval)**3)
    dlinelum = const.c**2 / (2*const.k_B) * rms * DLs**2 / (nuobs**2 * (1+zval)**3)

    # fix units
    linelum = linelum.to(u.K*u.km/u.s*u.pc**2)
    dlinelum = dlinelum.to(u.K*u.km/u.s*u.pc**2)

    # if summed, sum across channels for an overall line luminosity
    if summed:
        linelum = np.nansum(linelum)
        dlinelum = np.sqrt(np.nansum(dlinelum**2))

    return linelum, dlinelum

def linelum_to_flux(linelum, meanz, params):

    nuobs = nuem_to_nuobs(params.centfreq, meanz) * u.GHz

    flux = linelum*u.K*u.km/u.s*u.pc**2 * 2 * const.k_B / const.c**2
    flux = flux * nuobs**2 * (1+meanz)**3 / cosmo.luminosity_distance(meanz)**2

    return (flux).to(u.Jy*u.km/u.s)

def rho_h2(linelum, nuobs, params):
    """
    Function to calculate the (specifically CO) line luminosity of a line emitter from
    its flux. from Solomon et al. 1997 (https://iopscience.iop.org/article/10.1086/303765/pdf).
    Uses COMOVING distances
    -------
    INPUTS:
    -------
    linelum: line luminosity of the source in K km/s pc^2 (should be a quantity)
    nuobs:   observed frequency in frequency units (should be a quantity)
    params:  lim_stacker params object (used for central frequency and the size of the aperture,
             in order to calculate the cosmic volume covered by the aperture)
    --------
    OUTPUTS:
    --------
    rhoh2: molecular gas density in the aperture (Msun / Mpc^3; astropy quantity)
    """


    alphaco = 3.6*u.Msun / (u.K * u.km/u.s*u.pc**2)

    # h2 masses
    mh2 = linelum * alphaco

    nu1 = ((nuobs*u.GHz - params.freqwidth/2* params.chanwidth*u.GHz).to(u.GHz)).value
    nu2 = ((nuobs*u.GHz + params.freqwidth/2* params.chanwidth*u.GHz).to(u.GHz)).value

    (z, z1, z2) = freq_to_z(params.centfreq, np.array([nuobs, nu1, nu2]))

    distdiff = cosmo.luminosity_distance(z1)/(1+z1) - cosmo.luminosity_distance(z2)/(1+z2)

    # proper volume of the cube
    # volus = ((cosmo.kpc_proper_per_arcmin(z) * params.xwidth * 2*u.arcmin).to(u.Mpc))**2 * distdiff
    beamx = 4.5*u.arcmin/(2*np.sqrt(2*np.log(2)))
    volus = ((cosmo.kpc_comoving_per_arcmin(z) * params.xwidth * beamx).to(u.Mpc))**2 * distdiff

    rhoh2 = (mh2 / volus).to(u.Msun/u.Mpc**3)

    return rhoh2


def perchannel_flux_sum(tbvals, rmsvals, nuobs, params):
    """
    Function to determine the per-channel flux (in Jy km/s) of a cutout from a COMAP map.
    This will take the UNWEIGHTED SUM of all the spaxel values in each channel. If there
    are nans in a channel, it fill them in by interpolating them to be the mean value in
    the channel.
    -------
    INPUTS:
    -------
    tbvals:  array of brightness temperature values. first axis needs to be the spectral one
             should be unitless (NxMxL)
    rmsvals: the per-pixel rms uncertainties associated with tbvals. also a unitless array (NxMxL)
    nuobs:   observed frequency in frequency units (should be a float, in GHz)
    params:  lim_stacker params object (only used for central frequency)
    --------
    OUTPUTS:
    --------
    Sval_chans: the flux in the cutout in each individual spectral channel. length-N array of astropy
                quantities (in Jy km/s)
    Srms_chans: the rms associated with each flux value. length-N array of astropy quantities (in Jy km/s)
    """

    # number of frequency values we're dealing with
    freqwidth = tbvals.shape[0]

    # taking a straight sum, so not including certain voxels because they're NaNed
    # out will cause problems. interpolate to fill them (bad)
    tbvals, rmsvals = cubelet_fill_nans(tbvals, rmsvals, params)

    # correct for the primary beam response
    tbvals = tbvals / 0.72
    rmsvals = rmsvals / 0.72

    # not the COMAP beam but the angular size of the region over which the brightness
    # temperature is the given value (ie one spaxel)
    # if physical spacing can't use the hardcoded value
    try:
        redshift = params.centfreq / nuobs - 1
        res = (params.goalres / cosmo.kpc_proper_per_arcmin(redshift)).to(u.arcmin)
        omega_B (res**2).to(u.sr)
    except AttributeError:
        omega_B = ((2*u.arcmin)**2).to(u.sr)

    # central frequency of each individual spectral channel
    nuobsvals = (np.arange(params.freqwidth) - params.freqwidth//2) * params.chanwidth * u.GHz
    nuobsvals = nuobsvals + nuobs*u.GHz

    Sval_chans = np.ones(params.freqwidth)*u.Jy
    Srms_chans = np.ones(params.freqwidth)*u.Jy
    # flux in each spectral channel
    for i in range(params.freqwidth):
        Sval_chan = rayleigh_jeans(tbvals[i,:,:]*u.K, nuobsvals[i], omega_B)
        Sval_chan = np.nansum(Sval_chan)

        Srms_chan = rayleigh_jeans(rmsvals[i,:,:]*u.K, nuobsvals[i], omega_B)
        Srms_chan = np.sqrt(np.nansum(Srms_chan**2))

        Sval_chans[i] = Sval_chan
        Srms_chans[i] = Srms_chan

    # channel widths in km/s
    delnus = (params.chanwidth * u.GHz / nuobsvals * const.c).to(u.km/u.s)

    Snu_Delnu = Sval_chans * delnus
    d_Snu_Delnu = Srms_chans * delnus

    return Snu_Delnu, d_Snu_Delnu

def perchannel_flux_mean(tbvals, rmsvals, nuobs, params):
    """
    Function to determine the per-channel flux (in Jy km/s) of a cutout from a COMAP map.
    This will INVERSE-VARIANCE weight each spaxel by its associated RMS, and thus will ignore
    NaNs.
    -------
    INPUTS:
    -------
    tbvals:  array of brightness temperature values. first axis needs to be the spectral one
             should be unitless (NxMxL)
    rmsvals: the per-pixel rms uncertainties associated with tbvals. also a unitless array (NxMxL)
    nuobs:   observed frequency in frequency units (should be a float, in GHz)
    params:  lim_stacker params object (only used for central frequency)
    --------
    OUTPUTS:
    --------
    Sval_chans: the flux in the cutout in each individual spectral channel. length-N array of astropy
                quantities (in Jy km/s)
    Srms_chans: the rms associated with each flux value. length-N array of astropy quantities (in Jy km/s)
    """

    # number of frequency channels we're dealing with
    freqwidth = tbvals.shape[0]

    # correct for the primary beam response
    tbvals = tbvals / 0.72
    rmsvals = rmsvals / 0.72

    # not the COMAP beam but the angular size of the region over which the brightness
    # temperature is the given value (ie one spaxel)
    # omega_B = ((params.xwidth * 2*u.arcmin)**2).to(u.sr)

    # actual COMAP beam
    beam_fwhm = 4.5*u.arcmin
    sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = sigma_x
    omega_B = (2 * np.pi * sigma_x * sigma_y).to(u.sr)

    # central frequency of each individual spectral channel
    nuobsvals = (np.arange(freqwidth) - freqwidth//2) * params.chanwidth * u.GHz
    nuobsvals = nuobsvals + nuobs*u.GHz

    Sval_chans = np.ones(freqwidth)*u.Jy
    Srms_chans = np.ones(freqwidth)*u.Jy
    # flux in each spectral channel
    for i in range(freqwidth):
        Sval_chan = rayleigh_jeans(tbvals[i,:,:]*u.K, nuobsvals[i], omega_B)

        Srms_chan = rayleigh_jeans(rmsvals[i,:,:]*u.K, nuobsvals[i], omega_B)

        # per-channel flux density by taking the weighted mean
        Sval_chan, Srms_chan = weightmean(Sval_chan, Srms_chan)

        Sval_chans[i] = Sval_chan
        Srms_chans[i] = Srms_chan

    # channel widths in km/s
    delnus = (params.chanwidth * u.GHz / nuobsvals * const.c).to(u.km/u.s)

    Snu_Delnu = Sval_chans * delnus
    d_Snu_Delnu = Srms_chans * delnus

    return Snu_Delnu, d_Snu_Delnu


def perpixel_flux(tbvals, rmsvals, nuobs, params):
    """
    Function to determine the flux (in Jy km/s) IN EACH PIXEL of a cutout
    from a COMAP map. Unlike other functions, this function won't try to
    combine pixels in any way
    -------
    INPUTS:
    -------
    tbvals:  array of brightness temperature values. first axis needs to be the
             spectral one. should be unitless (NxMxL)
    rmsvals: the per-pixel rms uncertainties associated with tbvals. also a
             unitless array (NxMxL)
    nuobs:   observed frequency of the central pixel (index N/2) in frequency
             units (should be a float, in GHz)
    params:  lim_stacker params object (only used for central frequency)
    --------
    OUTPUTS:
    --------
    Snu_Delnu:  the flux in the cutout in each individual spectral channel.
                (NxMxL) array of astropy quantities (in Jy km/s)
    dSnu_Delnu: the rms associated with each flux value. (NxMxL) array of
                astropy quantities (in Jy km/s)
    """

    # number of frequency channels we're dealing with
    freqwidth = tbvals.shape[0]

    # correct for the primary beam response
    tbvals = tbvals / 0.72
    rmsvals = rmsvals / 0.72

    # not the COMAP beam but the angular size of the region over which the brightness
    # temperature is the given value (ie one spaxel)
    # omega_B = ((params.xwidth * 2*u.arcmin)**2).to(u.sr)

    # actual COMAP beam
    beam_fwhm = 4.5*u.arcmin
    sigma_x = beam_fwhm / (2 * np.sqrt(2 * np.log(2)))
    sigma_y = sigma_x
    omega_B = (2 * np.pi * sigma_x * sigma_y).to(u.sr)

    # central frequency of each individual spectral channel
    nuobsvals = (np.arange(freqwidth) - freqwidth//2) * params.chanwidth * u.GHz
    nuobsvals = nuobsvals + nuobs*u.GHz
    # reshaped to match the cubelet
    nuobsvals = np.tile(nuobsvals, (tbvals.shape[2], tbvals.shape[1], 1)).T

    # calculate fluxes in Jy
    Svals = rayleigh_jeans(tbvals*u.K, nuobsvals, omega_B)
    Srmss = rayleigh_jeans(rmsvals*u.K, nuobsvals, omega_B)

    # multiply by the channel width in km/s
    delnus = (params.chanwidth * u.GHz / nuobsvals * const.c).to(u.km/u.s)

    Snu_Delnu = Svals * delnus
    dSnu_Delnu = Srmss * delnus

    return Snu_Delnu, dSnu_Delnu


def observer_units_sum(tbvals, rmsvals, cutout, params):
    """
    calculate the more physical quantities associated with a single cutout. Uses
    an UNWEIGHTED SUM to get the per-channel flux, interpolating across NaNs.
    """

    # per-channel fluxes
    Sval_chan, Srms_chan = perchannel_flux_sum(tbvals, rmsvals, cutout.freq, params)

    # make the fluxes into line luminosities
    linelum, linelumrms = line_luminosity(Sval_chan, Srms_chan, cutout.freq, params)

    rhoh2 = rho_h2(linelum, cutout.freq, params)
    rhoh2rms = rho_h2(linelumrms, cutout.freq, params)

    cutout.flux = np.nansum(Sval_chan)
    cutout.dflux = np.sqrt(np.nansum(Srms_chan**2))

    cutout.linelum = linelum
    cutout.dlinelum = linelumrms

    cutout.rhoh2 = rhoh2
    cutout.drhoh2 = rhoh2rms

    return cutout


def observer_units_weightedsum(tbvals, rmsvals, cutout, params):
    """
    calculate the more physical quantities associated with a single cutout. Uses
    a WEIGHTED SUM to get the per-channel flux, and thus ignores NaNs.
    assumes cubelet is already in line luminosity units
    """

    # per-channel fluxes
    # Sval_chan, Srms_chan = perchannel_flux_mean(tbvals, rmsvals, cutout.freq, params)

    # make the fluxes into line luminosities
    # linelum, linelumrms = line_luminosity(Sval_chan, Srms_chan, cutout.freq, params)
    pcllum, dpcllum = weightmean(tbvals, rmsvals, axis=(1,2))
    # ap_area = params.xwidth * params.ywidth
    # pcllum, dpcllum = pcllum * ap_area, dpcllum * ap_area

    linelum = np.nansum(pcllum)*u.K*u.km/u.s*u.pc**2
    linelumrms = np.sqrt(np.nansum(dpcllum**2))*u.K*u.km/u.s*u.pc**2

    # rhoh2 = rho_h2(linelum, cutout.freq, params)
    # rhoh2rms = rho_h2(linelumrms, cutout.freq, params)
    rhoh2 = rho_h2(linelum, cutout.freq, params)
    rhoh2rms = rho_h2(linelumrms, cutout.freq, params)

    # cutout.flux = Sval_chan.value
    # cutout.dflux = Srms_chan.value

    cutout.linelum = linelum.value
    cutout.dlinelum = linelumrms.value

    cutout.rhoh2 = rhoh2.value
    cutout.drhoh2 = rhoh2rms.value

    return cutout


def cubelet_fill_nans(pixvals, rmsvals, params):
    """
    function to replace any NaN values in a cubelet aperture with the mean in that
    frequency channel. should obviously check to make sure there are a reasonable
    number of actual values first.
    """

    for i in range(pixvals.shape[0]):
        chanmean, chanrms = weightmean(pixvals[i,:,:], rmsvals[i,:,:])
        pixvals[i, np.where(np.isnan(pixvals[i,:,:]))] = chanmean
        rmsvals[i, np.where(np.isnan(rmsvals[i,:,:]))] = chanrms

    return pixvals, rmsvals


def observer_units(Tvals, rmsvals, zvals, nuobsvals, params):
    """
    unit change to physical units (**OLD***)
    """

    # main beam to full beam correction
    Tvals = Tvals / 0.72
    rmsvals = rmsvals / 0.72

    # actual beam FWHP is a function of frequency - listed values are 4.9,4.5,4.4 arcmin at 26, 30, 34GHz
    # set up a function to interpolate on
    # beamthetavals = np.array([4.9,4.5,4.4])
    # beamthetafreqs = np.array([26, 30, 34])

    # beamthetas = np.interp(nuobsvals, beamthetafreqs, beamthetavals)*u.arcmin
    # omega_Bs = 1.33*beamthetas**2

    # the 'beam' here is actually the stack aperture size
    # beamsigma = params.xwidth / 2 * 2*u.arcmin
    # omega_B = (2 / np.sqrt(2*np.log(2)))*np.pi*beamsigma**2
    # if physical spacing can't use the hardcoded value
    try:
        redshift = params.centfreq / nuobs - 1
        res = (params.goalres / cosmo.kpc_proper_per_arcmin(redshift)).to(u.arcmin)
        omega_B (res**2).to(u.sr)
    except AttributeError:
        omega_B = ((2*u.arcmin)**2).to(u.sr)

    nuobsvals = nuobsvals*u.GHz
    meannuobs = np.nanmean(nuobsvals)

    onesiglimvals = Tvals + rmsvals

    Sact = (Tvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(nuobsvals, omega_B))
    Ssig = (onesiglimvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(nuobsvals, omega_B))
    Srms = (rmsvals*u.K).to(u.Jy, equivalencies=u.brightness_temperature(nuobsvals, omega_B))

    # channel widths in km/s
    delnus = (params.chanwidth*u.GHz*params.freqwidth / nuobsvals * const.c).to(u.km/u.s)

    # luminosity distances in Mpc
    DLs = cosmo.luminosity_distance(zvals)

    # line luminosities
    linelumact = const.c**2/(2*const.k_B)*Sact*delnus*DLs**2/ (nuobsvals**2*(1+zvals)**3)
    linelumsig = const.c**2/(2*const.k_B)*Ssig*delnus*DLs**2/ (nuobsvals**2*(1+zvals)**3)
    linelumrms = const.c**2/(2*const.k_B)*Srms*delnus*DLs**2/ (nuobsvals**2*(1+zvals)**3)


    # fixing units
    beamvalobs = linelumact.to(u.K*u.km/u.s*u.pc**2)
    beamvalslim = linelumsig.to(u.K*u.km/u.s*u.pc**2)
    beamrmsobs = linelumrms.to(u.K*u.km/u.s*u.pc**2)

    # rho H2:
    alphaco = 3.6*u.Msun / (u.K * u.km/u.s*u.pc**2)
    mh2us = (linelumact + 2*linelumrms) * alphaco
    mh2obs = linelumact * alphaco
    mh2rms = linelumrms * alphaco

    nu1 = nuobsvals - params.freqwidth/2* params.chanwidth*u.GHz
    nu2 = nuobsvals + params.freqwidth/2* params.chanwidth*u.GHz

    z = (params.centfreq*u.GHz - nuobsvals) / nuobsvals
    z1 = (params.centfreq*u.GHz - nu1) / nu1
    z2 = (params.centfreq*u.GHz - nu2) / nu2
    meanz = np.nanmean(z)

    distdiff = cosmo.luminosity_distance(z1) - cosmo.luminosity_distance(z2)

    # proper volume of the cube
    # if physical spacing can't use the hardcoded value
    try:
        redshift = params.centfreq / nuobs - 1
        res = (params.goalres / cosmo.kpc_proper_per_arcmin(redshift)).to(u.arcmin)
    except AttributeError:
        res = 2*u.arcmin
    volus = ((cosmo.kpc_proper_per_arcmin(z1) * params.xwidth * res).to(u.Mpc))**2 * distdiff

    rhous = mh2us / volus
    rhousobs = mh2obs / volus
    rhousrms = mh2rms / volus
    # keep number
    beamrhoobs = rhousobs.to(u.Msun/u.Mpc**3)
    beamrhorms = rhousrms.to(u.Msun/u.Mpc**3)
    beamrholim = rhous.to(u.Msun/u.Mpc**3)

    obsunitdict = {'L': beamvalobs, 'dL': beamrmsobs,
                   'rho': beamrhoobs, 'drho': beamrhorms,
                   'nuobs_mean': meannuobs, 'z_mean': meanz}

    return obsunitdict
