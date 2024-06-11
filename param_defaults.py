""" size of the stack aperture used to calcuate values """
# number of pixels (RA)
xwidth 3
# number of pixels (Dec)
ywidth 3
# number of frequency channels at map freq resolution
freqwidth 3

""" properties of the map """
# CO(1-0) emitted frequency (GHz)
centfreq 115.27
# std of a gaussian 2D beam in arcmin
beamwidth 4.5
# cosmology to use (default 'comap' is the one from the ES papers)
cosmo comap

""" stacking metaparameters """
# only take a specific feed to stack on
# (load all feeds in if set > 20)
usefeed 100
 # save cubelets instead of separate spatial/spectral stacks for diagnostic
 # images. this will combine cutouts iteratively instead of returning the
 # full list of them
cubelet True
# return physical units
obsunits True
# rotate each cubelet by a random pi/2 angle before stacking
# this should help with asymmetric noise which is common when catalogue objects
# tend to fall near the edge of the map
rotate True
# random seed for this random rotation
rotseed 12345
# remove a 2D linear polynomial from each cutout before stacking
lowmodefilter False
# instaed, remove the per-channel mean of the region around the source before stacking
chanmeanfilter False
## if fitting, the number of beams to extend outwards from the stack aperture to fit
fitnbeams 3
## and to mask
fitmasknbeams 1
# find global mean in the cutout spectrum
specmeanfilter False
# if fitting, the number of times the frequency stack aperture to mask out in the center
freqmaskwidth 1
## and number of times the aperture to include total
frequsewidth 8
## if the c0_0 parameter part of the fit is above this value on either side of 0,
## assume the cutout as a whole is bad
fitmeanlimit 100
fitvallims 100 10 10
# min number of hits in a single voxel for inclusion in the stack
voxelhitlimit 50000
# max RMS in K to include in the stack
voxelrmslimit 0.01
# verbose output when running
verbose True
# save the stack data
savedata True
# file path for saving the stack data
savepath stack_output
# for simulations -- to hit a target number of objects exactly
# to split this up by field, pass a list [n_field1, n_field2, n_field3]
goalnumcutouts False
# return the actual cutout objects
returncutlist False

""" plotting parameters """
# save plots to disk
saveplots True
# save fields individually
savefields True
# do the spatial plot
plotspace True
# RADIUS of the output diagnostic spatial images in pixels
# only used if plotspace == True
spacestackwidth 15
 # do the spectral plot
plotfreq True
# TOTAL number of channels to include in the diagnostic spectral images
# only used if plotfreq == True
freqstackwidth 40
# extra diagnostic plots for the cubelet output
plotcubelet True
# units to make the plots ITO ('linelum' for line luminosity or 'flux')
plotunits linelum
