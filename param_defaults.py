""" size of the stack aperture used to calcuate values """
# number of pixels (RA)
xwidth 3
# number of pixels (Dec)
ywidth 3
# number of frequency channels at map freq resolution
freqwidth 1

""" properties of the map """
# CO(1-0) emitted frequency (GHz)
centfreq 115.27
# std of a gaussian 2D beam in pixels
beamwidth 1

""" stacking metaparameters """
 # save cubelets instead of separate spatial/spectral stacks for diagnostic
 # images. this will combine cutouts iteratively instead of returning the
 # full list of them
cubelet True
# return physical units
obsunits True
# verbose output when running
verbose True
# save the stack data
savedata True
# file path for saving the stack data
savepath stack_output

""" plotting parameters """
# save plots to disk
saveplots True
# do the spatial plot
plotspace True
# RADIUS of the output diagnostic spatial images in pixels
# only used if plotspace == True
spacestackwidth 10
 # do the spectral plot
plotfreq True
# TOTAL number of channels to include in the diagnostic spectral images
# only used if plotfreq == True
freqstackwidth 10