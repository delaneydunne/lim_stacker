# lim_stacker

Code to stack line-intensity mapping data cubes on galaxy catalogues.

**run_bootstraps.py** contains an example run of the stack on random locations
an example run of the regular stacker should be up soon


NOTES:
- weighting issue when params.beamscale is set to True -- returning results that seem to be self-consistent but about a factor of 10 too small
- Code to convert returned brightness temperatures to physical units isn't yet well-integrated but should be up soon
