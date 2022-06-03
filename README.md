# lim_stacker

Code to stack line-intensity mapping data cubes on galaxy catalogues.

**run_stack.py** contains an example run of the stack

**run_bootstraps.py** contains an example run of stacking on random locations in the 3D cube as a consistency check.


NOTES:
- weighting issue when params.beamscale is set to True -- returning results that seem to be self-consistent but about a factor of 10 too small
- Code to convert returned brightness temperatures to physical units isn't yet well-integrated but should be up soon
