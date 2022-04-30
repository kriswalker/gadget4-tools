import numpy as np
from box import Box, Subhalo
from utils import nfw_density, nfw_mass, vel_disp_nfw
from lmfit import Parameters

boxdir = 'sim_files/' # directory containing snapshot and other output files. Change this.

snap = 48 # snapshot number
box = Box(path=boxdir, snapshot=snap, verbose=True)
box.plot_mass_function(nbins=20)
box.plot_box()

grp = 10 # group number
sub = None # subhalo number (set group number to None if using this)
subh = Subhalo(boxdir, snap, grp, sub, verbose=True)
subh.plot_subhalo()

# initialize NFW fit parameters
params = Parameters()
params.add('central_density', value=np.random.uniform(low=0, high=1.5e6), min=0, max=1.5e6)
params.add('scale_radius', value=np.random.uniform(low=0, high=10), min=0, max=10)

nbins = 20 # number of logarithmic radial bins
cutoff_radii = (10**-2.7, 2) # inner and outer radius to calculate between (in units of R_200)

subh.plot_density_profile(nbins=nbins,
                          cutoff_radii=cutoff_radii,
                          plot_model=True,
                          model=(nfw_density, params),
                          title='density profile')
subh.plot_mass_profile(nbins=nbins,
                        cutoff_radii=cutoff_radii,
                        plot_model=True,
                        model=(nfw_mass, params),
                        title='mass profile')
subh.plot_dispersion_profile(nbins=nbins,
                        cutoff_radii=cutoff_radii,
                        plot_model=True,
                        model=(vel_disp_nfw, params),
                        title='velocity dispersion profile')

v, c = subh.get_v_c(params=None, v=200)
print('concentration =', c)

