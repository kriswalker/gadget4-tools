import numpy as np
from box import Box, Halo
from utils import nfw_density, nfw_mass, vel_disp_nfw
from lmfit import Parameters

# directory containing snapshot and other output files. Change this.
boxdir = 'sim_files/'

snap = 48  # snapshot number
box = Box(path=boxdir, snapshot=snap, verbose=True)
box.plot_mass_function(nbins=20)
box.plot_box(projection='xy')

grp = 10  # group number
sub = None  # subhalo number
halo = Halo(None, boxdir, snap, group_index=grp, subhalo_index=sub,
            verbose=True)
halo.plot_halo(projection='xy')

# initialize NFW fit parameters
params = Parameters()
params.add('central_density', value=np.random.uniform(low=0, high=1.5e6),
           min=0, max=1.5e6)
params.add('scale_radius', value=np.random.uniform(low=0, high=10),
           min=0, max=10)

# number of logarithmic radial bins between the cutoff radii
nbins = 20
cutoff_radii = (10**-2.7, 2)

halo.plot_density_profile(nbins=nbins,
                          cutoff_radii=cutoff_radii,
                          plot_model=True,
                          model=(nfw_density, params),
                          title='density profile')
halo.plot_mass_profile(nbins=nbins,
                       cutoff_radii=cutoff_radii,
                       plot_model=True,
                       model=(nfw_mass, params),
                       title='mass profile')
halo.plot_dispersion_profile(nbins=nbins,
                             cutoff_radii=cutoff_radii,
                             plot_model=True,
                             model=(vel_disp_nfw, params),
                             title='velocity dispersion profile')

print('concentration =', halo.get_concentration(v=200))
