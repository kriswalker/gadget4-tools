import glob
import numpy as np
import h5py
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sphviewer.tools import QuickView
from gadget4tools.utils import recenter, approx_concentration, pretty_print, \
    to_physical_velocity, interpolate2D, magnitude, radial_velocity, fit_model


class Box():

    def __init__(self, path, snapshot_prefix=None, snapshot=None,
                 particle_type=1, to_physical=False, group_list=None,
                 cutout='fof', cutout_radii=2, verbose=True):
        """
        Box object containing parameters and methods for the simulation box as
        a whole.

        Parameters
        ----------
        path : str
            Directory containing the simulation output.
        snapshot_prefix : str
            The prefix of the snapshot files.
        snapshot : int
            Number of the desired snapshot.
        verbose : bool, optional
            Print all the things. The default is True.

        """

        self.path = path
        self.snapshot = snapshot
        self.snapshot_prefix = snapshot_prefix
        self.particle_type = particle_type

        file = path + '{}_{:03d}*.hdf5'

        # print('Checking for halo catalogues...')
        ncat = len(glob.glob(file.format('fof*_tab', snapshot)))
        if ncat != 0:
            # print('Found {} halo catalogue{}.'.format(
            #     ncat, '' if ncat == 1 else 's'))
            self.group, self.subhalo = self.read_groups(glob.glob(
                file.format('fof*_tab', snapshot))[0], to_physical, verbose)
        elif group_list is None:
            print('No halo catalogues found.')
        else:
            raise ValueError('You have entered a value for group_list but I' +
                             ' cannot find any halo catalogues in the' +
                             ' specified directory. Aborting.')

        if snapshot is not None:
            # print('\nChecking for snapshot files...')
            nsnap = len(glob.glob(file.format(snapshot_prefix, snapshot)))
            if nsnap != 0:
                # print('Found {} snapshot file{}.'.format(
                #     nsnap, '' if ncat == 1 else 's'))
                self.read_snap(np.sort(glob.glob(file.format(snapshot_prefix,
                                                 snapshot))), to_physical,
                               group_list, cutout, cutout_radii, verbose)
            else:
                raise ValueError('No snapshot files found! Are you sure you' +
                                 ' have specified the correct directory?')

    def read_parameters(self, datafile, to_physical):

        self.scale_phys = 1.0
        if datafile['Parameters'].attrs['ComovingIntegrationOn'] == 1:
            self.scale_factor = datafile['Header'].attrs['Time']
            if to_physical:
                self.scale_phys = self.scale_factor
        else:
            self.time = datafile['Header'].attrs['Time']
        self.redshift = datafile['Header'].attrs['Redshift']
        self.box_size = datafile['Header'].attrs['BoxSize']
        if 'NSample' in datafile['Parameters'].attrs:
            self.nsample = datafile['Parameters'].attrs['NSample']
        else:
            self.nsample = 1
        self.hubble_constant = datafile['Parameters'].attrs['HubbleParam']
        self.h = datafile['Parameters'].attrs['Hubble']
        self.Omega0 = datafile['Parameters'].attrs['Omega0']
        self.OmegaBaryon = datafile['Parameters'].attrs['OmegaBaryon']
        self.OmegaLambda = datafile['Parameters'].attrs['OmegaLambda']
        self.unit_length = datafile['Parameters'].attrs['UnitLength_in_cm']
        self.unit_mass = datafile['Parameters'].attrs['UnitMass_in_g']
        self.unit_velocity = datafile['Parameters'].attrs[
            'UnitVelocity_in_cm_per_s']

        cm_per_Mpc = 3.085678e24
        g_per_1e10Msun = 1.989e43
        cmps_per_kmps = 1.0e5
        self.length_norm = self.unit_length / cm_per_Mpc
        self.mass_norm = self.unit_mass / g_per_1e10Msun
        self.velocity_norm = self.unit_velocity / cmps_per_kmps

        self.OmegaDM = self.Omega0 - self.OmegaBaryon
        self.mean_interparticle_spacing = self.box_size * self.scale_phys /\
            self.nsample
        self.convergence_radius = 0.77 * \
            (3 * self.OmegaDM / (800 * np.pi))**(1/3) \
            * self.mean_interparticle_spacing / (1 + self.redshift)

        self.gravitational_constant = 43.0071 * self.mass_norm / \
            (self.length_norm * self.velocity_norm**2)

        return

    def read_snap(self, filenames, to_physical, group_list, cutout,
                  cutout_radii, verbose):
        """
        Read in snapshot contents.

        Parameters
        ----------
        filename : str
            Snapshot file.
        verbose : bool
            Print all the things.

        """

        if verbose:
            start = time.time()
            print(("  {}\n"*len(filenames)).format(*filenames))

        coords_all, vels_all, ids_all, masses_all = [], [], [], []

        if group_list is not None and cutout == 'fof':
            offset = self.group['Offset']
            offset_sel_start = offset[np.array(group_list)][..., np.newaxis]
            offset_sel_end = offset[np.array(group_list)+1][..., np.newaxis]
            offset_sel = np.concatenate((offset_sel_start, offset_sel_end),
                                        axis=1)
        for i, f in enumerate(filenames):
            snap = h5py.File(f, 'r')
            if i == 0:
                self.read_parameters(snap, to_physical)

            ids = snap['PartType{}'.format(self.particle_type)][
                'ParticleIDs'][()]
            coords = snap['PartType{}'.format(self.particle_type)][
                'Coordinates'][()] * self.scale_phys
            vels = snap['PartType{}'.format(self.particle_type)][
                'Velocities'][()]
            if to_physical:
                vels = to_physical_velocity(vels, coords, self.redshift,
                                            self.hubble_constant * self.h,
                                            Omega_m=self.Omega0,
                                            Omega_Lambda=self.OmegaLambda,
                                            Omega_k=0)
            if 'Masses' in list(snap['PartType{}'.format(self.particle_type)]):
                masses = snap['PartType{}'.format(self.particle_type)][
                    'Masses'][()]
            else:
                masses = list(snap['Header'].attrs['MassTable'])[
                    self.particle_type] * np.ones(len(ids))

            if group_list is None:
                ids_all.append(ids)
                coords_all.append(coords)
                vels_all.append(vels)
                masses_all.append(masses)
            else:
                if cutout == 'fof':
                    offset_sel_ = offset_sel[np.argwhere(
                        (offset_sel[:, 0] >= 0) |
                        (offset_sel[:, 1] > 0)).flatten()]
                    offset_sel_[np.argwhere(offset_sel_[:, 0] < 0), 0] = 0
                    for idxs in offset_sel_:
                        ids_all.append(ids[slice(*idxs)])
                        coords_all.append(coords[slice(*idxs)])
                        vels_all.append(vels[slice(*idxs)])
                        masses_all.append(masses[slice(*idxs)])
                    offset_sel -= len(ids)
                elif cutout == 'sphere':
                    positions = self.group['Pos'][np.array(group_list)]
                    inds_all = []
                    for i, pos in enumerate(positions):
                        coords_rel = recenter(coords - pos, self.box_size *
                                              self.scale_phys)
                        radii = magnitude(coords_rel)[0]
                        rcut = cutout_radii if (
                            type(cutout_radii) == int or
                            type(cutout_radii) == float) else cutout_radii[i]
                        inds = np.argwhere(radii < rcut).flatten()
                        inds_all.append(inds)
                    inds_all = np.unique(np.hstack(inds_all))
                    ids_all.append(ids[inds_all])
                    coords_all.append(coords[inds_all])
                    vels_all.append(vels[inds_all])
                    masses_all.append(masses[inds_all])
                elif cutout == 'cube':
                    positions = self.group['Pos'][np.array(group_list)]
                    R200s = self.group['R200'][np.array(group_list)]
                    inds_all = []
                    for i, pos in enumerate(positions):
                        coords_rel = recenter(coords - pos, self.box_size *
                                              self.scale_phys)
                        rcut = cutout_radii if (
                            type(cutout_radii) == int or
                            type(cutout_radii) == float) else cutout_radii[i]
                        rcut *= R200s[i]
                        inds = np.argwhere((abs(coords_rel[:, 0]) < rcut) &
                                           (abs(coords_rel[:, 1]) < rcut) &
                                           (abs(coords_rel[:, 2]) < rcut)
                                           ).flatten()
                        inds_all.append(inds)
                    inds_all = np.unique(np.hstack(inds_all))
                    ids_all.append(ids[inds_all])
                    coords_all.append(coords[inds_all])
                    vels_all.append(vels[inds_all])
                    masses_all.append(masses[inds_all])
                else:
                    raise ValueError('Value of cutout not recognized. Must' +
                                     ' be either "fof" (friends-of-friends),' +
                                     ' "sphere", or "cube".')
        self.coords, self.vels, self.ids, self.masses = np.vstack(coords_all),\
            np.vstack(vels_all), np.hstack(ids_all), np.hstack(masses_all)

        tags = ['Gas', 'Dark matter', 'Disk', 'Bulge', 'Stars', 'Boundary']
        self.tag = tags[self.particle_type]

        snap.close()

        if verbose:
            end = time.time()
            print("...Loaded in {0} seconds\n".format(round(end-start, 2)))
            self.box_info()

        return

    def read_groups(self, filename, to_physical, verbose):
        """
        Read in group and subhalo data.

        Parameters
        ----------
        filename : str
            Group file.
        verbose : bool
            Print all the things.

        Returns
        -------
        group : dict
            Group data and parameters.
        subhalo : dict
            Subhalo data and parameters.

        """

        halo_cat = h5py.File(filename, 'r')
        if not hasattr(self, 'redshift'):
            self.read_parameters(halo_cat, to_physical)

        config_options = list(halo_cat['Config'].attrs)

        group = {}
        group['R200'] = halo_cat['Group']['Group_R_Crit200'][()] * \
            self.scale_phys
        group['R500'] = halo_cat['Group']['Group_R_Crit500'][()] * \
            self.scale_phys
        group['M200'] = halo_cat['Group']['Group_M_Crit200'][()]
        group['M500'] = halo_cat['Group']['Group_M_Crit500'][()]
        group['Mass'] = halo_cat['Group']['GroupMassType'][()][
            :, self.particle_type]
        group['Pos'] = halo_cat['Group']['GroupPos'][()] * self.scale_phys
        group['Vel'] = halo_cat['Group']['GroupVel'][()]
        if to_physical:
            group['Vel'] = to_physical_velocity(group['Vel'], group['Pos'],
                                                self.redshift,
                                                self.hubble_constant * self.h,
                                                Omega_m=self.Omega0,
                                                Omega_Lambda=self.OmegaLambda,
                                                Omega_k=0)
        group['Len'] = halo_cat['Group']['GroupLenType'][()][
            :, self.particle_type]
        group['Offset'] = halo_cat['Group']['GroupOffsetType'][()][
            :, self.particle_type]
        if 'SUBFIND' or 'SUBFIND_HBT' in config_options:
            group['FirstSub'] = halo_cat['Group']['GroupFirstSub'][()]
            group['Nsubs'] = halo_cat['Group']['GroupNsubs'][()]
        self.number_of_groups = halo_cat['Header'].attrs['Ngroups_Total']

        np.seterr(divide='ignore', invalid='ignore')
        group['V200'] = np.sqrt(self.gravitational_constant * group['M200'] /
                                group['R200'])
        group['V500'] = np.sqrt(self.gravitational_constant * group['M500'] /
                                group['R500'])
        group['A200'] = group['V200']**2 / group['R200']
        group['A500'] = group['V500']**2 / group['R500']
        group['Vol200'] = 4 * np.pi * group['R200']**3 / 3
        group['Vol500'] = 4 * np.pi * group['R500']**3 / 3
        critical_density_200 = (group['M200'] / (200 * group['Vol200']))[0]
        critical_density_500 = (group['M500'] / (500 * group['Vol500']))[0]
        self.critical_density = np.mean([critical_density_200,
                                         critical_density_500])
        np.seterr(divide='warn', invalid='warn')

        if 'SUBFIND' or 'SUBFIND_HBT' in config_options:
            subhalo = {}
            subhalo['Mass'] = halo_cat['Subhalo']['SubhaloMassType'][()][
                :, self.particle_type]
            subhalo['CM'] = halo_cat['Subhalo']['SubhaloCM'][()] * \
                self.scale_phys
            subhalo['Pos'] = halo_cat['Subhalo']['SubhaloPos'][()] * \
                self.scale_phys
            subhalo['Vel'] = halo_cat['Subhalo']['SubhaloVel'][()]
            if to_physical:
                subhalo['Vel'] = to_physical_velocity(
                    subhalo['Vel'], subhalo['Pos'],
                    self.redshift, self.hubble_constant * self.h,
                    Omega_m=self.Omega0, Omega_Lambda=self.OmegaLambda,
                    Omega_k=0)
            subhalo['HalfmassRad'] = halo_cat['Subhalo'][
                'SubhaloHalfmassRadType'][()][:, self.particle_type] * \
                self.scale_phys
            subhalo['Len'] = halo_cat['Subhalo']['SubhaloLenType'][()][
                :, self.particle_type]
            subhalo['Offset'] = halo_cat['Subhalo']['SubhaloOffsetType'][()][
                :, self.particle_type]
            subhalo['IDMostbound'] = halo_cat['Subhalo'][
                'SubhaloIDMostbound'][()]
            subhalo['SubhaloGroupNr'] = halo_cat['Subhalo'][
                'SubhaloGroupNr'][()]
            subhalo['SubhaloRankInGr'] = halo_cat['Subhalo'][
                'SubhaloRankInGr'][()]
            self.number_of_subhalos = halo_cat['Header'].attrs[
                'Nsubhalos_Total']

        halo_cat.close()

        return group, subhalo

    def calc_mass_function(self, nbins=20):
        """
        Compute the mass function of the box in Mpc^(-3).

        Parameters
        ----------
        nbins : int, optional
            Number of mass bins. The default is 20.

        """

        mass_func_vol, self.mass_bins = np.histogram(np.log10(
            self.group['Mass']), bins=nbins)
        self.mass_function = mass_func_vol / self.box_size**3

    def plot_mass_function(self, nbins=20, title=None, save=False,
                           savefile=None):
        """
        Plot the mass function of the box.

        Parameters
        ----------
        nbins : int, optional
            Number of mass bins. The default is 20.
        title : str, optional
            Title of the plot. The default is None.
        save : bool, optional
            Save the plot. The default is False.
        savefile : str, optional
            Filename of the saved plot. The default is None.

        """

        if not hasattr(self, 'mass_function'):
            self.calc_mass_function(nbins=nbins)

        lin_mass = 10**(self.mass_bins +
                        (self.mass_bins[1] - self.mass_bins[0]) / 2)

        f_mass = plt.subplots(figsize=(5, 5))
        f_mass[1].plot(lin_mass[:-1], self.mass_function)
        f_mass[1].set_xscale('log')
        f_mass[1].set_yscale('log')
        f_mass[1].set_xlabel(r'Halo mass [$10^{10}$ M$_\odot$]')
        f_mass[1].set_ylabel(r'Number density [Mpc$^{-3}$]')
        f_mass[0].suptitle(title)
        f_mass[0].tight_layout()
        if save:
            f_mass[0].savefig(savefile, dpi=300)
        else:
            f_mass[0].show()

    def plot_box(self, projection='xy', center=[0, 0, 0], extent=None,
                 sphviewer=False, bins=[1000, 1000], cmap=None, log=True,
                 title=None, figsize=(8, 8), dpi=500, save=False,
                 savefile=None, return_fig=False):
        """
        2D plot of the particle distribution.

        Parameters
        ----------
        projection : str, optional
            The plane in which to plot. The default is 'xy'.
        title : str, optional
            Title of the plot. The default is None.
        save : bool, optional
            Save the plot. The default is False.
        savefile : str, optional
            Filename of the saved plot. The default is None.

        """

        width = self.box_size * self.scale_phys / 2
        coords = self.coords - np.array([width]*3) - np.array(center)
        if type(extent) == float or type(extent) == int:
            extent = [extent, extent]
        if extent is None:
            extent = [width, width]
        else:
            if type(bins) == int:
                bins = [bins, bins]
            bins1 = np.linspace(-extent[0], extent[0], bins[0])
            bins2 = np.linspace(-extent[1], extent[1], bins[1])
            bins = [bins1, bins2]
        order = []
        for p in projection:
            order.append(
                0 if p == 'x' else 1 if p == 'y' else 2 if p == 'z' else 3)
        order.append(list(set([0, 1, 2]) - set(order))[0])
        coords[:, [0, 1, 2]] = coords[:, order]
        f_box = plt.subplots(figsize=figsize)
        if cmap is None:
            cmaps = [plt.cm.magma, plt.cm.inferno, plt.cm.twilight_shifted,
                     plt.cm.twilight_shifted, plt.cm.cividis,
                     plt.cm.twilight_shifted]
            cmap = cmaps[self.particle_type]
            cmap.set_bad('k', 1)
        if sphviewer:
            qv_parallel = QuickView(coords, r='infinity', plot=False,
                                    x=0, y=0, z=0,
                                    extent=[-extent[0], extent[0],
                                            -extent[1], extent[1]])
            norm = mpl.colors.LogNorm() if log else mpl.colors.Normalize()
            f_box[1].imshow(qv_parallel.get_image(),
                            extent=qv_parallel.get_extent(), cmap=cmap,
                            origin='lower', norm=norm)
        else:
            norm = mpl.colors.LogNorm() if log else mpl.colors.Normalize()
            f_box[1].hist2d(coords[:, 0], coords[:, 1], bins=bins,
                            cmap=cmap, norm=norm)
        f_box[1].set_xlabel(r'{} ($h^{{{}}}$ Mpc)'.format(projection[0], '-1'))
        f_box[1].set_ylabel(r'{} ($h^{{{}}}$ Mpc)'.format(projection[1], '-1'))
        metadata = ['${{{}}}^3$ particles'.format(self.nsample),
                    'Box size = {} Mpc'.format(self.box_size*self.length_norm),
                    'z = {}'.format(round(self.redshift, 3))]
        for mi, m in enumerate(metadata):
            f_box[0].text(0.11, 0.03*(mi+3), m, color='white')
        f_box[0].suptitle(title)
        f_box[0].tight_layout()
        if save:
            f_box[0].savefig(savefile, dpi=500)
        elif return_fig:
            return f_box
        else:
            f_box[0].show()

    def box_info(self):
        """
        Print some of the simulation parameters.

        """

        pretty_print([round(self.redshift, 3),
                      self.box_size * self.length_norm,
                      '{}^3'.format(self.nsample),
                      self.Omega0,
                      self.OmegaBaryon,
                      1-self.Omega0],
                     ['Redshift',
                      'Box size (Mpc)',
                      'Number of particles',
                      'Omega_0',
                      'Omega_Baryon',
                      'Omega_Lambda'],
                     'SIMULATION PARAMETERS')

        return


class Halo():

    def __init__(self, box=None, group_index=None, subhalo_index=None,
                 verbose=True):
        """
        Halo object containing parameters and methods pertaining to individual
        groups or subhalos.

        Parameters
        ----------
        box : Box object, optional
            Box object to load. The default is None.
        group_index : int, optional
            Index of the desired group. The default is None.
        subhalo_index : int, optional
            Index of the desired subhalo. If group_index is specified, only the
            subhalos within that group are considered (i.e. subhalo_index is
            the subhalo rank). The default is None.
        verbose : bool, optional
            Print all the things. The default is True.

        """

        if isinstance(box, Box):
            self.box = box
            self.path = box.path
            self.snapshot = box.snapshot
            self.snapshot_prefix = box.snapshot_prefix
            self.particle_type = box.particle_type
        else:
            raise ValueError("{} is not a \
                             gadget4tools.box.Box object".format(box))
        if group_index is not None:
            self.group_index = group_index
            self.get_group_params(verbose)
        if subhalo_index is not None:
            if group_index is not None:
                self.subhalo_index = np.argwhere(
                    (self.box.subhalo['SubhaloRankInGr'] == subhalo_index) &
                    (self.box.subhalo['SubhaloGroupNr'] == group_index))[0, 0]
            else:
                self.subhalo_index = subhalo_index
            self.get_subhalo_params(verbose)
            self.relative_coords = self.subhalo_relative_coords
            self.relative_vels = self.subhalo_relative_vels
        else:
            self.subhalo_index = None
            self.relative_coords = self.group_relative_coords
            self.relative_vels = self.group_relative_vels
        if self.box.particle_type == 1:
            self.get_halo_particle_positions_and_velocities()
        self.verbose = verbose
        if verbose and group_index is not None:
            self.group_info()

    def get_group_params(self, verbose):
        """
        Retrieve data for specified group.

        Parameters
        ----------
        verbose : bool
            Print all the things.

        """

        self.group_first_subhalo = self.box.group['FirstSub'][self.group_index]
        self.R_200 = self.box.group['R200'][self.group_index]
        self.R_500 = self.box.group['R500'][self.group_index]
        self.M_200 = self.box.group['M200'][self.group_index]
        self.M_500 = self.box.group['M500'][self.group_index]
        self.group_mass = self.box.group['Mass'][self.group_index]
        self.group_position = self.box.group['Pos'][self.group_index]
        self.group_velocity = self.box.group['Vel'][self.group_index]
        self.group_number_of_particles = self.box.group['Len'][
            self.group_index]
        self.group_number_of_subhalos = self.box.group['Nsubs'][
            self.group_index]
        self.group_subhalo_positions = self.box.subhalo['Pos'][
            self.group_first_subhalo:self.group_first_subhalo +
            self.group_number_of_subhalos]
        self.group_subhalo_velocities = self.box.subhalo['Vel'][
            self.group_first_subhalo:self.group_first_subhalo +
            self.group_number_of_subhalos]

        coords_rel = self.box.coords - self.group_position
        self.group_relative_coords = recenter(coords_rel, self.box.box_size *
                                              self.box.scale_phys)
        self.group_relative_vels = self.box.vels - self.group_velocity

        subhalo_pos_rel = self.group_subhalo_positions - self.group_position
        self.group_subhalo_relative_coords = recenter(subhalo_pos_rel,
                                                      self.box.box_size *
                                                      self.box.scale_phys)
        self.group_subhalo_relative_vels = self.group_subhalo_velocities -\
            self.group_velocity

        self.V_200 = self.box.group['V200'][self.group_index]
        self.V_200 = self.box.group['V500'][self.group_index]
        self.A_200 = self.box.group['A200'][self.group_index]
        self.A_200 = self.box.group['A500'][self.group_index]
        self.Vol_200 = self.box.group['Vol200'][self.group_index]
        self.Vol_200 = self.box.group['Vol500'][self.group_index]

        self.virial_overdensity = 200
        self.R_subscript = '200'

        self.R_scale = self.R_200
        self.M_scale = self.R_200
        self.V_scale = self.V_200

        return

    def get_subhalo_params(self, verbose):
        """
        Retrieve data for specified subhalo. If subhalo index is not specified,
        this is the parent (rank 0) subhalo of the group.

        Parameters
        ----------
        verbose : bool
            Print all the things.

        """

        if not hasattr(self, 'group_index'):
            self.group_index = self.box.subhalo['SubhaloGroupNr'][
                self.subhalo_index]
            self.get_group_params(verbose=False)

        self.halfmass_radius = self.box.subhalo['HalfmassRad'][
            self.subhalo_index]
        self.subhalo_mass = self.box.subhalo['Mass'][self.subhalo_index]
        self.subhalo_position = self.box.subhalo['Pos'][self.subhalo_index]
        self.subhalo_center_of_mass = self.box.subhalo['CM'][
            self.subhalo_index]
        self.subhalo_velocity = self.box.subhalo['Vel'][self.subhalo_index]
        self.subhalo_number_of_particles = self.box.subhalo['Len'][
            self.subhalo_index]
        if self.box.particle_type == 1:
            self.subhalo_index_most_bound = np.argwhere(
                (self.box.ids == self.box.subhalo['IDMostbound'][
                    self.subhalo_index])).flatten()[0]
        self.subhalo_rank = self.box.subhalo['SubhaloRankInGr'][
            self.subhalo_index]

        coords_rel = self.box.coords - self.subhalo_position
        self.subhalo_relative_coords = recenter(coords_rel, self.box.box_size *
                                                self.box.scale_phys)
        self.subhalo_relative_vels = self.box.vels - self.subhalo_velocity

        self.R_scale = self.halfmass_radius
        self.M_scale = self.subhalo_mass / 2
        self.V_scale = np.sqrt(self.box.gravitational_constant * self.M_scale /
                               self.R_scale)

        self.R_subscript = '1/2'

        return

    def get_halo_particle_positions_and_velocities(self):
        """
        returns positions of all the particles in the group

        """

        def index(inds):
            return self.box.ids[inds], self.relative_coords[inds], \
                self.relative_vels[inds]

        most_bound_particle = self.box.subhalo['IDMostbound'][
            self.group_first_subhalo]
        ind = np.where(self.box.ids == most_bound_particle)[0]
        if self.subhalo_index is not None:
            offset_sub_low = 0
            for s in range(self.subhalo_rank):
                offset_sub_low += self.box.subhalo['Len'][
                    self.group_first_subhalo + s]
            offset_sub_high = offset_sub_low + self.box.subhalo['Len'][
                self.group_first_subhalo + self.subhalo_rank]
            self.particle_inds = np.arange(ind + offset_sub_low,
                                           ind + offset_sub_high)
            self.particle_ids, self.particle_relative_coords, \
                self.particle_relative_vels = index(self.particle_inds)
            self.number_of_particles = self.subhalo_number_of_particles

        if self.group_index is not None:
            offset_group = self.group_number_of_particles
            self.particle_inds = np.arange(ind, ind + offset_group)
            self.particle_ids, self.particle_relative_coords, \
                self.particle_relative_vels = index(self.particle_inds)
            self.number_of_particles = self.group_number_of_particles

            offset_firstsub = self.box.subhalo['Len'][
                self.group_first_subhalo]
            self.particle_inds_firstsub = np.arange(ind,
                                                    ind + offset_firstsub)
            self.particle_ids_firstsub, \
                self.particle_relative_coords_firstsub, \
                self.particle_relative_vels_firstsub = index(
                    self.particle_inds_firstsub)

            offset_subs = offset_firstsub
            for s in range(1, self.group_number_of_subhalos):
                offset_subs += self.box.subhalo['Len'][
                    self.group_first_subhalo + s]
            self.particle_inds_subs = np.arange(ind + offset_firstsub,
                                                ind + offset_subs)
            self.particle_ids_subs, self.particle_relative_coords_subs, \
                self.particle_relative_vels_subs = index(
                    self.particle_inds_subs)

            offset_fuzz = offset_group
            self.particle_inds_fuzz = np.arange(ind + offset_subs,
                                                ind + offset_fuzz)
            self.particle_ids_fuzz, self.particle_relative_coords_fuzz, \
                self.particle_relative_vels_fuzz = index(
                    self.particle_inds_fuzz)

        return

    def get_merger_tree(self, descend=False):
        """
        Retrieve the merger tree of the subhalo.

        Parameters
        ----------
        descend : bool, optional
            Retrieve the descendants rather than the progenitors. The default
            is False.

        """

        if self.subhalo_index is None:
            raise ValueError('Cannot retrieve merger tree because no subhalo' +
                             ' index has been provided.')

        if self.verbose:
            print("LOADING {0}...".format(self.path + 'trees.hdf5'))

        tree_file = h5py.File(self.path + 'trees.hdf5', 'r')

        tree_data = {}
        tree_data['MainProgenitor'] = tree_file[
            'TreeHalos/TreeMainProgenitor'][()]
        tree_data['FirstDescendant'] = tree_file[
            'TreeHalos/TreeFirstDescendant'][()]
        tree_data['SubhaloMass'] = tree_file['TreeHalos/SubhaloMass'][()]
        tree_data['SubhaloNr'] = tree_file['TreeHalos/SubhaloNr'][()]
        tree_data['GroupNr'] = tree_file['TreeHalos/GroupNr'][()]
        tree_data['SnapNum'] = tree_file['TreeHalos/SnapNum'][()]
        tree_data['Redshift'] = tree_file['TreeTimes/Redshift'][()]
        tree_data['Time'] = tree_file['TreeTimes/Time'][()]
        tree_data['ID'] = tree_file['TreeHalos/TreeID'][()]
        tree_data['StartOffset'] = tree_file['TreeTable/StartOffset'][()]

        mass = []
        n = (np.argwhere((tree_data['SnapNum'] == self.snapshot) &
                         (tree_data['SubhaloNr'] == self.subhalo_index) &
                         (tree_data['GroupNr'] == self.group_index))
             .flatten())[0]
        inds = []
        nrs = []
        grpnrs = []
        snap_nums = []
        redshifts = []
        times = []
        if descend:
            key = 'FirstDescendant'
        else:
            key = 'MainProgenitor'

        def propagate(n):
            inds.append(n)
            mass.append(tree_data['SubhaloMass'][n])
            nrs.append(tree_data['SubhaloNr'][n])
            grpnrs.append(tree_data['GroupNr'][n])

            sn = tree_data['SnapNum'][n]
            snap_nums.append(sn)
            redshifts.append(tree_data['Redshift'][sn])
            times.append(tree_data['Time'][sn])

            return tree_data['StartOffset'][tree_data['ID'][n]] + \
                tree_data[key][n]

        while tree_data[key][n] > -1:
            n = propagate(n)
        if descend:
            n = propagate(n)
            self.merger_tree = np.array(inds)
            self.tree_subhalo_indices = np.array(nrs)
            self.tree_group_indices = np.array(grpnrs)
            self.tree_masses = np.array(mass)
            self.tree_snapshot_numbers = np.array(snap_nums)
            self.tree_redshifts = np.array(redshifts)
            self.tree_times = np.array(times)
        else:
            self.merger_tree = np.flip(np.array(inds))
            self.tree_subhalo_indices = np.flip(np.array(nrs))
            self.tree_group_indices = np.flip(np.array(grpnrs))
            self.tree_masses = np.flip(np.array(mass))
            self.tree_snapshot_numbers = np.flip(np.array(snap_nums))
            self.tree_redshifts = np.flip(np.array(redshifts))
            self.tree_times = np.flip(np.array(times))

    def histogram_halo(self, nbins, cutoff_radii, log=True):
        """
        Bin the particles of the group or subhalo by log radius.

        Parameters
        ----------
        nbins : int
            Number of bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.

        """

        r = magnitude(self.relative_coords)[0]
        # r > 0 so log() doesn't diverge
        inds = np.argwhere((r < cutoff_radii[1]*self.R_scale) & (r > 0))
        self.particle_indices = inds.flatten()
        self.ids_enclosed = self.box.ids[inds].flatten()
        self.r_all = r
        self.r = r[inds].flatten()

        self.coords_inside = self.relative_coords[inds].squeeze(axis=1)
        self.vels_inside = self.relative_vels[inds].squeeze(axis=1)

        if log:
            logr = np.log10(r)
            logrx = np.linspace(np.log10(cutoff_radii[0]*self.R_scale),
                                np.log10(cutoff_radii[1]*self.R_scale),
                                nbins+1)
            logrhist = []
            for k in range(nbins):
                npart = len(np.argwhere((logr > logrx[k]) &
                                        (logr < logrx[k+1])))
                logrhist.append(npart)
            self.rhist = np.array(logrhist)
            logrx_ = logrx + (logrx[1] - logrx[0])/2
            self.redge = 10**logrx
            self.rcenter = 10**logrx_
        else:
            rx = np.linspace(cutoff_radii[0]*self.R_scale,
                             cutoff_radii[1]*self.R_scale, nbins+1)
            rhist = []
            for k in range(nbins):
                npart = len(np.argwhere((r > rx[k]) & (r < rx[k+1])))
                rhist.append(npart)
            self.rhist = np.array(rhist)
            rx_ = rx + (rx[1] - rx[0])/2
            self.redge = rx
            self.rcenter = rx_

        return

    def calc_density_profile(self, nbins, cutoff_radii, model=None, log=True):
        """
        Compute the density profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        model : function, optional
            Model to fit to the profile. The default is None.
        log : bool, optional
            Use logarithmically-spaced bins. The default is True.

        """

        # TODO: use varible particle masses in density profile calculation.

        self.histogram_halo(nbins, cutoff_radii, log=log)

        # if not hasattr(self, 'density_profile'):
        self.density_profile = np.mean(self.box.masses) * self.rhist / \
            ((4 * np.pi / 3) * (self.redge[1:]**3 - self.redge[:-1]**3))

        if model is not None:
            rad = self.rcenter[:-1]
            inner = np.argmin((np.abs(rad - self.box.convergence_radius)))
            outer = np.argmin((np.abs(rad - 0.8*self.R_scale)))
            if outer - inner < 3:
                raise Exception('Subhalo radius too close to convergence' +
                                ' radius! Cannot perform reliable fit.')
            fit_radius = rad[inner:outer]
            fit_density = self.density_profile[inner:outer]
            self.density_fit_params = fit_model(
                model[0], model[1], fit_density/self.box.critical_density,
                {'r': fit_radius}, verbose=self.verbose)
            self.density_profile_model = self.box.critical_density * \
                model[0](self.density_fit_params, rad)
            self.c_200 = self.get_concentration(v=200)

        return

    def calc_mass_profile(self, nbins, cutoff_radii, model=None):
        """
        Compute the mass profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        model : function, optional
            Model to fit to the profile. The default is None.

        """

        # TODO: use varible particle masses in mass profile calculation.

        self.histogram_halo(nbins, cutoff_radii)

        self.mass_profile = np.mean(self.box.masses) * np.cumsum(self.rhist)

        if model is not None:
            rad = self.redge[1:]
            # if not hasattr(self, 'density_fit_params'):
            inner = np.argmin((np.abs(rad - self.box.convergence_radius)))
            outer = np.argmin((np.abs(rad - 0.8*self.R_scale)))
            if outer - inner < 3:
                raise Exception('Subhalo radius too close to convergence' +
                                ' radius! Cannot perform reliable fit.')
            fit_radius = rad[inner:outer]
            fit_mass = self.mass_profile[inner:outer]
            self.density_fit_params = fit_model(
                model[0], model[1], fit_mass/self.box.critical_density,
                {'r': fit_radius}, verbose=self.verbose)
            self.mass_profile_model = self.box.critical_density * model[0](
                self.density_fit_params, rad)

        return

    def calc_log_density_slope_profile(self, nbins, cutoff_radii, model=None):
        """
        Compute the log slope of the density profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        model : function, optional
            Model to fit to the profile. The default is None.

        """
        self.calc_density_profile(nbins, cutoff_radii, model=model,
                                  log=True)

        def density_slope(density_profile):
            return np.diff(np.log(density_profile)) / \
                np.diff(np.log(self.rcenter[:-1]))

        self.log_density_slope_profile = density_slope(self.density_profile)
        if model is not None:
            self.log_density_slope_profile_model = density_slope(
                self.density_profile_model)

        return

    def calc_velocity_dispersion(self, v, x=None):
        """
        Calculate the velocity dispersion of a group of particles.

        Parameters
        ----------
        v : array of shape (nparticles, 3)
            Velocities of the particles relative to the bulk motion of the
            halo.
        x : array of shape (nparticles, 3)
            Positions of the particles relative to the halo center.

        Returns
        -------
        disp : float
            The total velocity dipsersion.
        disprad : float
            The velocity dispersion in the radial direction.

        """
        v -= np.mean(v, axis=0)
        vsq = np.sum(v**2, axis=1)
        disp = np.sqrt(np.mean(vsq, axis=0))

        if x is not None:
            vrad = radial_velocity(x, v)
            vrad -= np.mean(vrad)
            vradsq = np.sum(vrad**2, axis=1)
            disprad = np.sqrt(np.mean(vradsq, axis=0))
            return disp, disprad
        else:
            return disp

    def calc_dispersion_profile(self, nbins, cutoff_radii, model=None,
                                concentration=None, beta=None):
        """
        Compute the radial and total velocity dispersion profile and the beta
        profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        model : function, optional
            Model to fit to the profile. The default is None.
        concentration : float, optional
            Value of the concentration to use in the theoretical profile
            calculation. If unspecified, uses the value from a fit to the
            density profile. The default is None.
        beta : float, optional
            Value of beta to use in the theoretical profile calculation. If
            unspecified, uses the mean of the beta profile. The default is
            None.

        """

        self.histogram_halo(nbins, cutoff_radii)

        disprad = []
        disptot = []
        beta_profile = []
        for b in range(nbins):
            inds_i = np.argwhere((self.r > self.redge[b]) &
                                 (self.r < self.redge[b+1])).flatten()
            vi = self.vels_inside[inds_i, :]
            xi = self.coords_inside[inds_i, :]

            disp_i, disprad_i = self.calc_velocity_dispersion(vi, xi)
            beta_i = 0.5 * (3 - (disp_i**2 / disprad_i**2))

            beta_profile.append(beta_i)
            disprad.append(disprad_i)
            disptot.append(disp_i)
        disprad = np.array(disprad)
        disptot = np.array(disptot)
        beta_profile = np.array(beta_profile)

        self.radial_dispersion_profile = disprad
        self.total_dispersion_profile = disptot
        self.beta_profile = beta_profile

        if model is not None:
            rad = self.rcenter[:-1]
            inner = np.argmin((np.abs(rad - self.box.convergence_radius)))
            outer = np.argmin((np.abs(rad - 0.8*self.R_scale)))
            if (not hasattr(self, 'density_fit_params')
                    and concentration is None):
                raise Exception('No concentration provided for velocity' +
                                ' dispersion model. Either specify with the' +
                                ' concentration argument or fit density' +
                                ' profile first.')
            elif concentration is None:
                concentration = self.get_concentration(v=200)
            if beta is None:
                beta = np.mean(self.beta_profile[inner:outer])
            self.radial_dispersion_profile_model = model[0](
                rad/self.R_scale, concentration, beta)

        return

    def calc_angular_momentum(self, v, x, m=None, specific=False):
        """
        Calculate the angular momentum of a group of particles.

        Parameters
        ----------
        v : array of shape (nparticles, 3)
            Velocities of the particles relative to the bulk motion of the
            halo.
        x : array of shape (nparticles, 3)
            Positions of the particles relative to the halo center.
        specific : bool, optional
            Calculate the specific angular momentum. The default is False.

        Returns
        -------
        float
            The angular momentum.

        """

        if specific:
            return np.sum(np.cross(x, v), axis=0) / len(x)
        else:
            return np.sum((m * np.cross(x, v).T).T, axis=0)

    def calc_angular_momentum_profile(self, nbins, cutoff_radii, model=None,
                                      proj=None, specific=False):
        """
        Compute the angular momentum profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        model : function, optional
            Model to fit to the profile. The default is None.
        proj : array of shape (3,), optional
            Unit vector along which to project the angular momentum. The
            default is None.
        specific : bool, optional
            Calculate the specific angular momentum. The default is False.

        """
        self.histogram_halo(nbins, cutoff_radii)

        angmom_profile = []
        for b in range(nbins):
            inds_i = np.argwhere((self.r > self.redge[b]) &
                                 (self.r < self.redge[b+1]))
            inds_i = inds_i[:, 0]

            vi = self.vels_inside[inds_i, :]
            xi = self.coords_inside[inds_i, :]

            angmom = self.calc_angular_momentum(vi, xi, specific=specific)
            if proj is None:
                angmom_scalar = np.sqrt(np.sum(angmom**2))
            else:
                angmom_scalar = np.sum(angmom * proj)
            angmom_profile.append(angmom_scalar)

        self.angular_momentum_profile = np.array(angmom_profile)

        return

    def plot_density_profile(self, nbins, cutoff_radii, plot_model=False,
                             model=None, style=('C0', '-'), xlim=None,
                             ylim=None, with_rsq=False, with_slope=False,
                             title=None, save=False, savefile=None):
        """
        Plot the density profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        plot_model : bool, optional
            Plot the model fit. The default is False.
        model : function, optional
            Model to fit to the profile. The default is None.
        style : tuple of str, optional
            Color and linestyle of the profile. The default is ('C0', '-').
        xlim : tuple of float, optional
            The x-axis limits of the plot. If None, uses cutoff_radii. The
            default is None.
        ylim : tuple of float, optional
            The y-axis limits of the plot. The default is None.
        title : str, optional
            Title of the plot. The default is None.
        save : bool, optional
            Save the plot. The default is False.
        savefile : str, optional
            Filename of the saved plot. The default is None.

        """

        if not hasattr(self, 'density_profile'):
            self.calc_density_profile(nbins, cutoff_radii, model)

        nrows = 1 + len(np.where([with_rsq, with_slope])[0])
        # f_dens = plt.subplots(nrows, 1, sharex=True, figsize=(6, nrows*6))
        fig = plt.figure(figsize=(5, nrows*5), dpi=300)
        rad = self.rcenter[:-1] / self.R_scale

        i = 1
        ax1 = fig.add_subplot(nrows, 1, i)
        ax1.plot(rad, self.density_profile/self.box.critical_density,
                 alpha=0.8, color=style[0], linestyle=style[1], linewidth=3,
                 label='data')
        if plot_model:
            ax1.plot(
                rad, self.density_profile_model/self.box.critical_density,
                alpha=0.8, color='k', linestyle='--', linewidth=2, label='NFW')
        if xlim is None:
            xlim = (min(rad), max(rad))
        if ylim is None:
            dens = self.density_profile/self.box.critical_density
            ylim = [dens[-1], 2*max(dens)]
        ax1.set_ylim(ylim[0], ylim[1])
        ax1.set_yscale('log')
        ax1.set_ylabel(r'$\rho/\rho_\mathrm{crit}$')
        axes = [ax1]

        if with_rsq:
            i += 1
            ax2 = fig.add_subplot(nrows, 1, i)
            ax2.plot(
                rad, (self.density_profile/self.box.critical_density) *
                rad**2, alpha=0.8, color=style[0], linestyle=style[1],
                linewidth=3)
            if plot_model:
                ax2.plot(
                    rad, (self.density_profile_model /
                          self.box.critical_density) * rad**2,
                    alpha=0.8, color='k', linestyle='--', linewidth=2)
            ax2.set_yscale('log')
            ax2.set_ylabel(
                r'$(\rho/\rho_\mathrm{{{0}}})(r/R_{{{1}}})^2$'
                .format('crit', self.R_subscript))
            axes.append(ax2)

        if with_slope:
            i += 1
            ax3 = fig.add_subplot(nrows, 1, i)
            self.calc_log_density_slope_profile(nbins, cutoff_radii,
                                                model=model)
            rad = self.redge[:-2] / self.R_scale
            ax3.plot(rad, self.log_density_slope_profile,
                     alpha=0.8, color=style[0], linestyle=style[1],
                     linewidth=3)
            if plot_model:
                ax3.plot(rad, self.log_density_slope_profile_model,
                         alpha=0.8, color='k', linewidth=2, linestyle='--')
            ax3.set_ylabel(r'${\rm d}\ln\rho/{\rm d}\ln r$')
            axes.append(ax3)

        for j, ax in enumerate(axes):
            if self.box.nsample != 1:
                label = 'collisional' if j == 0 else None
                ax.axvspan(xlim[0], self.box.convergence_radius/self.R_scale,
                           color='r', alpha=0.3, label=label)
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_xscale('log')
            ax.set_xlabel(r'$r/R_{{{}}}$'.format(self.R_subscript))
        ax1.legend()
        fig.suptitle(title)
        fig.tight_layout()
        if save:
            fig.savefig(savefile, dpi=300)
            plt.close(fig)

    def plot_mass_profile(self, nbins, cutoff_radii, plot_model=False,
                          model=None, style=('C0', '-'), xlim=None, ylim=None,
                          title=None, save=False, savefile=None):
        """
        Plot the mass profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        plot_model : bool, optional
            Plot the model fit. The default is False.
        model : function, optional
            Model to fit to the profile. The default is None.
        style : tuple of str, optional
            Color and linestyle of the profile. The default is ('C0', '-').
        xlim : tuple of float, optional
            The x-axis limits of the plot. If None, uses cutoff_radii. The
            default is None.
        ylim : tuple of float, optional
            The y-axis limits of the plot. The default is None.
        title : str, optional
            Title of the plot. The default is None.
        save : bool, optional
            Save the plot. The default is False.
        savefile : str, optional
            Filename of the saved plot. The default is None.

        """

        if not hasattr(self, 'mass_profile'):
            self.calc_mass_profile(nbins, cutoff_radii, model)

        f_mass = plt.subplots(figsize=(8, 5))
        rad = self.redge[1:] / self.R_scale
        f_mass[1].plot(rad, self.mass_profile/self.M_scale, alpha=0.8,
                       color=style[0], linestyle=style[1], label='data')
        if plot_model:
            f_mass[1].plot(rad, self.mass_profile_model/self.M_scale,
                           alpha=0.8, color='k', linestyle='--', label='NFW')
        f_mass[1].axvline(self.box.convergence_radius/self.R_scale, color='r',
                          linestyle='--', label='convergence radius')
        if xlim is None:
            xlim = cutoff_radii
        if ylim is None:
            mass = self.mass_profile/self.M_scale
            ylim = [min(mass), max(mass)]
        f_mass[1].set_xlim(xlim[0], xlim[1])
        f_mass[1].set_ylim(ylim[0], ylim[1])
        f_mass[1].set_xscale('log')
        f_mass[1].set_yscale('log')
        f_mass[1].set_xlabel(r'$r/R_{{{}}}$'.format(self.R_subscript))
        f_mass[1].set_ylabel(r'$M_\mathrm{{{0}}}/M_{{{1}}}$'
                             .format('enc', self.R_subscript))
        f_mass[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        f_mass[0].suptitle(title)
        f_mass[0].tight_layout()
        if save:
            f_mass[0].savefig(savefile, dpi=300)
            plt.close(f_mass[0])

    def plot_dispersion_profile(self, nbins, cutoff_radii, plot_model=False,
                                model=None, concentration=None, beta=None,
                                style=(('C0', 'C1'), ('-', '-')), xlim=None,
                                ylim=None, title=None, save=False,
                                savefile=None):
        """
        Plot the radial and total velocity dipsersion profile and the beta
        profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        plot_model : bool, optional
            Plot the model fit. The default is False.
        model : function, optional
            Model to fit to the profile. The default is None.
        concentration : float, optional
            Value of the concentration to use in the theoretical profile
            calculation. If unspecified, uses the value from a fit to the
            density profile. The default is None.
        style : tuple of str, optional
            Color and linestyle of the profile. The default is
            (('C0', 'C1'), ('-', '-')).
        xlim : tuple of float, optional
            The x-axis limits of the plot. If None, uses cutoff_radii. The
            default is None.
        ylim : tuple of float, optional
            The y-axis limits of the plot. The default is None.
        title : str, optional
            Title of the plot. The default is None.
        save : bool, optional
            Save the plot. The default is False.
        savefile : str, optional
            Filename of the saved plot. The default is None.

        """

        if not hasattr(self, 'radial_dispersion_profile'):
            self.calc_dispersion_profile(nbins, cutoff_radii,
                                         concentration=concentration,
                                         beta=beta, model=model)

        f_disp = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        rad = self.rcenter[:-1] / self.R_scale
        f_disp[1][0].plot(rad, self.radial_dispersion_profile/self.V_scale,
                          alpha=0.8, color=style[0][0], linestyle=style[1][0],
                          label='radial')
        if plot_model:
            f_disp[1][0].plot(rad, self.radial_dispersion_profile_model,
                              alpha=0.8, color='k', linestyle='--',
                              label='NFW')
        f_disp[1][0].plot(rad, self.total_dispersion_profile/self.V_scale,
                          alpha=0.8, color=style[0][1],
                          linestyle=style[1][1], label='total')
        f_disp[1][1].axhline(0, color='r', alpha=0.5)
        f_disp[1][1].plot(rad, self.beta_profile, alpha=0.8, color=style[0][0],
                          linestyle=style[1][0])
        f_disp[1][0].axvline(self.box.convergence_radius/self.R_scale,
                             color='r', linestyle='--',
                             label='convergence radius')
        f_disp[1][1].axvline(self.box.convergence_radius/self.R_scale,
                             color='r', linestyle='--')
        if xlim is None:
            xlim = cutoff_radii
        f_disp[1][0].set_xlim(xlim[0], xlim[1])
        if ylim is not None:
            f_disp[1][0].set_ylim(ylim[0], ylim[1])
        f_disp[1][1].set_ylim(-2, 2)
        f_disp[1][0].set_xscale('log')
        f_disp[1][1].set_xscale('log')
        f_disp[1][0].set_xlabel(r'$r/R_{{{}}}$'.format(self.R_subscript))
        f_disp[1][0].set_ylabel(r'$\sigma/V_{{{}}}$'.format(self.R_subscript))
        f_disp[1][1].set_xlabel(r'$r/R_{{{}}}$'.format(self.R_subscript))
        f_disp[1][1].set_ylabel(r'$\beta$')
        f_disp[1][0].legend(loc='center left', bbox_to_anchor=(1, 0))
        f_disp[0].suptitle(title)
        f_disp[0].tight_layout()
        if save:
            f_disp[0].savefig(savefile, dpi=300)
            plt.close(f_disp[0])

    def plot_angular_momentum_profile(self, nbins, cutoff_radii, proj=None,
                                      specific=False, save=False,
                                      savefile=None):
        """
        Compute the angular momentum profile of the group or subhalo.

        Parameters
        ----------
        nbins : int
            Number of radial bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.
        proj : array of shape (3,), optional
            Unit vector along which to project the angular momentum. The
            default is None.
        specific : bool, optional
            Calculate the specific angular momentum. The default is False.
        save : bool, optional
            Save the plot. The default is False.
        savefile : str, optional
            Filename of the saved plot. The default is None.

        """

        # if not hasattr(self, 'radial_dispersion_profile'):
        self.calc_angular_momentum_profile(nbins, cutoff_radii, proj=proj,
                                           specific=specific)

        f_angmom = plt.subplots(figsize=(10, 6))
        rad = self.rcenter[:-1] / self.R_scale
        f_angmom[1].plot(
            rad, self.angular_momentum_profile/(self.V_scale * self.R_scale),
            alpha=0.8)
        f_angmom[1].axvline(self.box.convergence_radius/self.R_scale,
                            color='r', linestyle='--',
                            label='convergence radius')
        f_angmom[1].set_xlabel(r'$r/R_{{{}}}$'.format(self.R_subscript))
        f_angmom[1].set_ylabel(r'$|j|/(V_{{{0}}}R_{{{1}}})$'
                               .format(self.R_subscript, self.R_subscript))
        f_angmom[1].legend(loc='center left', bbox_to_anchor=(1, 0))
        f_angmom[0].tight_layout()
        if save:
            f_angmom[0].savefig(savefile, dpi=300)
            plt.close(f_angmom[0])

    def plot_halo(self, projection='xy', extent=[-2, 2, -2, 2],
                  sphviewer=True, bins=1000, log=False, cmap=None,
                  title=None, dpi=500, save=False, savefile=None):
        """
        2D plot of the halo particle distribution.

        Parameters
        ----------
        projection : str, optional
            The plane in which to plot. The default is 'xy'.
        extent : array of shape (4,), optional
            x and y limits of the plot in units of the virial radius. The
            default is [-2, 2, -2, 2].
        title : str, optional
            Title of the plot. The default is None.
        save : bool, optional
            Save the plot. The default is False.
        savefile : str, optional
            Filename of the saved plot. The default is None.

        """

        extent = self.R_scale * np.array(extent)
        coords_ = np.copy(self.relative_coords)
        r = magnitude(coords_)[0]
        sx, sy = 2 * max(np.abs(extent[:2])), 2 * max(np.abs(extent[2:]))
        theta = np.arctan(sy / sx)
        Rlim = sy / (2 * np.sin(theta))
        fac = 20 if sphviewer else 2
        inds = np.argwhere((r < fac * Rlim)).flatten()
        coords = coords_[inds]
        order = []
        for p in projection:
            order.append(
                0 if p == 'x' else 1 if p == 'y' else 2 if p == 'z' else 3)
        order.append(list(set([0, 1, 2]) - set(order))[0])
        coords[:, [0, 1, 2]] = coords[:, order]
        f_subh = plt.subplots(figsize=(8, 8))
        if cmap is None:
            cmaps = [plt.cm.magma, plt.cm.inferno, plt.cm.twilight_shifted,
                     plt.cm.twilight_shifted, plt.cm.cividis,
                     plt.cm.twilight_shifted]
            cmap = cmaps[self.box.particle_type]
            cmap.set_bad('k', 1)
        if sphviewer:
            qv_parallel = QuickView(coords, r='infinity', plot=False,
                                    x=0, y=0, z=0, extent=list(extent))
            norm = mpl.colors.LogNorm() if log else mpl.colors.Normalize()
            f_subh[1].imshow(qv_parallel.get_image(),
                             extent=qv_parallel.get_extent(), cmap=cmap,
                             origin='lower', norm=norm)
        else:
            norm = mpl.colors.LogNorm() if log else mpl.colors.Normalize()
            f_subh[1].hist2d(coords[:, 0], coords[:, 1], bins=bins,
                             cmap=cmap, norm=norm)
        x = np.linspace(-self.R_scale, self.R_scale, 100)
        f_subh[1].plot(x, np.sqrt(self.R_scale**2 - x**2), color='r',
                       linestyle='--', linewidth=1,
                       label=r'$R_{{{}}}$'.format(self.R_subscript))
        f_subh[1].plot(x, -np.sqrt(self.R_scale**2 - x**2), color='r',
                       linestyle='--', linewidth=1)
        f_subh[1].set_xlim(*extent[:2])
        f_subh[1].set_ylim(*extent[2:])
        f_subh[1].set_xlabel(r'{} ($h^{{{}}}$ Mpc)'.format(
            projection[0], '-1'))
        f_subh[1].set_ylabel(r'{} ($h^{{{}}}$ Mpc)'.format(
            projection[1], '-1'))
        if self.box.particle_type == 1:
            metadata = ['{} particles'.format(self.number_of_particles),
                        '$R_{{{0}}}$ = {1} Mpc'.format(
                            self.R_subscript,
                            round(self.R_scale * self.box.length_norm, 3)),
                        'z = {}'.format(round(self.box.redshift, 3)),
                        'Group {}'.format(self.group_index)]
        else:
            metadata = ['$R_{{{0}}}$ = {1} Mpc'.format(
                            self.R_subscript,
                            round(self.R_scale * self.box.length_norm, 3)),
                        'z = {}'.format(round(self.box.redshift, 3)),
                        'Group {}'.format(self.group_index)]
        if self.subhalo_index is not None:
            metadata.insert(3, 'Subhalo {0} ({1})'.format(
                self.subhalo_rank, self.subhalo_index))
            mass = self.subhalo_mass * self.box.mass_norm
            exponent = np.floor(np.log10(mass))
            mass /= 10**exponent
            metadata.insert(2, r'$M={0}\times 10^{{{1}}}\,M_\odot$'.format(
                round(mass, 2), str(int(10 + exponent))))
        else:
            mass = self.M_200 * self.box.mass_norm
            exponent = np.floor(np.log10(mass))
            mass /= 10**exponent
            metadata.insert(2,
                            r'$M_{{{0}}}={1}\times10^{{{2}}}\,M_\odot$'.format(
                                200, round(mass, 2), str(int(10 + exponent))))
        for mi, m in enumerate(metadata):
            f_subh[0].text(0.12, 0.03*(mi+3.3), m, color='white')
        if title is None:
            title = self.box.tag
        f_subh[0].suptitle(title)
        f_subh[0].tight_layout()
        f_subh[0].legend()
        if save:
            f_subh[0].savefig(savefile, dpi=dpi)
            plt.close(f_subh[0])
        return f_subh

    def plot_phase_space(self, outer_radius=4, kde=False, histogram=True,
                         bins=100, by_components=False, comoving_velocity=True,
                         vlim=None, cmap=None, title=None, save=False,
                         savefile=None):

        def phase_space(inds):
            coords, vels = self.relative_coords[inds], self.relative_vels[inds]

            H0 = self.box.hubble_constant * self.box.h
            if comoving_velocity:
                vels = to_physical_velocity(vels, coords,
                                            self.box.redshift, H0,
                                            Omega_m=self.box.Omega0,
                                            Omega_Lambda=self.box.OmegaLambda,
                                            Omega_k=0)
            return magnitude(coords)[0], radial_velocity(coords, vels)

        r_all = magnitude(self.relative_coords)[0]
        interior = np.argwhere((r_all < outer_radius*self.R_scale) &
                               (r_all > 0)).flatten()

        fps = plt.subplots(figsize=(7, 5))
        if kde:
            r, vr = phase_space(interior)
            kde_data = np.vstack((r[~np.isnan(vr)], vr[~np.isnan(vr)])).T
            rax, vax, density = interpolate2D(kde_data, kernel='gaussian',
                                              bandwidth=0.02, resolution=100)
            fps[1].pcolormesh(rax/self.R_scale, vax, density)
        elif by_components:
            comp_inds = [self.particle_inds_firstsub,
                         self.particle_inds_subs,
                         self.particle_inds_fuzz]
            comp_inds_flat = np.hstack(comp_inds)
            comp_remain = np.setdiff1d(interior, comp_inds_flat)
            comp_inds.append(comp_remain)
            cmap = mpl.cm.get_cmap('inferno')
            colors = [cmap(0.4), cmap(0.6), cmap(0.8), 'grey']
            labels = ['main halo', 'subhalos', 'fuzz', None]
            for i, comp in enumerate(comp_inds):
                r, vr = phase_space(comp)
                fps[1].scatter(r/self.R_scale, vr, alpha=0.5, marker='.', s=1,
                               color=colors[i], label=labels[i])
            fps[1].spines['bottom'].set_position('zero')
            fps[1].spines['top'].set_color('none')
            fps[1].spines['right'].set_color('none')
            fps[1].legend()
        else:
            r, vr = phase_space(interior)
            if histogram:
                if cmap is None:
                    cmaps = [plt.cm.magma, plt.cm.inferno,
                             plt.cm.twilight_shifted, plt.cm.twilight_shifted,
                             plt.cm.cividis, plt.cm.twilight_shifted]
                    cmap = cmaps[self.box.particle_type]
                    cmap.set_bad('w', 1)
                fps[1].hist2d(r/self.R_scale, vr, bins=bins,
                              norm=mpl.colors.LogNorm(), cmap=cmap)
            else:
                fps[1].scatter(r/self.R_scale, vr, alpha=0.5, marker='.', s=1,
                               color='C1')
            fps[1].spines['bottom'].set_position('zero')
            fps[1].spines['top'].set_color('none')
            fps[1].spines['right'].set_color('none')
        fps[1].set_xlim(0, outer_radius)
        if vlim is not None:
            fps[1].set_ylim(-vlim, vlim)
        fps[1].set_xlabel(r'$r/R_{200}$')
        fps[1].set_ylabel(r'$v_r$')
        if title is None:
            title = self.box.tag
        fps[0].suptitle(title)
        fps[0].tight_layout()
        if save:
            fps[0].savefig(savefile)

    def get_formation_time(self, frac=0.5, redshift=True, time=False):
        self.get_merger_tree()
        if redshift:
            f = interp1d(self.tree_masses, self.tree_redshifts)
            self.formation_redshift = float(f(frac * self.tree_masses[-1]))
        if time:
            f = interp1d(self.tree_masses, self.tree_times)
            self.formation_time = float(f(frac * self.tree_masses[-1]))

    def get_concentration(self, v=200):
        """
        Calculate the concentration from the density fit parameters.

        Parameters
        ----------
        v : float, optional
            Virial overdensity parameter. The default is 200.

        Returns
        -------
        float
            The concentration.

        """

        return approx_concentration(
            self.density_fit_params['central_density'].value, v)

    def calc_potential(self, r, nbins):
        """
        Calculate the gravitational potential at a specified radius.

        Parameters
        ----------
        r : float
            The radius at which the potential is evaluated.
        nbins : int
            Number of radial bins interior to r and number exterior to r.

        Returns
        -------
        float
            The potential at r.

        """
        radii = magnitude(self.relative_coords)[0]
        bins_outer = np.linspace(r, 10 * self.R_scale, nbins)
        m_enc = sum(self.box.masses[np.argwhere((radii < r) & (radii > 0))])
        int2 = 0
        for i in range(len(bins_outer) - 1):
            dm = sum(self.box.masses[np.argwhere((radii < bins_outer[i+1]) &
                                                 (radii > bins_outer[i]))])
            r_ = (radii[i+1] + radii[i]) / 2
            int2 += dm / r_
        potential = -self.box.gravitational_constant * (m_enc / r + int2)
        self.potential = potential

        return potential

    def calc_overdensity_mass_and_radius(self, overdensity=200):
        radii = magnitude(self.relative_coords)[0]

        def M(r):
            return np.sum(self.box.masses[np.argwhere((radii < r))])

        ri = 0.01 * self.R_200
        avg_density = M(ri) / (4 * np.pi * ri**3 / 3)
        while avg_density > overdensity * self.box.critical_density:
            ri += 0.01 * self.R_200
            avg_density = M(ri) / (4 * np.pi * ri**3 / 3)

        return M(ri), ri

    def calc_velocity_dispersion_mass(self, tracer='subhalos',
                                      overdensity=None, radius=None):

        if overdensity is not None:
            M, aperture = self.calc_overdensity_mass_and_radius(overdensity)
        elif radius is not None:
            aperture = radius
            inds = np.argwhere(magnitude(self.relative_coords)[0]
                               < radius).flatten()
            M = sum(self.box.masses[inds])

        if tracer == 'particles':
            radii = magnitude(self.group_relative_coords)[0]
            particles_interior = np.argwhere(radii < aperture)
            particle_vel_disp = self.calc_velocity_dispersion(
                self.group_relative_vels[particles_interior])[0]
            Mvir = particle_vel_disp**2 * aperture / \
                self.box.gravitational_constant

        if tracer == 'subhalos':
            radii = magnitude(self.group_subhalo_relative_coords)[0]
            subhalos_interior = np.argwhere(radii < aperture).flatten()
            subhalo_vel_disp = self.calc_velocity_dispersion(
                self.group_subhalo_relative_vels[subhalos_interior])[0]
            Mvir = subhalo_vel_disp**2 * aperture / \
                self.box.gravitational_constant

        return Mvir, M

    def calc_center_of_mass(self, radius=None):
        radii = magnitude(self.relative_coords)[0]
        inds = np.argwhere(radii < radius).flatten()
        com = np.sum((self.box.masses[inds] *
                      self.relative_coords[inds].T).T, axis=0) / \
            np.sum(self.box.masses[inds])
        self.center_of_mass = com

        return com

    def group_info(self):
        """
        Print some of the group parameters.

        """

        cm_per_Mpc = 3.085678e+24
        g_per_1e10Msun = 1.989e43
        length_norm = self.box.unit_length/cm_per_Mpc
        mass_norm = self.box.unit_mass/g_per_1e10Msun
        pretty_print([self.group_index,
                      self.number_of_particles,
                      self.R_200*length_norm*1e3,
                      self.M_200 * mass_norm,
                      self.group_mass*mass_norm,
                      tuple(np.round(self.group_position, 3)),
                      tuple(np.round(self.group_velocity, 3))],
                     ['Group index',
                      'Number of particles',
                      'R_200 (kpc)',
                      'M_200 (10^10 Msol)',
                      'Group mass (10^10 Msol)',
                      'Group position',
                      'Group velocity'],
                     'GROUP PARAMETERS')

        return
