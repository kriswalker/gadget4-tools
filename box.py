import glob
import numpy as np
import h5py
import time
from utils import recenter, approx_concentration, pretty_print
import matplotlib.pyplot as plt
from sphviewer.tools import QuickView
from lmfit import Minimizer


class Box():

    def __init__(self, path, snapshot, snapshot_prefix='snapshot',
                 verbose=True):
        """
        Box object containing parameters and methods for the simulation box as
        a whole.

        Parameters
        ----------
        path : str
            Directory containing the simulation output.
        snapshot : int
            Number of the desired snapshot.
        snapshot_prefix : str, optional
            The prefix of the snapshot files. The default is 'snapshot'.
        verbose : bool, optional
            Print all the things. The default is True.

        """

        self.path = path
        self.snapshot = snapshot

        file = path + '{}_{:03d}.hdf5'
        if np.sort(glob.glob(file.format(snapshot_prefix,
                                         snapshot))).size != 0:
            self.snap = self.read_snap(file.format(snapshot_prefix, snapshot),
                                       verbose)
        if np.sort(glob.glob(file.format('fof_subhalo_tab',
                                         snapshot))).size != 0:
            self.group, self.subhalo = self.read_groups(file.format(
                'fof_subhalo_tab', snapshot), verbose)

    def read_snap(self, filename, verbose):
        """
        Read in snapshot contents.

        Parameters
        ----------
        filename : str
            Snapshot file.
        verbose : bool
            Print all the things.

        Returns
        -------
        params : dict
            Snapshot data and parameters.

        """

        if verbose:
            start = time.time()
            print("LOADING {0}...".format(filename))

        snap = h5py.File(filename, 'r')
        params = {}

        self.coords = snap['PartType1']['Coordinates'][()]
        self.vels = snap['PartType1']['Velocities'][()]
        self.ids = snap['PartType1']['ParticleIDs'][()]

        self.particle_mass = list(snap['Header'].attrs['MassTable'])[1]
        self.redshift = snap['Header'].attrs['Redshift']
        self.box_size = snap['Parameters'].attrs['BoxSize']
        self.nsample = snap['Parameters'].attrs['NSample']
        self.Omega0 = snap['Parameters'].attrs['Omega0']
        self.OmegaBaryon = snap['Parameters'].attrs['OmegaBaryon']
        self.unit_length = snap['Parameters'].attrs['UnitLength_in_cm']
        self.unit_mass = snap['Parameters'].attrs['UnitMass_in_g']

        cm_per_Mpc = 3.085678e+24
        g_per_1e10Msun = 1.989e43
        self.length_norm = self.unit_length / cm_per_Mpc
        self.mass_norm = self.unit_mass / g_per_1e10Msun

        self.OmegaDM = self.Omega0 - self.OmegaBaryon
        self.mean_interparticle_spacing = self.box_size / self.nsample
        self.convergence_radius = 0.77 * \
            (3 * self.OmegaDM / (800 * np.pi))**(1/3) \
            * self.mean_interparticle_spacing / (1 + self.redshift)

        self.gravitational_constant = 43.02

        snap.close()
        if verbose:
            end = time.time()
            print("...LOADED in {0} seconds\n".format(round(end-start, 2)))
            self.box_info()

        return params

    def read_groups(self, filename, verbose):
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

        subh = h5py.File(filename, 'r')

        group = {}
        group['R200'] = subh['Group']['Group_R_Crit200'][()]
        group['R500'] = subh['Group']['Group_R_Crit500'][()]
        group['M200'] = subh['Group']['Group_M_Crit200'][()]
        group['M500'] = subh['Group']['Group_M_Crit500'][()]
        group['Mass'] = subh['Group']['GroupMass'][()]
        group['Pos'] = subh['Group']['GroupPos'][()]
        group['Vel'] = subh['Group']['GroupVel'][()]
        group['Len'] = subh['Group']['GroupLen'][()]
        group['FirstSub'] = subh['Group']['GroupFirstSub'][()]

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

        subhalo = {}
        subhalo['Mass'] = subh['Subhalo']['SubhaloMass'][()]
        subhalo['CM'] = subh['Subhalo']['SubhaloCM'][()]
        subhalo['Pos'] = subh['Subhalo']['SubhaloPos'][()]
        subhalo['Vel'] = subh['Subhalo']['SubhaloVel'][()]
        subhalo['HalfmassRad'] = subh['Subhalo']['SubhaloHalfmassRad'][()]
        subhalo['Len'] = subh['Subhalo']['SubhaloLen'][()]
        subhalo['IDMostbound'] = subh['Subhalo']['SubhaloIDMostbound'][()]
        subhalo['SubhaloGroupNr'] = subh['Subhalo']['SubhaloGroupNr'][()]
        subhalo['SubhaloRankInGr'] = subh['Subhalo']['SubhaloRankInGr'][()]

        subh.close()

        self.number_of_groups = len(group['Pos'])
        self.number_of_subhalos = len(subhalo['Pos'])

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

        f_mass = plt.subplots(figsize=(7, 5))
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

    def plot_box(self, projection='xy', title=None, save=False, savefile=None):
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

        width = self.box_size/2
        coords = self.coords - np.array([width]*3)
        order = []
        for p in projection:
            order.append(
                0 if p == 'x' else 1 if p == 'y' else 2 if p == 'z' else 3)
        order.append(list(set([0, 1, 2]) - set(order))[0])
        coords[:, [0, 1, 2]] = coords[:, order]
        qv_parallel = QuickView(coords, r='infinity', plot=False,
                                x=0, y=0, z=0, extent=[-width, width,
                                                       -width, width])
        f_box = plt.subplots(figsize=(8, 8))
        f_box[1].imshow(qv_parallel.get_image(),
                        extent=qv_parallel.get_extent(),
                        cmap='inferno', origin='lower')
        f_box[1].set_xlabel(r'{} ($h^{{{}}}$ Mpc)'.format(projection[0], '-1'))
        f_box[1].set_ylabel(r'{} ($h^{{{}}}$ Mpc)'.format(projection[1], '-1'))
        metadata = [('No. of particles = {0}', '{}^3'.format(self.nsample)),
                    ('Box size = {0} Mpc', self.box_size * self.length_norm),
                    ('z = {0}', round(self.redshift, 3))]
        for mi, m in enumerate(metadata):
            f_box[0].text(0.11, 0.03*(mi+3), m[0].format(m[1]), color='white')
        f_box[0].suptitle(title)
        f_box[0].tight_layout()
        if save:
            f_box[0].savefig(savefile, dpi=500)
        else:
            f_box[0].show()

    def box_info(self):
        """
        Print some of the simulation parameters.

        """

        pretty_print([round(self.redshift, 3),
                      self.box_size * self.length_norm,
                      '{}^3'.format(self.nsample),
                      round(self.particle_mass * self.mass_norm * 1e4, 5),
                      self.Omega0,
                      self.OmegaBaryon,
                      1-self.Omega0],
                     ['Redshift',
                      'Box size (Mpc)',
                      'Number of particles',
                      'Particle mass (10^6 Msol)',
                      'Omega_0',
                      'Omega_Baryon',
                      'Omega_Lambda'],
                     'SIMULATION PARAMETERS')

        return


class Halo():

    def __init__(self, box=None, path=None, snapshot=None,
                 snapshot_prefix='snapshot', group_index=None,
                 subhalo_index=None, verbose=True):
        """
        Halo object containing parameters and methods pertaining to individual
        groups or subhalos.

        Parameters
        ----------
        box : Box object, optional
            Box object to load. The default is None.
        path : str
            Directory containing the simulation output.
        snapshot : int
            Number of the desired snapshot.
        snapshot_prefix : str, optional
            The prefix of the snapshot files. The default is 'snapshot'.
        group_index : int, optional
            Index of the desired group. The default is None.
        subhalo_index : int, optional
            Index of the desired subhalo. If group_index is specified, only the
            subhalos within that group are considered. The default is None.
        verbose : bool, optional
            Print all the things. The default is True.

        """

        if box is None:
            self.path = path
            self.snapshot = snapshot
            self.box = Box(path, snapshot, verbose=verbose)
        else:
            self.box = box
            self.path = box.path
            self.snapshot = box.snapshot
        if subhalo_index is not None:
            self.subhalo_index = subhalo_index
            self.get_subhalo_params(verbose)
        else:
            self.subhalo_index = None
        if group_index is not None:
            self.group_index = group_index
            self.get_group_params(verbose)
        self.get_halo_particle_positions()
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
        # if self.subhalo_index is None:
        #     self.subhalo_index = self.group_first_subhalo
        #     self.get_subhalo_params(verbose=False)

        self.R_200 = self.box.group['R200'][self.group_index]
        self.R_500 = self.box.group['R500'][self.group_index]
        self.M_200 = self.box.group['M200'][self.group_index]
        self.M_500 = self.box.group['M500'][self.group_index]
        self.group_mass = self.box.group['Mass'][self.group_index]
        self.group_position = self.box.group['Pos'][self.group_index]
        self.group_velocity = self.box.group['Vel'][self.group_index]
        self.group_len = self.box.group['Len'][self.group_index]

        coords_rel = self.box.coords - self.group_position
        self.centered_coords = recenter(coords_rel, self.box.box_size)
        self.relative_vels = self.box.vels - self.group_velocity

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
        self.subhalo_len = self.box.subhalo['Len'][self.subhalo_index]
        self.subhalo_index_most_bound = np.argwhere(
            (self.box.ids == self.box.subhalo['IDMostbound'][
                self.subhalo_index])).flatten()[0]
        self.subhalo_rank = self.box.subhalo['SubhaloRankInGr'][
            self.subhalo_index]

        coords_rel = self.box.coords - self.subhalo_position
        self.centered_coords = recenter(coords_rel, self.box.box_size)
        self.relative_vels = self.box.vels - self.subhalo_velocity

        self.R_scale = self.halfmass_radius
        self.M_scale = self.subhalo_mass / 2
        self.V_scale = np.sqrt(self.box.gravitational_constant * self.M_scale /
                               self.R_scale)

        self.R_subscript = '1/2'

        return

    def get_halo_particle_positions(self):
        """
        returns positions of all the particles in the group

        """

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
            inds_sub = np.arange(ind + offset_sub_low, ind + offset_sub_high)
            self.inds_sub = inds_sub
            # IDs_sub = self.box.ids[inds_sub]
            subhalo_coords = self.box.coords[inds_sub] - self.subhalo_position
            self.subhalo_coords = recenter(subhalo_coords, boxsize=25)
            self.number_of_particles = len(subhalo_coords)

        elif self.group_index is not None:
            offset_grp = self.group_len
            inds_grp = np.arange(ind, ind + offset_grp)
            # IDs_grp = self.box.ids[inds_grp]
            group_coords = self.box.coords[inds_grp] - self.group_position
            self.group_coords = recenter(group_coords, boxsize=25)
            self.number_of_particles = len(group_coords)

        # inds_rem = np.arange(ind + offset_sub, ind + offset_grp)
        # remain_coords = self.box.coords[inds_rem] - self.group_position
        # self.remain_coords = recenter(remain_coords, boxsize=25)

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
        if descend:
            key = 'FirstDescendant'
        else:
            key = 'MainProgenitor'
        while tree_data[key][n] > -1:
            inds.append(n)
            mass.append(tree_data['SubhaloMass'][n])
            nrs.append(tree_data['SubhaloNr'][n])
            grpnrs.append(tree_data['GroupNr'][n])
            snap_nums.append(tree_data['SnapNum'][n])
            n = tree_data['StartOffset'][tree_data['ID'][n]] + \
                tree_data[key][n]
        if descend:
            self.merger_tree = np.array(inds)
            self.tree_subhalo_indices = np.array(nrs)
            self.tree_group_indices = np.array(grpnrs)
            self.tree_masses = np.array(mass)
            self.tree_snapshot_numbers = np.array(snap_nums)
        else:
            self.merger_tree = np.flip(np.array(inds))
            self.tree_subhalo_indices = np.flip(np.array(nrs))
            self.tree_group_indices = np.flip(np.array(grpnrs))
            self.tree_masses = np.flip(np.array(mass))
            self.tree_snapshot_numbers = np.flip(np.array(snap_nums))
        self.tree_redshifts = tree_file['TreeTimes/Redshift'][()][
            -len(self.merger_tree):]
        self.tree_times = tree_file['TreeTimes/Time'][()][
            -len(self.merger_tree):]

    def histogram_halo(self, nbins, cutoff_radii):
        """
        Bin the particles of the group or subhalo by log radius.

        Parameters
        ----------
        nbins : int
            Number of bins.
        cutoff_radii : tuple of floats
            Inner and outer radii within which to perform the binning.

        """

        r = np.sqrt(self.centered_coords[:, 0]**2 +
                    self.centered_coords[:, 1]**2 +
                    self.centered_coords[:, 2]**2)
        # r > 0 so log() doesn't diverge
        inds = np.argwhere((r < cutoff_radii[1]*self.R_scale) & (r > 0))
        self.particle_indices = inds.flatten()
        self.r_all = r
        self.r = r[inds].flatten()

        self.coords_inside = self.centered_coords[inds].squeeze(axis=1)
        self.vels_inside = self.relative_vels[inds].squeeze(axis=1)

        logr = np.log10(r)
        logrx = np.linspace(np.log10(cutoff_radii[0]*self.R_scale),
                            np.log10(cutoff_radii[1]*self.R_scale), nbins+1)
        logrhist = []
        for k in range(nbins):
            npart = len(np.argwhere((logr > logrx[k]) & (logr < logrx[k+1])))
            logrhist.append(npart)
        self.logrhist = np.array(logrhist)
        logrx_ = logrx + (logrx[1] - logrx[0])/2
        self.redge = 10**logrx
        self.rcenter = 10**logrx_

        return

    def calc_density_profile(self, nbins, cutoff_radii, model=None):
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

        """

        self.histogram_halo(nbins, cutoff_radii)

        if not hasattr(self, 'density_profile'):
            self.density_profile = self.box.particle_mass * self.logrhist / \
                ((4 * np.pi / 3) * (self.redge[1:]**3 - self.redge[:-1]**3))

        if model is not None:
            rad = self.rcenter[:-1]
            inner = np.argmin((np.abs(rad - self.box.convergence_radius)))
            outer = np.argmin((np.abs(rad - 0.8*self.R_scale)))
            if outer - inner < 3:
                raise Exception('Subhalo radius too close to convergence',
                                'radius! Cannot perform reliable fit.')
            fit_radius = rad[inner:outer]
            fit_density = self.density_profile[inner:outer]
            self.density_fit_params = self.fit_model(
                model[0], model[1], fit_density/self.box.critical_density,
                {'r': fit_radius})
            self.density_profile_model = self.box.critical_density * \
                model[0](self.density_fit_params, rad)

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

        self.histogram_halo(nbins, cutoff_radii)

        self.mass_profile = self.box.particle_mass * np.cumsum(self.logrhist)

        if model is not None:
            rad = self.redge[1:]
            # if not hasattr(self, 'density_fit_params'):
            inner = np.argmin((np.abs(rad - self.box.convergence_radius)))
            outer = np.argmin((np.abs(rad - 0.8*self.R_scale)))
            if outer - inner < 3:
                raise Exception('Subhalo radius too close to convergence',
                                'radius! Cannot perform reliable fit.')
            fit_radius = rad[inner:outer]
            fit_mass = self.mass_profile[inner:outer]
            self.density_fit_params = self.fit_model(
                model[0], model[1], fit_mass/self.box.critical_density,
                {'r': fit_radius})
            self.mass_profile_model = self.box.critical_density * model[0](
                self.density_fit_params, rad)

        return

    def calc_vel_disp(self, v, x):
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
        sqmag = np.sum(x**2, axis=1)
        rhat = x / np.sqrt(sqmag.reshape(len(sqmag), 1))
        vrad = np.sum(v*rhat, axis=1)

        vm = np.mean(v, axis=0)
        vms = np.mean(v**2, axis=0)
        disp = np.sqrt(np.sum(vms - vm**2))

        vmrad = np.mean(vrad)
        vmsrad = np.mean(vrad**2)
        disprad = np.sqrt(vmsrad - vmrad**2)

        return disp, disprad

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
                                 (self.r < self.redge[b+1]))
            inds_i = inds_i[:, 0]

            vi = self.vels_inside[inds_i, :]
            xi = self.coords_inside[inds_i, :]

            disp_i, disprad_i = self.calc_vel_disp(vi, xi)
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
                raise Exception('No concentration provided for velocity',
                                'dispersion model. Either specify with the',
                                'concentration argument or fit density',
                                'profile first.')
            elif concentration is None:
                concentration = self.get_concentration(v=200)
            if beta is None:
                beta = np.mean(self.beta_profile[inner:outer])
            # print('\nbeta = {}\n'.format(b))
            self.radial_dispersion_profile_model = model[0](
                rad/self.R_scale, concentration, beta)

        return

    def calc_angular_momentum(self, v, x, specific=False):
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
            return self.box.particle_mass * np.sum(np.cross(x, v), axis=0)

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
                             ylim=None, title=None, save=False, savefile=None):
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

        f_dens = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        rad = self.rcenter[:-1] / self.R_scale
        f_dens[1][0].plot(rad, self.density_profile/self.box.critical_density,
                          alpha=0.8, color=style[0], linestyle=style[1],
                          label='data')
        f_dens[1][1].plot(
            rad, (self.density_profile/self.box.critical_density) *
            rad**2, alpha=0.8, color=style[0], linestyle=style[1])
        if plot_model:
            f_dens[1][0].plot(
                rad, self.density_profile_model/self.box.critical_density,
                alpha=0.8, color='k', linestyle='--', label='NFW')
            f_dens[1][1].plot(
                rad, (self.density_profile_model/self.box.critical_density) *
                rad**2, alpha=0.8, color='k', linestyle='--')
        f_dens[1][0].axvline(self.box.convergence_radius/self.R_scale,
                             color='r', linestyle='--',
                             label='convergence radius')
        f_dens[1][1].axvline(self.box.convergence_radius/self.R_scale,
                             color='r', linestyle='--')
        f_dens[1][0].set_xscale('log')
        f_dens[1][0].set_yscale('log')
        f_dens[1][1].set_xscale('log')
        f_dens[1][1].set_yscale('log')
        if xlim is None:
            xlim = cutoff_radii
        if ylim is None:
            dens = self.density_profile/self.box.critical_density
            ylim = [dens[-1], max(dens)]
        f_dens[1][0].set_xlim(xlim[0], xlim[1])
        f_dens[1][0].set_ylim(ylim[0], ylim[1])
        f_dens[1][1].set_xlim(xlim[0], xlim[1])
        f_dens[1][0].set_xlabel(r'$r/R_{{{}}}$'.format(self.R_subscript))
        f_dens[1][0].set_ylabel(r'$\rho/\rho_\mathrm{crit}$')
        f_dens[1][1].set_xlabel(r'$r/R_{{{}}}$'.format(self.R_subscript))
        f_dens[1][1].set_ylabel(r'$(\rho/\rho_\mathrm{{{0}}})(r/R_{{{1}}})^2$'
                                .format('crit', self.R_subscript))
        f_dens[1][0].legend(loc='center left', bbox_to_anchor=(1, 0))
        f_dens[0].suptitle(title)
        f_dens[0].tight_layout()
        if save:
            f_dens[0].savefig(savefile, dpi=300)
            plt.close(f_dens[0])

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
                                model=None, concentration=None,
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
                                         model=model)

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

    def plot_subhalo(self, projection='xy', extent=[-2, 2, -2, 2], title=None,
                     save=False, savefile=None):
        """
        2D plot of the subhalo particle distribution.

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
        coords = np.copy(self.centered_coords)
        order = []
        for p in projection:
            order.append(
                0 if p == 'x' else 1 if p == 'y' else 2 if p == 'z' else 3)
        order.append(list(set([0, 1, 2]) - set(order))[0])
        coords[:, [0, 1, 2]] = coords[:, order]
        qv_parallel = QuickView(coords, r='infinity', plot=False,
                                x=0, y=0, z=0, extent=list(extent))
        f_subh = plt.subplots(figsize=(8, 8))
        f_subh[1].imshow(qv_parallel.get_image(),
                         extent=qv_parallel.get_extent(), cmap='inferno',
                         origin='lower')
        x = np.linspace(-self.R_scale, self.R_scale, 100)
        f_subh[1].plot(x, np.sqrt(self.R_scale**2 - x**2), color='r',
                       linestyle='--', linewidth=1,
                       label=r'$R_{{{}}}$'.format(self.R_subscript))
        f_subh[1].plot(x, -np.sqrt(self.R_scale**2 - x**2), color='r',
                       linestyle='--', linewidth=1)
        f_subh[1].set_xlabel(r'{} ($h^{{{}}}$ Mpc)'.format(
            projection[0], '-1'))
        f_subh[1].set_ylabel(r'{} ($h^{{{}}}$ Mpc)'.format(
            projection[1], '-1'))
        metadata = [('Total no. of particles = {0}', '{}^3'
                     .format(self.box.nsample)),
                    ('Box size = {0} Mpc',
                     self.box.box_size * self.box.length_norm),
                    ('z = {0}', round(self.box.redshift, 3)),
                    ('Subhalo {0} ({1})'.format(self.subhalo_rank,
                                                self.subhalo_index), ''),
                    ('Group {0}', self.group_index)]
        for mi, m in enumerate(metadata):
            f_subh[0].text(0.12, 0.03*(mi+3.3), m[0].format(m[1]),
                           color='white')
        f_subh[0].suptitle(title)
        f_subh[0].tight_layout()
        f_subh[0].legend()
        if save:
            f_subh[0].savefig(savefile, dpi=500)
            plt.close(f_subh[0])

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

    def fit_model(self, model, params, quantity, args, log=True):
        """
        Fit a specified model to some data.

        Parameters
        ----------
        model : function
            The model.
        params : dict
            The model parameters.
        quantity : 1D array
            The data to fit the model to.
        args : dict
            Arguments of the model function.
        log : bool, optional
            Fit in log space. The default is True.

        Returns
        -------
        dict
            The fit parameter values.

        """

        def residual(p, q, args, log=True):
            if log:
                return np.log10(q) - np.log10(model(p, **args))
            else:
                return q - model(p, **args)

        func = Minimizer(residual, params, fcn_args=(quantity, args, log))

        results = func.minimize()
        fit_params = results.params

        fit_param_keys = list(fit_params.keys())
        if self.verbose:
            pretty_print([fit_params[key].value for key in fit_param_keys],
                         fit_param_keys, 'Fit results')

        return fit_params

    def group_info(self):
        """
        Print some of the group parameters.

        """

        cm_per_Mpc = 3.085678e+24
        g_per_1e10Msun = 1.989e43
        length_norm = self.box.unit_length/cm_per_Mpc
        mass_norm = self.box.unit_mass/g_per_1e10Msun
        pretty_print([self.group_index,
                      len(self.group_coords),
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
