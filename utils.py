import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import tabulate


def recenter(pos, boxsize):
    for dim in range(3):
        pos[np.argwhere((pos[:, dim] > boxsize/2)), dim] -= boxsize
        pos[np.argwhere((pos[:, dim] < -boxsize/2)), dim] += boxsize
    return pos


def g(c):
    return 1 / (np.log(1 + c) - c / (1 + c))


def nfw_density(params, r):
    delta_c = params['central_density'].value
    rs = params['scale_radius'].value

    x = r / rs
    density = delta_c / (x * (1 + x)**2)

    return density


def nfw_mass(params, r):
    delta_c = params['central_density'].value
    rs = params['scale_radius'].value

    x = r / rs
    mass = 4 * np.pi * delta_c * rs**3 * (1 / (1 + x) + np.log(1 + x) - 1)

    return mass


def calc_sp_ang_mom(x, v, npart):
    return np.sum(np.cross(x, v), axis=0) / npart


def specific_angular_momentum(nbins, r, rx, v, coords):
    angmom = []
    for b in range(nbins):
        inds_i = np.argwhere((r > rx[b]) & (r < rx[b+1]))
        inds_i = inds_i[:, 0]

        vi = v[inds_i, :]
        xi = coords[inds_i, :]
        angmom.append(calc_sp_ang_mom(xi, vi, len(inds_i)))
    return np.array(angmom)


def calc_delta_c(params, particle_radii, particle_mass, critical_density, Rs,
                 R_200):
    v = params['virial_overdensity'].value

    Rvir = 0.8*R_200
    tol = 1e-2
    res = 1
    dr = Rvir/500
    while res > tol:
        Rvir += dr
        Mvir = particle_mass * len(np.argwhere((particle_radii < Rvir))
                                   .flatten())
        Vol = 4 * np.pi * Rvir**3 / 3
        res = abs(1 - (Mvir / Vol) / (v*critical_density))
        # if res_ > res:
        #     dr *= -1
        # res = res_
    c = Rvir / Rs
    delta_c = (v / 3) * c**3 * g(c)

    print('delta_c', delta_c)

    return delta_c


def approx_concentration(delta_c, v):
    c = np.linspace(1, 100, 1000)
    y = c**3 * g(c)
    f = interp1d(y, c)
    return f(3 * delta_c / v)


def tcirc(r, Vc):
    return 2 * np.pi * r / Vc


def vel_disp_nfw(x, conc, beta):
    def integrand(s, b, c):
        return ((s**(2 * b - 3) * np.log(1 + c * s)) / (1 + c * s)**2) -\
            ((c * s**(2 * b - 2)) / (1 + c * s)**3)
    dispint = []
    for ri in x:
        dispint.append(quad(integrand, ri, np.inf, args=(beta, conc))[0])
    return np.sqrt(g(conc) * (1 + conc * x)**2 * x**(1 - 2 * beta) *
                   np.array(dispint))


def hubble_parameter(z, H0, Omega_m, Omega_Lambda, Omega_k):
    return H0 * np.sqrt(Omega_m * (1 + z)**3 +
                        Omega_k * (1 + z)**2 +
                        Omega_Lambda)


def to_physical_velocity(velocity, coord, z, H0, **Omega):
    scale_factor = 1 / (1 + z)
    return velocity * np.sqrt(scale_factor) + \
        coord * hubble_parameter(z, H0, **Omega)


def pretty_print(quantities, labels, title):

    info_table_labels = np.array(labels, dtype=object)
    info_table_quantities = np.array(quantities)
    info_table = np.vstack((info_table_labels, info_table_quantities)).T

    print("\n\n\t {}\n".format(title))
    print(tabulate.tabulate(info_table))

    return
