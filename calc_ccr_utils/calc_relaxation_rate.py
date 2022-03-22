import numpy as np


def calc_j_0(amplitudes,
             taus
             ):
    return sum(
        (2 / 5 * a * tau) for a, tau in zip(amplitudes, taus))


def calc_dipole_interaction_const(gyromagnetic_ratio_atom_1,
                                  gyromagnetic_ratio_atom_2,
                                  bond_length):
    """
    calculation dipole-interaction constant
    
    :param gyromagnetic_ratio_atom_1: 
    :param gyromagnetic_ratio_atom_2: 
    :param bond_length: bond legth between atom_1 and atom_2
    :return: dipole-interaction constant
    """
    h = 6.626069e-34
    mu_bohr = 4 * np.pi * 1e-7
    dipole_interaction_const = (mu_bohr / 4 / np.pi) * (h / 2 / np.pi)
    return dipole_interaction_const * gyromagnetic_ratio_atom_1 * gyromagnetic_ratio_atom_2 * bond_length ** (-3)


def calc_csa_axis_interaction_const(gyromagnetic_ratio_atom,
                                    csa_axis_atom,
                                    nmr_freq):
    """
    calculation csa-axis-interaction constant
    
    :param gyromagnetic_ratio_atom: 
    :param csa_axis_atom: csa_x, csa_y or csa_z value
    :param nmr_freq: frequency in nmr experiment
    :return: csa-axis-interaction constant
    """
    gammaH1 = 267.522e6
    B0 = nmr_freq / (gammaH1 / 2 / np.pi)
    csa_interaction_const = 2 * B0 / 3
    return csa_interaction_const * csa_axis_atom * gyromagnetic_ratio_atom


def calc_remote_ccr_rates(interaction_const, j_0):
    """
    calculation remote ccr relaxation rate = cons * J(0)
    
    :param interaction_const: interaction constant: dipole, csd and etc
    :param amplitude: amplitudes in exponential fit of correlation function
    :param taus_s: taus in exponential fit of correlation function
    :return: remote ccr relaxation rate
    """

    return interaction_const * j_0
