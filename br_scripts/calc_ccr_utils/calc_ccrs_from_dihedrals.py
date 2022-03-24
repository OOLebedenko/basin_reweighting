from br_scripts.calc_ccr_utils.calc_relaxation_rate import calc_csa_axis_interaction_const, \
    calc_dipole_interaction_const, calc_remote_ccr_rates
from br_scripts.calc_ccr_utils.dihedrals_to_cos_theta import dihedrals_to_cos_theta, \
    dihedrals_to_distance
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path


class BaseCcrs:
    from pyxmolpp2 import PdbFile
    path_to_tripeptide_pdb = (Path(__file__).parent / "../../data/ALA_tripeptide.pdb").resolve().as_posix()
    tripeptide = PdbFile(path_to_tripeptide_pdb).frames()[0]

    rCAHA = 1.09e-10
    rNH = 1.02e-10
    gyromagnetic_ratio_dict = {"H1": 267.522e6,
                               "N15": -27.126e6,
                               "C13": 67.2828e6}

    dipole_dict = {"N-H": [gyromagnetic_ratio_dict["N15"], gyromagnetic_ratio_dict["H1"], rNH],
                   "CA-HA": [gyromagnetic_ratio_dict["C13"], gyromagnetic_ratio_dict["H1"], rCAHA]}

    csa_x, csa_y, csa_z = 244e-6, 178e-6, 90e-6

    def __init__(self, ccr_name, phi, psi, nmr_freq):
        self.ccr_name = ccr_name
        self.phi = phi
        self.psi = psi
        self.nmr_freq = nmr_freq

    @property
    def interaction_const(self):
        return

    def ccr_value(self, j_0):
        return


class DipoleDipoleCcrs(BaseCcrs):

    def __init__(self, ccr_name, phi, psi, nmr_freq):
        super().__init__(ccr_name, phi, psi, nmr_freq)
        ccr_to_dipole_dict = {
            "02_CAHA_Np1Hp1": (self.dipole_dict["CA-HA"], self.dipole_dict["N-H"]),
            "04_CAHA_NH": (self.dipole_dict["CA-HA"], self.dipole_dict["N-H"]),
            "08_NH_Np1Hp1": (self.dipole_dict["N-H"], self.dipole_dict["N-H"])
        }

        assert ccr_name in ccr_to_dipole_dict.keys()
        self.ccr_name = ccr_name
        self.dipole_1 = ccr_to_dipole_dict[ccr_name][0]
        self.dipole_2 = ccr_to_dipole_dict[ccr_name][1]
        self.phi = phi
        self.psi = psi

    @property
    def cos_theta(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide, phi_degree=self.phi, psi_degree=self.psi)[0]

    @property
    def interaction_const(self):
        interaction_const = calc_dipole_interaction_const(*self.dipole_1)
        interaction_const *= calc_dipole_interaction_const(*self.dipole_2)
        return interaction_const * (3 * self.cos_theta ** 2 - 1) / 2

    def ccr_value(self, j_0):
        return calc_remote_ccr_rates(interaction_const=self.interaction_const,
                                     j_0=j_0
                                     )


class DipoleCsaCcrs(BaseCcrs):

    def __init__(self, ccr_name, phi, psi, nmr_freq):
        ccr_to_dipole_dict = {
            "00_HAHN_C": None,
            "01_HAHNp1_C": None,
            "03_CAHA_C": (self.dipole_dict["CA-HA"]),
            "05_CAHA_Cm1": (self.dipole_dict["CA-HA"]),
            "06_NH_C": (self.dipole_dict["N-H"])
        }

        super().__init__(ccr_name, phi, psi, nmr_freq)
        assert ccr_name in ccr_to_dipole_dict.keys()
        self.ccr_name = ccr_name
        self.phi = phi
        self.psi = psi
        self.dipole_1 = ccr_to_dipole_dict[ccr_name]
        self.nmr_freq = nmr_freq
        if ccr_name in ["00_HAHN_C", "01_HAHNp1_C"]:
            self.dipole_1 = [self.gyromagnetic_ratio_dict["H1"],
                             self.gyromagnetic_ratio_dict["H1"],
                             dihedrals_to_distance(self.ccr_name, self.tripeptide, self.phi, self.psi) * 1e-10]

    @property
    def cos_theta_x(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide,
                                      phi_degree=self.phi, psi_degree=self.psi)[0]

    @property
    def cos_theta_y(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide,
                                      phi_degree=self.phi, psi_degree=self.psi)[1]

    @property
    def cos_theta_z(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide,
                                      phi_degree=self.phi, psi_degree=self.psi)[2]

    @property
    def _csa_interaction_x(self):
        return calc_csa_axis_interaction_const(self.csa_x,
                                               self.gyromagnetic_ratio_dict["C13"],
                                               self.nmr_freq)

    @property
    def _csa_interaction_y(self):
        return calc_csa_axis_interaction_const(self.csa_y,
                                               self.gyromagnetic_ratio_dict["C13"],
                                               self.nmr_freq)

    @property
    def _csa_interaction_z(self):
        return calc_csa_axis_interaction_const(self.csa_z,
                                               self.gyromagnetic_ratio_dict["C13"],
                                               self.nmr_freq)

    @property
    def interaction_const(self):
        dipole_interaction_const = calc_dipole_interaction_const(*self.dipole_1)
        csa_interaction_const = (3 * self.cos_theta_x ** 2 - 1) / 2 * self._csa_interaction_x + \
                                (3 * self.cos_theta_y ** 2 - 1) / 2 * self._csa_interaction_y + \
                                (3 * self.cos_theta_z ** 2 - 1) / 2 * self._csa_interaction_z
        return csa_interaction_const * dipole_interaction_const

    def ccr_value(self, j_0):
        return calc_remote_ccr_rates(interaction_const=self.interaction_const,
                                     j_0=j_0
                                     )


class CsaCsaCcrs(BaseCcrs):

    def __init__(self, ccr_name, phi, psi, nmr_freq):
        super().__init__(ccr_name, phi, psi, nmr_freq)
        assert ccr_name in ["07_Cm1_C"]
        self.ccr_name = ccr_name
        self.phi = phi
        self.psi = psi
        self.nmr_freq = nmr_freq

    @property
    def _csa_interaction_x(self):
        return calc_csa_axis_interaction_const((self.csa_x - self.csa_z) / 3,
                                               self.gyromagnetic_ratio_dict["C13"],
                                               self.nmr_freq)

    @property
    def _csa_interaction_y(self):
        return calc_csa_axis_interaction_const((self.csa_y - self.csa_z) / 3,
                                               self.gyromagnetic_ratio_dict["C13"],
                                               self.nmr_freq)

    @property
    def cos_theta_xx(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide,
                                      phi_degree=self.phi, psi_degree=self.psi)[0]

    @property
    def cos_theta_xy(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide,
                                      phi_degree=self.phi, psi_degree=self.psi)[1]

    @property
    def cos_theta_yx(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide,
                                      phi_degree=self.phi, psi_degree=self.psi)[2]

    @property
    def cos_theta_yy(self):
        return dihedrals_to_cos_theta(self.ccr_name, self.tripeptide,
                                      phi_degree=self.phi, psi_degree=self.psi)[3]

    @property
    def interaction_const(self):
        csa_interaction_const = (3 * self.cos_theta_xx ** 2 - 1) * self._csa_interaction_x * self._csa_interaction_x + \
                                (3 * self.cos_theta_xy ** 2 - 1) * self._csa_interaction_x * self._csa_interaction_y + \
                                (3 * self.cos_theta_yx ** 2 - 1) * self._csa_interaction_y * self._csa_interaction_x + \
                                (3 * self.cos_theta_yy ** 2 - 1) * self._csa_interaction_y * self._csa_interaction_y

        return csa_interaction_const

    def ccr_value(self, j_0):
        return calc_remote_ccr_rates(interaction_const=self.interaction_const,
                                     j_0=j_0
                                     )


def grid(*arrays):
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points


class CcrsDihedralsCalculator:
    ccrs_interaction_type = {
        "00_HAHN_C": DipoleCsaCcrs,
        "01_HAHNp1_C": DipoleCsaCcrs,
        "02_CAHA_Np1Hp1": DipoleDipoleCcrs,
        "03_CAHA_C": DipoleCsaCcrs,
        "04_CAHA_NH": DipoleDipoleCcrs,
        "05_CAHA_Cm1": DipoleCsaCcrs,
        "06_NH_C": DipoleCsaCcrs,
        "07_Cm1_C": CsaCsaCcrs,
        "08_NH_Np1Hp1": DipoleDipoleCcrs
    }

    def __init__(self, j_0, nmr_freq):
        self.j_0 = j_0
        self.nmr_freq = nmr_freq

    def ccrs_values(self, phi, psi, ccr_names):
        ccr_values = []
        for ccr_name in ccr_names:
            assert ccr_name in self.ccrs_interaction_type.keys(), "Use ccr_names in 00_HAHN_C, 01_HAHNp1_C, 02_CAHA_Np1Hp1, " \
                                                                  "03_CAHA_C, 04_CAHA_NH, 05_CAHA_Cm1," \
                                                                  "06_NH_C, 07_Cm1_C, 08_NH_Np1Hp1"
            ccrs_calculator = self.ccrs_interaction_type[ccr_name](ccr_name, phi, psi, self.nmr_freq)
            ccr_values.append(ccrs_calculator.ccr_value(self.j_0))
        return ccr_values

    def ccrs_table(self, phi_array, psi_array, ccr_names=None):
        if isinstance(ccr_names, str):
            ccr_names = [ccr_names]
        ccr_names = ccr_names if ccr_names else self.ccrs_interaction_type.keys()
        ccrs_table = np.zeros((phi_array.size, len(ccr_names)))
        for ind, (phi, psi) in (enumerate(zip(tqdm(phi_array, desc="calc ccrs"), psi_array))):
            ccrs_table[ind, :] = self.ccrs_values(phi, psi, ccr_names)
        return pd.DataFrame(ccrs_table, columns=ccr_names)

    def ccrs_ramachandran(self, step=5, phi_min=-180, phi_max=180, psi_min=-180, psi_max=180, ccr_names=None):
        ccr_names = ccr_names if ccr_names else self.ccrs_interaction_type.keys()
        phi_ramachandran, psi_ramachandran = np.arange(phi_min, phi_max, step), np.arange(psi_min, psi_max, step)
        ccrs_table = self.ccrs_table(*grid(phi_ramachandran, psi_ramachandran).T, ccr_names)
        ccrs_table["phi"], ccrs_table["psi"] = grid(phi_ramachandran, psi_ramachandran).T
        return ccrs_table.reindex(columns=["phi", "psi", *ccr_names])
