from br_scripts.br_utils.grid_search import ErrorFunctionCcr, grid_search, permutations_w_constraints
from br_scripts.calc_ccr_utils.calc_ccrs_from_dihedrals import CcrsDihedralsCalculator
import numpy as np


class BasinReweightingModel:

    def __init__(self,
                 alpha_helix_cm,
                 beta_sheet_cm,
                 ppII_cm,
                 left_helix_cm,
                 grid_weight_step=0.01
                 ):
        self.alpha_helix_phi, self.alpha_helix_psi = alpha_helix_cm
        self.beta_sheet_phi, self.beta_sheet_psi = beta_sheet_cm
        self.left_helix_phi, self.left_helix_psi = left_helix_cm
        self.ppII_phi, self.ppII_psi = ppII_cm
        self.grid_weight_step = grid_weight_step

        self.ccrs_names = None
        self.ccrs_train_values = None
        self.grid_weight = None

    def train(self,
              nmr_freq,
              j_nh_0,
              ccr_names
              ):
        ccrs_calculator = CcrsDihedralsCalculator(j_0=j_nh_0, nmr_freq=nmr_freq)
        alpha_helix_ccr = ccrs_calculator.ccrs_values(phi=self.alpha_helix_phi, psi=self.alpha_helix_psi,
                                                      ccr_names=ccr_names)
        beta_sheet_ccr = ccrs_calculator.ccrs_values(phi=self.beta_sheet_phi, psi=self.beta_sheet_psi,
                                                     ccr_names=ccr_names)
        left_handed_helix_ccr = ccrs_calculator.ccrs_values(phi=self.left_helix_phi, psi=self.left_helix_psi,
                                                            ccr_names=ccr_names)
        ppII_ccr = ccrs_calculator.ccrs_values(phi=self.ppII_phi, psi=self.ppII_psi,
                                               ccr_names=ccr_names)

        self.ccrs_names = ccr_names
        self.ccrs_train_values = np.array([alpha_helix_ccr, beta_sheet_ccr, left_handed_helix_ccr, ppII_ccr])

    def fit(self,
            ccrs_experimental_values,
            ):
        assert len(self.ccrs_names) == len(ccrs_experimental_values)

        self.grid_weight = permutations_w_constraints(n_perm_elements=4,
                                                      sum_total=1,
                                                      min_value=0,
                                                      max_value=1,
                                                      step=self.grid_weight_step
                                                      )
        error_function = ErrorFunctionCcr(ccrs_experimental_values=ccrs_experimental_values,
                                          ccrs_train_values=self.ccrs_train_values
                                          )
        weight, err = grid_search(error_function=error_function,
                                  grid=self.grid_weight
                                  )

        return weight, err
