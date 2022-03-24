import numpy as np


def permutations_w_constraints(n_perm_elements, sum_total, min_value, max_value, step):
    if n_perm_elements == 1:
        if (sum_total <= max_value) & (sum_total >= min_value):
            yield (sum_total,)
    else:
        for value in np.arange(min_value, max_value + 1, step):
            for permutation in permutations_w_constraints(n_perm_elements - 1, sum_total - value, min_value, max_value,
                                                          step):
                yield (value,) + permutation


def grid_search(error_function, grid):
    best_err = np.inf
    best = None
    for point in grid:
        err = error_function(point)
        if err < best_err:
            best = point
            best_err = err
    return best, best_err


class ErrorFunctionCcr:

    def __init__(self, ccrs_experimental_values, ccrs_train_values):
        self.ccrs_experimental_values = ccrs_experimental_values
        self.ccrs_train_values = ccrs_train_values

    def __call__(self, weights):
        return self.calc_residual_sum_of_squares(weights)

    def calc_residual_sum_of_squares(self, weights):
        delta = []
        for ind, ccr_value in enumerate(self.ccrs_experimental_values):
            weighted_ss_ccr = np.array(weights) * self.ccrs_train_values[:, ind]
            current_delta = np.abs(ccr_value - sum(weighted_ss_ccr))
            delta.append(current_delta)
        return np.array(delta).sum()
