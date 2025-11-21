import numpy as np
from scipy import stats

def wilson_ci(successes, trials, confidence=0.95):
    """Calculate Wilson confidence interval"""
    if trials == 0:
        return 0, 0, 0
    p_hat = successes / trials
    z = stats.norm.ppf((1 + confidence) / 2)
    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator
    return center - margin, center, center + margin

def add_dkw_bands(ecdf_values, n_samples, alpha=0.05):
    """Add DKW confidence bands to ECDF"""
    epsilon = np.sqrt(np.log(2/alpha) / (2*n_samples))
    lower = np.maximum(ecdf_values - epsilon, 0)
    upper = np.minimum(ecdf_values + epsilon, 1)
    return lower, upper

print("Confidence interval functions added for uncertainty quantification")
