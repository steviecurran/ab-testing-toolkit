from scipy import stats
import numpy as np

# ---------------------------------------------------------------------
# Statistical calculations
# ---------------------------------------------------------------------

def get_probability(
    confidence_level,
    sidedness,
):
    """Return probability used to calculate the critical value."""

    alpha = 1.0 - confidence_level

    if sidedness == "Two-tailed":
        return 1.0 - alpha / 2.0

    return 1.0 - alpha


def confidence_margin_for_mean(
    n,
    mean,
    standard_deviation,
    confidence_level,
    distribution,
):
    """Calculate the confidence-interval half-width for one sample mean."""

    standard_error = standard_deviation / np.sqrt(n)
    probability = 1.0 - (1.0 - confidence_level) / 2.0

    if distribution == "z":
        critical_value = stats.norm.ppf(probability)
    else:
        critical_value = stats.t.ppf(probability, n - 1)

    margin = critical_value * standard_error
    return margin


def compare_means(
    n1,
    mean1,
    sd1,
    n2,
    mean2,
    sd2,
    confidence_percent,
    sidedness,
    equal_variances,
    z_threshold,
):
    """Compare two independent sample means."""

    if n1 < 2 or n2 < 2:
        raise ValueError("Each sample must contain at least two observations.")

    if sd1 < 0 or sd2 < 0:
        raise ValueError("Standard deviations cannot be negative.")

    confidence_level = confidence_percent / 100.0
    alpha = 1.0 - confidence_level
    probability = get_probability(confidence_level, sidedness)

    difference = mean1 - mean2
    use_z = (n1 > z_threshold and n2 > z_threshold)

    if use_z:
        distribution = "z"
        method = "z-statistics"
        degrees_of_freedom = None
        standard_error = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
        critical_value = stats.norm.ppf(probability)
        test_statistic = difference / standard_error

        if sidedness == "Two-tailed":
            p_value = 2.0 * stats.norm.sf(abs(test_statistic))
        else:
            p_value = stats.norm.sf(abs(test_statistic))
    else:
        distribution = "t"
        method = "t-statistics"

        if equal_variances:
            degrees_of_freedom = n1 + n2 - 2
            pooled_variance = ((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / degrees_of_freedom
            standard_error = np.sqrt(pooled_variance * (1.0 / n1 + 1.0 / n2))
        else:
            variance_1 = sd1**2 / n1
            variance_2 = sd2**2 / n2
            standard_error = np.sqrt(variance_1 + variance_2)
            degrees_of_freedom = (variance_1 + variance_2) ** 2 / (
                variance_1**2 / (n1 - 1) + variance_2**2 / (n2 - 1)
            )

        critical_value = stats.t.ppf(probability, degrees_of_freedom)
        test_statistic = difference / standard_error

        if sidedness == "Two-tailed":
            p_value = 2.0 * stats.t.sf(abs(test_statistic), degrees_of_freedom)
        else:
            p_value = stats.t.sf(abs(test_statistic), degrees_of_freedom)

    difference_margin = critical_value * standard_error
    lower_bound = difference - difference_margin
    upper_bound = difference + difference_margin
    significant = (lower_bound > 0 or upper_bound < 0)

    sample_1_margin = confidence_margin_for_mean(
        n=n1, mean=mean1, standard_deviation=sd1, confidence_level=confidence_level, distribution=distribution
    )
    sample_2_margin = confidence_margin_for_mean(
        n=n2, mean=mean2, standard_deviation=sd2, confidence_level=confidence_level, distribution=distribution
    )

    variance_ratio = np.nan if sd2 == 0 else (sd1**2 / sd2**2)

    return {
        "distribution": distribution,
        "method": method,
        "degrees_of_freedom": degrees_of_freedom,
        "difference": difference,
        "difference_margin": difference_margin,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "critical_value": critical_value,
        "test_statistic": test_statistic,
        "p_value": p_value,
        "significant": significant,
        "sample_1_margin": sample_1_margin,
        "sample_2_margin": sample_2_margin,
        "variance_ratio": variance_ratio,
        "alpha": alpha,
    }

