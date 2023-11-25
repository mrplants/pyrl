import numpy as np
from scipy import stats
import numpy as np
from scipy import stats

def stepwise_confidence_interval(data, confidence=0.95):
    """
    Calculates the confidence intervals for each column of a 2D NumPy array.

    Args:
        data (np.ndarray): 2D array of data points.
        confidence (float): The confidence level for the interval.

    Returns:
        means (np.ndarray): The mean of each column.
        lower_bounds (np.ndarray): The lower bound of the confidence interval for each column.
        upper_bounds (np.ndarray): The upper bound of the confidence interval for each column.
    """
    # Ensure data is a NumPy array
    data = np.array(data)

    # Calculate mean and standard error of the mean for each column
    means = np.mean(data, axis=0)
    sems = stats.sem(data, axis=0)

    # Degrees of freedom
    dof = data.shape[0] - 1

    # Critical value for the t-distribution
    critical_value = stats.t.ppf((1 + confidence) / 2., dof)

    # Margin of error
    margins_of_error = critical_value * sems

    # Confidence intervals
    lower_bounds = means - margins_of_error
    upper_bounds = means + margins_of_error

    # Combine means, lower bounds, and upper bounds into a single array
    return means, lower_bounds, upper_bounds