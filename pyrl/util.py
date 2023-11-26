import numpy as np
from scipy import stats
import numpy as np
from scipy import stats
import mlflow

def update_mlfow_metrics(avg_reward_per_rollout_per_step: np.ndarray) -> None:
    """
    Updates the MLflow metrics for the current rollout.

    Args:
        avg_reward_per_rollout_per_step (np.ndarray): The average reward per step for the current rollout.
    """
    means, lower_bounds, upper_bounds = stepwise_confidence_interval(avg_reward_per_rollout_per_step)

    for step, (mean, lower, upper) in enumerate(zip(means, lower_bounds, upper_bounds)):
        mlflow.log_metrics({
            'Mean Reward per Step': mean,
            'Mean Reward per Step - 95 Lower Bound': lower,
            'Mean Reward per Step - 95 Upper Bound': upper
        }, step=step+1)
    
    mlflow.log_metric('Overall Mean Reward', means[-1])

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