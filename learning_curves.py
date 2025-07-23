import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


def plot_learning_curve(
    train_sizes: Union[List[int], np.ndarray],
    train_errors: Union[List[float], np.ndarray],
    val_errors: Union[List[float], np.ndarray],
) -> plt.Figure:
    """
    Plot a learning curve from pre-computed errors.

    Parameters:
    -----------
    train_sizes : array-like
        List or array of training set sizes
    train_errors : array-like
        List or array of training errors corresponding to each training size
    val_errors : array-like
        List or array of validation errors corresponding to each training size

    Returns:
    --------
    plt.Figure
        The matplotlib figure containing the learning curve plot
    """
    # Convert inputs to numpy arrays
    train_sizes = np.array(train_sizes)
    train_errors = np.array(train_errors)
    val_errors = np.array(val_errors)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Error')

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Error')

    # Plot training errors
    plt.plot(train_sizes, train_errors,
             label='Training Error', color='blue', marker='o')

    # Plot validation errors
    plt.plot(train_sizes, val_errors,
             label='Validation Error', color='green', marker='o')

    plt.grid(True)
    plt.legend(loc='best')

    return plt.gcf()


# Example usage:
"""
import numpy as np

# Example with pre-computed errors
train_sizes = [100, 200, 300, 400, 500]
train_errors = [0.5, 0.3, 0.2, 0.15, 0.1]
val_errors = [0.6, 0.4, 0.35, 0.33, 0.32]

fig = plot_learning_curve(train_sizes, train_errors, val_errors)
plt.show()
"""
