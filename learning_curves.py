import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Generate and plot a learning curve for a given estimator.

    Parameters:
    -----------
    estimator : estimator object
        A scikit-learn estimator object implementing 'fit' and 'predict' methods
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,)
        Target values
    cv : int, cross-validation generator or iterable
        Determines the cross-validation splitting strategy (default=5)
    train_sizes : array-like
        Relative or absolute numbers of training examples to use to generate the learning curve
        (default=np.linspace(0.1, 1.0, 10))

    Returns:
    --------
    plt.Figure
        The matplotlib figure containing the learning curve plot
    """
    # Calculate learning curve values
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        n_jobs=-1,  # Use all available cores
        scoring='neg_mean_squared_error'  # Can be changed based on your metric
    )

    # Calculate mean and standard deviation for training and validation scores
    # Negative because of neg_mean_squared_error
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = -np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Error')

    # Plot training and validation scores
    plt.plot(train_sizes, train_scores_mean,
             label='Training Error', color='blue', marker='o')
    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color='blue')

    plt.plot(train_sizes, validation_scores_mean,
             label='Cross-validation Error', color='green', marker='o')
    plt.fill_between(train_sizes,
                     validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std,
                     alpha=0.1,
                     color='green')

    plt.grid(True)
    plt.legend(loc='best')

    return plt.gcf()


# Example usage:
"""
from sklearn.linear_model import LinearRegression
import numpy as np

# Generate sample data
X = np.random.rand(1000, 10)
y = np.random.rand(1000)

# Create and fit the model
model = LinearRegression()

# Generate and display the learning curve
fig = plot_learning_curve(model, X, y)
plt.show()
"""
