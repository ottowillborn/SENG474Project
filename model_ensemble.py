import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Add the 'neural-network' directory to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural-network'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'normal-equation'))
from neural_network_model import train_and_test_model
from normal_equation import train_and_test_normal_eq

# Convert floating point picks to unique draft numbers
def enforce_unique_picks(df: pd.DataFrame) -> pd.DataFrame:
    predicted_picks = df["Predicted Pick"].values
    sorted_indices = np.argsort(predicted_picks)
    unique_picks = np.zeros_like(predicted_picks, dtype=int)
    unique_picks[sorted_indices] = np.arange(1, len(predicted_picks) + 1)
    df = df.copy()
    df["Predicted Pick"] = unique_picks
    df["Pick Error"] = np.abs(df["Actual Pick"] - df["Predicted Pick"])
    return df

def ensemble_predictions(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    # Merge on player name
    merged = pd.merge(df1[["Player", "Actual Pick", "Predicted Pick"]],
                      df2[["Player", "Predicted Pick"]],
                      on="Player",
                      suffixes=("_model1", "_model2"))

    # Average the predictions
    merged["Ensembled Prediction"] = (merged["Predicted Pick_model1"] + merged["Predicted Pick_model2"]) / 2

    # Assign unique draft positions
    sorted_indices = np.argsort(merged["Ensembled Prediction"].values)
    unique_picks = np.zeros_like(merged["Ensembled Prediction"].values, dtype=int)
    unique_picks[sorted_indices] = np.arange(1, len(merged) + 1)

    merged["Predicted Pick"] = unique_picks
    merged["Pick Error"] = np.abs(merged["Actual Pick"] - merged["Predicted Pick"])

    # Final result
    results_df = merged[["Player", "Actual Pick", "Predicted Pick", "Pick Error"]]
    return results_df

def plot_predictions(results_df: pd.DataFrame, title: str = "Ensemble Model: Predicted vs. Actual Picks"):
    plt.figure(figsize=(10, 8))
    
    # Scatter plot of actual vs predicted
    plt.scatter(results_df["Actual Pick"], results_df["Predicted Pick"], alpha=0.8, label="Players")

    # Best-fit line
    m, b = np.polyfit(results_df["Actual Pick"], results_df["Predicted Pick"], 1)
    plt.plot(results_df["Actual Pick"], m * results_df["Actual Pick"] + b, color='red', linestyle='--', label=f"Best Fit: y = {m:.2f}x + {b:.2f}")

    # Diagonal for perfect prediction
    plt.plot([1, 61], [1, 61], linestyle=':', color='gray', label="Perfect Prediction")

    plt.xlabel("Actual Pick")
    plt.ylabel("Predicted Pick")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    _ , neural_net_pred = train_and_test_model("allUpdatedPlayerData/", "2025", show_plots=False)
    neural_net_pred = enforce_unique_picks(neural_net_pred)
    norm_eq_pred = train_and_test_normal_eq("allUpdatedPlayerData/", "2025")


    print(neural_net_pred)
    print(norm_eq_pred)

    # Ensemble the two predictions
    ensembled_df = ensemble_predictions(neural_net_pred, norm_eq_pred)

    print("Ensembled Prediction Results:")
    print(ensembled_df)

    # Compute average pick error
    avg_error = ensembled_df["Pick Error"].mean()
    print(f"\nEnsembled Average Pick Error: {avg_error:.2f}")

    # Plot the predictions
    plot_predictions(ensembled_df)
if __name__ == "__main__":
    main()
