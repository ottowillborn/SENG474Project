import sys
import os
import numpy as np
import pandas as pd

# Add the 'neural-network' directory to Python's search path
sys.path.append(os.path.join(os.path.dirname(__file__), 'neural-network'))
from neural_network_model import train_and_test_model

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

def main():
    _ , neural_net_pred = train_and_test_model("allUpdatedPlayerData/", "2025", show_plots=False)
    neural_net_pred = enforce_unique_picks(neural_net_pred)
    print(neural_net_pred)
if __name__ == "__main__":
    main()
