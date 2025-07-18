import numpy as np
import pandas as pd
import glob
import sys 
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

path = "playerData/"
team_le = LabelEncoder()
pos_le = LabelEncoder()

def plot_data(merged_names_and_picks):
    # Scatter plot
    print("Mean AVG pick error (for now this is a very unfavorably skewed metric):",merged_names_and_picks["Error (pick distance)"].mean())
    plt.figure(figsize=(10, 8))
    plt.scatter(merged_names_and_picks["Actual Pick"], merged_names_and_picks["Predicted Pick"], alpha=0.8)

    plt.plot([1, 60], [1, 60], linestyle='--', color='gray', label="Perfect Prediction")

    for _, row in merged_names_and_picks.iterrows():
        plt.text(
            row["Actual Pick"] + 0.5,  
            row["Predicted Pick"] + 0.5,  
            row["Player"],
            fontsize=8
        )

    plt.xlabel("Actual Pick")
    plt.ylabel("Predicted Pick")
    plt.title("Predicted vs. Actual NBA Draft Picks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_formated_player_data(filenmae):
    #func: create_formated_player_data
    #args:
    #Docs:
    pattern = os.path.join(path, "college_players_career_stats_*.csv")
    all_files = glob.glob(pattern)

    files = [f for f in all_files if not f.endswith(test_file_name)] #excluding the test file

    dfs = []
    for file in files:
        year = int(file.split("_")[-1].split(".")[0])  
        df = pd.read_csv(file)
        df["Year"] = year  
        dfs.append(df)

    #MEGA FRAME
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop(columns=["Draft Trades", "Age_y", "Class", "Season", "School"])
    # 999 labels as not picked
    combined_df["Pick"] = combined_df["Pick"].replace(0,999)
    combined_df["label"] = -combined_df["Pick"] #make a new column label which is just negative pick 

    combined_df["Team_encoded"] = team_le.fit_transform(combined_df["Pre-Draft Team"]) #encodes pre draft teams into numbers
    combined_df["Position_encoded"] = pos_le.fit_transform(combined_df["Pos"]) #encodes positions into numbers

    combined_df.to_csv(filenmae, index=False) 
    return combined_df