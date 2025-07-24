import numpy as np
import pandas as pd
import glob
import sys 
import os
import matplotlib.pyplot as plt
from plotData import plot_data
from formatData import create_formated_player_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Define file path
path = "playerData/"
team_le = LabelEncoder()
pos_le = LabelEncoder()



# Usage check
if len(sys.argv) != 2:
    print("Usage: python script.py <test_csv_filename>")
    sys.exit(1)

test_file_name = sys.argv[1]

# Load and preprocess data
combined_df = create_formated_player_data("combined_player_data_with_labels.csv",test_file_name)

desired_feats = ["WT","Age_x","GP","TS%","eFG%","ORB%","DRB%","TRB%","AST%","TOV%",
                 "STL%","BLK%","USG%","Total S %","PPR","PPS","ORtg","DRtg","PER",
                 "Team_encoded","Position_encoded"]

x_vector = combined_df[desired_feats].copy()
y_vector = combined_df["label"].copy()

# Clean feature values
x_vector.replace('-', pd.NA, inplace=True)
for col in x_vector.columns:
    x_vector[col] = pd.to_numeric(x_vector[col], errors="coerce")
x_vector = x_vector.fillna(x_vector.mean())



# Train Random Forest Regressor
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=3,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)
model.fit(x_vector, y_vector)

# Format test data (hardcoded for 2022)
test_path = os.path.join(path, test_file_name)
pre_tested_players = pd.read_csv(test_path)
pre_tested_players = pre_tested_players.drop(columns=["Draft Trades", "Age_y", "Class", "Season", "School"])
names_and_picks_pre_tested = pre_tested_players[["Player", "Pick"]].copy()
pre_tested_players["Pick"] = pre_tested_players["Pick"].replace(0, 61)
pre_tested_players["label"] = -pre_tested_players["Pick"]
pre_tested_players["Team_encoded"] = team_le.fit_transform(pre_tested_players["Pre-Draft Team"])
pre_tested_players["Position_encoded"] = pos_le.fit_transform(pre_tested_players["Pos"])
pre_tested_players.replace('-', pd.NA, inplace=True)
pre_tested_player_NAMES_SAVED = pre_tested_players["Player"].copy()
for col in pre_tested_players.columns:
    pre_tested_players[col] = pd.to_numeric(pre_tested_players[col], errors="coerce")
pre_tested_players = pre_tested_players.fillna(pre_tested_players.mean())
pre_tested_players.to_csv("PRE-TESTED.csv", index=False)

# Predict and sort by score
tested_players = pre_tested_players[desired_feats].copy()
predictions = model.predict(tested_players[desired_feats])
tested_players["PREDICTION"] = predictions
tested_players["Player"] = pre_tested_player_NAMES_SAVED
tested_players = tested_players.sort_values(by="PREDICTION", ascending=False)
tested_players["RowIndex"] = range(1, len(tested_players) + 1)
tested_players.to_csv("TESTED.csv", index=False)

# Merge predictions and actuals for analysis
names_and_picks_tested = tested_players[["Player", "RowIndex"]].copy()
merged_names_and_picks = pd.merge(names_and_picks_tested, names_and_picks_pre_tested, on="Player", how="left")
merged_names_and_picks = merged_names_and_picks.rename(columns={"RowIndex": "Predicted Pick", "Pick": "Actual Pick"})
merged_names_and_picks["Actual Pick"] = merged_names_and_picks["Actual Pick"].replace(0, 61)
merged_names_and_picks["Error (pick distance)"] = (merged_names_and_picks["Predicted Pick"] - merged_names_and_picks["Actual Pick"]).abs()

# Show results
print(merged_names_and_picks)
plot_data(merged_names_and_picks)
