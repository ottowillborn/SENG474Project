import numpy as np
import pandas as pd
import glob
import sys 
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from plotData import plot_data
from formatData import create_formated_player_data
#2002 is the earliest year with NCAA stats available
#2006 is when high school players were no longer eligible for the draft
# Player: Full name of the basketball player.
# Team: Team the player was drafted by or currently plays for.
# Pos: Player's position (e.g., PG, SG, SF, PF, C).

# HT: Height of the player ft-in.
# WT: Weight of the player lb.

# Pre-Draft Team: The last team the player played for before entering the draft
# Nationality: Country the player represents or is a citizen of.

# GP: Games Played — total number of games the player appeared in.
# TS%: True Shooting Percentage — a shooting efficiency metric that accounts for 2s, 3s, and free throws.
# eFG%: Effective Field Goal Percentage — adjusts FG% to account for the extra value of 3-point shots.

# ORB%: Offensive Rebound Percentage
# DRB%: Defensive Rebound Percentage
# TRB%: Total Rebound Percentage

# AST%: Assist Percentage
# TOV%: Turnover Percentage
# STL%: Steal Percentage
# BLK%: Block Percentage

# USG%: Usage Percentage — estimate of the team plays used by a player while on the court.
# Total S %: "Total Shooting %" combination of 3pt% and fg% and ft%
# PPR: Pure Point Rating — advanced stat to measure point guard play, weighing assists vs. turnovers.
# PPS: Points Per Shot — points scored per shot attempt (including free throws).
# ORtg: Offensive Rating — estimate of points produced per 100 possessions.
# DRtg: Defensive Rating — estimate of points allowed per 100 possessions.
# PER: Player Efficiency Rating — overall rating of a player’s per-minute statistical production (league average is 15.0).


#The following is an XGboost model set to pairwise ranking only american college players. Run with model.py <test file name in Player Data>. Rudimentary testing but it will just exclude your test file from training. 


#Global Variables:
path = "playerData/"
team_le = LabelEncoder()
pos_le = LabelEncoder()

if len(sys.argv) != 2:
    print("Usage: python script.py <test_csv_filename>")
    sys.exit(1)

#create path to draft class folder, match all individual file paths using glob
test_file_name = sys.argv[1]

combined_df = create_formated_player_data("combined_player_data_with_labels.csv",test_file_name)


desired_feats = ["WT","Age_x","GP","TS%",                   #add height, omit noationality for later, POS ENCODED

            "eFG%","ORB%","DRB%","TRB%","AST%","TOV%",
            "STL%","BLK%","USG%","Total S %","PPR",
            "PPS","ORtg","DRtg","PER", "Team_encoded","Position_encoded"]

#create feature matrix and label vector
x_vector = combined_df[desired_feats].copy()
y_vector = combined_df["label"].copy()

#creates a list of draft class sizes for each draft. Remember, at this point all drafts are in the same dataframe, and we 
#would like to train the model within each draft class. this will be used later as our QID to tell the model what to limit each set of comparisons to.
group_sizes = combined_df.groupby("Year").size().tolist()


# Replace dashes and other non-numeric values with NaN
x_vector.replace('-', pd.NA, inplace=True)

# Convert all columns to numeric, forcing anything bad to NaN. I know all columns should be numeric but for some reason we get errors without this.
for col in x_vector.columns:
    x_vector[col] = pd.to_numeric(x_vector[col], errors="coerce")

# Fill missing values with column means
x_vector = x_vector.fillna(x_vector.mean()) 

#created xgboost Dmatrix with our x matrix of features and y vector of labels (actual draft picks) from before
trained = xgb.DMatrix(x_vector, label=y_vector)

#this is where the group sizes we got from above come in. 
#We set the group sizes for the DMatrix so that it knows how to compare players within each draft class
trained.set_group(group_sizes)

#XGboost hyperparameters
params = {
    "objective": "rank:pairwise",
    "eta": 0.1,
    "max_depth": 7,
    "eval_metric": "ndcg"
}


#Money maker, train model with the DMatrix and the hyperparameters
model = xgb.train(params, trained, num_boost_round=50)

#create path to the testing file we omitted from training earlier
test_path = os.path.join(path, test_file_name)

#read testing file into its own dataframe
pre_tested_players = pd.read_csv(test_path)

#pre_tested_players = pre_tested_players.drop(columns=["Draft Trades", "Age_y", "Class", "Season", "School"])

#what we would like later for output is just the names and picks of tested players. save this one for later
names_and_picks_pre_tested = pre_tested_players[["Player", "Pick"]].copy() 

#same thing as above, replace 0 picks (undraftees) with 999 
pre_tested_players["Pick"] = pre_tested_players["Pick"].replace(0,999)

#make negative label, same as we did above for combined_df
pre_tested_players["label"] = -pre_tested_players["Pick"]

#encode college team and position using the same label encoders as we did above for training which is important. If a center -> 0 in training, then a center in testing must also be 0.
pre_tested_players["Team_encoded"] = team_le.fit_transform(pre_tested_players["Pre-Draft Team"])
pre_tested_players["Position_encoded"] = pos_le.fit_transform(pre_tested_players["Pos"]) 

#a similar checkpoint. This is going to be our testing draft set, encoded and labeled just as the training set was.
pre_tested_players.to_csv("combined_player_data_with_labels_TEST_DATA.csv", index=False) 

#replace dashes with NaN in tested player dataframe
pre_tested_players.replace('-', pd.NA, inplace=True)

#save their names at this point as a column we can attach later for output
pre_tested_player_NAMES_SAVED = pre_tested_players["Player"].copy() 

# Convert all columns to numeric, forcing anything bad to NaN like we did for training
for col in pre_tested_players.columns:
    pre_tested_players[col] = pd.to_numeric(pre_tested_players[col], errors="coerce")

# Fill missing values with column means 
pre_tested_players = pre_tested_players.fillna(pre_tested_players.mean()) 

#checkpoint. This is now labeled, encoded, cleaned, and ready to be tested.
pre_tested_players.to_csv("PRE-TESTED.csv", index=False) 

#pre_tested_players is now ready to be tested. We will use the same features as we did for training, so we can just copy the list from above
tested_players = pre_tested_players[desired_feats].copy()

#create xgboost DMatrix for the testing data
test = xgb.DMatrix(tested_players[desired_feats])

#input testing matrix into model
predictions = model.predict(test)

#slice off prediction for each player, and add a new column with player names
tested_players["PREDICTION"] = predictions 
tested_players["Player"] = pre_tested_player_NAMES_SAVED 

#sort tested players by prediction, in descending order
tested_players = tested_players.sort_values(by="PREDICTION", ascending=False)  

#add indexes to tested list because we want to compare draft picks and rthe model only gives us relative scores
tested_players["RowIndex"] = range(1, len(tested_players) + 1) 

#checkpoint. This is now the final tested players dataframe, with predictions and row indexes. We have predicted picks but still gotta compare them to their actual for our own interpretation.
tested_players.to_csv("TESTED.csv", index=False) 

#this is where names and picks of tested players comes in handy caquse we can merge this with pre-tested players dataframe to get a dataframe with both actual and predicted picks 
names_and_picks_tested = tested_players[["Player","RowIndex"]].copy()
merged_names_and_picks = pd.merge(
    names_and_picks_tested,
    names_and_picks_pre_tested,
    on="Player",
    how="left"
)

#some renaming
merged_names_and_picks = merged_names_and_picks.rename(columns={"RowIndex": "Predicted Pick", "Pick": "Actual Pick"})

# Set predicted pick to 61 if it's >= 60
merged_names_and_picks["Predicted Pick"] = merged_names_and_picks["Predicted Pick"].apply(lambda x: 61 if x >= 60 else x)

# Set actual pick to 61 if it's 0
merged_names_and_picks["Actual Pick"] = merged_names_and_picks["Actual Pick"].apply(lambda x: 61 if x == 0 else x)

# Remove rows where both predicted and actual picks are 61 as doesnt contribute to top 60 error 
merged_names_and_picks = merged_names_and_picks[
    (merged_names_and_picks["Predicted Pick"] != 61) | (merged_names_and_picks["Actual Pick"] != 61)
]


#pick error 
merged_names_and_picks["Error (pick distance)"] = (merged_names_and_picks["Predicted Pick"] - merged_names_and_picks["Actual Pick"]).abs()

#output
print(merged_names_and_picks)

plot_data(merged_names_and_picks)
 


#NOTES : WANT TO CHANGE PREDICTED PICK TO BE 0 IF EXCEEDS DRAFT SIZE 
#MAKE IT EASIER TO CHOOSE WHAT YEAR TO USE AS TEST SET

