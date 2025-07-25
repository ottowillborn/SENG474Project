import numpy as np
import pandas as pd
import glob
import sys 
import os
from plotData import plot_data
from formatData import create_formated_player_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb


path = "playerData/"
team_le = LabelEncoder()
pos_le = LabelEncoder()



i = 0
files = ["06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22"]
errord_tree_list = []
error_xgb_list = []
def make_prediction_xgb(i):
    test_file_name = f"college_players_career_stats_20{files[i]}.csv"

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
    print("Mean AVG pick error (for now this is a very unfavorably skewed metric):",merged_names_and_picks["Error (pick distance)"].mean())
    error_xgb_list.append(merged_names_and_picks["Error (pick distance)"].mean())

def make_predictionD(i):
    test_file_name = f"college_players_career_stats_20{files[i]}.csv"

    combined_df = create_formated_player_data("combined_player_data_with_labels.csv",test_file_name)


    desired_feats = ["WT","Age_x","GP","TS%",                   #add height, omit noationality for later, POS ENCODED
                "eFG%","ORB%","DRB%","TRB%","AST%","TOV%",
                "STL%","BLK%","USG%","Total S %","PPR",
                "PPS","ORtg","DRtg","PER", "Team_encoded","Position_encoded"]

    x_vector = combined_df[desired_feats].copy()

    y_vector = combined_df["label"].copy()
    #we need to rank within each draft year pairwise, not across years (learning is still global though)

    group_sizes = combined_df.groupby("Year").size().tolist()


    #Chats cleaning method below
    # Replace dashes and other non-numeric values with NaN
    x_vector.replace('-', pd.NA, inplace=True)
    # Convert all columns to numeric, forcing anything bad to NaN
    for col in x_vector.columns:
        x_vector[col] = pd.to_numeric(x_vector[col], errors="coerce")
    # Fill missing values with column means (or medians, or zero)
    x_vector = x_vector.fillna(x_vector.mean()) 
    #WARNING HAD TO FILL WITH MEAN 

    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model.fit(x_vector, y_vector)


    #Format Test data HARDCODED FOR 2022 this line is 
    test_path = os.path.join(path, test_file_name)
    pre_tested_players = pd.read_csv(test_path)

    pre_tested_players = pre_tested_players.drop(columns=["Draft Trades", "Age_y", "Class", "Season", "School"])

    names_and_picks_pre_tested = pre_tested_players[["Player", "Pick"]].copy() #keep names and indexes 

    pre_tested_players["Pick"] = pre_tested_players["Pick"].replace(0,999)
    pre_tested_players["label"] = -pre_tested_players["Pick"]

    pre_tested_players["Team_encoded"] = team_le.fit_transform(pre_tested_players["Pre-Draft Team"]) #encodes pre draft teams into numbers
    pre_tested_players["Position_encoded"] = pos_le.fit_transform(pre_tested_players["Pos"]) #encodes positions into numbers
    pre_tested_players.to_csv("combined_player_data_with_labels_TEST_DATA.csv", index=False) 

    pre_tested_players.replace('-', pd.NA, inplace=True)

    pre_tested_player_NAMES_SAVED = pre_tested_players["Player"].copy() #na

    # Convert all columns to numeric, forcing anything bad to NaN
    for col in pre_tested_players.columns:
        pre_tested_players[col] = pd.to_numeric(pre_tested_players[col], errors="coerce")
    # Fill missing values with column means (or medians, or zero)
    pre_tested_players = pre_tested_players.fillna(pre_tested_players.mean()) 

    pre_tested_players.to_csv("PRE-TESTED.csv", index=False) 

    tested_players = pre_tested_players[desired_feats].copy()

    #TESTING
    #Making prediction
    predictions = model.predict(tested_players[desired_feats])

    tested_players["PREDICTION"] = predictions 
    tested_players["Player"] = pre_tested_player_NAMES_SAVED 


    tested_players = tested_players.sort_values(by="PREDICTION", ascending=False) #DONT SORT UNTIL RIGHT NOW IT WILL FUCK ANALYSIS UP 

    tested_players["RowIndex"] = range(1, len(tested_players) + 1) #add indexes to tested list because we want to compare draft picks

    tested_players.to_csv("TESTED.csv", index=False) 

    #print(pre_tested_players.head())
    #print(combined_df.head())

    #ANALYSIS
    #new pandas dataframe with: name. need name in both tested and pre tested. need predicted index from tested. need actual pick from pre tested. 

    #print(names_and_picks_pre_tested)

    names_and_picks_tested = tested_players[["Player","RowIndex"]].copy()

    #print(names_and_picks_tested)

    merged_names_and_picks = pd.merge(
        names_and_picks_tested,
        names_and_picks_pre_tested,
        on="Player",
        how="left"
    )

    merged_names_and_picks = merged_names_and_picks.rename(columns={"RowIndex": "Predicted Pick", "Pick": "Actual Pick"})
    merged_names_and_picks["Actual Pick"] = merged_names_and_picks["Actual Pick"].replace(0, 61)
    merged_names_and_picks["Error (pick distance)"] = (merged_names_and_picks["Predicted Pick"] - merged_names_and_picks["Actual Pick"]).abs()
    print(f"Model for year: 20{files[i]}")
    print("Mean AVG pick error (for now this is a very unfavorably skewed metric):",merged_names_and_picks["Error (pick distance)"].mean())
    errord_tree_list.append(merged_names_and_picks["Error (pick distance)"].mean())

for x in range(0,len(files)):
    make_predictionD(x)
    make_prediction_xgb(x)

#Make graph for ave errors:
years = list(range(2006,2023))
# Plot
plt.figure(figsize=(12, 6))
plt.plot(years, errord_tree_list, marker='o', linestyle='-', linewidth=2, label="Sklearn Decision tree")
plt.plot(years, error_xgb_list, marker='o', linestyle='-', linewidth=2,label="XGBoost Decision tree")
plt.title('Mean Average Pick Error by Year (DecisionTree)')
plt.xlabel('Year')
plt.ylabel('Mean Pick Error')
plt.grid(True)
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.legend()

plt.show()