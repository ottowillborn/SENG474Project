import numpy as np
import pandas as pd
import glob
import sys 
import os
from plotData import plot_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeRegressor


path = "playerData/"
team_le = LabelEncoder()
pos_le = LabelEncoder()

def plot_data1(merged_names_and_picks):
    # Scatter plot
    print("Mean AVG pick error (for now this is a very unfavorably skewed metric):",merged_names_and_picks["Error (pick distance)"].mean())
    plt.figure(figsize=(10, 8))
    player_data_copy = []
    index = 0
    for _, row in merged_names_and_picks.iterrows():
        if(index > 50):
            break
        player_data_copy.append(row)
        index = index + 1
    print("------------------")
    print(player_data_copy)
    player_data_copy = pd.DataFrame(player_data_copy)

    plt.scatter(player_data_copy["Actual Pick"], player_data_copy["Predicted Pick"], alpha=0.8)

    plt.plot([1, 60], [1, 60], linestyle='--', color='gray', label="Perfect Prediction")
    index = 0
    for _, row in merged_names_and_picks.iterrows():
        #print(i)
        if(index > 50):
            break
        plt.text(
            row["Actual Pick"] + 0.5,  
            row["Predicted Pick"] + 0.5,  
            row["Player"],
            fontsize=8
        )
        index = index + 1

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

if len(sys.argv) != 2:
    print("Usage: python script.py <test_csv_filename>")
    sys.exit(1)

test_file_name = sys.argv[1]

combined_df = create_formated_player_data("combined_player_data_with_labels.csv")


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

print(names_and_picks_pre_tested)

names_and_picks_tested = tested_players[["Player","RowIndex"]].copy()

print(names_and_picks_tested)

merged_names_and_picks = pd.merge(
    names_and_picks_tested,
    names_and_picks_pre_tested,
    on="Player",
    how="left"
)

merged_names_and_picks = merged_names_and_picks.rename(columns={"RowIndex": "Predicted Pick", "Pick": "Actual Pick"})

merged_names_and_picks["Error (pick distance)"] = (merged_names_and_picks["Predicted Pick"] - merged_names_and_picks["Actual Pick"]).abs()

print(merged_names_and_picks)
print(merged_names_and_picks["Player"])
plot_data(merged_names_and_picks)

 


#NOTES : WANT TO CHANGE PREDICTED PICK TO BE 0 IF EXCEEDS DRAFT SIZE 
#MAKE IT EASIER TO CHOOSE WHAT YEAR TO USE AS TEST SET