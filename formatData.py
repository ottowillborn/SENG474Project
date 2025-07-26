import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import glob
path = "allUpdatedPlayerData/"
def create_formated_player_data(filename,test_file_name):
    #func: create_formated_player_data
    #args:
    #Docs:
    team_le = LabelEncoder()
    pos_le = LabelEncoder()
    pattern = os.path.join(path, "all_players_career_stats_*.csv")
    all_files = glob.glob(pattern)
    #exclude the files: 2010,2015,2020,2025
    #files = [f for f in all_files if not f.endswith(test_file_name)] #excluding the test file
    files = []
    files_to_exclude = [test_file_name,"all_players_career_stats_2010.csv","all_players_career_stats_2015.csv","all_players_career_stats_2020.csv","all_players_career_stats_2025.csv"]
    for f in all_files:
        if f.split("/")[1] not in files_to_exclude:
            files.append(f)
        else:
            print(f)
        
    

    dfs = []
    for file in files:
        year = int(file.split("_")[-1].split(".")[0])  
        df = pd.read_csv(file)
        df["Year"] = year  
        dfs.append(df)

    #MEGA FRAME
    combined_df = pd.concat(dfs, ignore_index=True)
    # 999 labels as not picked
    combined_df["Pick"] = combined_df["Pick"].replace(0,61)
    combined_df["label"] = -combined_df["Pick"] #make a new column label which is just negative pick 

    combined_df["Team_encoded"] = team_le.fit_transform(combined_df["Pre-Draft Team"]) #encodes pre draft teams into numbers
    combined_df["Position_encoded"] = pos_le.fit_transform(combined_df["Pos"]) #encodes positions into numbers

    combined_df.to_csv(filename, index=False) 
    return combined_df
def format_training_data():
    pass