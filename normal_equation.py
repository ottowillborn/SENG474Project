import pandas as pd
# import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import glob
import os
import sys

#argument error handling
if len(sys.argv) != 2:
    print("Usage: python script.py <test_csv_filename>")
    sys.exit(1)

#create path to draft class folder, match all individual file paths using glob
test_file_name = sys.argv[1]
path = "allUpdatedPlayerData/"
pattern = os.path.join(path, "all_players_career_stats_*.csv")
all_files = glob.glob(pattern)

# create list dfs with dataframed draft class csvs, but exclude the test file
files = [f for f in all_files if not f.endswith(test_file_name)] 
dfs = []
for file in files:
    year = int(file.split("_")[-1].split(".")[0])  
    df = pd.read_csv(file)
    df["Year"] = year  
    dfs.append(df)

#concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# label pick 0s in test data as 999
combined_df["Pick"] = combined_df["Pick"].replace(0,999)
#Get test data
test_path = f"allUpdatedPlayerData/{test_file_name}"
test_data = pd.read_csv(test_path)

#Make undrafted picks 61 (0 -> 61)
combined_df["Pick"] = combined_df["Pick"].replace(0, 61)
test_data["Pick"] = test_data["Pick"].replace(0, 61)

# desired features for training, these are the features we will use to train the model
desired_feats = ["HT", "WT","Age_x","GP","TS%",                   
            "eFG%","ORB%","DRB%","TRB%","AST%","TOV%",
            "STL%","BLK%","USG%","Total S %","PPR",
            "PPS","ORtg","DRtg","PER", "Team_encoded","Position_encoded"]

#Handling of team that is not in the combined df, but in the test df
#Adding a dummy player with no stats, so label encoder has an "Unknown Team" value available
if "Unknown" not in combined_df["Pre-Draft Team"].unique():
    combined_df = pd.concat([
        combined_df,
        pd.DataFrame([{
            "Pre-Draft Team": "Unknown",
            "Pos": "Unknown",
            "Pick": 999,
            **{feat: 0 for feat in desired_feats} # fill all desired feats to 0
        }])
    ], ignore_index=True)

#encode college team and position of player (must be done for any non numerical feature in params)
team_le = LabelEncoder()
pos_le = LabelEncoder()
combined_df["Team_encoded"] = team_le.fit_transform(combined_df["Pre-Draft Team"])
# Replace unseen teams in test_data with unknown
test_data["Pre-Draft Team"] = test_data["Pre-Draft Team"].apply(
    lambda x: x if x in team_le.classes_ else "Unknown"
)
test_data["Team_encoded"] = team_le.transform(test_data["Pre-Draft Team"]) 

combined_df["Position_encoded"] = pos_le.fit_transform(combined_df["Pos"])
#Replace unseen positions in test_data with Unknown
test_data["Pos"] = test_data["Pos"].apply(
    lambda x: x if x in pos_le.classes_ else "Unknown"
)
test_data["Position_encoded"] = pos_le.transform(test_data["Pos"])

#Drop dummy player before training
combined_df = combined_df[combined_df["Pick"] != 999]

# Convert height to inches
def convert_height(height):
    try:
        parts = height.strip().split('-')
        if len(parts)!=2:
            return np.nan
        feet = int(parts[0])
        inches = int(parts[1])
        return (feet * 12) + inches
    except:
        return np.nan
    
combined_df["HT"] = combined_df["HT"].apply(convert_height)
test_data["HT"] = test_data["HT"].apply(convert_height)

#create feature matrix and label vector
x_vector = combined_df[desired_feats].copy()
y_vector = combined_df["Pick"].copy()
x_test = test_data[desired_feats].copy()
y_test = test_data["Pick"].copy()

# Replace dashes and other non-numeric values with NaN
x_vector.replace('-', pd.NA, inplace=True)
x_test.replace('-', pd.NA, inplace=True)

# Convert all columns to numeric, forcing anything bad to NaN. I know all columns should be numeric but for some reason we get errors without this.
for col in x_vector.columns:
    x_vector[col] = pd.to_numeric(x_vector[col], errors="coerce")

for col in x_test.columns:
    x_test[col] = pd.to_numeric(x_test[col], errors="coerce")

# Fill missing values with column means
x_vector = x_vector.fillna(0.1 * x_vector.mean()) 
x_test = x_test.fillna(0.1 * x_vector.mean()) 

#normal equation calculation
def normal_equation(X, Y):
  #add bias term
  bias = np.ones((X.shape[0], 1))
  X_new = np.concatenate((bias, X), axis=1)
  #calculate thetas
  theta = np.linalg.pinv(X_new).dot(Y)
  return theta

#prediction function
def predict(X, theta):
    bias = np.ones((X.shape[0], 1))
    X_new = np.concatenate((bias, X), axis=1)
    return X_new.dot(theta)

# Train model
X_train_np = np.array(x_vector)
Y_train_np = np.array(y_vector)

#Calculate parameters
theta_normal = normal_equation(X_train_np, Y_train_np)

#----------------------------------------------
# This section just to find feature importance
# Create copies for scaled
# scaled_data = x_vector.copy()
# scaled_labels = y_vector.copy()
# #Scale data for feature importance
# scaler = StandardScaler()
# #Scale x input
# columns_to_scale = scaled_data.columns.difference(['Player'])
# scaled_data[columns_to_scale] = scaler.fit_transform(scaled_data[columns_to_scale])
# #scale labels
# y_scaler = StandardScaler()
# y_scaled = y_scaler.fit_transform(scaled_labels.values.reshape(-1, 1)).flatten()
# #for scaled values
# x_train_scaled = np.array(scaled_data)
# y_train_scaled = np.array(y_scaled)
# # Find scaled parameters (Feature importance scores)
# theta_scaled = normal_equation(x_train_scaled, y_train_scaled)
# print(f"Scaled parameters:")
# #print thetas
# for val in theta_scaled:
#     print(val)
#---------------------------------------------

# Make prediction
y_pred = predict(x_test, theta_normal)

#Rank players based on their "Regression score"
test_data["predicted_score"] = y_pred

test_data_sorted = test_data.sort_values("predicted_score").reset_index(drop=True)
test_data_sorted["predicted_pick"] = test_data_sorted.index + 1
test_data_sorted["predicted_pick"] = np.minimum(test_data_sorted["predicted_pick"], 61)

# Find error (MAE)
avg_error = np.mean(np.abs(test_data_sorted["Pick"] - test_data_sorted["predicted_pick"]))
print(f"avg error: {avg_error}")

# Find error (MSE)
# avg_MSE = np.mean((test_data_sorted["Pick"] - test_data_sorted["predicted_pick"])**2)
# print(f"avg MSE error: {avg_MSE}")


# Scatter plot

# plt.figure(figsize=(10, 8))
# plt.scatter(
#     test_data_sorted["Pick"],
#     test_data_sorted["predicted_pick"],
#     alpha=0.8
# )

# # Diagonal line (perfect prediction)
# plt.plot([1, 60], [1, 60], linestyle='--', color='gray', label="Perfect Prediction")
# # Axis labels and title

# plt.xlabel("Actual Pick")
# plt.ylabel("Predicted Pick")
# plt.title("Predicted vs. Actual NBA Draft Picks")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
