import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import glob
import os
import sys


# Path to all player data
path = "allUpdatedPlayerData/"
pattern = os.path.join(path, "all_players_career_stats_*.csv")
all_files = glob.glob(pattern)

# Convert player data to dataframes
dfs = []
for file in all_files:
    year = int(file.split("_")[-1].split(".")[0])  
    df = pd.read_csv(file)
    df["Year"] = year  
    dfs.append(df)

#concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# label pick 0s in test data as 61
combined_df["Pick"] = combined_df["Pick"].replace(0,61)

#Split data into training and testing dataframes
train_years = [year for year in range(2006, 2026) if year not in [2010, 2015, 2020, 2025]]
test_years = [2010, 2015, 2020, 2025]

train_data = combined_df[combined_df["Year"].isin(train_years)].copy()
test_data = combined_df[combined_df["Year"].isin(test_years)].copy()

# desired features for training, these are the features we will use to train the model
# This is features for normal equation, update them based on model
desired_feats = ["HT","Age_x","TS%",                   
            "eFG%","ORB%","DRB%","TRB%","USG%",
            "PPS","ORtg","DRtg","PER"]

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

# Convert height string to inches   
train_data["HT"] = train_data["HT"].apply(convert_height)
test_data["HT"] = test_data["HT"].apply(convert_height)


# Feature and label dataframes for train/test
x_vector = train_data[desired_feats].copy()
y_vector = train_data["Pick"].copy()
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
x_vector = x_vector.fillna(0.9 * x_vector.mean()) 
x_test = x_test.fillna(0.9 * x_vector.mean()) 

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

#choosing increment of data for each plot
# Increase data used by 2.5% each iteration for learning curves 
train_sizes = np.linspace(0.025, 1.0, 40)
#hold training and validation error
train_errors = []
test_errors = []

X_train_np = np.array(x_vector)
Y_train_np = np.array(y_vector)
X_test_np = np.array(x_test)
Y_test_np = np.array(y_test)

#Shuffle data for less bias
X_train_np, Y_train_np = shuffle(X_train_np, Y_train_np, random_state=42)

#train and get errors for each increment of data
for frac in train_sizes:
    # Number of training samples to include per iteration
    n_samples = int(len(X_train_np) * frac)
    if n_samples == 0:
        continue

    X_train_frac = X_train_np[:n_samples]
    Y_train_frac = Y_train_np[:n_samples]

    # Train model on training data
    theta = normal_equation(X_train_frac, Y_train_frac)

    # Predict on training subset
    y_train_pred = predict(X_train_frac, theta)
    #Training error
    train_mae = np.mean(np.abs(Y_train_frac - y_train_pred))

    # Predict on full test set
    y_test_pred = predict(X_test_np, theta)
    #Validation error
    test_mae = np.mean(np.abs(Y_test_np - y_test_pred))

    print(f"Train size: {n_samples} | Train MAE: {train_mae:.3f} | Test MAE: {test_mae:.3f}")

    train_errors.append(train_mae)
    test_errors.append(test_mae)

# Plotting
plt.figure(figsize=(8,6))
plt.plot(train_sizes * 100, train_errors, marker='o', label="Training Error")
plt.plot(train_sizes * 100, test_errors, marker='o', label="Validation Error")
plt.xlabel("Training Set Size (%)")
plt.ylabel("Mean Absolute Error (MAE)")
plt.title("Learning Curves for Normal Equation Linear Regression")
plt.legend()
plt.grid(True)
plt.show()