import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap


def train_and_test_xgboost(training_folder: str, test_file_path: str):
    # Use relative pattern assuming you're running from inside xgboost/
    pattern = os.path.join(training_folder, "all_players_career_stats_*.csv")
    train_files = glob.glob(pattern)

    if not train_files:
        raise FileNotFoundError(f"No training files found in: {training_folder}")

    dfs = []
    for file in train_files:
        year = int(file.split("_")[-1].split(".")[0])
        df = pd.read_csv(file)
        df["Year"] = year
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["Pick"] = combined_df["Pick"].replace(0, 999)
    combined_df["label"] = -combined_df["Pick"]

    # Manual label encoding to avoid unseen label error
    team_map = {team: idx for idx, team in enumerate(combined_df["Pre-Draft Team"].dropna().unique())}
    pos_map = {pos: idx for idx, pos in enumerate(combined_df["Pos"].dropna().unique())}

    combined_df["Team_encoded"] = combined_df["Pre-Draft Team"].map(team_map).fillna(-1).astype(int)
    combined_df["Position_encoded"] = combined_df["Pos"].map(pos_map).fillna(-1).astype(int)

    desired_feats = ["WT", "Age_x", "GP", "TS%", "eFG%", "ORB%", "DRB%", "TRB%",
                     "AST%", "TOV%", "STL%", "BLK%", "USG%", "Total S %", "PPR",
                     "PPS", "ORtg", "DRtg", "PER", "Team_encoded", "Position_encoded"]

    x_vector = combined_df[desired_feats].copy()
    y_vector = combined_df["label"].copy()

    x_vector.replace('-', pd.NA, inplace=True)
    for col in x_vector.columns:
        x_vector[col] = pd.to_numeric(x_vector[col], errors="coerce")
    x_vector = x_vector.fillna(x_vector.mean())

    trained = xgb.DMatrix(x_vector, label=y_vector)
    group_sizes = combined_df.groupby("Year").size().tolist()
    trained.set_group(group_sizes)

    params = {
        "objective": "rank:pairwise",
        "eta": 0.1,
        "max_depth": 6,
        "eval_metric": "ndcg"
    }

    model = xgb.train(params, trained, num_boost_round=50)
    """
    xgb.plot_importance(model, importance_type='gain')
    plt.title("Feature Importance (by Gain)")
    plt.tight_layout()
    plt.show()
    """
    # --- TESTING PHASE ---
    pre_tested_players = pd.read_csv(test_file_path)
    names_and_picks_pre_tested = pre_tested_players[["Player", "Pick"]].copy()
    pre_tested_players["Pick"] = pre_tested_players["Pick"].replace(0, 999)
    pre_tested_players["label"] = -pre_tested_players["Pick"]

    pre_tested_players["Team_encoded"] = pre_tested_players["Pre-Draft Team"].map(team_map).fillna(-1).astype(int)
    pre_tested_players["Position_encoded"] = pre_tested_players["Pos"].map(pos_map).fillna(-1).astype(int)

    pre_tested_players.replace('-', pd.NA, inplace=True)
    pre_tested_player_NAMES_SAVED = pre_tested_players["Player"].copy()

    for col in pre_tested_players.columns:
        pre_tested_players[col] = pd.to_numeric(pre_tested_players[col], errors="coerce")
    pre_tested_players = pre_tested_players.fillna(pre_tested_players.mean())

    tested_players = pre_tested_players[desired_feats].copy()
    test = xgb.DMatrix(tested_players)
    predictions = model.predict(test)
    """
    explainer = shap.TreeExplainer(model)

    # Get SHAP values for each player in the test set
    shap_values = explainer.shap_values(tested_players)

    # Convert back to player names for interpretation
    tested_players["Player"] = pre_tested_player_NAMES_SAVED
    shap.summary_plot(shap_values, tested_players[desired_feats])
    """

    tested_players["PREDICTION"] = predictions
    tested_players["Player"] = pre_tested_player_NAMES_SAVED
    tested_players = tested_players.sort_values(by="PREDICTION", ascending=False)
    tested_players["RowIndex"] = range(1, len(tested_players) + 1)

    names_and_picks_tested = tested_players[["Player", "RowIndex"]].copy()
    merged_names_and_picks = pd.merge(
        names_and_picks_tested,
        names_and_picks_pre_tested,
        on="Player",
        how="left"
    )

    merged_names_and_picks = merged_names_and_picks.rename(columns={"RowIndex": "Predicted Pick", "Pick": "Actual Pick"})
    merged_names_and_picks["Predicted Pick"] = merged_names_and_picks["Predicted Pick"].apply(lambda x: 61 if x >= 60 else x)
    merged_names_and_picks["Actual Pick"] = merged_names_and_picks["Actual Pick"].apply(lambda x: 61 if x == 0 else x)

    merged_names_and_picks = merged_names_and_picks[
        (merged_names_and_picks["Predicted Pick"] != 61) | (merged_names_and_picks["Actual Pick"] != 61)
    ]

    merged_names_and_picks["Error (pick distance)"] = (
        merged_names_and_picks["Predicted Pick"] - merged_names_and_picks["Actual Pick"]
    ).abs()

        #pick error 
    merged_names_and_picks["Error (pick distance)"] = (merged_names_and_picks["Predicted Pick"] - merged_names_and_picks["Actual Pick"]).abs()

    #output
    #print(merged_names_and_picks)

    return merged_names_and_picks


def main():

    training_path = "training"
    test_file = "test/all_players_career_stats_2015.csv"

    results_df = train_and_test_xgboost(training_path, test_file)

    results_df.to_csv("predicted_vs_actual.csv", index=False)
    print(f"\nAverage Pick Error: {results_df['Error (pick distance)'].mean():.2f}")

    #pick error 
    results_df["Error (pick distance)"] = (results_df["Predicted Pick"] - results_df["Actual Pick"]).abs()

    #output
    #print(results_df)
"""
    # Scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(
        results_df["Actual Pick"],
        results_df["Predicted Pick"],
        alpha=0.8
    )

    # Diagonal line (perfect prediction)
    plt.plot([1, 60], [1, 60], linestyle='--', color='gray', label="Perfect Prediction")

    # Annotate each player
    for _, row in results_df.iterrows():
        plt.text(
            row["Actual Pick"] + 0.5,
            row["Predicted Pick"] + 0.5,
            row["Player"],
            fontsize=8
        )

    # Add quadrant labels with background boxes for readability
    plt.text(
        10, 55, "predicted low, actual high", fontsize=8, color='red', weight='bold',
        alpha=0.7, bbox=dict(facecolor='white', edgecolor='none', alpha=0.3)
    )
    plt.text(
        45, 55, "predicted low, actual low", fontsize=8, color='green', weight='bold',
        alpha=0.7, bbox=dict(facecolor='white', edgecolor='none', alpha=0.3)
    )
    plt.text(
        10, 5, "predicted high, actual high", fontsize=8, color='green', weight='bold',
        alpha=0.7, bbox=dict(facecolor='white', edgecolor='none', alpha=0.3)
    )
    plt.text(
        45, 5, "predicted high, actual low", fontsize=8, color='red', weight='bold',
        alpha=0.7, bbox=dict(facecolor='white', edgecolor='none', alpha=0.3)
    )

    # Axis labels and title
    plt.xlabel("Actual Pick")
    plt.ylabel("Predicted Pick")
    plt.title("Predicted vs. Actual NBA Draft Picks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    #paosdgfasd

    results_df = pd.DataFrame({
        "Player": results_df["Player"],
        "Actual Pick": results_df["Actual Pick"],
        "Predicted Pick": results_df["Predicted Pick"],
        "Pick Error": results_df["Error (pick distance)"]
    })

    actual_top = set(results_df.sort_values("Actual Pick").head(10)["Player"])
    predicted_top = set(results_df.sort_values("Predicted Pick").head(10)["Player"])
    hit_rate = len(actual_top & predicted_top) / 10

    print("Top-10 hit rate:", hit_rate)
"""


if __name__ == "__main__":
    main()