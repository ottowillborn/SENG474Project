import numpy as np
import pandas as pd
import glob
import sys
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import logging
import torch.backends.cudnn as cudnn


class NBADraftNet(nn.Module):
    """Neural Network for NBA Draft Prediction"""

    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super(NBADraftNet, self).__init__()

        # First layer processes raw player data
        self.input_layer = nn.Linear(input_size, hidden_sizes[0])

        # Create the hierarchical structure as shown in the diagram
        # Stats/efficiency metrics layer
        self.stats_layer = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Physical measurements layer
        self.physical_layer = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Age and experience layer
        self.age_exp_layer = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Character/work ethic layer (will learn from overall patterns)
        self.character_layer = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Final combination layer
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_sizes[1] * 4, hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_sizes[2], 1)
        )

    def forward(self, x):
        # Initial processing
        x = torch.relu(self.input_layer(x))

        # Parallel processing through different aspects
        stats_out = self.stats_layer(x)
        physical_out = self.physical_layer(x)
        age_exp_out = self.age_exp_layer(x)
        character_out = self.character_layer(x)

        # Combine all aspects
        combined = torch.cat(
            [stats_out, physical_out, age_exp_out, character_out], dim=1)

        # Final prediction
        return self.final_layer(combined)


class NBADraftDataset(Dataset):
    """Dataset class for NBA Draft data"""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def convert_height(ht_str):
    """ Helper function to convert height safely """
    try:
        if pd.isna(ht_str) or ht_str == '':
            return np.nan
        feet, inches = ht_str.replace("'", "").replace(
            '"', '').replace('-', ' ').split()
        return int(feet) * 30.48 + int(inches) * 2.54
    except:
        return np.nan


def load_and_preprocess_data(data_path: str, test_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess the NBA draft data"""
    logging.info("Loading and preprocessing data...")

    pattern = os.path.join(data_path, "all_players_career_stats_*.csv")
    all_files = glob.glob(pattern)

    # Exclude test file from training data
    files = [f for f in all_files if not f.endswith(test_file)]
    dfs = []
    for file in files:
        year = int(file.split("_")[-1].split(".")[0])
        df = pd.read_csv(file)
        df["Year"] = year
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df["Pick"] = combined_df["Pick"].replace(0, 61)
    combined_df["label"] = -combined_df["Pick"]
    combined_df["HT"] = combined_df["HT"].apply(convert_height)

    # Load and process test data
    test_path = os.path.join(data_path, test_file)
    test_df = pd.read_csv(test_path)
    test_df["Pick"] = test_df["Pick"].replace(0, 61)
    test_df["HT"] = test_df["HT"].apply(convert_height)

    return combined_df, test_df


def prepare_features(df: pd.DataFrame, is_training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features and labels for the model"""
    desired_feats = ["HT", "WT", "Age_x", "GP", "TS%", "eFG%", "ORB%", "DRB%", "TRB%",
                     "AST%", "TOV%", "STL%", "BLK%", "USG%", "Total S %", "PPR",
                     "PPS", "ORtg", "DRtg", "PER"]

    # Create feature matrix
    X = df[desired_feats].copy()
    X = X.replace(['-', np.inf, -np.inf], np.nan)  # Handle more invalid values
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill NaN values with mean but print warning if there are many
    nan_counts = X.isna().sum()
    if nan_counts.any():
        print("Warning: NaN counts in features: \n",
              nan_counts[nan_counts > 0])
    X = X.fillna(X.mean())

    if is_training:
        # Negative because lower pick numbers are better
        y = -df["Pick"].values
    else:
        y = np.zeros(len(X))  # Dummy labels for test data

    return X.values, y


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int,
                learning_rate: float) -> Tuple[List[float], List[float], List[float]]:
    """Train the neural network model"""
    logging.info("Starting model training...")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    train_sizes = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure model is on GPU

    # Calculate total dataset size
    total_size = len(train_loader.dataset)
    # Generate training sizes (20%, 40%, 60%, 80%, 100% of data)
    subset_sizes = [int(total_size * p) for p in np.linspace(0.2, 1.0, 5)]

    for size in subset_sizes:
        # Create subset of training data
        subset_indices = torch.randperm(total_size)[:size]
        subset_train_data = torch.utils.data.Subset(
            train_loader.dataset, subset_indices)
        subset_train_loader = DataLoader(
            subset_train_data, batch_size=128, shuffle=True, pin_memory=True  # Enable pin_memory
        )

        model.train()
        train_loss = 0.0
        for _ in range(num_epochs // 5):  # Reduced number of epochs per subset
            for batch_X, batch_y in subset_train_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(
                    device, non_blocking=True)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.view(-1, 1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device, non_blocking=True), batch_y.to(
                    device, non_blocking=True)  # Move data to GPU
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y.view(-1, 1)).item()

        train_losses.append(train_loss / len(subset_train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_sizes.append(size)

        logging.info(
            f'Training Size [{size}/{total_size}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')

    return train_losses, val_losses, train_sizes


def evaluate_model(model: nn.Module, test_loader: DataLoader) -> np.ndarray:
    """Evaluate the model and return predictions"""
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device, non_blocking=True)  # Move data to GPU
            outputs = model(batch_X)
            # Move predictions back to CPU
            predictions.extend(outputs.cpu().numpy())

    return np.array(predictions)


def plot_results(actual_picks: np.ndarray, predicted_picks: np.ndarray, player_names: List[str]):
    """Plot the comparison between actual and predicted picks"""
    plt.figure(figsize=(10, 8))
    plt.scatter(actual_picks, predicted_picks, alpha=0.8)
    plt.plot([1, 60], [1, 60], linestyle='--',
             color='gray', label="Perfect Prediction")

    for i, name in enumerate(player_names):
        plt.annotate(name, (actual_picks[i], predicted_picks[i]), fontsize=8)

    plt.xlabel("Actual Pick")
    plt.ylabel("Predicted Pick")
    plt.title("Predicted vs. Actual NBA Draft Picks")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_learning_curves(train_sizes: List[int], train_losses: List[float], val_losses: List[float]):
    """Plot the learning curves showing training and validation loss vs training size"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_losses, 'o-',
             color='blue', label='Training Loss')
    plt.plot(train_sizes, val_losses, 'o-',
             color='green', label='Validation Loss')
    plt.title("Model Learning Curves")
    plt.xlabel("Number of Training Examples")
    plt.ylabel("Loss")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def train_and_test_model(data_path: str, year: str, show_plots: bool = True) -> float:
    test_file = f"all_players_career_stats_{year}.csv"

    # Load and preprocess data
    train_df, test_df = load_and_preprocess_data(data_path, test_file)

    # Prepare features
    X_train, y_train = prepare_features(train_df, is_training=True)
    X_test, _ = prepare_features(test_df, is_training=False)

    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Create datasets and dataloaders
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Enable cuDNN benchmarking and deterministic mode
    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = True

    # Move data to GPU immediately after creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = NBADraftDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(y_train)
    )
    val_dataset = NBADraftDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(y_val)
    )
    test_dataset = NBADraftDataset(
        torch.FloatTensor(X_test_scaled),
        torch.FloatTensor(np.zeros(len(X_test_scaled)))
    )

    # Increase batch size for better GPU utilization
    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True  # Enable pin_memory
    )
    val_loader = DataLoader(val_dataset, batch_size=128, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, pin_memory=True)

    # Initialize and train model
    input_size = X_train.shape[1]
    hidden_sizes = [128, 64, 32]
    model = NBADraftNet(input_size, hidden_sizes).to(
        device)  # Ensure model is on GPU

    train_losses, val_losses, train_sizes = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=0.001
    )

    # Plot learning curves
    if (show_plots):
        plot_learning_curves(train_sizes, train_losses, val_losses)

    # Make predictions
    predictions = evaluate_model(model, test_loader)

    # Process predictions
    predicted_picks = -predictions.flatten()  # Convert back to pick numbers
    actual_picks = test_df["Pick"].values
    player_names = test_df["Player"].values

    # Plot results
    if (show_plots):
        plot_results(actual_picks, predicted_picks, player_names)

    # Calculate and display error metrics
    pick_error = np.abs(predicted_picks - actual_picks)
    mean_error = np.mean(pick_error)
    print(f"Mean absolute pick error for test year {year}: {mean_error:.2f}\n")
    return mean_error


def main():
    # Enable logging
    # logging.basicConfig(level=logging.INFO)

    # Check if CUDA is available and print GPU info
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0)
          if torch.cuda.is_available() else "No GPU")

    data_path = "../allUpdatedPlayerData/"
    LOO = sys.argv[1] == "-loocv" if len(sys.argv) > 1 else False

    if LOO:
        years = ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015",
                 "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
        errors = []
        for year in years:
            error = train_and_test_model(data_path, year, show_plots=False)
            errors.append(error)

        # Final report
        print("-" * 40)
        for year in years:
            print(f"Year {year} error: {errors[years.index(year)]:.2f}")
        print(f"Average error across all years: {np.mean(errors):.2f}")
        print(f"Standard deviation of error: {np.std(errors):.2f}")

    else:
        train_and_test_model(data_path, "2025")


if __name__ == "__main__":
    main()
