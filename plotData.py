import matplotlib.pyplot as plt
import glob
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