from realgm_scraper import scrape_ncaa_player_stats, scrape_realgm_draft
import pandas as pd
#2002 is the earliest year with NCAA stats available
#2006 is when high school players were no longer eligible for the draft
draft_df = scrape_realgm_draft(2003)
output_filename = "draft_2024.csv"
draft_df.to_csv(output_filename, index=False)
print(f"Draft data for 2024 saved to {output_filename}")
print(draft_df.head())

stats_df = scrape_ncaa_player_stats(2003)
output_filename = "stats_2023.csv"
stats_df.to_csv(output_filename, index=False)
print(f"NCAA Player data for 2023 saved to {output_filename}")
print(stats_df.head())
print(stats_df.shape)

# Merge the DataFrames on the 'Player' column, keeping only players in both DataFrames
merged_df = pd.merge(draft_df, stats_df, on="Player", how="inner")
print("Merged DataFrame:")
print(merged_df.head())
print(merged_df.shape)
