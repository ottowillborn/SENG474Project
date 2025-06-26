from helpers import find_realgm_player_url
from realgm_scraper import scrape_ncaa_advanced_career_stats, scrape_realgm_draft
import pandas as pd
#2002 is the earliest year with NCAA stats available
#2006 is when high school players were no longer eligible for the draft
draft_year = 2012
draft_df = scrape_realgm_draft(draft_year)
college_players_df = draft_df[~draft_df['Class'].str.contains('DOB', na=False)].copy()
college_players_df = college_players_df.drop(columns=['YOS', 'Class', ], axis=1)
print(f"Found {len(college_players_df)} college players.")
print(college_players_df.head())

# Store career stats in a list
all_career_stats = []

for i, row in college_players_df.iterrows():
    player_name = row['Player']
    try:
        player_url = find_realgm_player_url(player_name)
        if player_url:
            career_stats = scrape_ncaa_advanced_career_stats(player_url)
            career_stats["Player"] = player_name  # add identifier
            all_career_stats.append(career_stats)
        else:
            print(f"URL not found for {player_name}")
    except Exception as e:
        print(f"Error processing {player_name}: {e}")

# Combine all into a DataFrame
career_stats_df = pd.DataFrame(all_career_stats)
career_stats_df = career_stats_df.dropna(axis=1, how='all')  # Remove columns with all NaN values
print(career_stats_df.head())

# Optional: merge with original draft info
combined_df = pd.merge(college_players_df, career_stats_df, on="Player", how="left")
print(combined_df.head())

# Save to CSV
combined_df.to_csv(f"college_players_career_stats_{draft_year}.csv", index=False)