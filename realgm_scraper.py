import requests
import pandas as pd
from bs4 import BeautifulSoup
from helpers import find_realgm_player_url
import pandas as pd
#2002 is the earliest year with NCAA stats available
#2006 is when high school players were no longer eligible for the draft
def scrape_realgm_draft(year: int) -> pd.DataFrame:
    url = f"https://basketball.realgm.com/nba/draft/past_drafts/{year}"
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.find_all("table")
    if not tables or len(tables) < 3:
        raise ValueError("Could not find the required tables on draft page.")

    # Parse both rounds and undrafted players
    all_rounds = []
    for table in tables[:3]:  
        headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cols:
                rows.append(cols)
        df = pd.DataFrame(rows, columns=headers)
        all_rounds.append(df)

    full_draft_df = pd.concat(all_rounds, ignore_index=True)

    # Set Pick to 0 where Pick is empty or missing (undrafted)
    if "Pick" in full_draft_df.columns:
        full_draft_df["Pick"] = full_draft_df["Pick"].replace("", 0)
        full_draft_df["Pick"] = full_draft_df["Pick"].fillna(0)

    return full_draft_df

def scrape_ncaa_advanced_career_stats(player_summary_url: str) -> pd.Series:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    resp = requests.get(player_summary_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    print(f"Getting advanced stats from {player_summary_url}")
    # Locate the NCAA Advanced Stats tab content
    advanced_tab_div = soup.find("div", id="tabs_ncaa_reg-4")
    if not advanced_tab_div:
        print(f"Advanced tab div not found for {player_summary_url}")
        return pd.Series(dtype="object")

    # Look for the table inside this div
    table = advanced_tab_div.find("table")
    if not table:
        print(f"No table found in NCAA Advanced Stats tab for {player_summary_url}")
        return pd.Series(dtype="object")

    # Get the header names
    header_row = table.find("thead")
    if not header_row:
        print("No thead found.")
        return pd.Series(dtype="object")

    headers = [th.get_text(strip=True) for th in header_row.find_all("th")]

    # Get the footer row (career stats)
    tfoot = table.find("tfoot")
    if not tfoot:
        print("No tfoot found.")
        return pd.Series(dtype="object")

    tr = tfoot.find("tr")
    if not tr:
        print("No row in tfoot.")
        return pd.Series(dtype="object")

    cols = [th.get_text(strip=True) for th in tr.find_all("th")]
    if len(headers) != len(cols):
        print("Header and footer column count mismatch.")
        return pd.Series(dtype="object")

    return pd.Series(dict(zip(headers, cols)))

draft_year = 2019
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