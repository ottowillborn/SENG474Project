import requests
import pandas as pd
from bs4 import BeautifulSoup

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

def scrape_ncaa_player_stats(season: int) -> pd.DataFrame:
    url = f"https://basketball.realgm.com/ncaa/stats/{season}/Averages/Qualified/All/Season/All/points/desc/1/"
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    # Find all tables and look for the one with column headers starting with 'Player'
    tables = soup.find_all("table")
    target_table = None
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if headers and "Player" in headers[1]:
            target_table = table
            break

    if not target_table:
        raise ValueError("Stats table not found.")

    headers = [th.get_text(strip=True) for th in target_table.find("thead").find_all("th")]
    rows = []
    for tr in target_table.find("tbody").find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cols:
            rows.append(cols)

    df = pd.DataFrame(rows, columns=headers)
    return df

