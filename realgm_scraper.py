import requests
import pandas as pd
from bs4 import BeautifulSoup

def scrape_realgm_draft(year: int) -> pd.DataFrame:
    url = f"https://basketball.realgm.com/nba/draft/past_drafts/{year}"
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.find_all("table")
    if not tables or len(tables) < 2:
        raise ValueError("Could not find both draft round tables on the page.")

    # Parse both rounds
    all_rounds = []
    for table in tables[:3]:  # round 1 and round 2 tables
        headers = [th.get_text(strip=True) for th in table.find("thead").find_all("th")]
        rows = []
        for tr in table.find("tbody").find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if cols:
                rows.append(cols)
        df = pd.DataFrame(rows, columns=headers)
        all_rounds.append(df)

    full_draft_df = pd.concat(all_rounds, ignore_index=True)

    # Set Pick to 0 where Pick is empty or missing
    if "Pick" in full_draft_df.columns:
        full_draft_df["Pick"] = full_draft_df["Pick"].replace("", 0)
        full_draft_df["Pick"] = full_draft_df["Pick"].fillna(0)

    return full_draft_df
