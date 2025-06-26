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

def scrape_ncaa_advanced_career_stats(player_summary_url: str) -> pd.Series:
    import requests
    from bs4 import BeautifulSoup
    import pandas as pd

    resp = requests.get(player_summary_url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

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