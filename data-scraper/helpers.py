import unicodedata
import requests
from bs4 import BeautifulSoup
import pandas as pd

def normalize_name(name):
    if pd.isnull(name):
        return ""
    # Remove accents and convert to lowercase
    return ''.join(
        c for c in unicodedata.normalize('NFKD', name)
        if not unicodedata.combining(c)
    ).strip()

def find_realgm_player_url(name: str) -> str:
    search_url = f"https://basketball.realgm.com/search?q={name.replace(' ', '+').replace('-', '+')}"
    resp = requests.get(search_url)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    for link in soup.select("a[href^='/player/']"):
        href = link.get("href")
        if "/Summary/" in href:
            return "https://basketball.realgm.com" + href
    return None
