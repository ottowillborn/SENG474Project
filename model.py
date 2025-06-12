from realgm_scraper import scrape_realgm_draft
import pandas as pd

df = scrape_realgm_draft(2023)
# Save the DataFrame to a CSV file
output_filename = "draft_2023.csv"
df.to_csv(output_filename, index=False)
print(f"Draft data for 2023 saved to {output_filename}")
