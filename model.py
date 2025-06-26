import numpy as np
import pandas as pd
#2002 is the earliest year with NCAA stats available
#2006 is when high school players were no longer eligible for the draft

df = pd.read_csv('college_players_career_stats_2012.csv')

# Display the first few rows
print(df.head())