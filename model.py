import numpy as np
import pandas as pd
#2002 is the earliest year with NCAA stats available
#2006 is when high school players were no longer eligible for the draft
# Player: Full name of the basketball player.
# Team: Team the player was drafted by or currently plays for.
# Pos: Player's position (e.g., PG, SG, SF, PF, C).

# HT: Height of the player ft-in.
# WT: Weight of the player lb.

# Pre-Draft Team: The last team the player played for before entering the draft
# Nationality: Country the player represents or is a citizen of.

# GP: Games Played — total number of games the player appeared in.
# TS%: True Shooting Percentage — a shooting efficiency metric that accounts for 2s, 3s, and free throws.
# eFG%: Effective Field Goal Percentage — adjusts FG% to account for the extra value of 3-point shots.

# ORB%: Offensive Rebound Percentage
# DRB%: Defensive Rebound Percentage
# TRB%: Total Rebound Percentage

# AST%: Assist Percentage
# TOV%: Turnover Percentage
# STL%: Steal Percentage
# BLK%: Block Percentage

# USG%: Usage Percentage — estimate of the team plays used by a player while on the court.
# Total S %: "Total Shooting %" combination of 3pt% and fg% and ft%
# PPR: Pure Point Rating — advanced stat to measure point guard play, weighing assists vs. turnovers.
# PPS: Points Per Shot — points scored per shot attempt (including free throws).
# ORtg: Offensive Rating — estimate of points produced per 100 possessions.
# DRtg: Defensive Rating — estimate of points allowed per 100 possessions.
# PER: Player Efficiency Rating — overall rating of a player’s per-minute statistical production (league average is 15.0).

df = pd.read_csv('playerData/college_players_career_stats_2012.csv')
columns_to_drop = ["Draft Trades", "Age_x", "Age_y", "Class", "Season", "School"]
df = df.drop(columns=columns_to_drop)
# Display the first few rows
print(df.head())