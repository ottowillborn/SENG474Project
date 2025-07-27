import matplotlib.pyplot as plt

# Data
dt_years = [2010, 2015, 2020, 2025]
dt_errors = [28.07, 29.1, 32.8, 13.1]

xgb_years = [2025, 2020, 2015, 2010]
xgb_errors = [10.8, 16.4, 15.2, 16.3]

# Sort XGB data by year to plot correctly
xgb_years, xgb_errors = zip(*sorted(zip(xgb_years, xgb_errors)))
years = [2010,2015,2020,2025]
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(dt_years, dt_errors, marker='o', label='Decision Tree')
plt.plot(xgb_years, xgb_errors, marker='s', label='XGBoost')

# Labels and title
plt.xlabel('Year')
plt.ylabel('Average Pick Error')
plt.title('Average Error rate: Sklearn Decision Tree vs XGBoost Decision Tree')
plt.xticks(years)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
