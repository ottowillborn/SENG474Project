import subprocess
import pandas as pd
import numpy as np

# ***************
# To use script: need output of model to be "{string}: error value int" (Splits by ": ")
# Also change model name in line 18
# ************** 

years = list(range(2006, 2023))
results = []

#Run for each year, extract error for each year
for year in years:
    print(f"Running for {year}")
    result = subprocess.run(
        # Change name of model below to run for different models
        ["python", "normal_equation.py", f"all_players_career_stats_{year}.csv"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text=True
    )

    output = result.stdout.strip()
    results.append(output)

#make list of all the errors
errors = []
for item in results:
    value = float(item.split(": ")[1])
    errors.append(value)

#print errors
# print(errors)

#make csv error file, for easy formatting to a spreadsheet
df = pd.DataFrame({"Avg Error": errors})
df.to_csv("errors.xlsx", index=False)

# Error metrics for total LOOCV
avg_error = np.mean(errors)
#can't remember if we want std or variance, I think std?
std_error = np.std(errors)
variance_error = np.var(errors)

print(f"Overall Error: {avg_error} +- {std_error}%")