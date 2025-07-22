import subprocess
import pandas as pd

# Comment out dataframe print in model.py before running script

years = list(range(2006, 2023))
results = []

#Run for each year, extract error for each year
for year in years:
    print(f"Running for {year}")
    result = subprocess.run(
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
print(errors)

#make csv error file, for easy formatting to a spreadsheet
df = pd.DataFrame({"Avg Error": errors})
df.to_csv("errors.xlsx", index=False)