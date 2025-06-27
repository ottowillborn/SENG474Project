# SENG474Project
Repo for Draft Prediction Model Course Project
# Installing required libraries
To install requirements from bash terminal: <pre> ```pip3 install -r requirements.txt``` </pre>

# Running the program
<pre> python model.py <file.csv> </pre>


model.py: Current working XGBoost algorithm on pairwise setting. Run with model.py "file name in playerData" will test on the file given and exclude it from training set.
Issues:
-No european, australian, g league etc players in dataset
-Naive assumption that teams pick on a best player available philosophy
-Assigning mean of feature columns for NaN values
-Skewed mean average pick error due to undrafted players
