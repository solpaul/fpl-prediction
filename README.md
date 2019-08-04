# FPL Prediction
This is a project to generate ongoing player forecasts for Fantasy Premier League. Credit to https://github.com/vaastav/Fantasy-Premier-League, from where all the FPL data is taken.

The data folder contains the following:
+ Player data for each gameweek since the start of the 2016/17 season (one folder with various csv files for each season)
+ fixtures.csv - Remaining fixtures for the 2019/20 season
+ teams.csv - Team names with IDs for each season
+ train.csv - The current training dataset containing all historic data (generated from the player data folders)
+ remaining_season.csv - The dataset to use for predicting the remainder of the 2019/20 season

There are two jupyter notebooks:
+ fpl_data_clean.ipynb - The process to take the raw data and create training and prediction datasets
+ fpl_predict_fastai_tabular.ipynb - The process to train a model and apply it to predict the remainder of the 2019/20 season

Finally, predictions.csv contains the latest predictions for the remainder of th 2019/20 season.
