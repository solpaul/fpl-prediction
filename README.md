# FPL Prediction
This is a project to generate ongoing player forecasts for Fantasy Premier League. Credit to https://github.com/vaastav/Fantasy-Premier-League, from which I have taken the script to scrape data from the FPL site each week.

Data and utility functions are contained in the fpl-predictor module. The following data can be found in the data directory:
+ Player data for each gameweek since the start of the 2016/17 season (one folder with various csv files for each season, including fixtures)
+ teams.csv - Team names and global IDs, plus specific IDs for each season
+ train_v8.csv - The current training dataset containing all historic data
+ remaining_season.csv - A dataset with rows for each player's remaining fixtures in the current season, for use in predicting the remainder of the current season each week

I have written notebooks that go through the entire process taken to train, validate and select the forecast model. I am in the process of transferring these to Google Colab (links given) - these online notebooks are the most up to date in terms of model training (and should run without any issues, tell me on [twitter](https://twitter.com/solpaul7) if not), but are not yet as well commented as the original versions:
+ [00_fpl_features](https://colab.research.google.com/drive/1xB8uQTJh8Q2geJXEieaz-KoOUPUqyslP?usp=sharing) - Explore the training dataset (fields, data types, null values, etc.), write functions to generate window/lagging features (e.g. points per game for each player over the last 5 fixtures), and understand the approach to assessing the performance of models (validation)
+ [01_fpl_predict_baseline](https://colab.research.google.com/drive/1v5aUlrodDsHsSjNOuJeTY9hPFPLQlwup?usp=sharing) - Build a simple model to predict players for use as a baseline, and write a function to transform the training data into a format that we can easily use to perform validation
+ [02_fpl_predict_random_forest](https://colab.research.google.com/drive/1KFx7qyIMui8hha66LD7-L7Cv0i0y-eBs?usp=sharing) - Build a random forest model and validate its performance
+ [03_fpl_predict_xgboost](https://colab.research.google.com/drive/1B7fYyLZ9KsWfz3yTdrubC5qug4QKpQs_?usp=sharing) - Build an XGBoost model, including parameter search, and validate its performance
+ [04_fpl_predict_fastai2_tabular](https://colab.research.google.com/drive/1Rm5h-4fLInSOb_7wlXdR2krfSIXPGjvO?usp=sharing) - Build a neural network model with embeddings for categorical features and validate its performance
+ [05_fpl_predict_lstm](https://colab.research.google.com/drive/1Z46MeQt9PSkAgZcIFgOuX3S1xBlUHD1v?usp=sharing) - Build a sequence model with LSTMs in tensorflow and validate its performance

These models have been validated by looking at their performance each gameweek of the 2020/21 season. For each gameweek we fit the model using all historical data prior to that week, and then calculate the mean absolute error for the following 6 gameweeks. The performance of each model across the season is summarised in the following chart:

![comparison chart](charts/comparison_chart.png)

The LSTM model is the top performer currently, so this is the approach used to generate forecasts prior to each gameweek.

There are a further three jupyter notebooks:
+ initial_fpl_data_clean.ipynb - The original process to take the raw data and create training and prediction datasets
+ update_data_weekly.ipynb - The notebook previously run each week with the XGBoost model to take the raw data and create updated training and prediction datasets
+ fpl_predict_fastai_tabular.ipynb - The notebook run each week to train a model using all historical data and predict the remainder of the current season

And one supporting python script:
+ util.py - various functions used throughout, all of which are described in one of the above process notebooks.

### Local setup:

I'd recommend using Colab, it's free and you don't need to worry much about setup. But if you want to run this locally or on your own cloud machine then for the non neural net notebooks I downloaded and installed anaconda and then set up an enivronment with jupyter, xgboost, pandas, matplotlib, requests, lxml and dtreeviz as follows:

```
conda create -n fplenv python=3.7
conda activate fplenv
conda install jupyter py-xgboost pandas matplotlib requests lxml
pip install dtreeviz
```

For neural nets (04_fpl_predict_fastai2_tabular.ipynb) I use fastai/PyTorch (installation instructions at https://docs.fast.ai/#Installing) and for sequence models I use tensorflow 2, but I recommend using a cloud instance with a GPU (e.g. AWS, GCP, Paperspace (has a fastai container), etc.).
