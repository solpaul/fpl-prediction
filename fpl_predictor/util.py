"""Utility functions for fpl_predictor module."""
# util.py
import pandas as pd
import numpy as np
import requests
import lxml.html as lh
import math
import json
from bs4 import BeautifulSoup
import re
import backoff



def build_players(path, season_paths, season_names, teams):
    # read in player information for each season and add to list
    season_players = []

    for season_path in season_paths:
        players = pd.read_csv(season_path/'players_raw.csv', 
                               usecols=['first_name', 'second_name', 'web_name', 'id', 
                                        'team_code', 'element_type', 'now_cost',
                                        'chance_of_playing_next_round'])
        season_players.append(players)

    if len(season_players) > 1:
        # two danny wards in 1819, rename the new one
        season_players[2].loc[143, 'second_name'] = 'Ward_2'

    # create full name field for each player
    for players in season_players:
        players['full_name'] = players['first_name'] + ' ' + players['second_name']
        players.drop(['first_name', 'second_name'], axis=1, inplace=True)

    # create series of all unique player names
    all_players = pd.concat(season_players, axis=0, ignore_index=True, sort=False)
    all_players = pd.DataFrame(all_players['full_name'].drop_duplicates())

    # create player dataset with their id, team code and position id for each season
    for players, season in zip(season_players, season_names):
        all_players = all_players.merge(players, on='full_name', how='left')
        all_players.rename(index=str,
                           columns={'id':'id_' + season,
                                    'team_code':'team_' + season,
                                    'element_type': 'position_' + season,
                                    'now_cost': 'cost_' + season,
                                    'chance_of_playing_next_round': 'play_proba_' + season,
                                    'web_name': 'web_name_' + season},
                           inplace=True)
        
    return all_players


# function to scrape market value of premier league teams at the start of each season
# from www.transfermarkt.com
def build_season_mv(season, header_row, team_rows):
    
    # url for page with team market value at start of season
    url=r'https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id=' + '20' + season[0:2]
    
    #Create a handle, page, to handle the contents of the website
    page = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    #Store the contents of the website under doc
    doc = lh.fromstring(page.content)
    
    #Parse data that are stored between <tr>..</tr> of HTML
    tr_elements = doc.xpath('//tr')
    
    #Create empty list
    col=[]
    i=0

    #For each row, store each first element (header) and an empty list
    for t in tr_elements[header_row]:
        i+=1
        name=t.text_content()
        col.append((name,[]))
        
    #data is stored on the second row onwards
    for j in team_rows:
        #T is our j'th row
        T=tr_elements[j]

        #If row is not of size 10, the //tr data is not from our table 
        if len(T)!=10:
            break

        #i is the index of our column
        i=0

        #Iterate through each element of the row
        for t in T.iterchildren():
            data=t.text_content() 
            #Check if row is empty
            if i>0:
            #Convert any numerical value to integers
                try:
                    data=int(data)
                except:
                    pass
            #Append the data to the empty list of the i'th column
            col[i][1].append(data)
            #Increment i for the next column
            i+=1
        
    # create market values dataframe
    Dict={title:column for (title,column) in col}
    df=pd.DataFrame(Dict)
        
    # convert market value string to float for millions of euros
    values = [float(item[0].replace(',', '.').replace('€', '').replace('bn', '').replace('m', '')) 
              for item in df['Total MV'].str.split(" ", 1)]
    values = [item*10**3 if item < 3 else item for item in values]
    
    # to remove effect of inflation, take relative market value for each season
    values = values/np.mean(values)
    
    # market value website has shortened team names
    # lookup dictionary of full names
    team_names = {'Man City': 'Manchester City',
                  'Spurs': 'Tottenham Hotspur',
                  'Man Utd': 'Manchester United',
                  'Leicester': 'Leicester City',
                  'West Ham': 'West Ham United',
                  'Wolves': 'Wolverhampton Wanderers',
                  'Brighton': 'Brighton and Hove Albion',
                  'Newcastle': 'Newcastle United',
                  'Sheff Utd': 'Sheffield United',
                  'West Brom': 'West Bromwich Albion',
                  'Swansea': 'Swansea City',
                  'Huddersfield': 'Huddersfield Town',
                  'Cardiff': 'Cardiff City'}
    
    # create smaller dataframe with team names, market value and the season
    df = df[['name']]
    df.replace(team_names, inplace=True)
    df['relative_market_value'] = values
    df['season'] = season
    
    return df

# function to create season training dataset
# each player has a row for each gameweek
def build_season(path, season, all_players, teams, teams_mv, gw=range(1, 39)):
    
    # season specific list and strings to use for merging
    df_season = []
    id_season = 'id_' + season
    id_team = 'team_' + season
    id_position = 'position_' + season
    
    # read in each gameweek and append to season list
    for i in gw:
        gw = 'gws/gw' + str(i) + '.csv'
        gw_df = pd.read_csv(path/gw, encoding='latin')
        gw_df['gw'] = i
        df_season.append(gw_df)
    
    # concatenate entire season
    df_season = pd.concat(df_season, axis=0)
    
    # remove team columns
    df_season.drop('team', axis=1, inplace=True)
    
    # join to player, team and team market value datasets to create season training set
    df_season = df_season.merge(all_players, left_on='element', right_on=id_season, how='left')
    df_season = df_season.merge(teams, left_on='opponent_team', right_on=id_team, how='left')
    df_season = df_season.merge(teams, left_on=id_team + '_x', right_on='team_code', how='left')
    df_season = df_season.merge(teams_mv[teams_mv['season'] == season], 
                                left_on='team_x', right_on='name', how='left')
    df_season = df_season.merge(teams_mv[teams_mv['season'] == season], 
                                left_on='team_y', right_on='name', how='left')
    df_season = df_season[['full_name', 'gw', 
                           id_position, 'minutes', 'team_y', 
                           'team_x', 'relative_market_value_y', 
                           'relative_market_value_x', 'was_home', 'total_points',
                           'assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 
                           'goals_conceded', 'goals_scored', 'ict_index', 'influence', 
                           'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 
                           'saves', 'selected', 'team_a_score', 'team_h_score', 'threat', 
                           'transfers_balance', 'transfers_in', 'transfers_out', 
                           'yellow_cards', 'kickoff_time']]
    df_season.columns = ['player', 'gw', 
                         'position', 'minutes', 'team', 
                         'opponent_team', 'relative_market_value_team', 
                         'relative_market_value_opponent_team', 'was_home', 'total_points',
                         'assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 
                         'goals_conceded', 'goals_scored', 'ict_index', 'influence', 
                         'own_goals', 'penalties_missed', 'penalties_saved', 'red_cards', 
                         'saves', 'selected', 'team_a_score', 'team_h_score', 'threat', 
                         'transfers_balance', 'transfers_in', 'transfers_out', 
                         'yellow_cards', 'kickoff_time']
    df_season['season'] = season
    df_season['position'] = df_season['position'].astype(int)
    
    return df_season


### functions to scrape fbref data
# get stats (given by features) for all players/teams
# for season pages given by season_urls
def get_stats(features, season_urls, url_base):
    seasons_list = []
    
    for url, season_name in zip(season_urls['url'], season_urls['season']):
        print(season_name)
        print(url)
        season_table = get_table(url)
        team_urls = get_urls(['squad'], 'squad', season_table, url_base)
        season_df = get_season_stats(features, team_urls, url_base)
        season_df.insert(0, 'season', season_name)
        seasons_list.append(season_df)
    
    df = pd.concat(seasons_list, ignore_index=True)
    return df

# get stats (given by features) for all players 
# for team pages given by team_urls
def get_season_stats(features, team_urls, url_base):
    teams_list = []
    
    for url, team_name in zip(team_urls['url'], team_urls['squad']):
        print(team_name)
        print(url)
        team_table = get_table(url)
        player_urls = get_urls(['player', 'games'], 'matches', team_table, url_base)
        team_df = get_team_stats(features, player_urls)
        teams_list.append(team_df)
        
    df = pd.concat(teams_list, ignore_index=True)
    return df

# get stats (given by features)
# for all players given by player_urls
def get_team_stats(features, player_urls):
    players_list = []
    player_urls_played = player_urls[player_urls['games'] != '0']

    for url, player_name in zip(player_urls_played['url'], player_urls_played['player']):
        print(player_name)
        player_table = get_table(url)
        player_df = get_player_stats(features, player_table)
        player_df.insert(0, 'player', player_name)
        if len(player_df) > 0:
            players_list.append(player_df[player_df['comp'] == 'Premier League'])

    df = pd.concat(players_list, ignore_index=True)
    return df

# get game by game player stats for one season
# date always included by default
def get_player_stats(features, player_table):
    pre_dict = dict()    
    table_rows = player_table.find_all('tr')    
    for row in table_rows:
        if(row.find('td',{"data-stat":'xg'}) != None):
            date = row.find('th',{"data-stat":'date'}).text.strip().encode().decode("utf-8")
            if date != '':
                if 'date' in pre_dict:
                    pre_dict['date'].append(date)
                else:
                    pre_dict['date'] = [date]
                for f in features:
                    text = row.find('td',{"data-stat":f}).text.strip().encode().decode("utf-8")
                    if f in pre_dict:
                        pre_dict[f].append(text)
                    else:
                        pre_dict[f] = [text]
    df = pd.DataFrame.from_dict(pre_dict)
    return df

def backoff_hdlr(details):
    print ("Backing off {wait:0.1f} seconds afters {tries} tries "
           "calling function {target} with args {args} and kwargs "
           "{kwargs}".format(**details))

# get a table on any page, defaults to first one
@backoff.on_exception(
    backoff.expo,
    (IndexError, requests.exceptions.RequestException),
    max_tries=5,
    on_backoff=backoff_hdlr
)
def get_table(url, table_no=0):
    res = requests.get(url)
    ## The next two lines get around the issue with comments breaking the parsing.
    comm = re.compile("<!--|-->")
    soup = BeautifulSoup(comm.sub("",res.text),'lxml')
    all_tables = soup.findAll("tbody")    
    table = all_tables[table_no]
    return table

# get text and urls for a given field
def get_urls(text_fields, url_field, table, url_base):
    pre_dict = dict()    
    table_rows = table.find_all('tr')
    for row in table_rows:
        if(row.find('th',{"scope":"row"}) != None):
            for f in text_fields:
                if(row.find('th',{"data-stat":f}) != None):
                    text = row.find('th',{"data-stat":f}).text.strip().encode().decode("utf-8")
                else:
                    text = row.find('td',{"data-stat":f}).text.strip().encode().decode("utf-8")
                if f in pre_dict:
                    pre_dict[f].append(text)
                else: 
                    pre_dict[f] = [text]
                
            if row.find('th',{"data-stat":url_field}) != None:
                url = url_base + row.find('th',{"data-stat":url_field}).find('a').get('href')
#                 url = url_base + row.find('a')['href']
            else:
                url = url_base + row.find('td',{"data-stat":url_field}).find('a').get('href')

            if 'url' in pre_dict:
                pre_dict['url'].append(url)
            else: 
                pre_dict['url'] = [url]
                
    df = pd.DataFrame.from_dict(pre_dict)
    return df

# function to generate player lag features
# returns a new dataframe and a list of the new per game features
def player_lag_features(df, features, lags):    
    df_new = df.copy()
    player_lag_vars = []
    
    # need minutes for per game stats, add to front of list
    features.insert(0, 'minutes')

    # calculate totals for each lag period
    for feature in features:
        for lag in lags:
            feature_name = feature + '_last_' + str(lag)
            minute_name = 'minutes_last_' + str(lag)
            
            if lag == 'all':
                df_new[feature_name] = df_new.groupby(['player'])[feature].apply(lambda x: x.cumsum() - x)
            else: 
                df_new[feature_name] = df_new.groupby(['player'])[feature].apply(lambda x: x.rolling(min_periods=1, 
                                                                                            window=lag+1).sum() - x)
            if feature != 'minutes':

                pg_feature_name = feature + '_pg_last_' + str(lag)
                player_lag_vars.append(pg_feature_name)
                
                df_new[pg_feature_name] = 90 * df_new[feature_name] / df_new[minute_name]
                
                # some cases of -1 points and 0 minutes cause -inf values
                # change these to NaN
                df_new[pg_feature_name] = df_new[pg_feature_name].replace([np.inf, -np.inf], np.nan)
            
            else: player_lag_vars.append(minute_name)
                
    return df_new, player_lag_vars

# function to generate team lag features
# returns a new dataframe and a list of the new per game features
def team_lag_features(df, features, lags):
    team_lag_vars = []
    df_new = df.copy()
    
    for feature in features:
        feature_team_name = feature + '_team'
        feature_conceded_team_name = feature_team_name + '_conceded'
        feature_team = (df.groupby(['team', 'season', 'gw',
                                   'kickoff_time', 'opponent_team'])
                        [feature].sum().rename(feature_team_name).reset_index())
        
        # join back for points conceded
        feature_team = feature_team.merge(feature_team,
                           left_on=['team', 'season', 'gw',
                                    'kickoff_time', 'opponent_team'],
                           right_on=['opponent_team', 'season', 'gw',
                                     'kickoff_time', 'team'],
                           how='left',
                           suffixes = ('', '_conceded'))
                
        feature_team.drop(['team_conceded', 'opponent_team_conceded'], axis=1, inplace=True)
                
        for lag in lags:
            feature_name = feature + '_team_last_' + str(lag)
            feature_conceded_name = feature + '_team_conceded_last_' + str(lag)
            pg_feature_name = feature + '_team_pg_last_' + str(lag)
            pg_feature_conceded_name = feature + '_team_conceded_pg_last_' + str(lag)
            
            team_lag_vars.extend([pg_feature_name])#, pg_feature_conceded_name])
            
            if lag == 'all':
                feature_team[feature_name] = (feature_team.groupby('team')[feature_team_name]
                                              .apply(lambda x: x.cumsum() - x))
                
                feature_team[feature_conceded_name] = (feature_team.groupby('team')[feature_conceded_team_name]
                                              .apply(lambda x: x.cumsum() - x))
                
                feature_team[pg_feature_name] = (feature_team[feature_name]
                                                 / feature_team.groupby('team').cumcount())
                
                feature_team[pg_feature_conceded_name] = (feature_team[feature_conceded_name]
                                                 / feature_team.groupby('team').cumcount())
                
            else:
                feature_team[feature_name] = (feature_team.groupby('team')[feature_team_name]
                                              .apply(lambda x: x.rolling(min_periods=1, 
                                                                         window=lag + 1).sum() - x))
                
                feature_team[feature_conceded_name] = (feature_team.groupby('team')[feature_conceded_team_name]
                                              .apply(lambda x: x.rolling(min_periods=1, 
                                                                         window=lag + 1).sum() - x))
                
                feature_team[pg_feature_name] = (feature_team[feature_name] / 
                                                 feature_team.groupby('team')[feature_team_name]
                                                 .apply(lambda x: x.rolling(min_periods=1, 
                                                                            window=lag + 1).count() - 1))
                
                feature_team[pg_feature_conceded_name] = (feature_team[feature_conceded_name] / 
                                                 feature_team.groupby('team')[feature_conceded_name]
                                                 .apply(lambda x: x.rolling(min_periods=1, 
                                                                            window=lag + 1).count() - 1))
        
        df_new = df_new.merge(feature_team, 
                          on=['team', 'season', 'gw', 'kickoff_time', 'opponent_team'], 
                          how='left')
        
        df_new = df_new.merge(feature_team,
                 left_on=['team', 'season', 'gw', 'kickoff_time', 'opponent_team'],
                 right_on=['opponent_team', 'season', 'gw', 'kickoff_time', 'team'],
                 how='left',
                 suffixes = ('', '_opponent'))
        
        df_new.drop(['team_opponent', 'opponent_team_opponent'], axis=1, inplace=True)
        
    team_lag_vars = team_lag_vars + [team_lag_var + '_opponent' for team_lag_var in team_lag_vars]  

    return df_new, team_lag_vars
    
# functions to get validation set indexes
# training will always be from start of data up to valid-start
# first function to get the validation set points for a given season and gameweek
def validation_gw_idx(df, season, gw, length):
    
    valid_start = df[(df['gw'] == gw) & (df['season'] == season)].index.min()
    valid_end = df[(df['gw'] == min(gw+length-1, 38)) & (df['season'] == season)].index.max()

    return (valid_start, valid_end)

# function to calculate root mean squared error for preds and targs
def r_mse(pred,y): return round(math.sqrt(((pred-y)**2).mean()), 6)

# function to calculate mean absolute error for preds and targs
def mae(pred, y):  return round(abs(pred-y).mean(), 6)

# function to correct lag variables after validation point in a dataset
# We can adapt this approach to also create validation sets with lag features
# When making predictions for gw +2 and beyond we cannot use those weeks's lag features
# This would be leakage if we did
# Instead, each subsequent validation week should have the same lag values as the first
def create_lag_train(df, cat_vars, cont_vars, player_lag_vars, team_lag_vars, dep_var, valid_season, valid_gw, valid_len):

    # get all the lag data for the current season up to the first validation gameweek
    player_lag_vals = df[(df['season'] == valid_season) & 
                         (df['gw'] >= valid_gw)][['player', 'kickoff_time'] + player_lag_vars]
    
    team_lag_vals = df[(df['season'] == valid_season) & 
                       (df['gw'] >= valid_gw)][['team', 'kickoff_time'] + 
                                               [x for x in team_lag_vars if "opponent" not in x]].drop_duplicates()
                                               
    opponent_team_lag_vals = df[(df['season'] == valid_season) & 
                                (df['gw'] >= valid_gw)][['opponent_team', 'kickoff_time'] + 
                                                        [x for x in team_lag_vars if "opponent" in x]].drop_duplicates()
    
    
    # get the last available lag data for each player
    # for most it will be the first validation week
    # but sometimes teams have blank gameweeks
    # in these cases it will be the previous gameweek
    player_lag_vals = player_lag_vals[player_lag_vals['kickoff_time'] == 
                                      player_lag_vals.groupby('player')['kickoff_time'].transform('min')]
    team_lag_vals = team_lag_vals[team_lag_vals['kickoff_time'] == 
                                  team_lag_vals.groupby('team')['kickoff_time'].transform('min')]
    opponent_team_lag_vals = opponent_team_lag_vals[opponent_team_lag_vals['kickoff_time'] == 
                                                    opponent_team_lag_vals.groupby('opponent_team')['kickoff_time'].transform('min')]
                                                                    
    player_lag_vals = player_lag_vals.drop('kickoff_time', axis=1)
    team_lag_vals = team_lag_vals.drop('kickoff_time', axis=1)
    opponent_team_lag_vals = opponent_team_lag_vals.drop('kickoff_time', axis=1)
    
    
    # get the validation start and end indexes
    valid_start, valid_end = validation_gw_idx(df, valid_season, valid_gw, valid_len)
    train_idx = range(valid_start)
    valid_idx = range(valid_start, valid_end + 1)    

    # split out train and validation sets
    # do not include lag vars in validation set
    cat_vars = list(set(['opponent_team', 'team', 'player'] + cat_vars))
    
    train = df[cat_vars + cont_vars + 
               player_lag_vars + team_lag_vars + 
               dep_var].iloc[train_idx]
    valid = df[cat_vars + cont_vars + dep_var].iloc[valid_idx]

    # add in lag vars
    # will be the same for all validation gameweeks
    valid = valid.merge(player_lag_vals, on='player', how='left')
    valid = valid.merge(team_lag_vals, on='team', how='left')
    valid = valid.merge(opponent_team_lag_vals, on='opponent_team', how='left')
        
    # concatenate train and test again
    lag_train_df = pd.concat([train, valid], sort=True).reset_index(drop=True)

    return lag_train_df, train_idx, valid_idx