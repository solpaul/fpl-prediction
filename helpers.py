# helpers.py
import pandas as pd
import numpy as np
import requests
import lxml.html as lh



def build_players(path, season_paths, season_names, teams):
    # read in player information for each season and add to list
    season_players = []

    for season_path in season_paths:
        players = pd.read_csv(season_path/'players_raw.csv', 
                               usecols=['first_name', 'second_name', 'id', 
                                        'team_code', 'element_type', 'now_cost',
                                        'chance_of_playing_next_round'])
        season_players.append(players)

    if len(season_players) > 1:
        # two danny wards in 1819, rename the new one
        season_players[2].loc[143, 'second_name'] = 'Ward_2'

    # create full name field for each player
    for players in season_players:
        players['full_name'] = players['first_name'] + '_' + players['second_name']
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
                                    'chance_of_playing_next_round': 'play_proba_' + season},
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
                  'Sheffield Utd.': 'Sheffield United',
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
                           'relative_market_value_x', 'was_home', 'total_points']]
    df_season.columns = ['player', 'gw', 
                          'position', 'minutes', 'team', 
                          'opponent_team', 'relative_market_value_team', 
                          'relative_market_value_opponent_team', 'was_home', 'total_points']
    df_season['season'] = season
    df_season['position'] = df_season['position'].astype(int)
    
    return df_season