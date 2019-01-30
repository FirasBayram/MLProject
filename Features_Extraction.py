import pandas as pd 
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import math


#1.Loading the data
#create the dataframe of the matches to be used for training, the data are contained in 'data' folder
path = 'data'
all_files = glob.glob(os.path.join(path, '*.csv'))
all_matches = pd.concat((pd.read_csv(f, header = 0) for f in all_files), ignore_index=True)



#2.Cleaning the data
#2.a. remove all the matches with missing data, or ended with a walkover (in this case no duration will be recorded for this match)
all_matches['w_ace'].replace('', np.nan, inplace=True)
all_matches = all_matches.dropna(axis=0, subset = ['w_ace'])
all_matches['minutes'].replace('', np.nan, inplace=True)
all_matches = all_matches.dropna(axis=0, subset = ['minutes'])

#2.b. extract the tournament ID from the first column by removing the year prefix, and then keep only the numeric part of the ID of the ones which starts with letter
all_matches['tourney_id'] = all_matches['tourney_id'].astype(str).str[5:]
all_matches['tourney_id'] = all_matches['tourney_id'].str.extract('(\d+)')
all_matches['tourney_id'] = all_matches['tourney_id'].astype(int)

#2.c. replace empty cells of winner_rank or loser_rank --unranked players-- with the max(rank)+1, the max is 1890
all_matches['winner_rank'] = np.where(all_matches['winner_rank'].isnull(), 1891, all_matches['winner_rank'])
all_matches['loser_rank'] = np.where(all_matches['loser_rank'].isnull(), 1891, all_matches['loser_rank'])

#2.d. replace empty cells of winner_seed and loser_seed with the player's rank
all_matches['winner_seed'] = np.where(all_matches['winner_seed'].isnull(), all_matches['winner_rank'], all_matches['winner_seed'])
all_matches['loser_seed'] = np.where(all_matches['loser_seed'].isnull(), all_matches['loser_rank'], all_matches['loser_seed'])

#2.e. removing Challenger tournaments included
all_matches  = all_matches[all_matches['tourney_level']!='C']

#2.f. remove Davis cup matches as their stats were not recorded before 2016
all_matches  = all_matches[all_matches['tourney_level']!='D']

#2.g. replace empty cells of winner_age or loser_age with the average of winner's --loser's-- age 
all_matches['winner_age'] = np.where(all_matches['winner_age'].isnull(), np.mean(all_matches['winner_age']), all_matches['winner_age'])
all_matches['loser_age'] = np.where(all_matches['loser_age'].isnull(), np.mean(all_matches['loser_age']), all_matches['loser_age'])

#2.h. remove matches with missing service games number data
all_matches  = all_matches[~((all_matches['w_SvGms']==0) & (all_matches['minutes']>10))]

#2.i. Sorting the data
#Sort the matches by tournament date and match number
all_matches=all_matches.sort_values(['tourney_date', 'match_num'], ascending=[True, True])

#2.j. Delete columns
#Delete irrelevant columns from the dataframe
all_matches.drop(['tourney_name', 'draw_size','tourney_date','match_num','winner_entry','winner_name','winner_hand','winner_ht','winner_ioc','winner_rank_points','loser_entry'
                  ,'loser_name','loser_hand','loser_ht','loser_ioc','loser_rank_points'], axis=1, inplace=True)

#2.k. Save the cleansed matches in a csv file
all_matches.to_csv("cleansed.csv",index = False )




#3. Applying map functions
#3.a map the surface dimension, we will map each class of surfaces to a number
surface_map = {'Hard': 1, 'Carpet':1, 'Grass': 2, 'Clay':3}
all_matches=all_matches.replace({'surface': surface_map})

#3.b map the tournament level dimension, we will map each class of levels to a number
surface_map = {'A': 1, 'M':2, 'F': 3, 'G':4}
all_matches=all_matches.replace({'tourney_level': surface_map})

#3.c map the round dimension, we will map each class of rounds to a number
surface_map = {'R128': 128, 'R64':64, 'R32': 32, 'R16':16, 'RR':8, 'QF':8, 'SF':4, 'F':2, 'BR':4}
all_matches=all_matches.replace({'round': surface_map})

#3.d. Save the mapped data in a csv file
all_matches.to_csv("mapped.csv",index = False )




#4. Creating players' profiles
#4.a. extract the players' IDs included in our dataset
wid = set(all_matches['winner_id'])
lid = set(all_matches['loser_id'])

#4.b extract the IDs of all the players
pid = set(wid | lid)

#4.c create player's serve and return profiles
d = []
for id in pid:
    df = all_matches[(all_matches.winner_id == id) | (all_matches.loser_id == id)]
    #4.c.1 Create serve profile
    #aces list
    aces = np.where(df['winner_id']==id, df['w_ace'], df['l_ace'])
    #double faults list    
    dfs = np.where(df['winner_id']==id, df['w_df'], df['l_df'])
    #service points list
    svpts=np.where(df['winner_id']==id, df['w_svpt'], df['l_svpt'])
    #service games list
    svgms=np.where(df['winner_id']==id, df['w_SvGms'], df['l_SvGms'])
    #break points saved list
    bps=np.where(df['winner_id']==id, df['w_bpSaved'], df['l_bpSaved'])
    #break points faced list
    bpf=np.where(df['winner_id']==id, df['w_bpFaced'], df['l_bpFaced'])
    bps_avg = np.mean(bps)
    bpf_avg=np.mean(bpf)
    svgms_avg = np.mean(svgms)
    
    #4.c.2 create return profile
    #aces faced list
    acesf = np.where(df['winner_id']==id, df['l_ace'], df['w_ace'])
    #return points received list
    rtpts=np.where(df['winner_id']==id, df['l_svpt'], df['w_svpt'])
    #return games list
    rtgms=np.where(df['winner_id']==id, df['l_SvGms'], df['w_SvGms'])
    #break points chances list
    bpc=np.where(df['winner_id']==id, df['l_bpFaced'], df['w_bpFaced'])  
    
    d.append((id,np.mean(aces),np.mean(dfs),np.mean(svpts),np.mean(svgms),1-(bpf_avg-bps_avg)/svgms_avg,np.mean(bpf),np.mean(acesf),np.mean(rtpts),np.mean(rtgms),np.mean(bpc)))
    
    #4.c.3. create new dataframe to contain each player's id and his serve and return profiles    
    profiles = pd.DataFrame(d, columns=('id', 'aces_avg','dfs_avg','svpts_avg','svgms_avg','svgms_pct_avg','bpf_avg','aces_faced_avg','return_pts_avg','return_gms_avg','bpc_avg'))             
    
profiles.to_csv("profiles.csv",index = False )

#4.d. create two datframes to distinguish between the data which are coming from winner's or loser's id
wdf = profiles.copy()
wdf.columns = [str(col) + '_w' for col in wdf.columns]
ldf = profiles.copy()
ldf.columns = [str(col) + '_l' for col in ldf.columns]

#4.e. merge our dataset to include the players' profiles
all_matches = pd.merge(all_matches, wdf, left_on='winner_id', right_on='id_w', how='left')
all_matches = pd.merge(all_matches, ldf, left_on='loser_id', right_on='id_l', how='left')
all_matches = all_matches.drop(['id_w', 'id_l'], axis=1)

#4.f. extract new features by calculating the absolute values of the differnece between the two players' profiles 
all_matches['aces_avg_abs']=abs(all_matches['aces_avg_w']-all_matches['aces_avg_l'])
all_matches['dfs_avg_abs']=abs(all_matches['dfs_avg_w']-all_matches['dfs_avg_l'])
all_matches['svpts_avg_abs']=abs(all_matches['svpts_avg_w']-all_matches['svpts_avg_l'])
all_matches['svgms_avg_abs']=abs(all_matches['svgms_avg_w']-all_matches['svgms_avg_l'])
all_matches['svgms_pct_avg_abs']=abs(all_matches['svgms_pct_avg_w']-all_matches['svgms_pct_avg_l'])
all_matches['bpf_avg_abs']=abs(all_matches['bpf_avg_w']-all_matches['bpf_avg_l'])
all_matches['aces_faced_avg_abs']=abs(all_matches['aces_faced_avg_w']-all_matches['aces_faced_avg_l'])
all_matches['return_pts_avg_abs']=abs(all_matches['return_pts_avg_w']-all_matches['return_pts_avg_l'])
all_matches['return_gms_avg_abs']=abs(all_matches['return_gms_avg_w']-all_matches['return_gms_avg_l'])
all_matches['bpc_avg_abs']=abs(all_matches['bpc_avg_w']-all_matches['bpc_avg_l'])
all_matches['rank_abs'] = abs((all_matches['winner_rank']+all_matches['winner_seed'])-(all_matches['loser_rank']+all_matches['loser_seed']))

#4.g. subset the datframe
all_matches = all_matches[['tourney_id','surface','tourney_level','rank_abs','round','score','minutes','aces_avg_abs','dfs_avg_abs','svpts_avg_abs','svgms_avg_abs','svgms_pct_avg_abs','bpf_avg_abs','aces_faced_avg_abs','return_pts_avg_abs','return_gms_avg_abs','bpc_avg_abs']]

all_matches.to_csv("matches_with_profiles.csv",index = False )




#5. Preparing the target variable 
#5.a normalize the minutes column to the average duration of the set in each match
all_matches['set_mins_avg'] = all_matches['minutes'] / all_matches.score.str.count('-')

#5.b show the distribution of the target variable In a box and whisker plot according to its quartiles
plt.boxplot(all_matches['set_mins_avg'])

#5.c map each entry of minutes column to the corresponding label
all_matches['label']= all_matches['set_mins_avg']
all_matches['label']= np.where( all_matches['label']<=np.mean(all_matches['set_mins_avg']) , -1 , all_matches['label'])
all_matches['label']= np.where( all_matches['label']>np.mean(all_matches['set_mins_avg']) , 1 , all_matches['label'])

#5.d remove unwanted columns
all_matches = all_matches.drop(['minutes' , 'score' ,'set_mins_avg'], axis =1 )

#5.e save the final dataset in csv file
all_matches.to_csv("dataset.csv",index = False )


#5.f save the description of the data in a csv file
all_matches.describe().to_csv("my_description.csv")


