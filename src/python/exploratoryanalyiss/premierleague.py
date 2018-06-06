import pandas as pd

import matplotlib.pyplot as plt

'''
Data exploration of premier league matches. These matches are supposedly
used in testing RJMCMC in Green, 2009.
'''

premier0506 = pd.read_csv(
    '../../../resources/data/premierleague/200506.csv')
premier0607 = pd.read_csv(
    '../../../resources/data/premierleague/200607.csv')
premier0708 = pd.read_csv(
    '../../../resources/data/premierleague/200708.csv')

framesFTHG = [premier0506.loc[:, ['FTHG', 'Date']],
              premier0607.loc[:, ['FTHG', 'Date']],
              premier0708.loc[:, ['FTHG', 'Date']]]

framesFTAG = [premier0506.loc[:, ['FTAG', 'Date']],
              premier0607.loc[:, ['FTAG', 'Date']],
              premier0708.loc[:, ['FTAG', 'Date']]]

frames = [premier0506,
          premier0607,
          premier0708]

# home goals
# HGs = pd.concat(framesFTHG, ignore_index=True)
# # away goals
# AGs = pd.concat(framesFTAG, ignore_index=True)

premier = pd.concat(frames, ignore_index=True)
premier.loc[:, 'TotalG'] = premier.loc[:, 'FTHG'] + premier.loc[:, 'FTAG']

premier.plot(y='TotalG')

plt.show()

