# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
import luigi as lg
import csv

class ReadEventData(d6t.tasks.TaskCache):  # read Metrica event data for game_id and return as a DataFrame
    datadir = lg.Parameter(default=r'C:\Users\USUARIO\Hugo\Sports_Analytics_Personal\Football\Data\Tracking\Metrica')
    gameid = lg.IntParameter()

    def run(self):
        eventfile = '/Sample_Game_%d/Sample_Game_%d_RawEventsData.csv' % (self.gameid, self.gameid)  # filename
        events = pd.read_csv('{}/{}'.format(self.datadir, eventfile))  # read data

        self.save(events)

class ReadTrackingData(d6t.tasks.TaskCache):  # read Metrica tracking data for game_id and return as a DataFrame
    datadir = lg.Parameter(default=r'C:\Users\USUARIO\Hugo\Sports_Analytics_Personal\Football\Data\Tracking\Metrica')
    gameid = lg.IntParameter()
    team = lg.Parameter()  # 'Home' or 'Away'

    def run(self):
        teamfile = '/Sample_Game_%d/Sample_Game_%d_RawTrackingData_%s_Team.csv' % (self.gameid, self.gameid, self.team)

        # First:  deal with file headers so that we can get the player names correct
        csvfile = open('{}/{}'.format(self.datadir, teamfile), 'r')  # create a csv file reader
        reader = csv.reader(csvfile)
        teamnamefull = next(reader)[3].lower()
        # construct column names
        jerseys = [x for x in next(reader) if x != '']  # extract player jersey numbers from second row
        columns = next(reader)
        for i, j in enumerate(jerseys):  # create x & y position column headers for each player
            columns[i * 2 + 3] = "{}_{}_x".format(self.team, j)
            columns[i * 2 + 4] = "{}_{}_y".format(self.team, j)
        columns[-2] = "ball_x"  # column headers for the x & y positions of the ball
        columns[-1] = "ball_y"
        # Second: read in tracking data and place into pandas Dataframe
        tracking = pd.read_csv('{}/{}'.format(self.datadir, teamfile), names=columns, index_col='Frame', skiprows=3)

        self.save(tracking)