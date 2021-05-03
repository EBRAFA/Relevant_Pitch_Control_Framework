# Library imports
import numpy as np
import d6tflow as d6t
import luigi

# Project imports
import dataimport as di
import preprocessing as pre
import parameters as pr
import player as pl

class GetPlayerIDs(d6t.tasks.TaskPickle):
    gameid = luigi.Parameter()

    def requires(self):
        return {'tracking_home': di.ReadTrackingData(gameid=self.gameid, team='Home'),
                'tracking_away': di.ReadTrackingData(gameid=self.gameid, team='Away')}

    def run(self):
        tracking_home = self.input()['tracking_home'].load()
        tracking_away = self.input()['tracking_away'].load()

        home_ids = np.unique([c[:-2] for c in tracking_home.columns if c[:4] == 'Home'])
        away_ids = np.unique([c[:-2] for c in tracking_away.columns if c[:4] == 'Away'])

        self.save({'home_ids': home_ids, 'away_ids': away_ids})

class PitchInfo(d6t.tasks.TaskCache):
    '''
    Interface for the other tasks that use these basic parameters
    '''

    field_dimen = luigi.Parameter(default=(106.0, 68.0))  # Field dimensions
    n_grid_cells_x = luigi.Parameter(default=50)  # Number of grid cells in the x axis (number of cells in y axis is dependent of this parameter)

class TrackingTupleInfo(d6t.tasks.TaskCache):
    '''

    Interface for the other tasks that use these basic parameters
    '''

    gameid = luigi.Parameter()
    rownumber = luigi.IntParameter()  # Index of tracking data tuple (default=None so it is not always required to fill this parameter)

@d6t.inherits(TrackingTupleInfo)
class GetRowTracking(d6t.tasks.TaskCachePandas):
    team = luigi.Parameter()  # 'Home' or 'Away'

    def requires(self):
        return self.clone(pre.PrepData)

    def run(self):
        if self.team == 'Home':
            tracking = self.input()['tracking_home'].load()

        else:
            tracking = self.input()['tracking_away'].load()

        self.save(tracking.loc[self.rownumber, tracking.columns])

@d6t.inherits(TrackingTupleInfo)
class GetBallPosition(d6t.tasks.TaskCache):
    '''
    Returns the position of the ball in a specific frame
    '''

    def requires(self):
        return GetRowTracking(gameid=self.gameid, rownumber=self.rownumber, team='Home')  # The team doesn't matter here. Dataframes for both teams have this information

    def run(self):
        tracking = self.input().load()

        ball_start_pos = np.array([tracking['ball_x'], tracking['ball_y']])

        self.save(ball_start_pos)

@d6t.inherits(TrackingTupleInfo)
class GetTeamInPossession(d6t.tasks.TaskCache):
    '''
    Returns the team in possession of the ball in a specific frame
    '''

    def requires(self):
        return self.clone(pre.PrepData)

    def run(self):
        tracking_home = self.input()['tracking_home'].load()

        team_in_possession = tracking_home.loc[self.rownumber, 'Team']

        self.save(team_in_possession)

@d6t.inherits(TrackingTupleInfo)
class InitialisePlayers(d6t.tasks.TaskCache):
    remove_offsides_players = luigi.BoolParameter(default=True)

    def requires(self):
        return {'tracking_home': GetRowTracking(gameid=self.gameid, team='Home', rownumber=self.rownumber),
                'tracking_away': GetRowTracking(gameid=self.gameid, team='Away', rownumber=self.rownumber),
                'ball_start_pos': GetBallPosition(gameid=self.gameid, rownumber=self.rownumber),
                'team_in_possession': GetTeamInPossession(gameid=self.gameid, rownumber=self.rownumber),
                'params': pr.PCModelParameters()}

    def run(self):
        tracking_home = self.input()['tracking_home'].load()  # Row of tracking data
        tracking_away = self.input()['tracking_away'].load()  # Row of tracking data
        ball_start_pos = self.input()['ball_start_pos'].load()
        team_in_possession = self.input()['team_in_possession'].load()
        params = self.input()['params'].load()  # Model parameters

        attacking_players = []
        defending_players = []

        # get player  ids
        player_ids = np.unique([c.split('_')[1] for c in tracking_home.keys() if c[:4] == 'Home'])
        for p in player_ids:
            # create a player object for player_id 'p'
            team_player = pl.player(p, tracking_home, 'Home', params)
            if team_player.inframe:
                if team_in_possession == 'Home':
                    attacking_players.append(team_player)
                else:
                    defending_players.append(team_player)

        # get player  ids
        player_ids = np.unique([c.split('_')[1] for c in tracking_away.keys() if c[:4] == 'Away'])
        for p in player_ids:
            # create a player object for player_id 'p'
            team_player = pl.player(p, tracking_away, 'Away', params)
            if team_player.inframe:
                if team_in_possession == 'Away':
                    attacking_players.append(team_player)
                else:
                    defending_players.append(team_player)

        if self.remove_offsides_players:
            offsides_players = []

            # find the second-last defender
            x_defending_players = []
            for player in defending_players:
                x_defending_players.append(player.position[0])

            x_defending_players = np.sort(x_defending_players)
            second_last_defender_x = x_defending_players[-2]

            for player in attacking_players:
                position = player.position
                # if player is nearer to the opponent's goal than the ball
                if position[0] > ball_start_pos[0] and position[0] > second_last_defender_x:
                    offsides_players.append(player)

            for op in offsides_players:
                attacking_players.remove(op)

        self.save({'attacking_players': attacking_players, 'defending_players': defending_players})

class FrameInfo(d6t.tasks.TaskCache):
    '''
    Interface with attributes of a single frame
    '''

    attacking_players = luigi.Parameter(default=None)  # List with attacking players
    defending_players = luigi.Parameter(default=None)  # List with defending players