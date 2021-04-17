# Library imports
import numpy as np
import d6tflow as d6t
import luigi as lg
import scipy.signal as sg

# Project imports
import dataimport as di

# transforms data to metric coordinates, to single playing field and calculates player velocities
class PrepData(d6t.tasks.TaskPickle):
    gameid = lg.IntParameter()
    smoothing = lg.BoolParameter(default=True)
    window = lg.IntParameter(default=7)
    maxspeed = lg.IntParameter(default=12)

    persist = ['events', 'tracking_home', 'tracking_away']  # declare what you will save

    def requires(self):
            return {'events': di.ReadEventData(gameid=self.gameid),
                    'tracking_home': di.ReadTrackingData(gameid=self.gameid, team='Home'),
                    'tracking_away': di.ReadTrackingData(gameid=self.gameid, team='Away')}

    def run(self):
        events = self.input()['events'].load()
        tracking_home = self.input()['tracking_home'].load()
        tracking_away = self.input()['tracking_away'].load()
        field_dimen = (106., 68.)

        for data in [events, tracking_home, tracking_away]:
            # to metric coordinates
            x_columns = [c for c in data.columns if c[-1].lower() == 'x']
            y_columns = [c for c in data.columns if c[-1].lower() == 'y']
            data.loc[:, x_columns] = (data.loc[:, x_columns] - 0.5) * field_dimen[0]
            data.loc[:, y_columns] = -1 * (data.loc[:, y_columns] - 0.5) * field_dimen[1]

        # calculate players velocities
        for team in [tracking_home, tracking_away]:
            # Get the player ids
            player_ids = np.unique([c[:-2] for c in team.columns if c[:4] in ['Home', 'Away']])

            # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
            dt = team.loc[:, 'Time [s]'].diff()

            # index of first frame in second half
            second_half_idx = team.Period.idxmax(2)

            # estimate velocities for players in team
            for player in player_ids:  # cycle through players individually
                # difference player positions in timestep dt to get unsmoothed estimate of velicity
                vx = team[player + "_x"].diff() / dt
                vy = team[player + "_y"].diff() / dt

                if self.maxspeed > 0:
                    # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
                    raw_speed = np.sqrt(vx.values ** 2 + vy.values ** 2)
                    vx.values[raw_speed > self.maxspeed] = np.nan
                    vy.values[raw_speed > self.maxspeed] = np.nan

                if self.smoothing:
                    ma_window = np.ones(self.window) / self.window
                    # calculate first half velocity
                    vx.loc[:second_half_idx] = np.convolve(vx.loc[:second_half_idx], ma_window, mode='same')
                    vy.loc[:second_half_idx] = np.convolve(vy.loc[:second_half_idx], ma_window, mode='same')
                    # calculate second half velocity
                    vx.loc[second_half_idx:] = np.convolve(vx.loc[second_half_idx:], ma_window, mode='same')
                    vy.loc[second_half_idx:] = np.convolve(vy.loc[second_half_idx:], ma_window, mode='same')

                # put player speed in x,y direction, and total speed back in the data frame
                team.loc[:, player + "_vx"] = vx
                team.loc[:, player + "_vy"] = vy
                team.loc[:, player + "_speed"] = np.sqrt(vx ** 2 + vy ** 2)

        # filter "event", "tracking_home" and "tracking_away" dataframes using "selected_events['frames']"
        events = events[events['Type'].isin(['PASS', 'SHOT'])]
        tracking_home = tracking_home.iloc[list(events['Start Frame'] - 1)]
        tracking_away = tracking_away.iloc[list(events['Start Frame'] - 1)]

        # all attacking left-to-right
        event_columns = [c for c in events.columns if c[-1].lower() in ['x', 'y']]
        tracking_home_columns = [c for c in tracking_home.columns if c[-1].lower() in ['x', 'y']]
        tracking_away_columns = [c for c in tracking_away.columns if c[-1].lower() in ['x', 'y']]
        events.loc[events.Period == 2, event_columns] *= -1
        events.loc[events.Team == 'Away', event_columns] *= -1
        tracking_home.loc[:, 'Team'] = np.array(events['Team'])
        tracking_away.loc[:, 'Team'] = np.array(events['Team'])
        tracking_home.loc[(tracking_home['Team'] == 'Away') & (tracking_home['Period'] == 1), tracking_home_columns] *= -1
        tracking_home.loc[(tracking_home['Team'] == 'Home') & (tracking_home['Period'] == 2), tracking_home_columns] *= -1
        tracking_away.loc[(tracking_away['Team'] == 'Away') & (tracking_away['Period'] == 1), tracking_away_columns] *= -1
        tracking_away.loc[(tracking_away['Team'] == 'Home') & (tracking_away['Period'] == 2), tracking_away_columns] *= -1

        self.save({'events': events, 'tracking_home': tracking_home, 'tracking_away': tracking_away})