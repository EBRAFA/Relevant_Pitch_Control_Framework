# Library imports
import numpy as np
import d6tflow as d6t
import luigi

# Project imports
import generaltasks as gt
import parameters as pr

@d6t.inherits(gt.TrackingTupleInfo, gt.FrameInfo)
class CalcPitchControlTarget(d6t.tasks.TaskCache):
    target_position = luigi.Parameter()

    def requires(self):
        return {'ball_start_pos': gt.GetBallPosition(gameid=self.gameid, rownumber=self.rownumber),
                'players': gt.InitialisePlayers(gameid=self.gameid, rownumber=self.rownumber, remove_offsides_players=True),
                'params': pr.PCModelParameters()}

    def run(self):
        attacking_players = self.input()['players'].load()['attacking_players']
        defending_players = self.input()['players'].load()['defending_players']
        ball_start_pos = self.input()['ball_start_pos'].load()
        params = self.input()['params'].load()  # Model parameters

        # calculate ball travel time from start position to end position.
        if ball_start_pos is None or any(np.isnan(ball_start_pos)):  # assume that ball is already at location
            ball_travel_time = 0.0
        else:
            # ball travel time is distance to target position from current ball position divided assumed average ball speed
            ball_travel_time = np.linalg.norm(self.target_position - ball_start_pos) / params['average_ball_speed']

        t_to_int_att = [p.simple_time_to_intercept(self.target_position) for p in attacking_players]
        t_to_int_def = [p.simple_time_to_intercept(self.target_position) for p in defending_players]

        # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
        tau_min_att = np.nanmin(t_to_int_att)
        tau_min_def = np.nanmin(t_to_int_def)

        # check whether we actually need to solve equation 3
        '''if tau_min_att - max(ball_travel_time, tau_min_def) >= params['time_to_control_def']:
            # if defending team can arrive significantly before attacking team, no need to solve pitch control model
            self.save({'PPCFatt': 0, 'PPCFdef': 1,
                       'attacking_players': attacking_players, 'defending_players': defending_players})
        elif tau_min_def - max(ball_travel_time, tau_min_att) >= params['time_to_control_att']:
            # if attacking team can arrive significantly before defending team, no need to solve pitch control model
            self.save({'PPCFatt': 1, 'PPCFdef': 0,
                       'attacking_players': attacking_players, 'defending_players': defending_players})
        else:'''
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if
                             p.time_to_intercept - tau_min_att < params['time_to_control_att']]
        defending_players = [p for p in defending_players if
                             p.time_to_intercept - tau_min_def < params['time_to_control_def']]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time - params['int_dt'], ball_travel_time + params['max_int_time'],
                             params['int_dt'])
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1 - ptot > params['model_converge_tol'] and i < dT_array.size:
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_intercept_ball(T) * params[
                    'lambda_att']
                # make sure it's greater than zero
                assert dPPCFdT >= 0, 'Invalid attacking player probability (CalcPitchControlTarget)'
                player.PPCF += dPPCFdT * params['int_dt']  # total contribution from individual player
                PPCFatt[
                    i] += player.PPCF  # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1 - PPCFatt[i - 1] - PPCFdef[i - 1]) * player.probability_intercept_ball(T) * params[
                    'lambda_def']
                # make sure it's greater than zero
                assert dPPCFdT >= 0, 'Invalid defending player probability (CalcPitchControlTarget)'
                player.PPCF += dPPCFdT * params['int_dt']  # total contribution from individual player
                PPCFdef[i] += player.PPCF  # add to sum over players in the defending team
            ptot = PPCFdef[i] + PPCFatt[i]  # total pitch control probability
            i += 1

        '''if i >= dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot))'''

        self.save({'PPCFatt': PPCFatt[i - 1], 'PPCFdef': PPCFdef[i - 1],
                   'attacking_players': attacking_players, 'defending_players': defending_players})

@d6t.inherits(gt.TrackingTupleInfo, gt.PitchInfo)
class CalcPitchControlFrame(d6t.tasks.TaskPickle):
    # Required parameters are gameid, rownumber and in_execution
    in_execution = luigi.BoolParameter(default=True)

    def requires(self):
        return {'params': pr.PCModelParameters(),
                'player_ids': self.clone(gt.GetPlayerIDs),
                'team_in_possession': self.clone(gt.GetTeamInPossession)}

    def run(self):
        params = self.input()['params'].load()
        home_ids = self.input()['player_ids'].load()['home_ids']
        away_ids = self.input()['player_ids'].load()['away_ids']
        team_in_possession = self.input()['team_in_possession'].load()

        # break the pitch down into a grid
        n_grid_cells_y = int(self.n_grid_cells_x * self.field_dimen[1] / self.field_dimen[0])
        xgrid = np.linspace(-self.field_dimen[0] / 2., self.field_dimen[0] / 2., self.n_grid_cells_x)
        ygrid = np.linspace(-self.field_dimen[1] / 2., self.field_dimen[1] / 2., n_grid_cells_y)

        if team_in_possession == 'Home':
            # initialise pitch control grids for attacking and defending teams
            PPCFa = np.zeros(shape=(len(home_ids) + 1, len(ygrid), len(xgrid)))
            PPCFd = np.zeros(shape=(len(away_ids) + 1, len(ygrid), len(xgrid)))

        else:
            # initialise pitch control grids for attacking and defending teams
            PPCFa = np.zeros(shape=(len(away_ids) + 1, len(ygrid), len(xgrid)))
            PPCFd = np.zeros(shape=(len(home_ids) + 1, len(ygrid), len(xgrid)))

        # calculate pitch pitch control model at each location on the pitch
        for i in range(len(ygrid)):
            for j in range(len(xgrid)):
                target_position = np.array([xgrid[j], ygrid[i]])
                d6t.settings.log_level = 'CRITICAL'
                d6t.run(CalcPitchControlTarget(gameid=self.gameid, rownumber=self.rownumber, target_position=target_position), execution_summary=False)
                PCT = CalcPitchControlTarget(gameid=self.gameid, rownumber=self.rownumber, target_position=target_position).output().load()
                PPCFa[-1, i, j] = PCT['PPCFatt']
                PPCFd[-1, i, j] = PCT['PPCFdef']
                attacking_players = PCT['attacking_players']
                for player in attacking_players:
                    for k in range(len(PPCFa) - 1):
                        if team_in_possession == 'Home':
                            if player.id == home_ids[k][5:]:
                                PPCFa[k, i, j] = player.PPCF
                        elif team_in_possession == 'Away':
                            if player.id == away_ids[k][5:]:
                                PPCFa[k, i, j] = player.PPCF
                defending_players = PCT['defending_players']
                for player in defending_players:
                    for k in range(len(PPCFd) - 1):
                        if team_in_possession == 'Home':
                            if player.id == away_ids[k][5:]:
                                PPCFd[k, i, j] = player.PPCF
                        elif team_in_possession == 'Away':
                            if player.id == home_ids[k][5:]:
                                PPCFd[k, i, j] = player.PPCF

        # check probabilitiy sums within convergence
        checksum = np.sum(PPCFa[-1] + PPCFd[-1]) / float(n_grid_cells_y * self.n_grid_cells_x)

        if self.in_execution:
            if 1 - checksum >= params['model_converge_tol']:
                if team_in_possession == 'Home':
                    # initialise pitch control grids for attacking and defending teams
                    PPCFa = np.zeros(shape=(len(home_ids) + 1, len(ygrid), len(xgrid)))
                    PPCFd = np.zeros(shape=(len(away_ids) + 1, len(ygrid), len(xgrid)))

                else:
                    # initialise pitch control grids for attacking and defending teams
                    PPCFa = np.zeros(shape=(len(away_ids) + 1, len(ygrid), len(xgrid)))
                    PPCFd = np.zeros(shape=(len(home_ids) + 1, len(ygrid), len(xgrid)))

        else:
            assert 1 - checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1 - checksum)

        self.save({'PPCFa': PPCFa, 'PPCFd': PPCFd, 'xgrid': xgrid, 'ygrid': ygrid})