# Library imports
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as colors
import d6tflow as d6t
import luigi
import cv2

# Project imports
import generaltasks as gt
import pitchcontrol as pc
import transitionprobability as tp
import relevantpitchcontrol as rpc
import execution as ex
import clustering as cl

class PlotPitch(d6t.tasks.TaskCache):
    field_dimen = luigi.Parameter(default=(106.0, 68.0))
    linewidth = luigi.IntParameter(default=2)
    markersize = luigi.IntParameter(default=20)
    field_color = luigi.Parameter(default='white')

    def run(self):
        fig, ax = plt.subplots(figsize=(12, 8))  # create a figure
        # decide what color we want the field to be. Default is green, but can also choose white
        if self.field_color == 'green':
            ax.set_facecolor('mediumseagreen')
            lc = 'whitesmoke'  # line color
            pc = 'w'  # 'spot' colors
        elif self.field_color == 'white':
            lc = 'k'
            pc = 'k'
        # ALL DIMENSIONS IN m
        border_dimen = (3, 3)  # include a border arround of the field of width 3m
        meters_per_yard = 0.9144  # unit conversion from yards to meters
        half_pitch_length = self.field_dimen[0] / 2.  # length of half pitch
        half_pitch_width = self.field_dimen[1] / 2.  # width of half pitch
        signs = [-1, 1]
        # Soccer field dimensions typically defined in yards, so we need to convert to meters
        goal_line_width = 8 * meters_per_yard
        box_width = 20 * meters_per_yard
        box_length = 6 * meters_per_yard
        area_width = 44 * meters_per_yard
        area_length = 18 * meters_per_yard
        penalty_spot = 12 * meters_per_yard
        corner_radius = 1 * meters_per_yard
        D_length = 8 * meters_per_yard
        D_radius = 10 * meters_per_yard
        D_pos = 12 * meters_per_yard
        centre_circle_radius = 10 * meters_per_yard
        # plot half way line # center circle
        ax.plot([0, 0], [-half_pitch_width, half_pitch_width], lc, linewidth=self.linewidth)
        ax.scatter(0.0, 0.0, marker='o', facecolor=lc, linewidth=0, s=self.markersize)
        y = np.linspace(-1, 1, 50) * centre_circle_radius
        x = np.sqrt(centre_circle_radius ** 2 - y ** 2)
        ax.plot(x, y, lc, linewidth=self.linewidth)
        ax.plot(-x, y, lc, linewidth=self.linewidth)
        for s in signs:  # plots each line seperately
            # plot pitch boundary
            ax.plot([-half_pitch_length, half_pitch_length], [s * half_pitch_width, s * half_pitch_width], lc,
                    linewidth=self.linewidth)
            ax.plot([s * half_pitch_length, s * half_pitch_length], [-half_pitch_width, half_pitch_width], lc,
                    linewidth=self.linewidth)
            # goal posts & line
            ax.plot([s * half_pitch_length, s * half_pitch_length], [-goal_line_width / 2., goal_line_width / 2.],
                    pc + 's',
                    markersize=6 * self.markersize / 20., linewidth=self.linewidth)
            # 6 yard box
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * box_length], [box_width / 2., box_width / 2.],
                    lc,
                    linewidth=self.linewidth)
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * box_length], [-box_width / 2., -box_width / 2.],
                    lc,
                    linewidth=self.linewidth)
            ax.plot([s * half_pitch_length - s * box_length, s * half_pitch_length - s * box_length],
                    [-box_width / 2., box_width / 2.], lc, linewidth=self.linewidth)
            # penalty area
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * area_length],
                    [area_width / 2., area_width / 2.],
                    lc, linewidth=self.linewidth)
            ax.plot([s * half_pitch_length, s * half_pitch_length - s * area_length],
                    [-area_width / 2., -area_width / 2.],
                    lc, linewidth=self.linewidth)
            ax.plot([s * half_pitch_length - s * area_length, s * half_pitch_length - s * area_length],
                    [-area_width / 2., area_width / 2.], lc, linewidth=self.linewidth)
            # penalty spot
            ax.scatter(s * half_pitch_length - s * penalty_spot, 0.0, marker='o', facecolor=lc, linewidth=0,
                       s=self.markersize)
            # corner flags
            y = np.linspace(0, 1, 50) * corner_radius
            x = np.sqrt(corner_radius ** 2 - y ** 2)
            ax.plot(s * half_pitch_length - s * x, -half_pitch_width + y, lc, linewidth=self.linewidth)
            ax.plot(s * half_pitch_length - s * x, half_pitch_width - y, lc, linewidth=self.linewidth)
            # draw the D
            y = np.linspace(-1, 1, 50) * D_length  # D_length is the chord of the circle that defines the D
            x = np.sqrt(D_radius ** 2 - y ** 2) + D_pos
            ax.plot(s * half_pitch_length - s * x, y, lc, linewidth=self.linewidth)

        # remove axis labels and ticks
        '''ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])'''
        # set axis limits
        xmax = self.field_dimen[0] / 2. + border_dimen[0]
        ymax = self.field_dimen[1] / 2. + border_dimen[1]
        ax.set_xlim([-xmax, xmax])
        ax.set_ylim([-ymax, ymax])
        ax.set_axisbelow(True)

        self.save({'fig': fig, 'ax': ax})

@d6t.inherits(PlotPitch, pc.CalcPitchControlFrame)
class PlotFrameTracking(d6t.tasks.TaskCache):
    include_player_velocities = luigi.BoolParameter(default=True)  # Option to include player velocities
    annotate = luigi.BoolParameter(default=True)  # Option to plot player id
    do_pitchcontrol = luigi.BoolParameter(default=False)
    do_transitionprobability = luigi.BoolParameter(default=False)
    do_relevantpitchcontrol = luigi.Parameter(default=False)  # Team to calculate, 'Home' or 'Away'
    do_player_pc = luigi.Parameter(default=None)  # Option to plot player Pitch Control i. e. 'Home_8'
    do_player_rpc = luigi.Parameter(default=None)  # Option to plot player Relevant Pitch Control i. e. 'Home_8'

    def requires(self):
        return {'tracking_home': gt.GetRowTracking(gameid=self.gameid, team='Home', rownumber=self.rownumber),
                'tracking_away': gt.GetRowTracking(gameid=self.gameid, team='Away', rownumber=self.rownumber),
                'team_in_possession': self.clone(gt.GetTeamInPossession),
                'player_ids': self.clone(gt.GetPlayerIDs)}

    def run(self):
        tracking_home = self.input()['tracking_home'].load()  # Row of tracking data
        tracking_away = self.input()['tracking_away'].load()  # Row of tracking data
        home_ids = self.input()['player_ids'].load()['home_ids']
        away_ids = self.input()['player_ids'].load()['away_ids']

        d6t.run(PlotPitch(), forced_all=True)
        pitch = PlotPitch().output().load()
        fig = pitch['fig']
        ax = pitch['ax']

        team_in_possession = self.input()['team_in_possession'].load()
        if team_in_possession == 'Home':
            player_colors = ('r', 'b')  # Player colors (red and blue)
        else:
            player_colors = ('b', 'r')  # Player colors (red and blue)

        # plot home & away teams in order
        for team, color in zip([tracking_home, tracking_away], player_colors):
            x_columns = [c for c in team.keys() if
                         c[-2:].lower() == '_x' and c != 'ball_x']  # column header for player x positions
            y_columns = [c for c in team.keys() if
                         c[-2:].lower() == '_y' and c != 'ball_y']  # column header for player y positions
            ax.plot(team[x_columns], team[y_columns], color + 'o', MarkerSize=10,
                    alpha=0.7)  # plot player positions
            if self.include_player_velocities:
                vx_columns = ['{}_vx'.format(c[:-2]) for c in x_columns]  # column header for player x positions
                vy_columns = ['{}_vy'.format(c[:-2]) for c in y_columns]  # column header for player y positions
                ax.quiver(team[x_columns].astype('float64'), team[y_columns].astype('float64'),
                          team[vx_columns].astype('float64'), team[vy_columns].astype('float64'), color=color,
                          scale_units='inches', scale=10., width=0.0015, headlength=5, headwidth=3, alpha=0.7)
            if self.annotate:
                [ax.text(team[x] + 0.5, team[y] + 0.5, x.split('_')[1], fontsize=10, color=color) for x, y in
                 zip(x_columns, y_columns) if not (np.isnan(team[x]) or np.isnan(team[y]))]

        # plot ball
        ax.plot(tracking_home['ball_x'], tracking_home['ball_y'], 'ko', MarkerSize=6, alpha=1.0, LineWidth=0)

        # plot axis labels
        ax.set_xlabel('x (m)', fontsize=20)
        ax.set_ylabel('y (m)', fontsize=20)
        ax.tick_params(labelsize=14)

        if self.do_pitchcontrol or self.do_transitionprobability or self.do_relevantpitchcontrol or self.do_player_pc or self.do_player_rpc:
            d6t.settings.check_dependencies = False

            if self.do_pitchcontrol or self.do_player_pc:
                d6t.run(pc.CalcPitchControlFrame(gameid=self.gameid, rownumber=self.rownumber))
                pitchcontrol = pc.CalcPitchControlFrame(gameid=self.gameid, rownumber=self.rownumber).output().load()
                PPCFa = pitchcontrol['PPCFa']
                PPCFd = pitchcontrol['PPCFd']
                xgrid = pitchcontrol['xgrid']
                ygrid = pitchcontrol['ygrid']

                if self.do_player_pc == None:
                    ax.imshow(np.flipud(PPCFa[-1,:,:]), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid),
                                                        np.amax(ygrid)), interpolation='hanning', vmin=0.0,
                              vmax=1.0, cmap='seismic', alpha=0.625)
                    ax.set_title('Potential Pitch Control Field (PPCF)', fontdict={'fontsize': 30})
                    cb = fig.colorbar(plt.cm.ScalarMappable(norm=None, cmap='seismic'), ax=ax, alpha=0.625, shrink=0.80)
                    cb.ax.tick_params(labelsize=14)

                else:
                    if self.do_player_pc[:4] == 'Home':
                        k = np.where(home_ids == self.do_player_pc)[0][0]  # Index of player id in ids array
                        if team_in_possession == 'Home':
                            surface_color = 'Reds'
                            ax.imshow(np.flipud(PPCFa[k]),
                                      extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(PPCFa[k]), cmap=surface_color,
                                      alpha=0.625)
                        else:
                            surface_color = 'Blues'
                            ax.imshow(np.flipud(PPCFd[k]),
                                      extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(PPCFa[k]), cmap=surface_color,
                                      alpha=0.625)
                    elif self.do_player_pc[:4] == 'Away':
                        k = np.where(away_ids == self.do_player_pc)[0][0]  # Index of player id in ids array
                        if team_in_possession == 'Away':
                            surface_color = 'Reds'
                            ax.imshow(np.flipud(PPCFa[k]),
                                      extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(PPCFa[k]), cmap=surface_color,
                                      alpha=0.625)
                        else:
                            surface_color = 'Blues'
                            ax.imshow(np.flipud(PPCFd[k]), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(PPCFa[k]), cmap=surface_color, alpha=0.625)
                    if team_in_possession == self.do_player_pc[:4]:
                        norm = colors.Normalize(vmin=0, vmax=np.max(PPCFa[k]))
                    else:
                        norm = colors.Normalize(vmin=0, vmax=np.max(PPCFd[k]))

                    ax.set_title('{} Potential Pitch Control Field (PPCF)'.format(self.do_player_pc), fontdict={'fontsize': 30})

            elif self.do_transitionprobability:
                surface_color = 'Reds'
                d6t.run(tp.CalcTransitionProbabilityFrame(gameid=self.gameid, rownumber=self.rownumber))
                transitionprobability = tp.CalcTransitionProbabilityFrame(gameid=self.gameid, rownumber=self.rownumber).output().load()
                TP = transitionprobability['TP']
                xgrid = transitionprobability['xgrid']
                ygrid = transitionprobability['ygrid']

                ax.imshow(np.flipud(TP), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                          interpolation='hanning', vmin=0.0, cmap=surface_color, alpha=0.625)
                norm = colors.Normalize(vmin=0, vmax=np.max(TP))
                ax.set_title('Potential Transition Probability Field (PTPF)', fontdict={'fontsize': 30})

            elif self.do_relevantpitchcontrol in ['Home', 'Away'] or self.do_player_rpc:
                surface_color = 'Reds'
                d6t.run(rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=self.rownumber))
                relevantpitchcontrol = rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=self.rownumber).output().load()
                RPCa = relevantpitchcontrol['RPCa']
                RPCd = relevantpitchcontrol['RPCd']
                xgrid = relevantpitchcontrol['xgrid']
                ygrid = relevantpitchcontrol['ygrid']

                if self.do_player_rpc == None:
                    if self.do_relevantpitchcontrol == team_in_possession:
                        surface_color = 'Reds'
                        ax.imshow(np.flipud(RPCa[-1, :, :]), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid),
                                                                     np.amax(ygrid)), interpolation='hanning', vmin=0.0,
                                                                     vmax=np.max(RPCa[-1, :, :]), cmap=surface_color, alpha=0.625)
                    else:
                        surface_color = 'Blues'
                        ax.imshow(np.flipud(RPCd[-1, :, :]), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid),
                                                                     np.amax(ygrid)), interpolation='hanning', vmin=0.0,
                                                                     vmax=np.max(RPCd[-1, :, :]), cmap=surface_color, alpha=0.625)
                    if team_in_possession == self.do_relevantpitchcontrol:
                        norm = colors.Normalize(vmin=0, vmax=np.max(RPCa[-1]))
                        ax.set_title('{} RPC Total = {}'.format(self.do_relevantpitchcontrol, round(np.sum(RPCa[-1]), 4)),
                                     fontdict={'fontsize': 30})
                    else:
                        norm = colors.Normalize(vmin=0, vmax=np.max(RPCd[-1]))
                        ax.set_title('{} RPC Total = {}'.format(self.do_relevantpitchcontrol, round(np.sum(RPCd[-1]), 4)),
                                     fontdict={'fontsize': 30})

                else:
                    if self.do_player_rpc[:4] == 'Home':
                        k = np.where(home_ids == self.do_player_rpc)[0][0]  # Index of player id in ids array
                        if team_in_possession == 'Home':
                            surface_color = 'Reds'
                            ax.imshow(np.flipud(RPCa[k]),
                                      extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(RPCa[k]), cmap=surface_color,
                                      alpha=0.625)
                        else:
                            surface_color = 'Blues'
                            ax.imshow(np.flipud(RPCd[k]),
                                      extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(RPCd[k]), cmap=surface_color,
                                      alpha=0.625)
                    elif self.do_player_rpc[:4] == 'Away':
                        k = np.where(away_ids == self.do_player_rpc)[0][0]  # Index of player id in ids array
                        if team_in_possession == 'Away':
                            surface_color = 'Reds'
                            ax.imshow(np.flipud(RPCa[k]),
                                      extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(RPCa[k]), cmap=surface_color,
                                      alpha=0.625)
                        else:
                            surface_color = 'Blues'
                            ax.imshow(np.flipud(RPCd[k]), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                                      interpolation='hanning', vmin=0.0, vmax=np.max(RPCd[k]), cmap=surface_color, alpha=0.625)
                    if team_in_possession == self.do_player_rpc[:4]:
                        norm = colors.Normalize(vmin=0, vmax=np.max(RPCa[k]))
                        ax.set_title('{} RPC Total = {}'.format(self.do_player_rpc, round(np.sum(RPCa[k]), 4)),
                                     fontdict={'fontsize': 30})
                    else:
                        norm = colors.Normalize(vmin=0, vmax=np.max(RPCd[k]))
                        ax.set_title('{} RPC Total = {}'.format(self.do_player_rpc, round(np.sum(RPCd[k]), 4)),
                                     fontdict={'fontsize': 30})

            if not self.do_pitchcontrol:
                cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=surface_color), ax=ax, alpha=0.625, shrink=0.80)
                cb.ax.tick_params(labelsize=14)

        fig.show()

        self.save({'fig': fig, 'ax': ax})

@d6t.inherits(cl.ClusteringRPCVectors, PlotPitch)
class PlotRPCSurfaces(d6t.tasks.TaskCache):
    clusterid = luigi.IntParameter()
    surface_color = luigi.Parameter()
    n_grid_cells_x = luigi.IntParameter(default=50)

    def requires(self):
        return {'clustering': self.clone(cl.ClusteringRPCVectors)}

    def run(self):
        centers = self.input()['clustering'].load()['centers']
        rpc_surface = centers[self.clusterid].reshape((50, 32), order='F')

        d6t.run(PlotPitch(), forced_all=True)
        pitch = PlotPitch().output().load()
        fig = pitch['fig']
        ax = pitch['ax']

        # plot axis labels
        ax.set_xlabel('x (m)', fontsize=20)
        ax.set_ylabel('y (m)', fontsize=20)
        ax.tick_params(labelsize=14)

        # break the pitch down into a grid
        n_grid_cells_y = int(self.n_grid_cells_x * self.field_dimen[1] / self.field_dimen[0])
        xgrid = np.linspace(-self.field_dimen[0] / 2., self.field_dimen[0] / 2., self.n_grid_cells_x)
        ygrid = np.linspace(-self.field_dimen[1] / 2., self.field_dimen[1] / 2., n_grid_cells_y)

        ax.imshow(np.flipud(rpc_surface), extent=(np.amin(xgrid), np.amax(xgrid), np.amin(ygrid), np.amax(ygrid)),
                  interpolation='hanning', vmin=0.0, cmap=self.surface_color, alpha=0.625)
        norm = colors.Normalize(vmin=0, vmax=np.max(rpc_surface))
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=self.surface_color), ax=ax, alpha=0.625, shrink=0.80)
        cb.ax.tick_params(labelsize=14)
        ax.set_title('Relevant Pitch Control - Cluster {} Centroid'.format(self.clusterid), fontdict={'fontsize': 30})

        self.save({'fig': fig, 'ax': ax})