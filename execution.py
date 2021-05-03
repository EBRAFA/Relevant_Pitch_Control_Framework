# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
import luigi as lg
from tqdm import tqdm

# Project imports
import preprocessing as pre
import generaltasks as gt
import relevantpitchcontrol as rpc

@d6t.inherits(gt.PitchInfo)
class RPCExecution(d6t.tasks.TaskPickle):
    gameid = lg.IntParameter()

    def requires(self):
        return {'events': self.clone(pre.PrepData),
                'ids': self.clone(gt.GetPlayerIDs)}

    def run(self):
        events = self.input()['events']['events'].load()
        home_ids = self.input()['ids'].load()['home_ids']
        away_ids = self.input()['ids'].load()['away_ids']

        # break the pitch down into a grid
        n_grid_cells_y = int(self.n_grid_cells_x * self.field_dimen[1] / self.field_dimen[0])
        xgrid = np.linspace(-self.field_dimen[0] / 2., self.field_dimen[0] / 2., self.n_grid_cells_x)
        ygrid = np.linspace(-self.field_dimen[1] / 2., self.field_dimen[1] / 2., n_grid_cells_y)

        RPCa_Home = np.zeros(shape=(len(events[events.Team == 'Home']), len(home_ids) + 1, len(ygrid), len(xgrid)))
        RPCd_Home = np.zeros(shape=(len(events[events.Team == 'Away']), len(home_ids) + 1, len(ygrid), len(xgrid)))
        RPCa_Away = np.zeros(shape=(len(events[events.Team == 'Away']), len(away_ids) + 1, len(ygrid), len(xgrid)))
        RPCd_Away = np.zeros(shape=(len(events[events.Team == 'Home']), len(away_ids) + 1, len(ygrid), len(xgrid)))

        home_rows = events[events.Team == 'Home'].index
        away_rows = events[events.Team == 'Away'].index

        home_int_fail = []
        away_int_fail = []

        for i in tqdm(range(len(home_rows)), desc="executing game{} home events".format(self.gameid)):
            d6t.settings.check_dependencies = False
            d6t.settings.log_level = 'ERROR'
            d6t.run(rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=events.loc[home_rows[i], 'Start Frame'], in_execution=True))
            RPCa_Home[i] = rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=events.loc[home_rows[i], 'Start Frame'], in_execution=True).output().load()['RPCa']
            RPCd_Away[i] = rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=events.loc[home_rows[i], 'Start Frame'], in_execution=True).output().load()['RPCd']
            if np.sum(RPCa_Home[i][-1]) == 0:
                home_int_fail.append(i)

        if len(home_int_fail) > 0:
            RPCa_Home = np.delete(RPCa_Home, obj=np.array(home_int_fail), axis=0)
            RPCd_Away = np.delete(RPCd_Away, obj=np.array(home_int_fail), axis=0)

        for i in tqdm(range(len(away_rows)), desc="executing game{} away events".format(self.gameid)):
            d6t.settings.check_dependencies = False
            d6t.settings.log_level = 'ERROR'
            d6t.run(rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=events.loc[away_rows[i], 'Start Frame'], in_execution=True))
            RPCa_Away[i] = rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=events.loc[away_rows[i], 'Start Frame'], in_execution=True).output().load()['RPCa']
            RPCd_Home[i] = rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=events.loc[away_rows[i], 'Start Frame'], in_execution=True).output().load()['RPCd']
            if np.sum(RPCa_Away[i][-1]) == 0:
                away_int_fail.append(i)

        if len(away_int_fail) > 0:
            RPCa_Away = np.delete(RPCa_Away, obj=np.array(away_int_fail), axis=0)
            RPCd_Home = np.delete(RPCd_Home, obj=np.array(away_int_fail), axis=0)

        self.save({'RPCa_Home': RPCa_Home, 'RPCd_Home': RPCd_Home, 'RPCa_Away': RPCa_Away, 'RPCd_Away': RPCd_Away,
                   'home_int_fail': home_int_fail, 'away_int_fail': away_int_fail})