# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
import luigi as lg
from tqdm import tqdm

# Project imports
import generaltasks as gt
import execution as ex

class TeamInputFormatting(d6t.tasks.TaskPickle):
    do_normalization = lg.BoolParameter(default=True)

    def requires(self):
        return {'game_1_surfaces': ex.RPCExecution(gameid=1), 'game_2_surfaces': ex.RPCExecution(gameid=2)}

    def run(self):
        game_1_surfaces = self.input()['game_1_surfaces'].load()
        game_2_surfaces = self.input()['game_2_surfaces'].load()

        n_1_RPCa_Home = game_1_surfaces['RPCa_Home'].shape[0]
        n_1_RPCa_Away = game_1_surfaces['RPCa_Away'].shape[0]
        n_2_RPCa_Home = game_2_surfaces['RPCa_Home'].shape[0]
        n_2_RPCa_Away = game_2_surfaces['RPCa_Away'].shape[0]

        RPCa_Team = np.zeros(shape=(n_1_RPCa_Home + n_1_RPCa_Away + n_2_RPCa_Home + n_2_RPCa_Away,
                                    game_1_surfaces['RPCa_Home'].shape[2] * game_1_surfaces['RPCa_Home'].shape[3]))
        RPCd_Team = np.zeros(shape=(n_1_RPCa_Home + n_1_RPCa_Away + n_2_RPCa_Home + n_2_RPCa_Away,
                                    game_1_surfaces['RPCd_Home'].shape[2] * game_1_surfaces['RPCd_Home'].shape[3]))

        for i in tqdm(range(n_1_RPCa_Home), desc="game1 home events"):
            RPCa_Team[i] = game_1_surfaces['RPCa_Home'][i][-1].flatten()
            RPCd_Team[i] = game_1_surfaces['RPCd_Away'][i][-1].flatten()

        for i in tqdm(range(n_1_RPCa_Home, n_1_RPCa_Home + n_1_RPCa_Away), desc="game1 away events"):
            RPCa_Team[i] = game_1_surfaces['RPCa_Away'][i - n_1_RPCa_Home][-1].flatten()
            RPCd_Team[i] = game_1_surfaces['RPCd_Home'][i - n_1_RPCa_Home][-1].flatten()

        for i in tqdm(range(n_1_RPCa_Home + n_1_RPCa_Away,
                            n_1_RPCa_Home + n_1_RPCa_Away + n_2_RPCa_Home), desc="game2 home events"):
            RPCa_Team[i] = game_2_surfaces['RPCa_Home'][i - n_1_RPCa_Home - n_1_RPCa_Away][-1].flatten()
            RPCd_Team[i] = game_2_surfaces['RPCd_Away'][i - n_1_RPCa_Home - n_1_RPCa_Away][-1].flatten()

        for i in tqdm(range(n_1_RPCa_Home + n_1_RPCa_Away + n_2_RPCa_Home,
                            n_1_RPCa_Home + n_1_RPCa_Away + n_2_RPCa_Home + n_2_RPCa_Away), desc="game2 away events"):
            RPCa_Team[i] = game_2_surfaces['RPCa_Away'][i - n_1_RPCa_Home - n_1_RPCa_Away - n_2_RPCa_Home][-1].flatten()
            RPCd_Team[i] = game_2_surfaces['RPCd_Home'][i - n_1_RPCa_Home - n_1_RPCa_Away - n_2_RPCa_Home][-1].flatten()

        if self.do_normalization:
            RPCa_Team *= 100
            RPCd_Team *= 100

        self.save({'RPCa_Team': RPCa_Team, 'RPCd_Team': RPCd_Team})

class PlayerInputFormatting(d6t.tasks.TaskCache):
    def requires(self):
        return {'game_1_ids': gt.GetPlayerIDs(gameid=1), 'game_2_ids': gt.GetPlayerIDs(gameid=2),
                'game_1_surfaces': ex.RPCExecution(gameid=1), 'game_2_surfaces': ex.RPCExecution(gameid=2)}

    def run(self):
        game_1_ids = self.input()['game_1_ids'].load()
        game_2_ids = self.input()['game_2_ids'].load()
        game_1_surfaces = self.input()['game_1_surfaces'].load()
        game_2_surfaces = self.input()['game_2_surfaces'].load()

        game_1_players = {'home_a': {}, 'home_d': {}, 'away_a': {}, 'away_d': {}}
        game_2_players = {'home_a': {}, 'home_d': {}, 'away_a': {}, 'away_d': {}}

        for i in tqdm(range(game_1_surfaces['RPCa_Home'].shape[0])):
            for j in range(len(game_1_ids['home_ids'])):
                game_1_players['home_a'][game_1_ids['home_ids'][j]] = game_1_surfaces['RPCa_Home'][i][j].flatten()
            for j in range(len(game_1_ids['away_ids'])):
                game_1_players['away_d'][game_1_ids['away_ids'][j]] = game_1_surfaces['RPCd_Away'][i][j].flatten()

        for i in tqdm(range(game_1_surfaces['RPCa_Away'].shape[0])):
            for j in range(len(game_1_ids['away_ids'])):
                game_1_players['away_a'][game_1_ids['away_ids'][j]] = game_1_surfaces['RPCa_Away'][i][j].flatten()
            for j in range(len(game_1_ids['home_ids'])):
                game_1_players['home_d'][game_1_ids['home_ids'][j]] = game_1_surfaces['RPCd_Home'][i][j].flatten()

        for i in tqdm(range(game_2_surfaces['RPCa_Home'].shape[0])):
            for j in range(len(game_2_ids['home_ids'])):
                game_2_players['home_a'][game_2_ids['home_ids'][j]] = game_2_surfaces['RPCa_Home'][i][j].flatten()
            for j in range(len(game_2_ids['away_ids'])):
                game_2_players['away_d'][game_2_ids['away_ids'][j]] = game_2_surfaces['RPCd_Away'][i][j].flatten()

        for i in tqdm(range(game_2_surfaces['RPCa_Away'].shape[0])):
            for j in range(len(game_2_ids['away_ids'])):
                game_2_players['away_a'][game_2_ids['away_ids'][j]] = game_2_surfaces['RPCa_Away'][i][j].flatten()
            for j in range(len(game_2_ids['home_ids'])):
                game_2_players['home_d'][game_2_ids['home_ids'][j]] = game_2_surfaces['RPCd_Home'][i][j].flatten()

        self.save({'game_1_players': game_1_players, 'game_2_players': game_2_players})