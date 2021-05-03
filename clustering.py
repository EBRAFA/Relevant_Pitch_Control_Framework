# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
import luigi
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

# Project imports
import execution as ex
import postprocessing as pos

class KMeansTeam1(d6t.tasks.TaskPickle):
    n_clusters = luigi.IntParameter()

    def requires(self):
        return ex.RPCExecution(gameid=2)

    def run(self):
        RPCa_Team = self.input().load()['RPCa_Home']

        V_RPCa_Team = np.zeros(shape=(RPCa_Team.shape[0], 1600))
        for i in tqdm(range(RPCa_Team.shape[0]), desc="game1 home events"):
            V_RPCa_Team[i] = RPCa_Team[i][-1].flatten()

        algorithm = KMeans(n_clusters=self.n_clusters, random_state=0).fit(V_RPCa_Team)
        labels = algorithm.labels_
        centers = algorithm.cluster_centers_
        distances = algorithm.transform(V_RPCa_Team)

        self.save({'algorithm': algorithm, 'labels': labels, 'centers': centers, 'distances': distances})

class KMeansTeam(d6t.tasks.TaskPickle):
    n_clusters = luigi.IntParameter()

    def requires(self):
        return pos.TeamInputFormatting()

    def run(self):
        RPCa_Team = self.input().load()['RPCa_Team']

        algorithm = KMeans(n_clusters=self.n_clusters, random_state=0).fit(RPCa_Team)
        labels = algorithm.labels_
        centers = algorithm.cluster_centers_
        distances = algorithm.transform(RPCa_Team)

        self.save({'algorithm': algorithm, 'labels': labels, 'centers': centers, 'distances': distances})

class HierarchicalTeam(d6t.tasks.TaskPickle):
    n_clusters = luigi.IntParameter()

    def requires(self):
        return pos.TeamInputFormatting()

    def run(self):
        RPCa_Team = self.input().load()['RPCa_Team']

        algorithm = AgglomerativeClustering(n_clusters=self.n_clusters).fit(RPCa_Team)
        labels = algorithm.labels_

        self.save({'algorithm': algorithm, 'labels': labels})

class KMeansPlayer(d6t.tasks.TaskPickle):
    n_clusters = luigi.IntParameter()
    player_id = luigi.Parameter(default=None)  # Player id for individual player clustering (i.e. 'Home_8')
    gameid = luigi.Parameter(default=None)  # Game id in which the player in 'player_id' was chosen

    def requires(self):
        return {''}

    def run(self):
        RPCa_Home = self.input().load()['RPCa_Home']
        RPCd_Away = self.input().load()['RPCd_Away']
        RPCa_Away = self.input().load()['RPCa_Away']
        RPCd_Home = self.input().load()['RPCd_Home']

        RPCa_Team = np.zeros(shape=(RPCa_Home.shape[0] + RPCa_Away.shape[0], RPCa_Home.shape[2] * RPCa_Home.shape[3]))  # Vectorized team surfaces

        for i in tqdm(range(RPCa_Home.shape[0])):
            RPCa_Team[i] = RPCa_Home[i][-1].flatten()

        for i in tqdm(range(RPCa_Away.shape[0])):
            RPCa_Team[i + RPCa_Home.shape[0]] = RPCa_Away[i][-1].flatten()

        algorithm = KMeans(n_clusters=self.n_clusters, random_state=0).fit(RPCa_Team)
        labels = algorithm.labels_
        centers = algorithm.cluster_centers_
        distances = algorithm.transform(RPCa_Team)

        self.save({'algorithm': algorithm, 'labels': labels, 'centers': centers, 'distances': distances})