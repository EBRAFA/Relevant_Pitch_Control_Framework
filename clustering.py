# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
import luigi
from sklearn.cluster import KMeans

# Project imports
import execution as ex

class ClusteringRPCVectors(d6t.tasks.TaskPickle):
    gameid = luigi.IntParameter()
    n_clusters = luigi.IntParameter()

    def requires(self):
        return self.clone(ex.RPCVectorization)

    def run(self):
        rpc_v = self.input().load()['rpc_v']
        frames = self.input().load()['frames']
        possessions = self.input().load()['possessions']

        algorithm = KMeans(n_clusters=self.n_clusters, random_state=0).fit(rpc_v)
        labels = algorithm.labels_
        centers = algorithm.cluster_centers_
        distances = algorithm.transform(rpc_v)

        self.save({'algorithm': algorithm, 'labels': labels, 'centers': centers, 'distances': distances})