# Library imports
import numpy as np
import pandas as pd
import d6tflow as d6t
import luigi

# Project imports
import dataprocessing as dp
import relevantpitchcontrol as rpc

class SelectEvents(d6t.tasks.TaskPickle):
    gameid = luigi.IntParameter()

    def requires(self):
        return dp.PrepData(gameid=self.gameid)

    def run(self):
        events = self.input()['events'].load()
        unwanted_events = ['SET PIECE', 'BALL OUT', 'FAULT RECEIVED', 'CARD']
        frames = []
        possessions = []  # List with team in possession at the moment

        for i in range(len(events)):
            if events.Type[i] not in unwanted_events:
                frames.append(events['Start Frame'][i])
                possessions.append(events['Team'][i])

        possessions.pop(frames.index(38232))  # Pitch Control integration failed
        possessions.pop(frames.index(93670))  # Pitch Control integration failed
        frames.remove(38232)  # Pitch Control integration failed
        frames.remove(93670)  # Pitch Control integration failed

        self.save({'frames': frames, 'possessions': possessions})

class RPCVectorization(d6t.tasks.TaskPickle):
    gameid = luigi.IntParameter()

    def requires(self):
        return self.clone(SelectEvents)

    def run(self):
        frames = self.input().load()['frames']
        possessions = self.input().load()['possessions']
        rpc_v = np.zeros(shape=(len(frames), 50*32))  # Size of pitch surface
        #rpc_v_max = []

        for i in range(len(frames)):
            d6t.settings.check_dependencies = False
            #d6t.show(rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=frames[i]))
            d6t.run(rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=frames[i]))
            rpc_i = rpc.CalcRelevantPitchControlFrame(gameid=self.gameid, rownumber=frames[i]).output().load()['RPC']
            # rpc_v_max.append(rpc_i.max())

            if possessions[i] == 'Home':
                rpc_v[i] = rpc_i.flatten('C')
            else:
                rpc_v[i] = np.flip(rpc_i.flatten('C'), 0)

        nani = np.unique(np.argwhere(np.isnan(rpc_v))[:, 0])
        frames = np.array(frames)
        frames = np.delete(frames, nani)
        possessions = np.array(possessions)
        possessions = np.delete(possessions, nani)
        rpc_v = np.delete(rpc_v, nani, axis=0)

        '''home_i = np.argwhere(possessions=='Home')
        rpc_v = np.delete(rpc_v, home_i, axis=0)
        frames = np.delete(frames, home_i)
        possessions = np.delete(possessions, home_i)'''

        self.save({'rpc_v': rpc_v, 'frames': frames, 'possessions': possessions})