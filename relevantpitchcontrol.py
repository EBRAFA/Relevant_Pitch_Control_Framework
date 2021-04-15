# Library imports
import d6tflow as d6t
import numpy as np

# Project imports
import pitchcontrol as pc
import transitionprobability as tp

@d6t.inherits(pc.CalcPitchControlFrame)
class CalcRelevantPitchControlFrame(d6t.tasks.TaskPickle):
    '''
    Calculates the relevant pitch control for an entire frame.
    Returns a (32, 50) matrix with the values corresponding to
    each location on the field.
    '''

    def requires(self):
        return {'pitchcontrol': self.clone(pc.CalcPitchControlFrame),
                'transitionprobability': self.clone(tp.CalcTransitionProbabilityFrame)}

    def run(self):
        pitchcontrol = self.input()['pitchcontrol'].load()
        transitionprobability = self.input()['transitionprobability'].load()

        PPCFa = pitchcontrol['PPCFa']
        PPCFd = pitchcontrol['PPCFd']
        TP = transitionprobability['TP']
        xgrid = pitchcontrol['xgrid']
        ygrid = pitchcontrol['ygrid']

        #print('RPCa:')
        RPCa = np.zeros(shape=(len(PPCFa), len(ygrid), len(xgrid)))
        for i in range(len(PPCFa)):
            RPCa[i] = PPCFa[i] * TP
            print(np.sum(RPCa[i]))

        #print()
        #print('RPCd:')
        RPCd = np.zeros(shape=(len(PPCFd), len(ygrid), len(xgrid)))
        for i in range(len(PPCFd)):
            RPCd[i] = PPCFd[i] * TP
            #print(np.sum(RPCd[i]))

        self.save({'RPCa': RPCa, 'RPCd': RPCd, 'xgrid': xgrid, 'ygrid': ygrid})