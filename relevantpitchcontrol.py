# Library imports
import d6tflow as d6t
import luigi as lg
import numpy as np

# Project imports
import pitchcontrol as pc
import transitionprobability as tp

@d6t.inherits(pc.CalcPitchControlFrame)
class CalcRelevantPitchControlFrame(d6t.tasks.TaskCache):
    '''
    Calculates the relevant pitch control for an entire frame.
    Returns a (32, 50) matrix with the values corresponding to
    each location on the field.
    '''

    tp_norm = lg.BoolParameter(default=True)  # Parameter to normalized or raw transition probability surface

    def requires(self):
        return {'pitchcontrol': self.clone(pc.CalcPitchControlFrame),
                'transitionprobability': self.clone(tp.CalcTransitionProbabilityFrame)}

    def run(self):
        PPCFa = self.input()['pitchcontrol'].load()['PPCFa']
        PPCFd = self.input()['pitchcontrol'].load()['PPCFd']
        xgrid = self.input()['pitchcontrol'].load()['xgrid']
        ygrid = self.input()['pitchcontrol'].load()['ygrid']

        if self.tp_norm:
            TP = self.input()['transitionprobability'].load()['N_TP']
        else:
            TP = self.input()['transitionprobability'].load()['TP']

        checksum = np.sum(PPCFa[-1] + PPCFd[-1]) / float(50 * 32)
        assert 1 - checksum < 0.01, "Checksum failed: %1.3f" % (1 - checksum)

        RPCa = np.zeros(shape=(len(PPCFa), len(ygrid), len(xgrid)))
        for i in range(len(PPCFa)):
            RPCa[i] = PPCFa[i] * TP

        RPCd = np.zeros(shape=(len(PPCFd), len(ygrid), len(xgrid)))
        for i in range(len(PPCFd)):
            RPCd[i] = PPCFd[i] * TP

        self.save({'RPCa': RPCa, 'RPCd': RPCd, 'xgrid': xgrid, 'ygrid': ygrid})