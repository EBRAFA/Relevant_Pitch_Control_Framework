# Library imports
import numpy as np
import d6tflow as d6t
import luigi
from scipy.stats import multivariate_normal

# Project imports
import generaltasks as gt
import parameters as pr
import pitchcontrol as pc

class CalcTransitionProbabilityTarget(d6t.tasks.TaskCache):
    '''
    Given the ball position and the pitch control for both teams at a certain location,
    calculates the probability of the ball transitioning there as the next event.
    '''

    target_position = luigi.Parameter()
    ball_start_pos = luigi.Parameter()
    PPCFa = luigi.Parameter()

    def requires(self):
        return {'params': pr.TModelParameters()}

    def run(self):
        params = self.input()['params'].load()

        sigma_2 = params['sigma_normal']**2
        normal_distrib = multivariate_normal(mean=tuple(self.ball_start_pos), cov=[[sigma_2, 0], [0, sigma_2]])
        T_proba = self.PPCFa**(params['alpha']) * normal_distrib.pdf(tuple(self.target_position))

        self.save(T_proba)

@d6t.inherits(pc.CalcPitchControlFrame)
class CalcTransitionProbabilityFrame(d6t.tasks.TaskPickle):
    '''
    Calculates the transition probability for every
    point on the pitch. You can't calculate the transition for a single player.
    '''

    def requires(self):
        return {'ball_start_pos': self.clone(gt.GetBallPosition),  # Loaded to get ball position and team in possession
                'pitch_control_frame': self.clone(pc.CalcPitchControlFrame),
                'params': pr.PCModelParameters()}

    def run(self):
        ball_start_pos = self.input()['ball_start_pos'].load()
        PPCFa = self.input()['pitch_control_frame'].load()['PPCFa']
        xgrid = self.input()['pitch_control_frame'].load()['xgrid']
        ygrid = self.input()['pitch_control_frame'].load()['ygrid']

        # initialise transition grid
        TP = np.zeros(shape=(len(ygrid), len(xgrid)))

        # calculate transition model at each location on the pitch
        for i in range(len(ygrid)):
            for j in range(len(xgrid)):
                target_position = np.array([xgrid[j], ygrid[i]])
                d6t.run(CalcTransitionProbabilityTarget(target_position=tuple(target_position), ball_start_pos=tuple(ball_start_pos), PPCFa=PPCFa[-1, i, j]), execution_summary=False)
                TP[i, j] = CalcTransitionProbabilityTarget(target_position=tuple(target_position), ball_start_pos=tuple(ball_start_pos),
                                                           PPCFa=PPCFa[-1, i, j]).output().load()

        # normalize T to unity
        N_TP = TP / np.sum(TP)

        self.save({'N_TP': N_TP, 'TP': TP, 'xgrid': xgrid, 'ygrid': ygrid})