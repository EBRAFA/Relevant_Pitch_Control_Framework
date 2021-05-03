# Library imports
import numpy as np

# Project imports
import d6tflow as d6t
import preprocessing as pre
import generaltasks as gt
import pitchcontrol as pc
import transitionprobability as tp
import relevantpitchcontrol as rpc
import execution as ex
import postprocessing as pos
import clustering as cl
import visualize as viz

d6t.settings.check_dependencies = False

#d6t.run(rpc.CalcRelevantPitchControlFrame(gameid=2, rownumber=146))

#d6t.run(pre.PrepData(gameid=1))

d6t.run(cl.KMeansTeam(n_clusters=6))

'''d6t.run(viz.PlotRPCSurfaces(n_clusters=10, clusterid=2, surface_color='Greens'))
fig = viz.PlotRPCSurfaces(n_clusters=10, clusterid=2, surface_color='Greens').output().load()['fig']
fig.show()'''

#d6t.run(viz.PlotFrameTracking(gameid=2, rownumber=5929, do_pitchcontrol=True))
#d6t.run(viz.PlotFrameTracking(gameid=1, rownumber=124504, do_transitionprobability=True))
