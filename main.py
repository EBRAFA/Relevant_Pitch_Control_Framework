# Library imports
import numpy as np

# Project imports
import d6tflow as d6t
import dataprocessing as dp
import generaltasks as gt
import pitchcontrol as pc
import transitionprobability as tp
import relevantpitchcontrol as rpc
import execution as ex
import visualize as viz

#d6t.settings.check_dependencies = False

#d6t.run(dp.PrepData(gameid=2))

d6t.run(ex.RPCExecution(gameid=2))

#d6t.run(viz.PlotFrameTracking(gameid=1, rownumber=5663, do_relevantpitchcontrol='Away'))
#d6t.run(viz.PlotFrameTracking(gameid=1, rownumber=124504, do_transitionprobability=True))
