from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from crustshot import CrustDB

import numpy as np

print 'plot America'
world = CrustDB()

us_frame = {
    'west': -130,
    'east': -50,
    'south': 17.5,
    'north': 52.5
}
us = world.selectRegion(**us_frame)
'''
# Example selection methods
us = us.selectMinLayers(4)
us = us.selectMinDepth(1)
us = us.selectMaxDepth(60).selectVp()

# Example plots
us.plotProfiles()
us.plotHistogram2d()
us.plotHistogram()
us.plotStats()
us.plotMap()
'''
