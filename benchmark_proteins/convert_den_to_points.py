import numpy as np
import biobox as bb
import sys, os

R = bb.Density()
L = bb.Density()

R._import_dx(sys.argv[1])
L._import_dx(sys.argv[2])

p_name = sys.argv[1].split('_')[0]

points_R = R.properties['density']
points_L = L.properties['density']

np.save(p_name + '_r', points_R)
np.save(p_name + '_l', points_L)
