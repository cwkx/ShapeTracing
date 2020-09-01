# Run this script to create two numpy arrays of receptor and ligand proteins
import numpy as np

proteins = np.loadtxt('proteins.dat', dtype=str)
R = []
L = []
for p in proteins:
    R.append(np.load('proteins/%s/%s_r.npy'%(p,p)))
    L.append(np.load('proteins/%s/%s_l.npy'%(p,p)))

np.save('receptors.npy', R)
np.save('ligands.npy', L)
