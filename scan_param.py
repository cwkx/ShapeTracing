import sys
sys.path.append('/home2/dclw29/JabberDock2/mini_protein_benchmark/analyse_param')
import ncc_functions as f
import numpy as np
import time
import torch 
import pymanopt # latest install from git https://github.com/pymanopt/pymanopt
from collections import deque, defaultdict
from skimage import measure
from pymanopt import Problem
from pymanopt.manifolds import Sphere, SpecialOrthogonalGroup, Euclidean, Product
from pymanopt.solvers.solver import Solver
from pymanopt.solvers import ParticleSwarm, SteepestDescent, TrustRegions, ConjugateGradient, NelderMead
from collections import deque, defaultdict
import geoopt
from sklearn.cluster import KMeans
import torch.nn as nn
import plotly.graph_objs as go
import scipy.spatial.transform
import skfmm

@pymanopt.function.PyTorch
def cost(t,r):

    t = t.float().to(device)
    r = r.float().to(device)

    a_p = a-ext # this first line shifts the ligand on the origin
    a_r = r.repeat(a_p.size(0),1,1)
    a_p = torch.einsum('ijk,ik->ij', a_r, a_p)
    a_p += t # finally its shifted

    phi_sample = sample_bounds(rec, a_p)
    surface_area = f.energy(phi_sample, k_const).sum() # warp_delta(phi_sample).sum()

    global best_t
    global best_r
    global best_scores
    global best_rmsd

    # this is actually the one(s) we care about, as its using our metric, not a test rmsd
    if surface_area > np.min(best_scores): # best_surface_area
        arg = np.argmin(best_scores)
        del best_t[arg], best_r[arg], best_scores[arg], best_rmsd[arg]
        best_t.append(t.detach().cpu().numpy())
        best_r.append(r.detach().cpu().numpy())
        best_scores.append(surface_area.item())
        best_rmsd.append(torch.sqrt(((a_p-a)**2).mean()))

    return -surface_area.cpu()#+ 0.0001*dist.cpu()

def sample_bounds(rec,p):
    # p expects [-1, 1] so we apply the transformation
    box = torch.tensor(rec.size()).to(device).float()
    t = p/(box-1)
    t = torch.stack([t[:,2],t[:,1],t[:,0]], dim=-1) * 2.0 - 1.0
    return torch.nn.functional.grid_sample(rec.unsqueeze(0).unsqueeze(0), t.unsqueeze(0).unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True).squeeze()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

protein = "1EAW" 
iso = 0.43 
k_const = 2.0 

rec = np.load(protein + "/1EAW_r_u.npy")
lig = np.load(protein + "/1EAW_l_u.npy")

rec_np = -skfmm.distance(rec-iso)
lig_np = -skfmm.distance(lig-iso)

rec = torch.from_numpy(rec_np).float().to(device)
lig = torch.from_numpy(lig_np).float().to(device)

rec = f.set_bounds(rec)

rec_vtx, rec_fac, rec_nrm, rec_vlx = measure.marching_cubes_lewiner(rec_np, 0, step_size=1, allow_degenerate=False)
lig_vtx, lig_fac, lig_nrm, lig_vlx = measure.marching_cubes_lewiner(lig_np, 0, step_size=1, allow_degenerate=False)
rec_sze = torch.tensor((rec.size(0),rec.size(1),rec.size(2)), dtype=torch.float).to(device)
lig_sze = torch.tensor((lig.size(0),lig.size(1),lig.size(2)), dtype=torch.float).to(device)

# send to torch
a = torch.from_numpy(lig_vtx.copy()).to(device)

# protein dist params
distance = rec_sze.max()
ext = 0.5*lig_sze # we get the extents of the ligand

manifold = Product([Euclidean(3), SpecialOrthogonalGroup(3)])

# setup arrays to save
ts = []
rs = []

# number of epochs necessary - we'll say 15 for now
epochs = 15
# no. of translations and rotations
no_trans = 80
no_rot = 80

for e in range(epochs):
    # get starting translations and rotation
    s = torch.randn(no_trans, 3).to(device)
    s /= torch.sqrt((s**2).sum(dim=1).unsqueeze(1))
    s *= distance/2. # radius around receptor
    s += 0.5*rec_sze # center on receptor
    
    rot = []
    # generate rotations to couple with
    for i in range(no_rot):
        mat = f.quat2mat(f.uniform_quat(np.random.rand(1)[0], np.random.rand(1)[0], np.random.rand(1)[0]))
        rot.append(mat)

    for ir in range(no_rot):
        # setup points (same r for all)
        outer_ring = []
        # generate rotations to couple with
        for i in range(len(s)):
            outer_ring.append([s[i].detach().cpu().numpy(), rot[i]])
    
        ##### Build big list then cluster geometrically at the end ####
        list_length = 50
        best_t = deque(maxlen=list_length); best_t.extend(np.zeros((list_length,3)))
        tmp = np.zeros((list_length,3,3))
        tmp[:,0,0] = 1.0; tmp[:,1,1] = 1.0; tmp[:,2,2] = 1.0
        best_r = deque(maxlen=list_length); best_r.extend(tmp)
        best_scores = deque(maxlen=list_length); best_scores.extend(np.zeros(list_length))
        min_score = np.min(best_scores)
        best_rmsd = deque(maxlen=list_length); best_rmsd.extend(np.zeros(list_length))
    
        time0 = time.time()
        solver = ParticleSwarm(maxcostevals=50000, maxiter=50000, social=1.4, maxtime=float('inf'), populationsize=80)
        problem = Problem(manifold=manifold, cost=cost, verbosity=2)
        solution = solver.solve(problem, x=outer_ring)

        ts.append(best_t)
        rs.append(best_r)


