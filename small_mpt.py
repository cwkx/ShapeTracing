import json
import sys

import numpy as np
import skfmm
import torch
import time
from pymanopt.manifolds import Sphere, SpecialOrthogonalGroup, Product
from skimage import measure
from tqdm import tqdm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
path = sys.argv[1]
prot = path.split('/')[1]

rec = np.load(path + "_r.npy")
lig = np.load(path + "_l.npy")

rec_np = skfmm.distance(rec-0.43)
lig_np = skfmm.distance(lig-0.43)

rec = -torch.from_numpy(rec_np).float().to(device)
lig = -torch.from_numpy(lig_np).float().to(device)

rec_pad = torch.nn.functional.pad(rec, (1,1,1,1,1,1), mode='constant', value=1e10)

rec_vtx, rec_fac, rec_nrm, rec_vlx = measure.marching_cubes_lewiner(rec_np, 0, step_size=1, allow_degenerate=False)
lig_vtx, lig_fac, lig_nrm, lig_vlx = measure.marching_cubes_lewiner(lig_np, 0, step_size=1, allow_degenerate=False)
rec_sze = torch.tensor((rec.size(0),rec.size(1),rec.size(2)), dtype=torch.float).to(device)
lig_sze = torch.tensor((lig.size(0),lig.size(1),lig.size(2)), dtype=torch.float).to(device)

# send to torch
a = torch.from_numpy(lig_vtx.copy()).to(device)

# protein tracing params
distance = rec_sze.max()
cone_base = 0.1 # perentage of random cone angle
eps = 0.0001 # terminate when convergence smaller than this


def sample_bounds(rec,p):
    # p expects [-1, 1] so we apply the transformation
    # im not sure if this function is perfect, so wrote a unit test to try to improve it
    box = (torch.tensor((rec.size(0),rec.size(1),rec.size(2)))).float().to(device)
    xt = box-(1+2)
    t = p/xt
    gx = t[:,0]
    gy = t[:,1]
    gz = t[:,2]
    t = torch.stack([gz,gy,gx], dim=-1)
    t *= (2-1.0/(2*xt))
    t -= (1-1.0/(4*xt))
    return torch.nn.functional.grid_sample(rec.unsqueeze(0).unsqueeze(0), t.unsqueeze(0).unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True).squeeze()

def dirac_eps(x, epsilon=1):
    return (epsilon/np.pi)/(epsilon**2+x**2)


def cost(t ,r ,c):
    global misses
    # t = t.float().to(device)
    # r = r.float().to(device)
    # c = c.float().to(device)

    # 1) get a starting translation and rotation on a ball around the receptor
    s = t.unsqueeze(0)
    v = -s.clone()
    s *= distance # radius around receptor
    s += 0.5 *rec_sze # center on receptor

    # 2) rotate the ligand and translate it to the boundary
    ext = 0.5 *lig_sze # we get the extents of the ligand
    a_p = a- ext  # this first line shifts the ligand on the origin
    r = r.unsqueeze(0).repeat(a_p.size(0), 1, 1)
    a_p = torch.einsum('ijk,ik->ij', r, a_p)
    a_p += s  # finally its shifted

    # 3) cone angle (linear interpolation, is this correct?)
    v = (1 - cone_base) * v + cone_base * c
    v /= torch.sqrt((v ** 2).sum(dim=1).unsqueeze(1))

    # 4) analytical ray-hit box
    r_v = 1.0 / v[0]
    in1 = r_v * (0 - a_p)
    in2 = r_v * (rec_sze - a_p)
    t_nears = torch.max(torch.min(in1, in2), dim=1)[0]
    t_fars = torch.min(torch.max(in1, in2), dim=1)[0]
    intersects = t_fars > t_nears

    surface_area = torch.Tensor([0])

    if torch.any(intersects):
        delta = torch.min(t_nears[intersects])
        a_p = a_p + delta * v[0] + 2 * torch.sign(v[0])

        phi_sample = sample_bounds(rec_pad, a_p)
        delta = phi_sample.min()

        while delta > eps and delta < 1e7:
            a_p = a_p + 0.5 * delta * v[0]
            phi_sample = sample_bounds(rec_pad, a_p)
            delta = phi_sample.min()

            if delta < 0:
                pass
                # print('error! delta went negative: ' + str(delta.item()))

        surface_area = dirac_eps(phi_sample, 1.0).sum()

        # print('resetting, converged delta: ' + str(delta) + " - visualising solution...")
        if delta > 1e5:
            misses += 1

    else:
        pass
        # print('no analytical rays hit the box')


    return torch.Tensor((-surface_area, (((a - a_p) ** 2).mean().mean()).sqrt()))


manifold = Product([Sphere(3), SpecialOrthogonalGroup(3), Sphere(3)])

cone_angles = [a.item() for a in np.arange(0,0.31,0.01)] # [0.0, 0.05, 0.10, 0.15, 0.20 ,0.25 ,0.30]
jsondict = {}
jsondict['angles'] = cone_angles
jsondict['results'] = []
iterations = 5000
runs = 10

print(f'angle,runtime,iter/s,misses/s')

for angle in cone_angles:
    cone_base = angle
    minscore = []
    minrmse = []
    misses = 0
    start_time = time.time()
    for _ in range(runs):
        points = [[torch.from_numpy(x).float().to(device) for x in manifold.rand()] for _ in range(iterations)]

        with torch.no_grad():
            results = torch.stack([cost(*args) for args in points])

        score, rmse = results.t().cpu().numpy()
        minscore += [[a.item() for a in np.minimum.accumulate(score)]]
        minrmse += [[a.item() for a in np.minimum.accumulate(rmse)]]

    end_time = time.time()
    runtime = end_time-start_time
    print(f'{angle},{runtime},{runs*iterations/runtime},{misses/runtime}')
    jsondict['results'].append(dict(scores=minscore, rmses=minrmse))
with open("results/" + prot + "_mpt.json", 'w') as outfile:
    json.dump(jsondict, outfile, indent=4)