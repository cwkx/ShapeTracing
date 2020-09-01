import os
import pickle
import sys
from itertools import count

import numpy as np
import skfmm
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from pymanopt.manifolds import Sphere, SpecialOrthogonalGroup, Product
from skimage import measure
from torch import nn
from torch import optim
from torch.utils import data

# We do all alignments in a batch at once,
# this limit controls memory usage of the peicewise function by varying batch size
MATRIX_LIMIT = 2000

PW_PIECES = 5
# Number of samples in each epoch
SAMPLES_PER_PROTEIN = 1_000
EPOCHS = 100


BATCH_SIZE = MATRIX_LIMIT//PW_PIECES
BATCHES_PER_PROTEIN = SAMPLES_PER_PROTEIN//BATCH_SIZE

# This allows for code to run cuda devices (GPU) if available, otherwise it'll run on the CPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

eps = 0.0001  # shape tracing, terminate when convergence smaller than this


def sample_bounds(rec, p):
    # p expects [-1, 1] so we apply the transformation
    box = torch.tensor(rec.size()).to(device).float()
    t = p / (box - 1)
    t = torch.stack([t[:, 2], t[:, 1], t[:, 0]], dim=-1) * 2.0 - 1.0
    return torch.nn.functional.grid_sample(rec.unsqueeze(0).unsqueeze(0), t.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                           mode='bilinear', padding_mode='border', align_corners=True).squeeze()


def project(t, r, v, lig_vtx, rec):
    """Project an array of points `lig_vtx` towards the SDF `rec` until they collide
    `t` - initial translation
    `r` - rotation matrix
    `v` - projection direction

    returns: `t_new` - translation of collided ligand"""

    # This is the core protein tracing algorithm
    # tweaked a bit so we can get t_new out
    lig_vtx = lig_vtx.to(device)
    rec = rec.to(device)
    rec_sze = torch.tensor(rec.shape, dtype=torch.float32).to(device)

    t = t.to(device)
    r = r.to(device)
    v = v.to(device)
    # 2) rotate the ligand and translate it to the boundary (defined by t)
    a = (r @ lig_vtx.T).T  # rotation matmul for ligand
    a += t.unsqueeze(0)  # offset by t

    # 4) analytical ray-hit box

    r_v = 1.0 / v[0]
    in1 = r_v * (0 - a)
    in2 = r_v * (rec_sze - a)
    t_nears = torch.max(torch.min(in1, in2), dim=1)[0]
    t_fars = torch.min(torch.max(in1, in2), dim=1)[0]
    intersects = t_fars > t_nears

    if torch.any(intersects):
        offset = torch.min(t_nears[intersects])
        a_p = a + offset * v + 2 * torch.sign(v)

        phi_sample = sample_bounds(rec, a_p)
        delta = phi_sample.min()

        while delta > eps:
            offset += 0.5 * delta
            a_p = a + offset * v
            phi_sample = sample_bounds(rec, a_p)
            delta = phi_sample.min()

            if delta > 1e7:
                return None

        return (t + offset * v).detach().cpu()
    else:
        return None
        # print('no analytical rays hit the box')


class ProteinDocks(data.Dataset):
    def __init__(self, recs, ligs, batch_size=10, batches_per_protein=10, fixed=False, exclusion_radius=5):
        super(ProteinDocks, self).__init__()
        self.recs, self.lig_vtxs = recs, ligs
        self.per_prot = batches_per_protein
        self.manifold = Product([Sphere(3), SpecialOrthogonalGroup(3), Sphere(3)])
        self.batch_size = batch_size
        self.fixed = fixed
        self.exclusion_radius = exclusion_radius
        # identity transform, first value in every batch
        self.identity_t = torch.tensor([0, 0, 0], dtype=torch.float32)
        self.identity_r = torch.tensor([[1, 0, 0],
                                       [0, 1, 0],
                                       [0, 0, 1]], dtype=torch.float32)


        if fixed:
            # reuse batches so we can compare like-for-like between iterations
            self.batches = [None] * (batches_per_protein * len(recs))
            # Pregen batches by iterating through self
            for res in self:
                pass

    def __getitem__(self, item):
        prot_idx = item // self.per_prot
        rec = self.recs[prot_idx]
        lig_vtx = self.lig_vtxs[prot_idx]
        # If fixed iterator and we've already calculated it:
        if self.fixed and self.batches[item] != None:
            ts_rs = self.batches[item]
        else:
            ts_rs = self.gen_batch(lig_vtx, rec)
            if self.fixed:
                self.batches[item] = ts_rs

        return ts_rs, lig_vtx, rec

    def gen_batch(self, lig_vtx, rec):
        distance = max(rec.shape)
        ts = torch.empty(self.batch_size, 3)
        rs = torch.empty(self.batch_size, 3, 3)
        # first half of the batch is made up of random translations
        # slowly increasing in size (bias towards smaller ones w/ **2)
        split_batch = 3*self.batch_size//4
        ts[:split_batch] = torch.randn(split_batch, 3)
        ts_norms = ts[:split_batch].norm(dim=1)
        ts_proj_vecs = (ts[:split_batch]/ts_norms.unsqueeze(1))*self.exclusion_radius
        ts[:split_batch] *= (torch.linspace(0,5, steps=split_batch)**2).unsqueeze(1)
        # offset by radius
        ts[:split_batch] += ts_proj_vecs

        # I don't know how to do small random rotations, so let's just do identity
        rs[:split_batch] = self.identity_r.unsqueeze(0).expand_as(rs[:split_batch])
        # Also, the first element in the batch should be the identity transform
        ts[0] = self.identity_t

        # alternative far-away docks for the other half
        i = split_batch
        while i < self.batch_size:
            pro = self.manifold.rand()
            t, r, v = [torch.tensor(x, dtype=torch.float32) for x in pro]
            v = (1 - 0.05) * v + 0.05 * -t  # force into cone, 0.05 base
            v /= v.norm()  # normalize
            t_new = project(t * distance, r, v, lig_vtx, rec)
            # sometimes we don't hit, so only add when we do
            if t_new is not None:
                ts[i] = t_new
                rs[i] = r
                i += 1
        ts_rs = (ts, rs)
        return ts_rs

    def __len__(self):
        return self.per_prot * len(self.recs)

def gen_prot(iso, ligpath, recpath):
    lig = np.load(ligpath)
    lig_np = skfmm.distance(lig - iso)
    lig_vtx, _, _, _ = measure.marching_cubes_lewiner(lig_np, 0, step_size=1, allow_degenerate=False)
    lig_vtx = torch.from_numpy(lig_vtx.astype(dtype=np.float32))

    rec_orig = np.load(recpath)
    rec = torch.from_numpy(skfmm.distance(rec_orig - iso).astype(dtype=np.float32))

    return rec, lig_vtx


class PieceWiseEnergy(nn.Module):
    def __init__(self, pieces=5):
        super(PieceWiseEnergy, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, pieces),
            nn.LeakyReLU(inplace=True),
            nn.Linear(pieces, pieces),
            nn.LeakyReLU(inplace=True),
            nn.Linear(pieces, pieces),
            nn.Hardtanh(inplace=True),
            nn.Linear(pieces, 1),
            )

    def forward(self, x):
        return self.layers(x)


class ChiSquaredEnergy(nn.Module):
    def __init__(self, k_start=2):
        super(ChiSquaredEnergy, self).__init__()
        # We use nn.Parameter to make K a parameter that optimisers can "see"

        self.k = nn.Parameter(torch.tensor(k_start, dtype=torch.float32))

    def forward(self, x):
        return - (x + self.k) * torch.exp(-(x + self.k) / self.k) / self.k ** 2


if __name__ == "__main__":
    # Incantations to make CUDA work across multiple processes
    mp.set_start_method('forkserver')
    mp.set_sharing_strategy('file_system')

    import visdom

    # get iso value as argument
    iso = int(sys.argv[1]) / 100

    # first, we need to organise all of our proteins
    # into a list of (receptor.npy, ligand.npy) file paths.
    # this creates a list of all .npy filepaths in current directory + subfolders
    npy_files = [os.path.join(r, f) for r, _, fs in os.walk('.') for f in fs if f.endswith('npy')]
    # (we're relying on r and l existing in each protein dir for this to work)
    ligands = [f for f in npy_files if f.endswith('l.npy')]
    receptors = [f for f in npy_files if f.endswith('r.npy')]

    # use zip to combine the iso values, and two lists
    proteins = list(zip([iso] * len(ligands), ligands, receptors))


    # Debug with fewer proteins
    # proteins = proteins[:10]
    # Calculate iso surfaces & SDFs, as well as ligand vertexes
    # We use multiprocessing here as it's CPU intensive,
    # but each prot can only run on one core.
    # equivalent to:
    # result = [gen_prot(*p) for p in proteins]
    # but uses multiple cores

    # Figure out how many cores to use,
    # Slurm sets an environment var that says how many cores are allocated
    # to the current job, using more just wastes memory and slows down execution (slightly)
    try:
        corecount = int(os.environ['SLURM_CPUS_PER_TASK'])
    except KeyError:
        corecount = None
    # If processes=None, mp.Pool defaults to the number of cores on the system
    print(f'cores: {corecount}')
    with mp.Pool(processes=corecount) as p:
        result = p.starmap(gen_prot, proteins)
    pass

    # Initialise both chi-squared energy and piecewise energy "functions"
    chisq_en = ChiSquaredEnergy().to(device)
    pw_en = PieceWiseEnergy(pieces=PW_PIECES).to(device)
    # Initialise repsective optimisers,
    chisq_optim = optim.Adam(params=chisq_en.parameters(), lr=3e-4)
    pw_optim = optim.Adam(params=pw_en.parameters(), lr=3e-4)

    # split into test and train lists
    split_size = len(result) // 10
    test, train = result[:split_size], result[split_size:]

    # separate out ligands and receptor arrays, put into lists
    test_rec, test_lig = list(zip(*test))
    train_rec, train_lig = list(zip(*train))

    # pickle.dump((test_rec, test_lig), open(f"test_{iso}", 'wb'))
    # pickle.dump((train_rec, train_lig), open(f"train_{iso}", 'wb'))
    prot_settings = dict(
        batch_size=BATCH_SIZE,
        batches_per_protein=BATCHES_PER_PROTEIN,
        )
    test_dataset = ProteinDocks(test_rec, test_lig, fixed=True, **prot_settings)
    train_dataset = ProteinDocks(train_rec, train_lig, fixed=False, **prot_settings)
    test_loader = data.DataLoader(test_dataset, num_workers=4, shuffle=False, pin_memory=True)
    train_loader = data.DataLoader(train_dataset, num_workers=4, shuffle=True, pin_memory=True)



    vis_dict = dict()
    # check if slurm job (NCC) if so, assume visdom is running
    # on the node that submitted the job
    if 'SLURM_SUBMIT_HOST' in os.environ:
        vis_dict['server'] = os.environ['SLURM_SUBMIT_HOST']
    # specify visdom port with env. variable, else use default
    if 'VIS_PORT' in os.environ:
        vis_dict['port'] = int(os.environ['VIS_PORT'])
    # which visdom environment to use, plotting calls will use this by default
    vis_dict['env'] = f'prot_iso={iso:1.2f}'  # e.g. prot_iso=0.45
    print(vis_dict)
    vis = visdom.Visdom(**vis_dict)

    vis.line(
        X=np.array([[float("nan"), float("nan")]]),
        Y=np.array([[float("nan"), float("nan")]]),
        opts=dict(showlegend=True, title="Loss", legend=["chisq", "piecewise"]),
        win="Loss"
        )

    x_big = torch.linspace(-5, 40).unsqueeze(1).to(device)
    x_small = torch.linspace(-2, 4).unsqueeze(1).to(device)
    far_vals = torch.tensor([[50],[70],[200]], dtype=torch.float32).to(device)
    zero = torch.zeros_like(far_vals).to(device)

    # grad target[0] = 0 for reasons we'll discuss later.
    grad_target = torch.ones(BATCH_SIZE, dtype=torch.float32).to(device)
    grad_target[0] = 0
    for epoch in range(EPOCHS):  # loop indefinitely (we use itertools.count() to track epoch)
        # torch does packing because of batches,
        # we're making our own batches in the dataset class
        # because sampling different proteins at the same time is inefficient
        # so we need to have all this unpacking stuff e.g. (t,) to get neat values
        for (ts, rs), (lig_vtx,), (rec,) in train_loader:
            lig_vtx = lig_vtx.to(device)
            rec = rec.to(device)
            rankings = -torch.ones(BATCH_SIZE - 1).to(device)
            # extract "batch" from tensors in ts, rs and push both to device
            ts = ts[0].to(device)
            rs = rs[0].to(device)
            ts.requires_grad = True
            rs.requires_grad = True
            # turns out you can vectorise everything if you try hard enough
            transf = ((rs @ lig_vtx.T) + ts.unsqueeze(2)).transpose(1, 2).reshape(-1,3)
            dists = sample_bounds(rec, transf).unsqueeze(1)
            chi_energies = chisq_en(dists).reshape(BATCH_SIZE,-1).mean(dim=1)
            pw_energies = pw_en(dists).reshape(BATCH_SIZE,-1).mean(dim=1)
            # Our first sample is *always* the identity transform,
            # AKA the ground truth loss
            chi_gt = chi_energies[0]
            chi_bad = chi_energies[1:]
            # we expand chi_gt as we want it the same size as chi_bad
            # for margin ranking loss (comparing pairs of results)
            chi_gt = chi_gt.expand_as(chi_bad)
            # use margin ranking loss
            # ranking = -1 means we want the second value to be bigger
            # (greater energy)
            chi_loss = F.margin_ranking_loss(chi_gt, chi_bad, rankings)
            #
            # # Here we calc gradients w.r.t translation and rotation.
            # # We always want there to be a decent gradient to find minima
            # # So we force these values to have norm(gradient) = 1
            # # (apart from the G.T, which we want as a minima, so norm(gradient) = 0)
            # ts_grads = torch.autograd.grad(chi_energies, ts,
            #                                grad_outputs=torch.empty_like(chi_energies),
            #                                retain_graph=True,
            #                                create_graph=False)[0]
            # rs_grads = torch.autograd.grad(chi_energies, rs,
            #                                grad_outputs=torch.empty_like(chi_energies),
            #                                retain_graph=True,
            #                                create_graph=False)[0]
            # ts_g_norm = ts_grads.norm(dim=1)
            # rs_g_norm = rs_grads.norm(dim=(1, 2)) # Frobenius norm over each matrix.
            # g_norm = torch.stack((ts_g_norm, rs_g_norm), dim=1).norm(dim=1)
            #
            # g_err = g_norm - grad_target
            # # Sometimes these vals have nans or infs in, remove them
            # # this comes from backprop though gridsample and how we define OOB values
            # # not going to touch that code atm.
            # g_err_no_inf = g_err[torch.isfinite(g_err)]
            # g_loss =  ((10e-30*g_err_no_inf) ** 2).mean()
            # print(g_loss)
            # # Add gradient errors
            # chi_loss += g_loss


            # repeat for piecewise
            # Our first sample is *always* the identity transform,
            # AKA the ground truth loss
            pw_gt = pw_energies[0]
            pw_bad = pw_energies[1:]
            # we expand chi_gt as we want it the same size as chi_bad
            # for margin ranking loss (comparing pairs of results)
            pw_gt = pw_gt.expand_as(pw_bad)
            # use margin ranking loss
            # ranking = -1 means we want the second value to be bigger
            # (greater energy) for non-GT docks.
            pw_loss = F.margin_ranking_loss(pw_gt, pw_bad, rankings)

            ts_grads = torch.autograd.grad(pw_energies, ts,
                                           grad_outputs=torch.empty_like(pw_energies),
                                           retain_graph=True,
                                           create_graph=False)[0]
            rs_grads = torch.autograd.grad(pw_energies, rs,
                                           grad_outputs=torch.empty_like(pw_energies),
                                           retain_graph=True,
                                           create_graph=False)[0]
            ts_g_norm = ts_grads.norm(dim=1)
            rs_g_norm = rs_grads.norm(dim=(1, 2)) # Frobenius norm over each matrix.
            ts_g_err = ts_g_norm - grad_target
            rs_g_err = rs_g_norm - grad_target
            ts_g_err = ts_g_err[torch.isfinite(ts_g_err)]
            rs_g_err = rs_g_err[torch.isfinite(rs_g_err)]

            # pw_loss += g_loss



            # optimiser step
            chi_loss.backward(retain_graph=True)
            chisq_optim.step()


            # optimiser step
            pw_loss.backward()
            pw_optim.step()

            # zero grads.
            chisq_optim.zero_grad()
            pw_optim.zero_grad()

        # Disable grad calculation, don't need it for evaluation
        with torch.no_grad():
            y_big = torch.cat((chisq_en(x_big), pw_en(x_big)), axis=1).cpu()
            vis.line(y_big, x_big.cpu().expand_as(y_big), win='big',
                     opts=dict(showlegend=True, title="Energy (wide view)", legend=["chisq", "piecewise"]))
            y_small = torch.cat((chisq_en(x_small), pw_en(x_small)), axis=1).cpu()
            vis.line(y_small, x_small.cpu().expand_as(y_big), win='small',
                     opts=dict(showlegend=True, title="Energy (narrow view)", legend=["chisq", "piecewise"]))

            chi_test_loss = []
            pw_test_loss = []

            for (ts, rs), (lig_vtx,), (rec,) in test_loader:
                lig_vtx = lig_vtx.to(device)
                rec = rec.to(device)
                rankings = -torch.ones(BATCH_SIZE - 1).to(device)
                # extract "batch" from tensors in ts, rs and push both to device
                ts = ts[0].to(device)
                rs = rs[0].to(device)
                transf = ((rs @ lig_vtx.T) + ts.unsqueeze(2)).transpose(1, 2).reshape(-1, 3)
                dists = sample_bounds(rec, transf).unsqueeze(1)
                chi_energies = chisq_en(dists).reshape(BATCH_SIZE, -1).mean(dim=1)
                pw_energies = pw_en(dists).reshape(BATCH_SIZE, -1).mean(dim=1)
                chi_gt = chi_energies[0]
                chi_bad = chi_energies[1:]
                chi_gt = chi_gt.expand_as(chi_bad)
                chi_test_loss.append(F.margin_ranking_loss(chi_gt, chi_bad, rankings))
                pw_gt = pw_energies[0]
                pw_bad = pw_energies[1:]
                pw_gt = pw_gt.expand_as(pw_bad)
                pw_test_loss.append(F.margin_ranking_loss(pw_gt, pw_bad, rankings))

            res = torch.stack((torch.stack(chi_test_loss).mean(),
                               torch.stack(pw_test_loss).mean(),
                               )).unsqueeze(0)
            vis.line(
                X=np.array([[epoch, epoch]]),
                Y=np.log(res.cpu()),
                opts=dict(showlegend=True, title="Loss", legend=["chisq", "piecewise"]),
                win="Loss",
                update="append",
                )
        # save every 10 epochs
        if epoch % 10 == 0:
            torch.save(chisq_en, f"chi_res_{int(sys.argv[1]):03}_{epoch:03}.pt")
            torch.save(pw_en, f"pw_res_{int(sys.argv[1]):03}_{epoch:03}.pt")