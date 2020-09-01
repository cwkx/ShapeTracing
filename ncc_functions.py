"""
functions that can only be run on NCC
"""

import skfmm
import torch
import torch.nn as nn
import scipy.spatial.transform
import pymanopt # latest install from git https://github.com/pymanopt/pymanopt
from collections import deque, defaultdict
from pymanopt import Problem
from pymanopt.manifolds import Sphere, SpecialOrthogonalGroup, Euclidean, Product
from pymanopt.solvers.solver import Solver
from pymanopt.solvers import ParticleSwarm, SteepestDescent, TrustRegions, ConjugateGradient, NelderMead
from collections import deque, defaultdict
import geoopt
from skimage import measure
import numpy as np
import time
import sys, os
import calendar
import scipy.spatial.distance as dist

######## Spacial and graphing ########

def plot_stats(stats):
    upper_bound = go.Scatter(
        name='Max',
        y=stats['max'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        fillcolor='rgba(65, 68, 78, 0.3)',
        fill='tonexty')

    trace = go.Scatter(
        name='Mean',
        y=stats['mean'],
        mode='markers',
        marker=dict(
            size=8,
            color=stats['div']/np.max(stats['div']),
            colorbar=dict(
                title="Diversity"
            ),
            colorscale="Viridis"
        ),
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty')

    lower_bound = go.Scatter(
        name='Min',
        y=stats['min'],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines')

    data = [lower_bound, trace, upper_bound]

    layout = go.Layout(
        yaxis=dict(title='Surface area'),
        title='Diverse solution plotting',
        showlegend = False)

    figure = {
            'data': data,
            'win': 'r',
            'eid': None,
            'layout': layout,
            'opts': {'title':'stats plot'},
        }

    fig = go.Figure(figure)

    vis.plotlyplot(fig, win='stats')

def visualise(rec_vtx, rec_fac, gt_vtx, lig_vtx, lig_fac, pts, win="render"):
    
    data = [
    {
        'x': rec_vtx[:,0].tolist(),
        'y': rec_vtx[:,1].tolist(),
        'z': rec_vtx[:,2].tolist(),
        'mode':'markers',
        'marker':{
            'size':1,
            'color':'black'
        },
        'type':'scatter3d'
    },
    {
        'x': rec_vtx[:,0].tolist(),
        'y': rec_vtx[:,1].tolist(),
        'z': rec_vtx[:,2].tolist(),
        'i': rec_fac[:,0].tolist(),
        'j': rec_fac[:,1].tolist(),
        'k': rec_fac[:,2].tolist(),
        'color':'gray',
        'opacity': 0.7,
        'type':'mesh3d'
    },
    {
        'x': lig_vtx[:,0].tolist(),
        'y': lig_vtx[:,1].tolist(),
        'z': lig_vtx[:,2].tolist(),
        'i': lig_fac[:,0].tolist(),
        'j': lig_fac[:,1].tolist(),
        'k': lig_fac[:,2].tolist(),
        'color':'red',
        'opacity': 1.0,
        'type':'mesh3d'
    },
    {
        'x': gt_vtx[:,0].tolist(),
        'y': gt_vtx[:,1].tolist(),
        'z': gt_vtx[:,2].tolist(),
        'i': lig_fac[:,0].tolist(),
        'j': lig_fac[:,1].tolist(),
        'k': lig_fac[:,2].tolist(),
        'color':'green',
        'opacity': 0.5,
        'type':'mesh3d'
    },
    {
        'x': pts[:,0].tolist(),
        'y': pts[:,1].tolist(),
        'z': pts[:,2].tolist(),
        'mode':'markers',
        'marker':{
            'size':1,
            'opacity': 0.2,
            'color':'black'
        },
        'type':'scatter3d'
    }]
    
    layout = go.Layout(title='render')
    figure = {
            'data': data,
            'win': 'r2',
            'eid': None,
            'layout': layout,
            'opts': {'title':'iso'},
        }

    fig = go.Figure(figure)

    vis.plotlyplot(fig, win=win)

# accurate, allows for 2nd order differentiation, slow
# error = 0.0000037
def sample_bounds_slow(rec, p):

    grid_3d = rec.unsqueeze(3).unsqueeze(0) # B and C support
    sampling_points = p

    voxel_cube_shape = grid_3d.size()[-4:-1]
    
    batch_dims = sampling_points.size()[:-2]
    num_points = sampling_points.size()[-2]

    bottom_left = torch.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = bottom_left.long()
    top_right_index = top_right.long()
    x0_index, y0_index, z0_index = torch.unbind(bottom_left_index, dim=-1)
    x1_index, y1_index, z1_index = torch.unbind(top_right_index, dim=-1)
    index_x = torch.cat([x0_index, x1_index, x0_index, x1_index,
                         x0_index, x1_index, x0_index, x1_index], dim=-1)
    index_y = torch.cat([y0_index, y0_index, y1_index, y1_index,
                         y0_index, y0_index, y1_index, y1_index], dim=-1)
    index_z = torch.cat([z0_index, z0_index, z0_index, z0_index,
                         z1_index, z1_index, z1_index, z1_index], dim=-1)
    indices = torch.stack([index_x, index_y, index_z], dim=-1)
    clip_value = torch.tensor(voxel_cube_shape)-1
    indices[:,0] = torch.clamp(indices[:,0], 0, clip_value[0])
    indices[:,1] = torch.clamp(indices[:,1], 0, clip_value[1])
    indices[:,2] = torch.clamp(indices[:,2], 0, clip_value[2])
    content = grid_3d[:, indices[:,0],indices[:,1],indices[:,2]] # gather
    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0, z_z0 = torch.unbind(distance_to_bottom_left, dim=-1)
    x1_x, y1_y, z1_z = torch.unbind(distance_to_top_right, dim=-1)
    weights_x = torch.cat([x1_x, x_x0, x1_x, x_x0,
                           x1_x, x_x0, x1_x, x_x0], dim=-1)
    weights_y = torch.cat([y1_y, y1_y, y_y0, y_y0,
                           y1_y, y1_y, y_y0, y_y0], dim=-1)
    weights_z = torch.cat([z1_z, z1_z, z1_z, z1_z,
                           z_z0, z_z0, z_z0, z_z0], dim=-1)
    weights = (weights_x * weights_y * weights_z).view_as(content)

    interpolated_values = weights * content
    return interpolated_values.view(-1,interpolated_values.size(1)//8).sum(dim=0)

# accurate, does not allow for 2nd order differentation (pytorch doesn't support it yet here), fast
# error = 0.0000071
def sample_bounds(rec,p):
    # p expects [-1, 1] so we apply the transformation
    box = torch.tensor(rec.size()).to(device).float()
    t = p/(box-1)
    t = torch.stack([t[:,2],t[:,1],t[:,0]], dim=-1) * 2.0 - 1.0
    return torch.nn.functional.grid_sample(rec.unsqueeze(0).unsqueeze(0), t.unsqueeze(0).unsqueeze(0).unsqueeze(0), mode='bilinear', padding_mode='border', align_corners=True).squeeze()

def random_rotate(pts):
    r = scipy.spatial.transform.Rotation.random(1)
    m = r.as_matrix().astype(np.float32)
    m = torch.from_numpy(m).to(device).repeat(pts.size(0),1,1)
    result = torch.einsum('ijk,ik->ij', m, pts)
    return result, r.as_quat() # can return rotation in whatever format is most useful

def set_bounds(rec, bound_value=1e10):
    rec[0,:,:] = bound_value
    rec[:,0,:] = bound_value
    rec[:,:,0] = bound_value
    rec[rec.size(0)-1,:,:] = bound_value
    rec[:,rec.size(1)-1,:] = bound_value
    rec[:,:,rec.size(2)-1] = bound_value
    return rec

def show_solution(solution, win="render"):
    t,r = solution
    t = torch.tensor(t).float().to(device)
    r = torch.tensor(r).float().to(device)

    ext = 0.5*lig_sze # we get the extents of the ligand
    a_p = a-ext # this first line shifts the ligand on the origin
    a_r = r.unsqueeze(0).repeat(a_p.size(0),1,1)
    a_p = torch.einsum('ijk,ik->ij', a_r, a_p)
    a_p += t # finally its shifted

    phi_sample = sample_bounds(rec, a_p)
    surface_area = chris_new(phi_sample, k_const).sum() # warp_delta(phi_sample).sum()

    print('showing solution with score: '+str(surface_area.item()))
    print('rmsd of solution with gt: '+str( np.sqrt(((a_p-a)**2).mean().item()) ))

    visualise(rec_vtx, rec_fac, a.squeeze().cpu().numpy(), a_p.squeeze().cpu().numpy(), lig_fac, s.cpu().numpy(), win="render")
    return surface_area.item(), np.sqrt(((a_p-a)**2).mean().item())

######## Clustering methods ########

def group_solutions(t_params_tmp, q_params_tmp, thresh_t = 2., thresh_q = 0.1):
    """
    group solutions based on how close translations and quaternions are (given by thresh - this we loop through to see when error grows too big)
    This is identical to sub_manifold.dist, but much quicker and can be applied on large array of points
    The only difficulty is what you use to define the thresh with

    Follow https://engineering.purdue.edu/kak/Tutorials/ClusteringDataOnManifolds.pdf page 23
    cluster data on manifold abitrarily (by geodesic distance)
    begin by creating a distjoint series of K clusters (that we minimise an error for later)

    :params t_params_tmp: Translation parameters (n x 3)
    :params q_params_tmp: Corresponding quaternionos (n x 4), created prior using mat2quat()
    :params rmsd_params_tmp: rmsds corresponding to the above parameters (n)
    :params scores_params_tmp: scores corresponding to the above parameters (n)
    :params thresh_t: Translation distance cutoff (under default of 5 ang., t_params are considered grouped, provided...)
    :params thresh_q: Secondary group check with quaternion. 1 - the dot product of two quaternions must be < 0.1 (as well as trans group) to be groupped
    :returns: All the grouped parameters, rmsds and scores (scores, rmsd, t, q)
    """

    groups_t = []
    groups_q = []
    trans = True
    cnt = 0

    # group by translation and rotation
    while trans:
        rot = True # check through rotations
    
        norm = np.linalg.norm(t_params_tmp - t_params_tmp[0], axis=1)
        idx_t = np.where(norm < thresh_t)
        q_tmp = q_params_tmp[idx_t]
        t_tmp = t_params_tmp[idx_t]
    
        while rot:
            prod = np.dot(q_tmp, q_tmp[0])
            sim = np.abs(prod - 1.0) < thresh_q
            idx_q = np.where(sim)
            groups_t.append(t_tmp[idx_q])
            groups_q.append(q_tmp[idx_q])
    
            q_tmp = q_tmp[~np.asarray(sim)]
            t_tmp = t_tmp[~np.asarray(sim)]

            cnt+=1
            if len(q_tmp) == 0:
                rot = False
                break
            else:
                continue
    
        t_params_tmp = t_params_tmp[~np.asarray(norm < thresh_t)]
        q_params_tmp = q_params_tmp[~np.asarray(norm < thresh_t)]

        if len(t_params_tmp) == 0:
            trans = False
            break
        else:
            continue
        
    params = []
    mean = []
    for i in range(len(groups_t)):
        params.append(np.hstack((groups_t[i], groups_q[i])).astype('float32'))
        mean.append(np.mean(params[-1], axis=0).astype('float32'))

    # return the parameters and means of each cluster
    return params, mean

def construct_subspace(X, err = "recon"):
    """
    Construct the subspace given pre-clustered t and q parameters
    The mean and covariance of datapoints in subspace S, grouped_t and grouped_q are the roughly pre-grouped parameters
    for subspace S.
    See https://engineering.purdue.edu/kak/Tutorials/ClusteringDataOnManifolds.pdf page 23

    # Since the recon error is so temperamental, there is also the option to return the mean error (i.e. how far data points are from centre of cluster),
    # using the norm as the error itself.
    # Need to normalise this wrt the translation and quaternion norms being so different though
    """

    if err == "recon":
        if type(X) == int:
            m = np.zeros((1, 7))
            C0 = np.zeros(7)
            C = np.zeros((7, 7))        
        elif np.shape(X)[0] == 1 or X.ndim == 1:
            # X is the mean in this case
            m = X
            C0 = np.zeros(7)
            C = np.zeros((7, 7))
        else:
            m = np.mean(X, axis=0) # mean of X
            C0 = X - m
            C1 = np.transpose(X - m)
            C =  np.matmul(C1, C0) / (float(len(X)))# covariance of data points in X
        
        # eigendecomposition of C
        decomp = np.linalg.eigh(C) # use eigh as matrix is symmetric
        P = decomp[1][np.argmax(decomp[0])] # get the leading eigenvecor
        # then get the trailing eigenvectors
        mask = np.ones(decomp[1].shape[0], bool) # decomp[1] square matrix, so can select either row or column
        mask[np.argmax(decomp[0])] = False
        trail = np.transpose(decomp[1][mask]) # do the transpose here to save memory #### DON'T REDO LATER ####
    
        # Now the error by representing projection of xk in subspace using trailing eigenvectors
        e = np.matmul(C0, trail)
        d2 = np.linalg.norm(np.matmul(e, np.transpose(e)))**2
    else: 
        if np.shape(X)[0] == 1 or X.ndim == 1:
            # X is the mean in this case
            m = X
        else:
            m = np.mean(X, axis=0) # mean of X
        
        d2 = np.linalg.norm(X - m)
        trail = np.zeros(7)

    # we need to return the trailing eigenvectors so we can construct our similarity matrix in phase 2
    # What is you give high error when in a cluster by itself?
    return list(m.squeeze()), d2, trail # return the mean for iterative checks and an error for associated data k with subspace S

def regroup(space, params, mean, trail, error):
    # reiterate by reassigning all parameters to where they minimise their recon error (norm of mean distance also calc. just in case)
    # error is an array of the recon errors for each subspace defined by params and trail
    # Then recalculate the subspace / reconstruction errors
    # space is all raw params
    regrouped_params = np.ones(len(params), dtype=list)
    dist_array = dist.cdist(mean, space) # get all norm distances in one swoop
    for cnt, p in enumerate(space):
        pdist =  dist_array[:,cnt] # calc. distance between means

        C = p - mean
        products = np.einsum('ij,ijk->ik',C,trail) #  remove loop using einsum (same can be done with matmul but need to extract diagonals)
        d_tmp = np.einsum('ij,ij->i', products, products)**2

        d_norm = np.sum(d_tmp) / float(np.shape(d_tmp)[0])
        mean_norm = np.sum(pdist) / float(np.shape(pdist)[0])
        sim_d = np.exp(-1.0 * d_tmp / d_norm)
        sim_m = np.exp(-1.0 * pdist / mean_norm)
        # d better as doesn't reduce impact of quaternion...?
        W = (sim_d + sim_m) / 2 # use the combination of the two

        # only consider mean if we have one data point in the cluster
        if np.min(sim_d) == 0:
            W = sim_m
        else:
            W = (sim_d + sim_m) / 2 # use the combination of the two (when we have cluster of 1 data point, recon doesn't work) 

        if np.any(regrouped_params[np.argmax(W)] == 1):
            regrouped_params[np.argmax(W)] = p[None]     # Now the error by representing projection of xk in subspace using trailing eigenvectors
        else:       
            regrouped_params[np.argmax(W)] = np.vstack((regrouped_params[np.argmax(W)], p))
        #try just recon error
        #if np.any(regrouped_params[np.argmin(d_tmp)] == 1):
        #    regrouped_params[np.argmin(d_tmp)] = p[None]     # Now the error by representing projection of xk in subspace using trailing eigenvectors
        #else:       
        #    regrouped_params[np.argmin(d_tmp)] = np.vstack((regrouped_params[np.argmin(d_tmp)], p))   
        
    regrouped_params = regrouped_params.tolist()

    error = []
    mean = []
    trail = []
    for i in range(len(regrouped_params)):
        m, er, tr = construct_subspace(regrouped_params[i])
        trail.append(tr)
        error.append(er)
        mean.append(m)
    mean = np.asarray(mean)

    # If a cluster is destroyed, we need to break up the cluster with the largest error into how many necessary clusters we need
    dead_cluster = []
    for i, j in enumerate(regrouped_params):
        if np.any(j == 1):
            dead_cluster.append(i)
    
    if len(dead_cluster) != 0:
        # split each param in the worse cluster sequentially by how far removed they are from the mean of that cluster (translation larger impact here than quat)
        worst_list = top(np.asarray(error), len(dead_cluster)) # get n worst clusters to extract from
        cluster_mean = mean[worst_list]

        for cnt, idx in enumerate(dead_cluster):
            # reassign the worst coordiante in this cluster to a new cluster each iteration
            X = regrouped_params[worst_list[cnt]]
            C = np.linalg.norm(X - cluster_mean[cnt], axis=1)
            worst_mean = np.argmax(C)
            regrouped_params[idx] = X[worst_mean]
            X = np.delete(X, worst_mean, axis=0)

            # Now correct old bad cluster to account for removed coordinates
            regrouped_params[worst_list[cnt]] = X
        
        # Now reconstruct the subspace accordingly
        error = []
        mean = []
        trail = []
        for i in range(len(regrouped_params)):
            m, er, tr = construct_subspace(regrouped_params[i])
            trail.append(tr)
            error.append(er)
            mean.append(m)
        mean = np.asarray(mean)
    
        # The alternative as well is just to simply remove these clusters completely from the set
        # to to reassign n dead clusters to n worst clusters (remove one each time maybe...?)
  
    return mean, error, regrouped_params, trail

def reseed_single(params, mean, mean_cap = 5.):
    # reseed clusters that have only one data point into pre-existing ones
    single = []
    for cnt, p in enumerate(params):
        if np.shape(p)[0] == 1:
            single.append(cnt)

    # keep a record of which parameters were reset to 1
    reset_record = []
    for s in single:
        # Account for if we reseed two single data point clusters together
        if np.shape(params[s])[0] == 1:
            dist = np.linalg.norm(params[s] - mean, axis=1)
            # need second distance index to reseed to (latter one so we can skip later if its reseeded to single point cluster)
            idx = np.max(bot(dist, 2))
            if dist[idx] < mean_cap:
                # only reseed when the distance between two isn't too bad
                params[idx] = np.vstack((params[idx], params[s]))
    
                # keep clusters that have been removed (to preserve indexing for loop) but just reassign to one so we can remove later
                params[s] = 1            
                reset_record.append(s)
            else:
                continue
        else: 
            continue
    
    # now delete params == 1
    params = np.asarray(params)
    params = np.delete(params, reset_record)
    return params

def reduce_cluster(params, mean, trail):
    """
    Reduce a cluster to one representative vector per cluster (i.e. the one with the least recon. error)
    """

    small_params = [] # reduced clusters
    for cnt, p in enumerate(params):
        # loop through each params and find which vector has the best recon error as representative
        if np.shape(p)[0] == 1:
            small_params.append(p.squeeze())
        else:
            # loop through each set of params, find the coordinate with the least recon error and append as representative coord of cluster
            d_tmp = []
            for s in p:
                C = s - mean[cnt]
                e = np.matmul(C, trail[cnt])
                d2 = np.dot(e, e)**2
                d_tmp.append(d2)
            small_params.append(p[np.argmin(d_tmp)])

    return np.asarray(small_params)

def cluster_setup(p, list_range, t=2., q = 0.1, folder = '/home2/dclw29/JabberDock2/mini_protein_benchmark/convergence_check/'):
    """
    Take protein p and list_range (i.e. how many epochs) and perform all necessary setups for a cluster, including reoptimisation,
    before returning the necessary parameters in translation and rotation form

    :params p: protein name
    :params list_range: numbers corresponding to epochs from previous calculations (subject to change)
    :params t: threshold t (translation sphere) for initial clustering
    :params q: threshold q (quaternion dot product difference) for initial clustering
    :params folder: location of files with translation, quaternion etc. data
    :returns params: Returns the best coordinate corresponding to each cluster
    """

    best_t_tmp = []
    best_r_tmp = []
    for i in list_range: # change these numbers depending on what has been run
        best_t_tmp.append(np.load(folder + 'ts_' + p + str(i) + '.npy'))
        best_r_tmp.append(np.load(folder + 'rs_' + p + str(i) + '.npy'))

    best_t = cat(best_t_tmp)
    best_r = cat(best_r_tmp)

    reshape = np.shape(best_t)[0] * np.shape(best_t)[1]

    t_params = best_t.reshape(reshape, 3)
    r_params = best_r.reshape(reshape, 3, 3)
    # convert rotations to quaternions
    quat = []
    for i in range(len(r_params)):
        quat.append(list(mat2quat(r_params[i])))
    t_params_tmp = t_params
    q_params_tmp = np.asarray(quat)

    # create 7 dimensional manifold
    space = np.hstack((t_params_tmp, q_params_tmp)).astype('float32')

    # begin clustering
    print(">> Intialising cluster...")
    params, mean = group_solutions(t_params_tmp, q_params_tmp, thresh_t = t, thresh_q=q)
    params = reseed_single(params, mean)
    print(">> %i individual clusters found"%(len(params)))

    error = []
    mean = []
    trail = []
    print(">> Constructing initial subspace...")
    for i in range(len(params)):
        # jsut disregard trailing eigenvectors for now
        m, er, tr = construct_subspace(params[i], err="recon")
        error.append(er) # recon error for each subclass
        mean.append(m)
        trail.append(tr)
    mean = np.asarray(mean)
    error_total = np.sum(error)
    
    error_keep = error_total
    error_diff = True
    print("<===== Beginning optimisation of clusters =====>")
    time0 = time.time()
    cnt = 0
    while error_diff:
            mean, error, params, trail = regroup(space, params, mean, trail, error)
            error_total = np.sum(error)
            print(">> Error for round %i is %i"%(cnt, int(error_total)))
            cnt += 1
            if np.abs(error_keep - error_total) < 1.:
                error_diff = False
                break
            else:
                error_keep = error_total

    print(">> Optimisation complete. Total time taken: %fs"%(time.time() - time0))
    print(">> Reducing cluster size to best representative...")
    new_params = reduce_cluster(params, mean, trail)

    print(">> Done!")
    return new_params

######## Maths ########

def uniform_quat(s, t1, t2):
    """
    Generate a random quaternion (for uniform sampling)
    """

    sig1 = np.sqrt(1-s)
    sig2 = np.sqrt(s)
    theta1 = 2 * np.pi * t1
    theta2 = 2 * np.pi * t2

    w = np.cos(theta2) * sig2
    x = np.sin(theta1) * sig1
    y = np.cos(theta1) * sig1
    z = np.sin(theta2) * sig2
    return torch.tensor([w, x, y, z])

def dirac(x):
    return (1/np.pi)/(1+x**2)

def heaviside(x,epsilon=1):
    return 0.5*(1+(2./np.pi)*torch.atan(x/epsilon))

def dirac_eps(x, epsilon=1):
    return (epsilon/np.pi)/(epsilon**2+x**2)

def warp_phi(x, epsilon=1): # not c_infty
    return torch.relu(x)-torch.relu(torch.exp(-x)-1)

def warp_delta(x):
    return np.pi/(1.0+torch.relu(x)**2) -4.0*torch.relu(-x)**2

def energy(x, k=2.0):
    return (x+k) * torch.exp(-(x+k) / k) / k**2

def quat2mat(quat):
    a, b, c, d = quat[0], quat[1], quat[2], quat[3]

    a2, b2, c2, d2 = pow(a, 2), pow(b, 2), pow(c, 2), pow(d, 2)

    rotMat = np.stack([a2 + b2 - c2 - d2, 2*b*c - 2*a*d, 2*b*d + 2*a*c,
                         2*b*c + 2*a*d, a2 - b2 + c2 - d2, 2*c*d - 2*a*b,
                         2*b*d - 2*a*c, 2*c*d + 2*a*b, a2 - b2 - c2 + d2], axis=0).reshape(3,3)

    return rotMat

def mat2quat(mat):
    # taken from https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/

    m00 = mat[0][0]; m01 = mat[0][1]; m02 = mat[0][2]
    m10 = mat[1][0]; m11 = mat[1][1]; m12 = mat[1][2]
    m20 = mat[2][0]; m21 = mat[2][1]; m22 = mat[2][2]

    # First check trace to make sure we avoid gimble lock
    tr = m00 + m11 + m22
    # ensure resiliance of code
    if tr > 0:
        S = np.sqrt(tr+1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S 
    elif m11 > m22: 
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S 
        qy = 0.25 * S
        qz = (m12 + m21) / S 
    else: 
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

######## Array manipulation ########

def cat(list_array):
    # cat everything in the list
    cat = np.concatenate((list_array[0], list_array[1]))
        
    for l in list_array[2:]:
        cat = np.concatenate((cat, l))
    
    return cat

def top(arr, x = 10):
    #return the indices for the top x values of an array (ordered in terms of largest first)
    return arr.argsort()[-x:][::-1]

def bot(arr, x = 2):
    #return the indices for the bottom x values (ordered in terms of largest first)
    return arr.argsort()[:x][::-1]

def check_symmetric(a, tol=1e-8):
    # check if matrix is symmetric or not
    return np.all(np.abs(a-a.T) < tol)