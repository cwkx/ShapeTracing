import biobox as bb
import numpy as np
import parse_alignment as pa
import sys, os
import JabberDock as jd
from copy import deepcopy 

def apply_global_transformation(pdb, M):
    '''
    Apply a 3x3 transformation matrix and loop through frames

    :param pdb: Multi pdb structures
    :param M: rotation matrix
    '''
    no_frames = np.shape(pdb.coordinates)[0]

    for f in range(no_frames):
        pdb.coordinates[f, :, :] = np.dot(pdb.coordinates[f, :, :], M)

def translate(pdb, x, y, z):
    '''
    translate the whole structure by a given amount.
    :param x: translation around x axis
    :param y: translation around y axis
    :param z: translation around z axis
    '''

    # if center has not been defined yet (may happen when using
    # subclasses), compute it
    if 'center' not in pdb.properties:
        pdb.get_center()

    # translate all points
    pdb.properties['center'][0] += x
    pdb.properties['center'][1] += y
    pdb.properties['center'][2] += z

    pdb.coordinates[:, :, 0] += x 
    pdb.coordinates[:, :, 1] += y 
    pdb.coordinates[:, :, 2] += z

def align_molecule(fname1, fname2, fname3):
    '''
    align two molecules (given two filenames)
    Aligns molecule given by fname1 onto fname2, then writes a new aligned simulation
    :param fname1: simulation data
    :param fname2: bound structure
    :param fname3: single frame structure of simulation (sim_map)
    '''

    # name for outfile
    outname = fname1.split('.')[0]

    pdb1 = bb.Molecule()
    pdb2 = bb.Molecule()

    pdb1.import_pdb(fname1)
    pdb2.import_pdb(fname2)

    # Use the parse alignment module to generate an alignment file get the sequence mapping 
    pa.get_alignment(fname3, fname2)
    R_match = pa.get_seq_map()
    
    # just in case the counting is off:
    res_match = pdb1.match_residue(pdb2)
    if res_match[0][0] == R_match[:,0][0] and res_match[1][0] == R_match[0,1]:
        Ru_idxs = pdb1.atomselect("*", R_match[:, 0], "CA", get_index=True)[1]
        Rb_idxs = pdb2.atomselect("*", R_match[:, 1], "CA", get_index=True)[1]
    else:
        Ru_idxs = pdb1.atomselect("*", res_match[0], "CA", get_index=True)[1]
        Rb_idxs = pdb2.atomselect("*", res_match[1], "CA", get_index=True)[1]        
    
    # center all molecules to the origin before applying rotations etc
    Ru_center = deepcopy(pdb1.get_center())
    Rb_center = deepcopy(pdb2.get_center())
    translate(pdb1, -Ru_center[0], -Ru_center[1], -Ru_center[2])
    translate(pdb2, -Rb_center[0], -Rb_center[1], -Rb_center[2])
    
    # We need to first calculate the rotation matrix to apply to the entire molecule
    # we can do this with the RMSD of the respective CA in unbound and bound states
    Ru_subset = pdb1.get_subset(Ru_idxs, conformations=[0])
    Rb_subset = pdb2.get_subset(Rb_idxs)
    
    Rb_subset.add_xyz(Ru_subset.coordinates)
    # just get rotation matrix
    rot_mat = Rb_subset.rmsd(0, 1, full=True)[1]
    
    # now we can apply the rotation to the unbound simulation
    apply_global_transformation(pdb1, rot_mat)
    
    # apply transformation back to original position before realigning
    translate(pdb1, Ru_center[0], Ru_center[1], Ru_center[2])
    translate(pdb2, Rb_center[0], Rb_center[1], Rb_center[2])
    
    # need to apply transformation to every frame I guess.... (only works on first)
    # but need to move by translation of the subset (i.e. matching!)
    Ru_subset = pdb1.get_subset(Ru_idxs, conformations=[0])
    Rb_subset = pdb2.get_subset(Rb_idxs)
    trans = Ru_subset.get_center() - Rb_subset.get_center()
    translate(pdb1, -trans[0], -trans[1], -trans[2])

    # okay... this works
    pdb1.write_pdb(outname + '_align.pdb')

fname1 = str(sys.argv[1]) # simulation
fname2 = str(sys.argv[2]) # bound molecule
fname3 = str(sys.argv[3]) # fixed structure for parse alignment

align_molecule(fname1, fname2, fname3)
