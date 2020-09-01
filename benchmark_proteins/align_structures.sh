#!/bin/bash

filename=${1}
filelines=$(cat ${filename})

for line in ${filelines} ; do
    echo $line
    cd $line

    python ~/JabberDock2/benchmark_proteins/align_unbound_bound.py ${line}_r_u_sim.pdb ${line}_r_b.pdb ${line}_r_u_sim_map.pdb
    python ~/JabberDock2/benchmark_proteins/align_unbound_bound.py ${line}_l_u_sim.pdb ${line}_l_b.pdb ${line}_l_u_sim_map.pdb
    rm tmp1.pdb tmp1.seq tmp.ali tmp2.pdb tmp2.seq 

    cd ~/JabberDock2/benchmark_proteins/proteins/
done

