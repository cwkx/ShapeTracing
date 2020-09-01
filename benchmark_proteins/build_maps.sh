#!/bin/bash

filename=${1}
filelines=$(cat ${filename})

for line in ${filelines} ; do
    echo $line
    cd $line

    #build_map.py -i ${line}_r_u_sim_align.pdb -ff ~/biobox/classes/amber14sb.dat
    #build_map.py -i ${line}_l_u_sim_align.pdb -ff ~/biobox/classes/amber14sb.dat
    ~/JabberDock2/benchmark_proteins/build_map_duel.py -i1  ${line}_r_u_sim_align.pdb -i2 ${line}_l_u_sim_align.pdb -ff_dat ~/biobox/classes/amber14sb.dat

    cd ~/JabberDock2/benchmark_proteins/proteins/

done
