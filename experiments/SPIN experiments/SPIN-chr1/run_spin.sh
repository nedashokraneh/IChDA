#!/bin/bash
#SBATCH --time=4:00:00


for edge in 'oe'; do
    for i in 1 2 4 8; do
        for signal in 'tsa' 'chip'; do
            python ../../src/main.py -i "data/chr1_${signal}.txt" --hic "data/chr1_edges_${edge}${i}.txt" -w 25000 -n 8 -o "chr1_results/${signal}_${edge}${i}" -g "data/chr1_bins.txt" --save
            ../../src/merge2bed.sh "data/chr1_bins.txt" "chr1_results/${signal}_${edge}${i}/state_8" "chr1_results/${signal}_${edge}${i}/annotation.bed"
        done
    done
done
