# This script gets folder of 1D binned bedgraph signals and hic file and create 3 files,
# 1) Input signals, 2) bin, 3) HiC significant interactions given different thresholds
# (10^-5, 5*10^-6, 10^-6, 5*10^-7, 10^-7)
# It also requires resolution of input signal

from subprocess import call
import math
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
import os
import argparse

parser = argparse.ArgumentParser(description='Create files to input to SPIN')
parser.add_argument('-s', "--signals_dir", required = True,
                   help='path of directory including binned bedgraph files')
parser.add_argument('-i', '--hic_file', required = True,
                   help='path of hic file to extract significant interactions from')
parser.add_argument('-j', '--juicer', required = True,
                   help='path of juicer tool')
parser.add_argument('-r', '--resolution', required = True, type = int,
                   help='resolution of bedgraphs')
parser.add_argument('-o', '--output_path', required = True,
                   help='folder path to save output files in')
args = parser.parse_args()

signals_dir = args.signals_dir
hic_path = args.hic_file
resolution = args.resolution
juicer_tool_path = args.juicer
out_path = args.output_path
input_path = os.path.join(out_path, "input.txt")
bin_path = os.path.join(out_path, "bin.txt")
hic_edges_paths = [os.path.join(out_path, "Hi-C-" + th + ".txt") for th in ["1e-5", "5e-6", "1e-6", "5e-7", "1e-7"]]
chain_edges_path = os.path.join(out_path, "chain_edges.txt")



def make_signals_df(signals_folder_path):
    signals_df = pd.DataFrame()
    for file in os.listdir(signals_folder_path):
        if file.endswith(".bedgraph"):
            signal = pd.read_csv(os.path.join(signals_folder_path, file), sep = "\t", header = None)
            signal = signal.iloc[:,[0,1,3]]
            signal_name = os.path.splitext(file)[0]
            signal.columns = ["chr_name", "pos", signal_name]
            if signals_df.empty:
                signals_df = signal
            else:
                signals_df = pd.merge(signals_df, signal, how = 'inner', on = ['chr_name','pos'])
    return(signals_df)

valid_chromosomes = ["chr" + str(i+1) for i in range(22)]
signals_df = make_signals_df(signals_dir)
signals_df = signals_df[signals_df['chr_name'].isin(valid_chromosomes)]
signals_df = signals_df[(signals_df.iloc[:,2]!=0)]
signals_df.iloc[:,2:].to_csv(input_path, header = None, index = False, sep = "\t")

bin = signals_df.iloc[:,[0,1]]
bin.loc[:,'start'] = bin.loc[:,'pos']
bin.loc[:,'end'] = bin.loc[:,'pos']+resolution
bin.loc[:,'index'] = np.arange(bin.shape[0])
bin.drop(columns = ['pos'], inplace = True)
bin.to_csv(bin_path, header = None, index = False, sep = "\t")

chr_pos_dict = {}
for i,row in bin.iterrows():
    chr_pos_dict[(row['chr_name'],row['start'])] = row['index']

HiC_dfs = []

for i in range(5):
    HiC_dfs[i] = pd.DataFrame({"source":[], "target":[], "weight":[]})
    HiC_dfs[i]['source'] = HiC_dfs[i]['source'].astype(int)
    HiC_dfs[i]['target'] = HiC_dfs[i]['target'].astype(int)
    HiC_dfs[i]['weight'] = HiC_dfs[i]['weight'].astype(int)

for chrm1 in np.arange(1,23):
    for chrm2 in np.arange(chrm1,23):
        cmd = "java -jar " + juicer_tool_path + " dump oe KR " + hic_path + " " + str(chrm1) + " " + str(chrm2) + " BP " + str(resolution) + " curr_file.txt"
        call([cmd],shell=True)
        chrm1_name = "chr" + str(chrm1)
        chrm2_name = "chr" + str(chrm2)
        edges = pd.read_csv("curr_file.txt", sep = "\t", header = None)
        edges = edges[~np.isnan(edges.iloc[:,2])]
        weights = np.log2(edges.iloc[:,2])
        norm_params = sp.stats.norm.fit(weights)
        th = sp.stats.norm.ppf(0.99999,norm_params[0], norm_params[1])
        edges = edges[weights>th]
        edges['source'] = [chr_pos_dict[(chrm1_name,source_pos)] for source_pos in edges.iloc[:,0]]
        edges['target'] = [chr_pos_dict[(chrm2_name,target_pos)] for target_pos in edges.iloc[:,1]]
        edges = edges.loc[:,['source', 'target']]
        edges['weight'] = 1
        HiC_df = HiC_df.append(edges)
        edges = edges[edges['source']!=edges['target']]
        temp = edges['source'].copy()
        edges['source'] = edges['target']
        edges['target'] = temp
        HiC_df = HiC_df.append(edges)
chain_edges = pd.DataFrame({"source":[], "target":[], "weight":[]})
for chr_name in valid_chromosomes:
    indices = bin[bin['chr_name'] == chr_name]['index']
    chain_edges = chain_edges.append(pd.DataFrame({"source":indices[:-1], "target":indices[:-1]+1, "weight":1}))
    chain_edges = chain_edges.append(pd.DataFrame({"source":indices[:-1]+1, "target":indices[:-1], "weight":1}))
HiC_df = HiC_df.append(chain_edges)
HiC_df['source'] = HiC_df['source'].astype(int)
HiC_df['target'] = HiC_df['target'].astype(int)
HiC_df['weight'] = HiC_df['weight'].astype(int)
HiC_df = HiC_df.sort_values(['source', 'target'])
HiC_df.to_csv(hic_edges_path, header = None, index = False, sep = "\t")
chain_edges['source'] = chain_edges['source'].astype(int)
chain_edges['target'] = chain_edges['target'].astype(int)
chain_edges['weight'] = chain_edges['weight'].astype(int)
chain_edges = chain_edges.sort_values(['source', 'target'])
chain_edges.to_csv(chain_edges_path, header = None, index = False, sep = "\t")
