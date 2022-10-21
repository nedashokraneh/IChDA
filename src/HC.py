import sys
import os
sys.path.append('../utilities')
import data_utils
import json
import argparse
import pandas as pd
import numpy as np
import math
import subprocess
from dataset_class import dataset
from hmmlearn import hmm

def parse_arg():
    p = argparse.ArgumentParser()
    p.add_argument('-c', help="The config file including different datasets paths.")
    return p.parse_args()

def load_pos2ind_and_ind2pos_maps(valid_bins_file, valid_chroms, chr_sizes, resolution):

    valid_bins_df = pd.read_csv(valid_bins_file, sep = "\t", header = None)
    valid_bins_df.columns = ['chr_name', 'start', 'end', 'index']
    total_valid_bins = valid_bins_df.shape[0]
    valid_bins_df['pos'] = (valid_bins_df['start']/resolution).astype(int)
    valid_bins_df['index'] = np.arange(valid_bins_df.shape[0])
    pos2ind_dict = {}
    ind2pos_dict = []
    for chr_name in valid_chroms:
        chrom_size = math.ceil(chr_sizes[chr_name]/resolution)
        pos2ind_dict[chr_name] = np.empty(chrom_size)
        pos2ind_dict[chr_name][:] = np.nan
    for i,row in valid_bins_df.iterrows():
        pos2ind_dict[row['chr_name']][row['pos']] = row['index']
        ind2pos_dict.append((row['chr_name'],row['pos']))
    return total_valid_bins, pos2ind_dict, ind2pos_dict

def get_chunks_lengths(valid_chroms, total_valid_bins, pos2ind_dict):

    start_inds = []
    for chr_name in valid_chroms:
        isnan = np.isnan(pos2ind_dict[chr_name]).astype(int)
        ls_isnan = np.concatenate((np.array([True]),isnan[:-1]))
        #rs_isnan = np.concatenate((isnan[1:],np.array([True])))
        start_poses = np.where((isnan - ls_isnan) == -1)[0]
        start_inds.extend(list(pos2ind_dict[chr_name][start_poses]))
    start_inds = [int(start_ind) for start_ind in start_inds]
    end_inds = start_inds[1:] + [total_valid_bins]
    chunks_lengths = [e-s for s,e in zip(start_inds,end_inds)]
    return chunks_lengths

def write_annotation(labels, label_name, out_dir, resolution, ind2pos_dict):


    label_out_path = os.path.join(out_dir, '{}_annotation.txt'.format(label_name))
    label_out = open(label_out_path, 'w')
    last_chr, last_pos = ind2pos_dict[0]
    last_label = labels[0]
    last_start_pos = last_pos
    last_added = False
    for l_ind, label in enumerate(labels[1:]):
        chr_name, pos = ind2pos_dict[l_ind+1]
        if (last_chr != chr_name) or (last_pos+1 != pos) or (last_label != label):
            start = last_start_pos * resolution
            end = (last_pos+1) * resolution
            if (l_ind == labels[1:].shape[0]-1) and (last_label == label):
                end = (pos+1) * resolution
                last_added = True
            label_out.write("{}\t{}\t{}\t{}\n".format(last_chr, start, end, last_label))
            last_chr = chr_name
            last_start_pos = pos
            last_label = label
        if (l_ind == labels[1:].shape[0]-1) and (last_added == False):
            start = last_start_pos * resolution
            end = (pos+1) * resolution
            label_out.write("{}\t{}\t{}\t{}\n".format(chr_name, start, end, label))
        last_pos = pos
    label_out.close()

def run(args):
    f = open(args.c, "r")
    config = json.load(f)

    embeddings_dir = os.path.join(config["processed_dir"], "embeddings")
    if not os.path.exists(embeddings_dir):
        os.mkdir(embeddings_dir)
    embedding_path = os.path.join(embeddings_dir, "LINE_embedding.txt")
    if not os.path.exists(embedding_path):
        print("Starting the embedding process...")
        cmd = [config["line_path"], "-train", config["interactions_file"], "-order", "2",
                "-sample", "50", "-size", "8", "-output", embedding_path]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        p.wait()
        print("Finished the embedding process...")
    else:
        print("Line embedding exists...")

    print("Loading input datasets...")
    embeddings = pd.read_csv(embedding_path, skiprows = 1, sep = " ", header = None).iloc[:,:-1]
    embeddings.sort_values([0], inplace=True)
    embeddings = embeddings.iloc[:,1:]
    #embeddings.to_csv(os.path.join(config["processed_dir"], "structural_signals.txt"), sep = "\t", index = False, header = None)
    chr_sizes = data_utils.read_chr_sizes(config["chr_size_path"])
    total_valid_bins, pos2ind_dict, ind2pos_dict = load_pos2ind_and_ind2pos_maps(config["bins_file"], config["valid_chroms"], chr_sizes, config["resolution"])
    signals = np.arcsinh(np.loadtxt(config["signals_file"]))
    if signals.ndim == 1:
        signals = signals.reshape(-1,1)
    print("Finished loading input datasets...")

    print("Running HMM...")
    lengths = get_chunks_lengths(config["valid_chroms"], total_valid_bins, pos2ind_dict)
    all_signals = np.concatenate([signals,np.array(embeddings)], axis = 1)
    gmm_hmm_viterbi = hmm.GaussianHMM(n_components=config["num_labels"], covariance_type = 'full', algorithm='viterbi')
    gmm_hmm_viterbi.fit(all_signals, lengths)
    l = gmm_hmm_viterbi.predict(all_signals, lengths)

    print("Writing HMM_combined_k{} annotation...".format(config["num_labels"]))
    annotations_dir = os.path.join(config["processed_dir"], "annotations")
    if not os.path.exists(annotations_dir):
        os.mkdir(annotations_dir)
    write_annotation(l, "HMM_combined_k{}".format(config["num_labels"]), annotations_dir,
                    config["resolution"], ind2pos_dict)


if __name__ == '__main__':
    args = parse_arg()
    run(args)
