import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix
import math
import os
import json
import subprocess
import shutil
import gzip
import HiCKRy

hg19_chr_sizes = {'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276,
 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431,
  'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540,
   'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983,
    'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566}

hg38_chr_sizes = {'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
 'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636, 'chr9': 138394717,
  'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309, 'chr13': 114364328, 'chr14': 107043718,
   'chr15': 101991189, 'chr16': 90338345, 'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616,
    'chr20': 64444167, 'chr21': 46709983, 'chr22': 50818468}

project_path = "/media/nedooshki/f4f0aea6-900a-437f-82e1-238569330477/genome-structure-function-aggregation"
juicer_path = os.path.join(project_path, 'utilities/juicer_tools_1.22.01.jar')


def is_file_genomedata(filepath):
    filename = os.path.basename(filepath)
    file_extension = filename.split(".")[-1]
    return file_extension == 'genomedata'


def read_chr_sizes(chr_size_file):
    chr_sizes = {}
    with open(chr_size_file) as f:
        for line in f:
            chr_name, chr_size = line.rstrip("\n").split("\t")
            chr_sizes[chr_name] = int(chr_size)
    return chr_sizes

def read_chr_arm_sizes(chr_arm_size_file):
    chr_arm_sizes = {}
    with open(chr_arm_size_file) as f:
        for line in f:
            chr_name, chr_arm, arm_size = line.rstrip("\n").split("\t")
            chr_arm_sizes[chr_name,chr_arm] = int(arm_size)
    return chr_arm_sizes

def normalize_dict(dict):
    total = 0
    for key in dict:
        total = total + dict[key]
    for key in dict:
        dict[key] = dict[key]/total
    return dict

def make_coverage(labels):
    counts = np.unique(labels[labels!=None],return_counts = True)
    labels_counts = {}
    for l,label in enumerate(counts[0]):
        labels_counts[label] = counts[1][l]
    labels_counts = normalize_dict(labels_counts)
    return labels_counts

def make_sym_table(pd_table):

    indices = pd_table.index
    columns = pd_table.columns
    if list(indices) != list(columns):
        print('this is a non symmetric matrix...')
        return
    #labels = np.union1d(indices, columns)
    new_pd_table = pd.DataFrame(columns = columns, index = indices)
    for l,label1 in enumerate(columns):
        for label2 in columns[l:]:
            if label1 == label2:
                new_pd_table.loc[label1, label2] = pd_table.loc[label1, label2]
                if np.isnan(new_pd_table.loc[label1, label2]):
                    new_pd_table.loc[label1, label2] = 0
            else:
                count1 = pd_table.loc[label1,label2]
                if np.isnan(count1):
                    count1 = 0
                count2 = pd_table.loc[label2,label1]
                if np.isnan(count2):
                    count2 = 0
                new_pd_table.loc[label1, label2] = count1 + count2
                new_pd_table.loc[label2, label1] = count1 + count2
    return new_pd_table

def normalize(table_df, row_coverages, col_coverages):
    total_counts = np.nansum(np.array(table_df))
    oe_table_df = pd.DataFrame(index = table_df.index, columns = table_df.columns)
    for ind in table_df.index:
        for col in table_df.columns:
            ind_coverage = row_coverages[ind]
            col_coverage = col_coverages[col]
            expected_count = total_counts * ind_coverage * col_coverage
            oe_table_df.loc[ind,col] = round(table_df.loc[ind,col]/expected_count,2)
    return oe_table_df

def read_config(config_path):
    out_config = {}
    in_config = json.load(open(config_path))
    #if 'valid_chroms' in in_config:
    #    out_config['valid_chroms'] = ['chr{}'.format(c) for c in in_config['valid_chroms']]
    #if 'hg19_chr_sizes_file' in in_config:
    #    out_config['hg19_chr_sizes_file'] = in_config['hg19_chr_sizes_file']
    #return out_config
    return in_config

def read_gene_info_file(file_path):
    genes_info = []
    with open(file_path) as f:
        for line in f:
            gene_info = line.split("\t")[0:6]
            genes_info.append(gene_info)
    genes_info = pd.DataFrame(genes_info)
    genes_info.columns = ['gene_id', 'chr', 'start', 'end', 'strand', 'type']
    return genes_info

def read_genes_rpkm_file(file_path):
    genes_RPKM = []
    with open(file_path) as f:
        for line in f:
            gene_RPKM = line.rstrip().split("\t")
            genes_RPKM.append(gene_RPKM)
    column_names = genes_RPKM[0]
    genes_RPKM = pd.DataFrame(genes_RPKM[1:])
    genes_RPKM.columns = column_names
    return genes_RPKM

def read_annot_file(annot_path, vir_res, resolution, skip_first_row):
    if skip_first_row:
        annot_df = pd.read_csv(annot_path, sep = "\t", header = None, skiprows = 1)
    else:
        annot_df = pd.read_csv(annot_path, sep = "\t", header = None)
    annot_df = annot_df.iloc[:,0:4]
    if not vir_res:
        annot_df.iloc[:,1:3] = (annot_df.iloc[:,1:3]/resolution).astype(int)
    annot_df.columns = ['chr_name', 'start', 'end', 'label']
    return(annot_df)

def melt_annotation(annot_df):
    annot_df = annot_df.reset_index()
    melt = annot_df.melt(id_vars=['index', 'chr_name', 'label'], value_name = 'pos').drop('variable', axis=1)
    melt_annot_df = melt.groupby('index').apply(lambda x: x.set_index('pos', drop = False)\
                                    .reindex(range(x.loc[:,'pos'].values[0], x.loc[:,'pos'].values[1])))\
                                .ffill().drop(['pos','index'], axis = 1).reset_index(level = 'pos').reset_index(drop=True)
    return(melt_annot_df)

# valid_bin1 is assumed to have a higher resolution (smaller resolution)
def align_two_resolution(valid_bin_path1, valid_bin_path2, resolution1, resolution2):
    gcd_res = math.gcd(resolution1, resolution2)
    replicate_value = int(resolution1/gcd_res)
    valid_bins1 = pd.read_csv(valid_bin_path1, sep = "\t", header = None)
    valid_bins1.columns = ['chr_name', 'start', 'end', 'index1']
    valid_bins1 = valid_bins1.loc[valid_bins1.index.repeat(replicate_value),:]
    bias_list = [i*gcd_res for i in range(replicate_value)]
    valid_bins1['start'] = valid_bins1['start'] + bias_list*int(valid_bins1.shape[0]/replicate_value)
    valid_bins1['start'] = [int(s/resolution2)*resolution2 for s in valid_bins1['start']]
    valid_bins2 = pd.read_csv(valid_bin_path2, sep = "\t", header = None)
    valid_bins2.columns = ['chr_name', 'start', 'end', 'index2']
    valid_bins = pd.merge(valid_bins1, valid_bins2, on = ['chr_name', 'start'], how = 'left')
    return valid_bins.loc[:,['index1','index2']]

def get_subset(list, indices):

    sublist = []
    for index in indices:
        if np.isfinite(index):
            sublist.append(list[int(index)])
        else:
            sublist.append(None)
    return sublist

def get_aligned_labels(labels1, labels2, index_mapper):

    labels1 = get_subset(labels1, index_mapper['index1'])
    labels2 = get_subset(labels2, index_mapper['index2'])
    labels = pd.DataFrame({'label1': labels1, 'label2': labels2})
    labels = labels.dropna()
    labels = labels.astype(int)
    return labels

def InputsAndBins_to_genomedata(inputs_file_path, signals_names, bins_file_path, assembly, resolution, genomedata_path):
    if not os.path.exists("dump_inputs"):
        os.mkdir("dump_inputs")
    bins = pd.read_csv(bins_file_path, sep = "\t", header = None)
    bins.columns = ['chr_name', 'start', 'end', 'index']
    inputs = pd.read_csv(inputs_file_path, sep = "\t", header = None)
    inputs.columns = signals_names
    for i in range(len(signals_names)):
        signal_bed = pd.concat([bins.iloc[:,0:3],inputs.loc[:,signals_names[i]]], axis = 1)
        signal_bed['start'] = (signal_bed['start']/resolution).astype(int)
        signal_bed['end'] = (signal_bed['end']/resolution).astype(int)
        signal_bed.to_csv(os.path.join("dump_inputs", signals_names[i]+".bed"), index = False, header = None, sep = "\t")

    ### creating chromosomes size file with valid chromosomes and their virtual sizes ###
    chr_size_file = open('chr_size_file.txt', 'w')
    for chr_name in np.unique(bins['chr_name']):
        if assembly == 'hg19':
            chr_size = math.ceil(hg19_chr_sizes[chr_name]/resolution)
        elif assembly == 'hg38':
            chr_size = math.ceil(hg38_chr_sizes[chr_name]/resolution)
        else:
            print ('invalid assembly')
            chr_size_file.close()
            os.remove("chr_size_file.txt")
            os.rmdir("dump_inputs")
            return
        chr_size_file.write(chr_name + '\t' + str(chr_size) + '\n')
    chr_size_file.close()
    #######################################################################################

    genomedata_load_cmd = ['genomedata-load', '-s', 'chr_size_file.txt']
    for signal_name in signals_names:
        signal_path = os.path.join("dump_inputs", signal_name+".bed")
        genomedata_load_cmd.append('-t')
        genomedata_load_cmd.append('{}={}'.format(signal_name,signal_path))
    genomedata_load_cmd.append('--sizes')
    genomedata_load_cmd.append(genomedata_path)
    p = subprocess.Popen(genomedata_load_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    os.remove("chr_size_file.txt")
    shutil.rmtree("dump_inputs")

def get_chr_pos_to_index(data_path, cell, resolution):
    valid_bins_path = os.path.join(data_path, "valid_bins", cell+"_"+str(resolution)+"_bins.txt")
    valid_bins = pd.read_csv(valid_bins_path, sep = "\t", header = None)
    valid_bins.columns = ['chr_name', 'start', 'end', 'index']
    valid_bins['pos'] = [int(p/resolution) for p in valid_bins['start']]
    valid_bins = valid_bins[['chr_name','pos','index']]
    chr_pos_to_index = {}
    for chr_name in ['chr' + str(c) for c in np.arange(1,23)]:
        chr_pos_to_index[chr_name] = np.empty(math.ceil(hg19_chr_sizes[chr_name]/resolution))
        chr_pos_to_index[chr_name][:] = np.nan
        curr_chr_indices = valid_bins[valid_bins['chr_name']==chr_name]
        chr_pos_to_index[chr_name][curr_chr_indices['pos'].values] = curr_chr_indices['index'].values
    return chr_pos_to_index

def get_chr_pos_to_index2(bin_file, resolution, assembly):
    valid_bins = pd.read_csv(bin_file, sep = "\t", header = None)
    valid_bins.columns = ['chr_name', 'start', 'end', 'index']
    valid_bins['pos'] = [int(p/resolution) for p in valid_bins['start']]
    valid_bins = valid_bins[['chr_name','pos','index']]
    chr_pos_to_index = {}
    for chr_name in ['chr' + str(c) for c in np.arange(1,23)]:
        if assembly == 'hg19':
            chr_pos_to_index[chr_name] = np.empty(math.ceil(hg19_chr_sizes[chr_name]/resolution))
        elif assembly == 'hg38':
            chr_pos_to_index[chr_name] = np.empty(math.ceil(hg38_chr_sizes[chr_name]/resolution))
        else:
            print('invalid assembly...')
            return
        chr_pos_to_index[chr_name][:] = np.nan
        curr_chr_indices = valid_bins[valid_bins['chr_name']==chr_name]
        chr_pos_to_index[chr_name][curr_chr_indices['pos'].values] = curr_chr_indices['index'].values
    return chr_pos_to_index

def get_index_to_chr_pos(data_path, cell, resolution):
    valid_bins_path = os.path.join(data_path, "valid_bins", cell+"_"+str(resolution)+"_bins.txt")
    valid_bins = pd.read_csv(valid_bins_path, sep = "\t", header = None)
    valid_bins.columns = ['chr_name', 'start', 'end', 'index']
    valid_bins['pos'] = [int(p/resolution) for p in valid_bins['start']]
    valid_bins = valid_bins[['chr_name','pos']]
    index_to_chr_pos = []
    for i,row in valid_bins.iterrows():
        index_to_chr_pos.append((row['chr_name'],row['pos']))
    return index_to_chr_pos

def dump_hic(hic_path, first_chr, second_chr, resolution, type, dump_dir):
    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)
    if not os.path.exists(os.path.join(dump_dir,type+"_"+str(resolution))):
        os.mkdir(os.path.join(dump_dir,type+"_"+str(resolution)))
    dump_name = "chr" + str(first_chr) + "_chr" + str(second_chr) + ".txt"
    dump_path = os.path.join(dump_dir,type+"_"+str(resolution),dump_name)
    if not os.path.exists(dump_path):
        cmd = ["java", "-jar", juicer_path, "dump", type, "VC", hic_path, first_chr, second_chr, "BP", resolution, dump_path]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #print('stderr: {}'.format(p.communicate()[1]))

def create_contact_list(data_path, cell, first_chr, second_chr, resolution, type, assembly):
    chr_pos_to_index = get_chr_pos_to_index(data_path, cell, resolution)
    hic_path = os.path.join(data_path, cell, "Hi-C", assembly, type + "_" + str(resolution), first_chr + "_" + second_chr + ".txt")
    contact_list = pd.read_csv(hic_path, sep = "\t", header = None)
    contact_list.columns = ['source', 'target', 'weight']
    contact_list['id1'] = [chr_pos_to_index[first_chr][int(p/resolution)] for p in contact_list['source']]
    contact_list['id2'] = [chr_pos_to_index[second_chr][int(p/resolution)] for p in contact_list['target']]
    contact_list = contact_list[['id1','id2','weight']]
    contact_list = contact_list.dropna()
    contact_list[['id1','id2']] = contact_list[['id1','id2']].astype(int)
    return contact_list

def create_contact_list_woInd(data_path, cell, first_chr, second_chr, resolution, type, assembly):
    hic_path = os.path.join(data_path, cell, "Hi-C", assembly, type + "_" + str(resolution), first_chr + "_" + second_chr + ".txt")
    contact_list = pd.read_csv(hic_path, sep = "\t", header = None)
    contact_list.columns = ['source', 'target', 'weight']
    contact_list['source'] = [int(s/resolution) for s in contact_list['source']]
    contact_list['target'] = [int(s/resolution) for s in contact_list['target']]
    contact_list = contact_list.dropna()
    return contact_list

def sym_mat(mat):
    mat_size = mat.shape[0]
    i_upper = np.triu_indices(mat_size, 1)
    i_lower = (i_upper[1],i_upper[0])
    mat[i_lower] = mat[i_upper]
    return mat

def filter_long_interactions(mat,k):
    mat2 = mat.copy()
    for kk in np.arange(k,dim):
        mat2[np.arange(0,dim-kk),np.arange(kk,dim)] = 0
        mat2[np.arange(kk,dim),np.arange(0,dim-kk)] = 0
    return mat2

def create_matrix(data_path, cell, first_chr, second_chr, resolution, type, assembly):
    row_num = math.ceil(hg19_chr_sizes[first_chr]/resolution)
    col_num = math.ceil(hg19_chr_sizes[second_chr]/resolution)
    mat = np.zeros([row_num,col_num])
    contact_list = create_contact_list_woInd(data_path, cell, first_chr, second_chr, resolution, type, assembly)
    for index,row in contact_list.iterrows():
        if not np.isnan(row[2]):
            mat[int(row[0]),int(row[1])] = row[2]
    if first_chr == second_chr:
        mat = sym_mat(mat)
    return mat

def create_matrix_from_file(file_path, first_chr, second_chr, resolution, assembly):
    if assembly == 'hg19':
        row_num = math.ceil(hg19_chr_sizes[first_chr]/resolution)
        col_num = math.ceil(hg19_chr_sizes[second_chr]/resolution)
    elif assembly == 'hg38':
        row_num = math.ceil(hg38_chr_sizes[first_chr]/resolution)
        col_num = math.ceil(hg38_chr_sizes[second_chr]/resolution)
    else:
        print('invalid assembly...')
        return
    contact_list = pd.read_csv(file_path, sep = "\t", header = None)
    contact_list.columns = ['source', 'target', 'weight']
    contact_list['source'] = [int(s/resolution) for s in contact_list['source']]
    contact_list['target'] = [int(s/resolution) for s in contact_list['target']]
    contact_list = contact_list.dropna()
    mat = csc_matrix((contact_list['weight'], (contact_list['source'], contact_list['target'])), shape=(row_num, col_num)).toarray()
    if first_chr == second_chr:
        mat = sym_mat(mat)
    return mat

def remove_zero_rc(mat):
    valid_indices = np.where(mat.sum(0)!=0)[0]
    return valid_indices, mat[np.ix_(valid_indices, valid_indices)]

def create_matrix_from_df(df,chr1, chr2, resolution, assembly):
    if assembly == 'hg19':
        chr1_size = math.ceil(hg19_chr_sizes[chr1]/resolution)
        chr2_size = math.ceil(hg19_chr_sizes[chr2]/resolution)
    elif assembly == 'hg38':
        chr1_size = math.ceil(hg38_chr_sizes[chr1]/resolution)
        chr2_size = math.ceil(hg38_chr_sizes[chr2]/resolution)
    else:
        print ('invalie assembly')
        return
    df_mat = np.zeros((chr1_size,chr2_size))
    for i, row in df.iterrows():
        ind1 = int(df['source']/resolution)
        ind2 = int(df['target']/resolution)
        df_mat[ind1,ind2] = df['weight']
        df_mat[ind2,ind1] = df['weight']
    return df_mat


def hic_to_fithic_files(hic_file, first_chr, second_chr, resolution, out_dir):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    interactions_file_path = os.path.join(out_dir,"interactions.txt.gz")
    fragments_file_path = os.path.join(out_dir, "fragments.txt.gz")
    biases_file_path = os.path.join(out_dir, "biases.txt.gz")
    interactions_file = gzip.open(interactions_file_path, "wt")
    fragments_sum = {}
    with open(hic_file) as f:
        for line in f:
            pos1, pos2, weight = line.split("\t")
            pos1 = int(pos1)
            pos2 = int(pos2)
            weight = float(weight)
            mid1 = pos1 + int(resolution/2)
            mid2 = pos2 + int(resolution/2)

            if (first_chr, mid1) in fragments_sum:
                fragments_sum[first_chr, mid1] = fragments_sum[first_chr, mid1] + weight
            else:
                fragments_sum[first_chr, mid1] = weight
            if not((first_chr == second_chr) and (mid1 == mid2)):
                if (second_chr, mid2) in fragments_sum:
                    fragments_sum[second_chr, mid2] = fragments_sum[second_chr, mid2] + weight
                else:
                    fragments_sum[second_chr, mid2] = weight

            interactions_file.write("{}\t{}\t{}\t{}\t{}\n".format(first_chr,str(mid1),second_chr,str(mid2),str(weight)))
    interactions_file.close()
    fragments_sum_df = []
    for (chr,mid) in fragments_sum:
        fragments_sum_df.append({'chr':chr, 'mid':mid, 'weight':fragments_sum[chr,mid]})
    fragments_sum_df = pd.DataFrame(fragments_sum_df)
    fragments_sum_df.sort_values(['chr','mid'], inplace = True)
    fragments_sum_df['extra'] = 0
    fragments_sum_df['mappable'] = 1
    fragments_sum_df = fragments_sum_df[['chr','extra','mid','weight','mappable']]
    fragments_sum_df['weight'] = fragments_sum_df['weight'].astype(int)
    fragments_sum_df.to_csv(fragments_file_path, sep = "\t", header = None, index = False, compression='gzip')

    matrix,revFrag = HiCKRy.loadfastfithicInteractions(interactions_file_path, fragments_file_path)
    bias = HiCKRy.returnBias(matrix, 0.05)
    HiCKRy.checkBias(bias)
    HiCKRy.outputBias(bias, revFrag, biases_file_path)

    if first_chr == second_chr:
        fithic_cmd = ['fithic', '-i', interactions_file_path, '-f', fragments_file_path, '-o', out_dir, '-r', str(resolution), '-t', biases_file_path, '-x', 'intraOnly']
    else:
        fithic_cmd = ['fithic', '-i', interactions_file_path, '-f', fragments_file_path, '-o', out_dir, '-r', str(resolution), '-t', biases_file_path, '-x', 'interOnly']

    p = subprocess.Popen(fithic_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    os.remove(interactions_file_path)
    os.remove(fragments_file_path)
    os.remove(biases_file_path)

def bin_bedgraph_file(infile, resolution, outfile, save_vir_res = False):
    o = open(outfile, "w")
    with open(infile, "r") as input:
        last_chr = ""
        last_pos = -1
        curr_values = []
        curr_weights = []
        for line in input:
            if not line.startswith("#"):
                chr_name, start, end, value = line.split()
                start_pos = int(int(start)/resolution)
                end_pos = int(int(end)/resolution)
                #pos = (int(start)+int(end))/2
                #pos = int(pos/resolution)
                if start_pos != last_pos or chr_name != last_chr :
                    if last_chr != "":
                        average = np.average(curr_values, weights = curr_weights)
                        if save_vir_res:
                            o.write(last_chr + "\t" + str(last_pos) +
                                    "\t" + str(last_pos+1) + "\t" + str(average) + "\n")
                        else:
                            o.write(last_chr + "\t" + str(last_pos*resolution) +
                                    "\t" + str((last_pos+1)*resolution) + "\t" + str(average) + "\n")
                    curr_values = []
                    curr_weights = []
                if end_pos == start_pos:
                    curr_values.append(float(value))
                    curr_weights.append(int(end)-int(start))
                else:
                    while(start_pos != end_pos):
                        curr_values.append(float(value))
                        curr_weights.append(((start_pos+1)*resolution) - int(start))
                        average = np.average(curr_values, weights = curr_weights)
                        if save_vir_res:
                            o.write(chr_name + "\t" + str(start_pos) +
                                    "\t" + str(start_pos+1) + "\t" + str(average) + "\n")
                        else:
                            o.write(chr_name + "\t" + str(start_pos*resolution) +
                                    "\t" + str((start_pos+1)*resolution) + "\t" + str(average) + "\n")
                        start_pos = start_pos + 1
                        curr_values = []
                        curr_weights= []
                    curr_values.append(float(value))
                    curr_weights.append(int(end) - (start_pos * resolution))

                last_pos = end_pos
                last_chr = chr_name
        if np.sum(curr_weights) > 0:
            average = np.average(curr_values, weights = curr_weights)
            if save_vir_res:
                o.write(last_chr + "\t" + str(last_pos) +
                        "\t" + str(last_pos+1) + "\t" + str(average) + "\n")
            else:
                o.write(last_chr + "\t" + str(last_pos*resolution) +
                        "\t" + str((last_pos+1)*resolution) + "\t" + str(average) + "\n")
    o.close()
'''
def create_assays_df(data_path,cell_type,resolution,assay_types):
    assays_df = pd.DataFrame()
    chip_seq_signals_dir = os.path.join(data_path, cell_type, "genomic-assays/ChIP-seq/bin_" + str(resolution))
    tsa_seq_signals_dir = os.path.join(data_path, cell_type, "genomic-assays/TSA-seq")
    if os.path.exists(chip_seq_signals_dir) and 'chip' in assay_types:
        for file in os.listdir(chip_seq_signals_dir):
            assay_name = file.split(".fc")[0].split("-")[1]
            assay_df = pd.read_csv(os.path.join(chip_seq_signals_dir,file), sep = "\t", header = None).drop(2,axis=1)
            assay_df.columns = ['chr_name', 'pos', assay_name]
            if assays_df.empty:
                assays_df = assay_df
            else:
                assays_df = pd.merge(assays_df, assay_df, on = ["chr_name", "pos"])
    if os.path.exists(tsa_seq_signals_dir) and 'tsa' in assay_types:
        for assay_name in ['LaminA','SON']:
            assay_df = pd.read_csv(os.path.join(tsa_seq_signals_dir, assay_name + "_TSA-seq_binned" + str(resolution) + ".bedgraph"), sep = "\t", header = None).drop(2,axis=1)
            assay_df.columns = ['chr_name', 'pos', assay_name]
            if assays_df.empty:
                assays_df = assay_df
            else:
                assays_df = pd.merge(assays_df, assay_df, on = ["chr_name", "pos"])
    return assays_df
'''

def create_assays_df(signals_dir, resolution):

    assays_df = pd.DataFrame()
    for signal_file in os.listdir(signals_dir):
        signal_name = signal_file[:-9]
        assay_df = pd.read_csv(os.path.join(signals_dir, signal_file), sep = "\t", header = None)
        assay_df.columns = ['chr_name', 'start', 'end', signal_name]
        assay_df.drop(columns=['end'])
        assay_df['pos'] = (assay_df['start']/resolution).astype(int)
        assay_df = assay_df.loc[:,['chr_name', 'pos', signal_name]]
        if assays_df.empty:
            assays_df = assay_df
        else:
            assays_df = pd.merge(assays_df, assay_df, on = ["chr_name", "pos"])
    return assays_df




def make_embedding_df(embeddings_dict, data_path, cell, resolution):
    index_to_chr_pos = get_index_to_chr_pos(data_path, cell, resolution)
    embeddings_df = []
    for key in embeddings_dict.keys():
        chr_name, pos = index_to_chr_pos[int(key)]
        emb_dict = {'chr_name': chr_name, 'pos': pos}
        for e, emb in enumerate(embeddings_dict[key]):
            emb_dict['emb'+str(e+1)] = emb
        embeddings_df.append(emb_dict)
    embeddings_df = pd.DataFrame(embeddings_df)
    dim = embeddings_df.shape[1]-2
    columns_order = ['chr_name', 'pos']
    for e in np.arange(1,dim+1):
        columns_order.append('emb'+str(e))
    embeddings_df = embeddings_df[columns_order]
    embeddings_df.sort_values(by=['chr_name','pos'], inplace = True)
    return embeddings_df

def chain_mat(N):
    mat = np.zeros((N,N))
    for i in range(N-1):
        mat[i,i+1] = 1
        mat = sym_mat(mat)
    return mat


################################### bedpe lifting utils ################################

def bedpe_to_bed(data_path, drop_folder):

    if not os.path.exists(drop_folder):
        os.mkdir(drop_folder)
    data_dir, data_name = os.path.split(data_path)
    data_name = os.path.splitext(data_name)[0]
    first_bed_file_path = os.path.join(drop_folder, data_name + "_1.bed")
    second_bed_file_path = os.path.join(drop_folder, data_name + "_2.bed")
    first_bed = open(first_bed_file_path, "w")
    second_bed = open(second_bed_file_path, "w")
    cnt = 0
    weights = []
    with open(data_path) as f:
        for line in f:
            if not line.startswith("#"):
                chr1, start1, end1, chr2, start2, end2, weight = line.rstrip("\n").split("\t")
                first_bed.write("{}\t{}\t{}\t{}\n".format(chr1,start1,end1,cnt))
                second_bed.write("{}\t{}\t{}\t{}\n".format(chr2,start2,end2,cnt))
            weights.append(weight)
            cnt = cnt + 1
    first_bed.close()
    second_bed.close()
    return([first_bed_file_path, second_bed_file_path], weights)


def lift(bed_path, liftover_path, chain_path, drop_folder):
    bed_dir, bed_name = os.path.split(bed_path)
    bed_name = os.path.splitext(bed_name)[0]
    out_path = os.path.join(drop_folder, bed_name + "_output.bed")
    if not os.path.exists(drop_folder):
        os.makedirs(drop_folder)
    if not os.path.isfile(out_path):
        unlifted_path = os.path.join(drop_folder, bed_name + "_unlifted.bed")
        cmd = [liftover_path, bed_path, chain_path, out_path, unlifted_path]
        MyOut = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).wait()
        #stdout,stderr = MyOut.communicate()
    return (out_path)


def lift_bedpe(bedpe_path, liftover_path, chain_path, drop_folder):

    bed_paths, weights = bedpe_to_bed(bedpe_path, drop_folder)
    lifted_bed_paths = []
    for bed_path in bed_paths:
        lifted_bed_paths.append(lift(bed_path, liftover_path, chain_path, drop_folder))
    first_lifted_bed = pd.read_csv(lifted_bed_paths[0], sep = "\t", header = None)
    second_lifted_bed = pd.read_csv(lifted_bed_paths[1], sep = "\t", header = None)
    first_lifted_bed.columns = ['chr_name1', 'start1', 'end1', 'index']
    second_lifted_bed.columns = ['chr_name2', 'start2', 'end2', 'index']
    lifted_bedpe = pd.merge(first_lifted_bed, second_lifted_bed, how = 'inner', on = 'index')
    weights = [weights[int(w)] for w in lifted_bedpe['index'].values]
    lifted_bedpe = lifted_bedpe.drop(['index'], axis = 1)
    lifted_bedpe['weight'] = weights
    bedpe_dir, bedpe_name = os.path.split(bedpe_path)
    lifted_bedpe_path = os.path.join(bedpe_dir, 'lifted_{}'.format(bedpe_name))
    lifted_bedpe.to_csv(lifted_bedpe_path, header=False, index=False, sep = "\t")
    shutil.rmtree(drop_folder)



########################################################################################

def sym_list(contact_list):

    contact_list.columns = ['id1', 'id2', 'weight']
    reverse_contact_list = contact_list.copy()
    reverse_contact_list = reverse_contact_list[['id2','id1','weight']]
    reverse_contact_list.columns = ['id1','id2','weight']
    contact_list = pd.concat([contact_list,reverse_contact_list])
    return contact_list

def uni_dir(contact_list):
    contact_list = contact_list[contact_list['id1'] != contact_list['id2']]
    contact_list.sort_values(['id1','id2'], inplace = True)
    return contact_list

def bi_dir(contact_list):
    non_loops = contact_list[contact_list['id1'] != contact_list['id2']]
    reverse_non_loops = non_loops[['id2','id1','weight']]
    reverse_non_loops.columns = ['id1','id2','weight']
    sym_non_loops = pd.concat([non_loops,reverse_non_loops])
    sym_non_loops.sort_values(['id1','id2'], inplace = True)
    return sym_non_loops

def loop_bi_dir(contact_list):
    loops = contact_list[contact_list['id1'] == contact_list['id2']]
    non_loops = contact_list[contact_list['id1'] != contact_list['id2']]
    reverse_non_loops = non_loops[['id2','id1','weight']]
    reverse_non_loops.columns = ['id1','id2','weight']
    sym_non_loops = pd.concat([non_loops,reverse_non_loops])
    full_contact_lists = pd.concat([loops,sym_non_loops])
    full_contact_lists.sort_values(['id1','id2'], inplace = True)
    return full_contact_lists
