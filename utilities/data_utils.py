import pandas as pd
import numpy as np
import math
import os

hg19_chr_sizes = {'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276,
 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431,
  'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540,
   'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983,
    'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566}

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


def create_contact_list(data_path, cell, first_chr, second_chr, resolution):
    chr_pos_to_index = get_chr_pos_to_index(data_path, cell, resolution)
    hic_path = os.path.join(data_path, cell, "HiC", "res" + str(resolution), "oe_" + first_chr + "_" + second_chr + ".txt")
    contact_list = pd.read_csv(hic_path, sep = "\t", header = None)
    contact_list.columns = ['source', 'target', 'weight']
    contact_list['id1'] = [chr_pos_to_index[first_chr][int(p/resolution)] for p in contact_list['source']]
    contact_list['id2'] = [chr_pos_to_index[second_chr][int(p/resolution)] for p in contact_list['target']]
    contact_list = contact_list[['id1','id2','weight']]
    contact_list = contact_list.dropna()
    contact_list[['id1','id2']] = contact_list[['id1','id2']].astype(int)
    return contact_list

def create_contact_list_woInd(data_path, cell, first_chr, second_chr, resolution):
    hic_path = os.path.join(data_path, cell, "HiC", "res" + str(resolution), "oe_" + first_chr + "_" + second_chr + ".txt")
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

def create_matrix(data_path, cell, first_chr, second_chr, resolution):
    row_num = math.ceil(hg19_chr_sizes[first_chr]/resolution)
    col_num = math.ceil(hg19_chr_sizes[second_chr]/resolution)
    mat = np.zeros([row_num,col_num])
    contact_list = create_contact_list_woInd(data_path, cell, first_chr, second_chr, resolution)
    for index,row in contact_list.iterrows():
        if not np.isnan(row[2]):
            mat[int(row[0]),int(row[1])] = row[2]
    if first_chr == second_chr:
        mat = sym_mat(mat)
    return mat


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
