import math
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from scipy.spatial import distance
import random
import data_utils
from collections import Counter
from scipy.stats import entropy

#chr_sizes = data_utils.read_chr_sizes('../data/supp/hg19.chrom.sizes')
chr_sizes = {'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276,
 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431,
  'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540,
   'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983,
    'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566}

#chr_arms_sizes = data_utils.read_chr_arm_sizes('../data/supp/hg19.chrom.arms.sizes')
chr_arms_sizes = {('chr1', 'p'): 125000000, ('chr1', 'q'): 124250621, ('chr10', 'p'): 40200000,
 ('chr10', 'q'): 95334747, ('chr11', 'p'): 53700000, ('chr11', 'q'): 81306516,
 ('chr12', 'p'): 35800000, ('chr12', 'q'): 98051895, ('chr13', 'p'): 17900000,
 ('chr13', 'q'): 97269878, ('chr14', 'p'): 17600000, ('chr14', 'q'): 89749540,
 ('chr15', 'p'): 19000000, ('chr15', 'q'): 83531392, ('chr16', 'p'): 36600000,
 ('chr16', 'q'): 53754753, ('chr17', 'p'): 24000000, ('chr17', 'q'): 57195210,
 ('chr18', 'p'): 17200000, ('chr18', 'q'): 60877248, ('chr19', 'p'): 26500000,
 ('chr19', 'q'): 32628983, ('chr2', 'p'): 93300000, ('chr2', 'q'): 149899373,
 ('chr20', 'p'): 27500000, ('chr20', 'q'): 35525520, ('chr21', 'p'): 13200000,
 ('chr21', 'q'): 34929895, ('chr22', 'p'): 14700000, ('chr22', 'q'): 36604566,
 ('chr3', 'p'): 91000000, ('chr3', 'q'): 107022430, ('chr4', 'p'): 50400000,
 ('chr4', 'q'): 140754276, ('chr5', 'p'): 48400000, ('chr5', 'q'): 132515260,
 ('chr6', 'p'): 61000000, ('chr6', 'q'): 110115067, ('chr7', 'p'): 59900000,
 ('chr7', 'q'): 99238663, ('chr8', 'p'): 45600000, ('chr8', 'q'): 100764022,
 ('chr9', 'p'): 49000000, ('chr9', 'q'): 92213431}


def is_valid_chr(chr_name):
    return chr_name in ['chr{}'.format(c) for c in np.arange(1,23)]

def get_pos(chr_name, pos, resolution):
    arm_p_size = chr_arms_sizes[chr_name,'p']
    if pos < arm_p_size:
        pos = math.floor(pos/resolution)
        return('p',pos)
    else:
        q_base_pos = math.ceil(arm_p_size/resolution)
        pos = math.floor(pos/resolution)
        return('q',pos-q_base_pos)

def make_annotation_dict(annotation_path, annotation_resolution):

    annotation_dict = {}
    for chr_name in ['chr{}'.format(c) for c in np.arange(1,23)]:
        chr_size = math.ceil(chr_sizes[chr_name]/annotation_resolution)
        annotation_dict[chr_name] = np.empty(chr_size, dtype=object)
    with open(annotation_path, 'r') as f:
        for line in f:
            chr_name, start, end, label, *_ = line.rstrip("\n").split("\t")
            if not is_valid_chr(chr_name):
                continue
            if label != 'NA':
                start_pos =  math.floor(int(start)/annotation_resolution)
                end_pos = math.floor(int(end)/annotation_resolution)
                annotation_dict[chr_name][start_pos:end_pos] = label
    return annotation_dict


def make_annotation_dict_chr_arm(annotation_path, annotation_resolution):
    annotation_dict = {}
    for chr_name in ['chr{}'.format(c) for c in np.arange(1,23)]:
        for arm in ['p', 'q']:
            arm_size = math.ceil(chr_arms_sizes[chr_name, arm]/annotation_resolution)
            annotation_dict[chr_name, arm] = np.empty(arm_size, dtype=object)
    with open(annotation_path, 'r') as f:
        for line in f:
            chr_name, start, end, label, *_ = line.rstrip("\n").split("\t")
            if not is_valid_chr(chr_name):
                continue
            if label != 'NA':
                arm, start_pos =  get_pos(chr_name, int(start), annotation_resolution)
                arm, end_pos = get_pos(chr_name, int(end), annotation_resolution)
                annotation_dict[chr_name, arm][start_pos:end_pos] = label
    return annotation_dict

def permute_annotation_dict(annotation_dict):

    r = random.uniform(0,1)
    for key in annotation_dict:
        shift_size = int(len(annotation_dict[key])*r)
        annotation_dict[key] = np.roll(annotation_dict[key],shift_size)
    return annotation_dict

def annot_freq_for_region_by_arm(chr_name, start, end, annotation_dict, annotation_resolution):

    start2 = math.floor(start/annotation_resolution)*annotation_resolution
    end2 = math.ceil(end/annotation_resolution)*annotation_resolution
    range_ = list(np.arange(start2,end2,annotation_resolution))
    range_[0] = start
    poses = [get_pos(chr_name, p, annotation_resolution) for p in range_]
    labels = [annotation_dict[chr_name,p[0]][p[1]] for p in poses]
    if end > range_[-1]:
        range_.append(end)
    lengths = [l1-l2 for l1,l2 in zip(range_[1:],range_[:-1])]
    labels_weights = [[(label,length),] for label,length in zip(labels,lengths)]
    labels_weights = dict(sum(map(lambda l: Counter(dict(l)), labels_weights), Counter()))
    return labels_weights

def get_most_freq_label(chr_name, start, end, annotation_dict, annotation_resolution):

    start2 = math.floor(start/annotation_resolution)*annotation_resolution
    end2 = math.ceil(end/annotation_resolution)*annotation_resolution
    range_ = list(np.arange(start2,end2,annotation_resolution))
    poses = [math.floor(p/annotation_resolution) for p in range_]
    labels = [annotation_dict[chr_name][p] for p in poses]
    range_[0] = start
    if end > range_[-1]:
        range_.append(end)
    lengths = [l1-l2 for l1,l2 in zip(range_[1:],range_[:-1])]
    labels_weights = [[(label,length),] for label,length in zip(labels,lengths)]
    labels_weights = dict(sum(map(lambda l: Counter(dict(l)), labels_weights), Counter()))
    most_freq_label = max(labels_weights, key=labels_weights.get)
    return most_freq_label



##### annotation evaluations #####

## it is assumed that signals_dir include binned bedgraphs with the same resolution as annotation
def enrichment_scores(signals_dir, annotation_path, resolution):

    assays_df = data_utils.create_assays_df(signals_dir, resolution)
    valid_rows = [is_valid_chr(chr_name) for chr_name in assays_df['chr_name']]
    assays_df = assays_df[valid_rows]
    signals_names = assays_df.columns[2:]
    annotation_df = make_annotation_dict(annotation_path, resolution)
    labels = []
    for i, row in assays_df.iterrows():
        chr_name = row['chr_name']
        pos = int(row['pos'])
        if not is_valid_chr(chr_name):
            continue
        label = annotation_df[chr_name][pos]
        labels.append(label)
    assays_df['label'] = labels
    en = assays_df.groupby('label')[signals_names].mean()
    mean_en = assays_df.loc[:,signals_names].mean(axis=0)
    en = round(en.divide(mean_en, axis=1),2)
    #sns.heatmap(en, annot = True, cmap="PiYG", center = 1, linewidths=.5, square = True, cbar = False)
    return en

def enrichment_scores_(signal_label_df, signals_names, label_name):
    en = signal_label_df.groupby(label_name)[signals_names].mean()
    mean_en = signal_label_df.loc[:,signals_names].mean(axis=0)
    en = round(en.divide(mean_en, axis=1),2)
    return en

def get_coverage(annotation_path):

    coverages = {}
    total = 0
    with open(annotation_path, 'r') as f:
        for line in f:
            chr_name, start, end, label, *_ = line.rstrip("\n").split("\t")
            length = int(end) - int(start)
            if label == 'NA':
                continue
            if label in coverages:
                coverages[label] = coverages[label] + length
            else:
                coverages[label] = length
            total = total + length
    for key in coverages:
        coverages[key] = coverages[key] / total
    #sns.barplot(x = freq.index.values, y = freq.values)
    return coverages

def num_domains(labels):
    previous_labels = labels[:-1]
    next_labels = labels[1:]
    is_changed = [l1 != l2 for l1,l2 in zip(previous_labels, next_labels)]
    return np.sum(is_changed)

def average_length(labels, resolution):
    previous_labels = labels[:-1]
    next_labels = labels[1:]
    is_changed = [l1 != l2 for l1,l2 in zip(previous_labels, next_labels)]
    changed_locs = np.where(np.array(is_changed)==True)[0]+1
    previous_locs = np.insert(changed_locs[:-1], 0, 0)
    next_locs = changed_locs
    lengths = [resolution*(n-p) for p,n in zip(previous_locs, next_locs)]
    return np.mean(lengths)

def get_length_dist(annotation_path):

    lengths = {}
    with open(annotation_path, 'r') as f:
        for line in f:
            chr_name, start, end, label, *_ = line.rstrip("\n").split("\t")
            length = (int(end) - int(start)) / 1000000
            if label in lengths:
                lengths[label].append(length)
            else:
                lengths[label] = [length]
    #sns.violinplot(x="label", y="length", data=annotation_bedgraph).set(yscale="log")
    return lengths


def variance_explained(in_signal, in_label):
    assert len(in_signal) == len(in_label), "signal and label lists should have the same length..."
    '''
    labels = np.unique(in_label)
    signal_mean = np.mean(in_signal)
    labels_means = {}
    for label in labels:
        labels_means[label] = np.mean(in_signal[in_label == label])
    within_ds = []
    total_ds = []
    for ind,s in enumerate(in_signal):
        within_d = s - labels_means[in_label[ind]]
        total_d = s - signal_mean
        within_ds.append(within_d)
        total_ds.append(total_d)
    within_ds_sum_sq = np.sum(np.power(within_ds,2))
    total_ds_sum_sq = np.sum(np.power(total_ds,2))
    #signal_sum_sq = np.sum(np.power(in_signal,2))
    VE = 1 - (within_ds_sum_sq/total_ds_sum_sq)
    return VE
    '''
    p = pd.DataFrame({'signal': in_signal, 'label': in_label})
    clusters_means = p.groupby('label')['signal'].mean()
    p['pred'] = [clusters_means[l] for l in in_label]
    p['d_from_pred'] = np.power(p['signal'] - p['pred'],2)
    p['d_from_mean'] = np.power((p['signal'] - np.mean(p['signal'])), 2)
    VE = 1 - (np.sum(p['d_from_pred'])/np.sum(p['d_from_mean']))
    return VE

def clustering_score(in_signals, in_label):
    SI = metrics.silhouette_score(in_signals, in_label, metric='euclidean')
    DB = metrics.davies_bouldin_score(in_signals, in_label)
    CH = metrics.calinski_harabasz_score(in_signals, in_label)
    return {'SI': SI, 'DB': DB, 'CH': CH}

def within_cluster_SSE(X, labels):
    labels_means = {}
    for label in np.unique(labels):
        label_indices = np.where(labels==label)
        labels_means[label] = np.mean(X[label_indices,:], axis = 1)
    SSE = 0
    for x_ind, x in enumerate(X):
        label_mean = labels_means[labels[x_ind]]
        d = distance.euclidean(x, label_mean)
        SSE = SSE + np.power(d,2)
    return SSE

def TAD_agreement(annotation_dict, TAD_path, annotation_resolution):

    #shift_sizes = {}
    #for chr_name in ['chr{}'.format(c) for c in np.arange(1,23)]:
    #    shift_sizes[chr_name] = int(chr_sizes[chr_name]*r)
    #valid_TADs = []
    f_ds = []
    #most_freq_labels = []
    #cnt = -1
    with open(TAD_path, 'r') as f:
        next(f)
        for line in f:
            chr_num, start, end , *_ = line.split("\t")
            chr_name = 'chr{}'.format(chr_num)
            if not is_valid_chr(chr_name):
                continue
            #start = (int(start) + shift_sizes[chr_name])%chr_sizes[chr_name]
            #end = (int(end) + shift_sizes[chr_name])%chr_sizes[chr_name]
            #cnt = cnt+1
            #if start > end:
            #    continue

            labels_weights = annot_freq_for_region_by_arm(chr_name, int(start), int(end), annotation_dict, annotation_resolution)
            most_freq_label = max(labels_weights, key=labels_weights.get)
            common_freq = labels_weights[most_freq_label]/np.sum(list(labels_weights.values()))
            labels_weights.pop(None, None)
            non_na_regions_sum = np.sum(list(labels_weights.values()))
            if ((non_na_regions_sum / annotation_resolution) > 3) & (most_freq_label != None):
                f_ds.append(common_freq)

    return np.mean(f_ds)

def normalized_TAD_agreement(annotation_path, TAD_path, annotation_resolution, p_num):


    annotation_dict = make_annotation_dict_chr_arm(annotation_path, annotation_resolution)
    f_raw = TAD_agreement(annotation_dict, TAD_path, annotation_resolution)
    p_f_raws = []
    for p in range(p_num):
        annotation_dict = permute_annotation_dict(annotation_dict)
        p_f_raw = TAD_agreement(annotation_dict, TAD_path, annotation_resolution)
        p_f_raws.append(p_f_raw)
    p_f_raws_mean = np.mean(p_f_raws)
    return f_raw/p_f_raws_mean
    '''
    annotation_dict = make_annotation_dict_chr_arm(annotation_path, annotation_resolution)
    f_raw = TAD_agreement(annotation_dict, TAD_path, annotation_resolution, 0)
    p_f_raws = []
    for i in range(p_num):
        r = random.uniform(0,1)
        p_f_raw = TAD_agreement(annotation_path, TAD_path, annotation_resolution, r)
        p_f_raws.append(p_f_raw)
    return f_raw/np.mean(p_f_raws)
    '''

def variance_explained_from_file(signal_path, annotation_path, annotation_resolution, is_binned_signal, dump_dir = None):

    if not is_binned_signal:
        if not os.path.exists(dump_dir):
            os.mkdir(dump_dir)
        dumped_file = os.path.join(dump_dir, 'signal.txt')
        data_utils.bin_bedgraph_file(signal_path, annotation_resolution, dumped_file, save_vir_res = False)
        signal_path = dumped_file
    signal_df = pd.read_csv(signal_path, sep = "\t", header = None)
    signal_df.columns = ['chr_name', 'start', 'end', 'signal']
    signal_df['pos'] = (signal_df['start']/annotation_resolution).astype(int)
    signal_df = signal_df[['chr_name', 'pos', 'signal']]
    annotation_df = pd.read_csv(annotation_path, sep = "\t", header = None)
    annotation_df.iloc[:,[1,2]] = (annotation_df.iloc[:,[1,2]]/annotation_resolution).astype(int)
    annotation_df.columns = ['chr_name', 'start', 'end', 'label']
    annotation_df = data_utils.melt_annotation(annotation_df)
    annotation_df = pd.merge(annotation_df,signal_df, on = ['chr_name', 'pos'])
    signals = annotation_df['signal']
    labels = annotation_df['label']
    ve = variance_explained(signals, labels)
    return ve


def create_pivot_table_from_bedpe(bedpe_path, annotation_path, annotation_resolution):

    annotation_dict = make_annotation_dict(annotation_path, annotation_resolution)
    labels1 = []
    labels2 = []
    weights = []
    with open(bedpe_path, 'r') as f:
        for line in f:
            chr1, start1, end1, chr2, start2, end2, weight = line.split("\t")
            if not (is_valid_chr(chr1)) & (is_valid_chr(chr2)):
                continue
            labels1.append(get_most_freq_label(chr1, int(start1), int(end1), annotation_dict, annotation_resolution))
            labels2.append(get_most_freq_label(chr2, int(start2), int(end2), annotation_dict, annotation_resolution))
            weights.append(int(weight))
    contacts = pd.DataFrame({'label1': labels1, 'label2': labels2, 'weight': weights}).groupby(['label1','label2'])['weight'].sum().reset_index()
    contacts = contacts.pivot_table(columns = 'label2', index = 'label1', values = 'weight')
    contacts = contacts.fillna(0)
    return contacts

def OE_intra_of_bedpe(bedpe_path, annotation_path, annotation_resolution):
    contacts = create_pivot_table_from_bedpe(bedpe_path, annotation_path, annotation_resolution)
    coverages = get_coverage(annotation_path)
    total_contacts = np.sum(contacts.to_numpy())
    expected_intra_contacts = 0
    observed_intra_contacts = 0
    for key in coverages:
        expected_intra_contacts = expected_intra_contacts + coverages[key]**2
        observed_intra_contacts = observed_intra_contacts + contacts.loc[key,key]/total_contacts
    return observed_intra_contacts/expected_intra_contacts
#def create_pivot_table_from_hic(hic_path, bin_path, annotation_path, annotation_resolution):




def get_phases_dist(RT_dict, chr_name, pos):
    phases_dist = [RT_dict[chr_name,phase][pos] for phase in phases]
    return phases_dist

def aggregate_RT_files(wigs_path, outfile):

    RT_dict = {}
    phases = ['G1', 'S1', 'S2', 'S3', 'S4', 'G2']
    for chr_name in ['chr{}'.format(c) for c in np.arange(1,23)]:
        chr_size = math.ceil(chr_sizes[chr_name]/1000)
        for phase in phases:
            RT_dict[chr_name,phase] = np.zeros(chr_size)
    for phase in phases:
        wig_path = os.path.join(wigs_path, '{}.wig'.format(phase))
        with open(wig_path, 'r') as f:
            for line in f:
                line = line.rstrip("\n").split("\t")
                if len(line) == 2:
                    if not is_valid_chr(chr_name):
                        continue
                    pos = int(int(line[0])/1000)
                    RT_dict[chr_name,phase][pos] = int(line[1])
                else:
                    chr_name = line[0].split(" ")[1][6:]
    RT_data = []
    for chr_name in ['chr{}'.format(c) for c in np.arange(1,23)]:
        chr_size = math.ceil(chr_sizes[chr_name]/1000)
        for p in np.arange(chr_size):
            phases_dist = get_phases_dist(RT_dict, chr_name, p)
            if abs(sum(phases_dist)-100) <= 2:
                pos = p*1000+500
                RT_data.append([chr_name,pos] + phases_dist)

    out = open(outfile, 'w')
    for r in RT_data:
        line = "\t".join([str(x) for x in r])+'\n'
        out.write(line)
    out.close()

def RT_scores(RT_path, annotation_path, annotation_resolution):

    annotation_dict = make_annotation_dict(annotation_path, annotation_resolution)
    RT = pd.read_csv(RT_path, sep = "\t", header = None)
    RT.columns = ['chr_name', 'pos', 'G1', 'S1', 'S2', 'S3', 'S4', 'G2']
    RT['label'] = [annotation_dict[chr_name][int(pos)] for chr_name, pos in zip(RT['chr_name'].values, RT['pos'].values/annotation_resolution)]
    RT = RT[RT['label'].values != None]
    RT['RT_label'] = RT.iloc[:,2:8].to_numpy().argmax(axis=1)
    VEs = []
    for phase in ['G1', 'S1', 'S2', 'S3', 'S4', 'G2']:
        VEs.append(variance_explained(RT[phase],RT['label']))
    mean_VE = np.mean(VEs)
    ARI = metrics.adjusted_rand_score(RT['RT_label'],RT['label'])
    return mean_VE, ARI


#def RT_evaluation(RT_path, annotation_path, annotation_resolution):


def gene_expression_ve(gene_expression_path, annotation_path, annotation_resolution):

    annotation_dict = make_annotation_dict(annotation_path, annotation_resolution)
    expressions = []
    labels = []
    with open(gene_expression_path, 'r') as f:
        for line in f:
            gene_id, chr_num, pos1, pos2, _, gene_expr = line.rstrip("\n").split("\t")
            if not chr_num in [str(c) for c in np.arange(1,23)]:
                continue
            chr_name = 'chr{}'.format(chr_num)
            start = min(int(pos1),int(pos2))
            end = max(int(pos1),int(pos2))
            label = get_most_freq_label(chr_name, start, end, annotation_dict, annotation_resolution)
            if label == None:
                continue
            expressions.append(float(gene_expr))
            labels.append(label)
    VE = variance_explained(np.arcsinh(expressions),labels)
    return VE

def brief_stats(annotation_path, annotation_resolution, gene_expression_path, RT_path):
    df = pd.read_csv(annotation_path, header = None, sep = "\t")
    df['length'] = df.iloc[:,2] - df.iloc[:,1]
    avg_length = np.mean(df['length'])
    num_domains = df.shape[0]
    ge_ve = gene_expression_ve(gene_expression_path, annotation_path, annotation_resolution)
    RT_score1, RT_score2 = RT_scores(RT_path, annotation_path, annotation_resolution)
    coverage = get_coverage(annotation_path)
    coverage = [coverage[label] for label in coverage.keys()]
    ent = entropy(coverage, base=None)
    return {'avg_length': avg_length, 'num_domains': num_domains, 'ge_ve': ge_ve, 'RT_score1': RT_score1, 'RT_score2': RT_score2, 'ent': ent}


def stats(annotation_path, annotation_resolution, gene_expression_path, RT_path, CTCF_bedpe_path, RNAPII_bedpe_path):
    df = pd.read_csv(annotation_path, header = None, sep = "\t")
    df['length'] = df.iloc[:,2] - df.iloc[:,1]
    avg_length = np.mean(df['length'])
    num_domains = df.shape[0]
    ge_ve = gene_expression_ve(gene_expression_path, annotation_path, annotation_resolution)
    RT_score1, RT_score2 = RT_scores(RT_path, annotation_path, annotation_resolution)
    CTCF_OE = OE_intra_of_bedpe(CTCF_bedpe_path, annotation_path, annotation_resolution)
    RNAPII_OE = OE_intra_of_bedpe(RNAPII_bedpe_path, annotation_path, annotation_resolution)
    coverage = get_coverage(annotation_path)
    coverage = [coverage[label] for label in coverage.keys()]
    ent = entropy(coverage, base=None)
    return {'avg_length': avg_length, 'num_domains': num_domains, 'ge_ve': ge_ve, 'RT_score1': RT_score1, 'RT_score2': RT_score2, 'CTCF_OE': CTCF_OE, 'RNAPII_OE': RNAPII_OE, 'ent': ent}

####################################


def OE_enrichment(label1, label2, labels):
    if len(label1) != len(label2):
        print('The size of label1 and label2 is different...')
        return
    #labels = np.concatenate([label1,label2])
    total_contacts = len(label1)
    coverage = pd.DataFrame({'label': labels})
    coverage = pd.DataFrame(coverage.groupby('label').size())
    coverage = coverage / np.sum(coverage)
    contacts = pd.DataFrame({'label1': label1, 'label2': label2})
    contacts = pd.DataFrame(contacts.groupby(['label1','label2']).size())
    contacts = contacts.pivot_table(columns = 'label2', index = 'label1', values = 0)
    contacts = contacts.fillna(0)
    expected_intra_contacts = 0
    observed_intra_contacts = 0
    for i,key in coverage.iterrows():
        if (not i in contacts.index) or (not i in contacts.columns):
            continue
        expected_intra_contacts = expected_intra_contacts + coverage.loc[i].item()**2
        observed_intra_contacts = observed_intra_contacts + contacts.loc[i,i].item()/total_contacts
    return observed_intra_contacts/expected_intra_contacts


##### comparison of two annotations #####

def percentage_ratio(table):
    table = table.fillna(0)
    return(table.div(table.sum(axis=0), axis = 1))

def EO_ratio(table):
    x_probs = table.sum(axis=0)/np.nansum(table.values)
    y_probs = table.sum(axis=1)/np.nansum(table.values)
    expected = np.array(y_probs).reshape(-1, 1).dot(np.array(x_probs).reshape(1,-1))*np.nansum(table.values)
    return(table.fillna(0)/expected)

def overlap(in_label1, in_label2):
    in_labels = pd.DataFrame({'in_label1': in_label1, 'in_label2': in_label2})
    a = in_labels.groupby(['in_label1','in_label2']).size()
    a_table = a.unstack(level=0)
    a_table_EO = EO_ratio(a_table)
    #a_table_percentage = percentage_ratio(a_table)
    return a_table_EO

def overlap_per(in_label1, in_label2):
    in_labels = pd.DataFrame({'in_label1': in_label1, 'in_label2': in_label2})
    a = in_labels.groupby(['in_label1','in_label2']).size()
    a_table = a.unstack(level=0)
    #a_table_EO = EO_ratio(a_table)
    a_table_percentage = percentage_ratio(a_table)
    return a_table_percentage

#########################################





def spatial_correlation(signal,adj_mat):

    N = signal.shape[0]
    W = adj_mat.sum()
    signal_mean = np.mean(signal)
    signal_res = signal - signal_mean
    sum_sq_signal_res = np.sum(np.power(signal_res, 2))
    signal_res_mul = np.sum(np.multiply(np.matmul(signal_res.reshape(-1,1), signal_res.reshape(1,-1)), adj_mat))
    moran = (N*signal_res_mul) / (W*sum_sq_signal_res)
    E = -1/(N-1)
    S1 = np.sum(np.power((adj_mat + adj_mat.transpose()),2))/2
    S2 = np.sum(np.power((np.sum(adj_mat, axis = 0) + np.sum(adj_mat, axis = 1)),2))
    S3 = (np.sum(np.power(signal_res,4))/N) / np.power((np.sum(np.power(signal_res,2))/N),2)
    S4 = (np.power(N,2)-(3*N)+3)*S1 - N*S2 + 3*np.power(W,2)
    S5 = ((np.power(N,2)-N)*S1) - 2*N*S2 + 6*np.power(W,2)
    var = ((N*S4 - S3*S5)/((N-1)*(N-2)*(N-3)*np.power(W,2))) - np.power(E,2)
    sd = np.sqrt(var)
    z_score = (moran-E)/sd
    p_value = stats.norm.sf(z_score)
    return {'moran': moran, 'z_score': z_score, 'p_value': p_value}

'''
def spatial_correlation(signal,adj_mat):
    dim = len(signal)
    signal_mean = np.mean(signal)
    mat_sum = np.sum(adj_mat)
    frac_nom = 0
    frac_denom = 0
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[0]):
            frac_nom = frac_nom + (adj_mat[i,j] * (signal[i] - signal_mean) * (signal[j] - signal_mean))
        frac_denom = frac_denom + ((signal[i] - signal_mean) * (signal[i] - signal_mean))
    moran = frac_nom * dim / (mat_sum * frac_denom)
    return(moran)

def spatial_correlation2(signal1,signal2,adj_mat):
    signal = signal1 + signal2
    signal_mean = np.mean(signal)
    signal_var = np.var(signal)
    mat_sum = np.sum(adj_mat)
    frac_nom = 0
    frac_denom = 0
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            frac_nom = frac_nom + (adj_mat[i,j] * (signal1[i] - signal_mean) * (signal2[j] - signal_mean))
    moran = frac_nom / (mat_sum * signal_var)
    return(moran)

'''
