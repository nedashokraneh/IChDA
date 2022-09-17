# This script is from pytadbit package, https://3dgenomes.github.io/TADbit/_modules/pytadbit/tadbit.html

from os import path, listdir
from math import isnan, sqrt
from scipy.sparse.csr import csr_matrix
from scipy.stats import mannwhitneyu
import numpy as np

def TopDom(csr_mat,window_size,statFilter=True):
    """
    Python implementation of the algorithm TopDom for the identification of TADs. See http://www.ncbi.nlm.nih.gov/pubmed/26704975 and http://zhoulab.usc.edu/TopDom/

    :param hic_data: a list corresponding to the Hi-C data
    :param window_size: window size parameter for the TopDom algorithm
    :param True statFilter: whether to apply or not statistical filtering for false detection of TADs

    :returns: the :py:func:`list` of topologically associated domains, boundaries and gaps. Domains include the mean value
        of computed p-values by Wilcox Ranksum Test as score while boundaries and gaps have a score of zero.
    """
    n_bins = csr_mat.shape[0]
    mean_cf = np.zeros(n_bins)
    pvalue = np.ones(n_bins)

    local_ext = np.ones(n_bins)*(-0.5)

    #Step 1
    for i in range(n_bins):
        diamond_mean = Get_Diamond_Matrix_Mean(data=csr_mat, i=i, size=window_size)
        mean_cf[i] = diamond_mean

    #Step 2
    gap_idx = Which_Gap_Region(data=csr_mat)
    proc_regions = Which_process_region(rmv_idx=gap_idx, n_bins=n_bins, min_size=3)

    for key in proc_regions:

        start = proc_regions[key]["start"]
        end = proc_regions[key]["end"]

        #print "Process Regions from " + str(start) + " to " + str(end)

        local_ext[start:end+1] = Detect_Local_Extreme(x=mean_cf[start:end+1])

    if statFilter:
        #Step 3
        lil_mat = csr_mat.todense()
        for k in range(1,(2*window_size)):

            mat_row = []
            mat_column = []
            my_range = list(range((n_bins*k), (n_bins*n_bins), 1+n_bins))
            mat_values = np.empty(len(my_range))
            col_arr = 0
            for j in my_range:
                mat_row.append(int(round(j//lil_mat.shape[0])))
                mat_column.append(j%lil_mat.shape[0])
                mat_values[col_arr] = lil_mat[int(round(j//lil_mat.shape[0])),j%lil_mat.shape[0]]
                col_arr = col_arr + 1

            scale_values = scale(mat_values)

            for i in range(len(mat_row)):
                lil_mat[mat_column[i],mat_row[i]] = scale_values[i]


        for key in proc_regions:
            start = proc_regions[key]['start']
            end = proc_regions[key]['end']

            pvalue[start:end] = Get_Pvalue(data=lil_mat[start:end+1, start:end+1], size=window_size, scale=1)

        for i in range(len(local_ext)):
            if local_ext[i] == -1 and pvalue[i] < 0.05:
                local_ext[i] = -2
        local_ext[local_ext==-1] = 0
        local_ext[local_ext==-2] = -1

        pvalue_cut=0.05
    else:
        pvalue = None
        pvalue_cut=None

    domains = Convert_Bin_To_Domain_TMP(n_bins=n_bins,
                                  signal_idx=np.where(local_ext==-1)[0],
                                  gap_idx=np.where(local_ext==-0.5)[0],
                                  pvalues=pvalue,
                                  pvalue_cut=pvalue_cut)


    return domains

def Get_Diamond_Matrix_Mean(data, i, size):

    n_bins = data.shape[1]
    if i==n_bins-1:
        return

    lowerbound = max( 0, i-size+1 )
    upperbound = min( i+size+1, n_bins)

    return (data[lowerbound:(i+1),(i+1):upperbound].mean())

def Which_Gap_Region(data):

    n_bins = data.shape[1]

    gap = np.zeros(n_bins)

    i=0
    while i < n_bins:

        j = i + 1
        while j < n_bins:
            if data[i:j+1, i:j+1].sum() == 0:
                gap[i:j+1] = -0.5
                j = j+1
            else:
                break

        i = j

    idx = np.where(gap==-0.5)[0]

    #return dict(zip(idx,idx))
    return idx

def Which_process_region(rmv_idx, n_bins, min_size):

    gap_idx = rmv_idx

    proc_regions = dict()
    proc_set = np.arange(n_bins)
    proc_set = np.setdiff1d(proc_set,gap_idx)

    n_proc_set = proc_set.shape[0]

    i=0
    while i < n_proc_set:
        start = proc_set[i]
        j = i+1

        while j < n_proc_set:
            if proc_set[j] - proc_set[j-1] <= 1:
                j = j + 1
            else:
                tmp_dict = {'start':start,'end':proc_set[j-1]}
                if abs(proc_set[j-1]-start) >= min_size:
                    proc_regions[start]=tmp_dict
                i = j
                break

        if j >= n_proc_set:
            tmp_dict = {'start':start,'end':proc_set[j-1]}
            if abs(proc_set[j-1]-start) >= min_size:
                proc_regions[start]=tmp_dict
            break

    return(proc_regions)

def Detect_Local_Extreme(x):

    n_bins = len(x)
    ret = np.zeros(n_bins)
    x[np.isnan(x)]=0

    if n_bins <= 3:
        ret[np.argmin(x)]=-1
        ret[np.argmax(x)]=1

        return ret
    # Norm##################################################3
    new_point_x, new_point_y = Data_Norm(x=np.arange(n_bins), y=x)


    x=new_point_y
    cp,Fv,Ev = Change_Point(x=np.arange(n_bins), y=x)

    if len(cp) <= 2:
        return ret
    for i in range(1,len(cp)-1):
        if x[cp[i]] >= x[cp[i]-1] and x[cp[i]] >= x[cp[i]+1]:
            ret[cp[i]] = 1
        else:
            if x[cp[i]] < x[cp[i]-1] and x[cp[i]] < x[cp[i]+1]:
                ret[cp[i]] = -1

        min_val = min( x[ cp[i-1] ], x[ cp[i] ] )
        max_val = max( x[ cp[i-1] ], x[ cp[i] ] )

        if np.min( x[cp[i-1]:cp[i]+1] ) < min_val:
            ret[ cp[i-1] + np.argmin( x[cp[i-1]:cp[i]+1] ) ] = -1
        if np.max( x[cp[i-1]:cp[i]+1] ) > max_val:
            ret[ cp[i-1] + np.argmax( x[cp[i-1]:cp[i]+1] ) ] = 1

    return ret

def Data_Norm(x, y):
    ret_x = np.zeros(len(x))
    ret_y = np.zeros(len(y))

    ret_x[0] = x[0]
    ret_y[0] = y[0]

    diff_x = np.diff(x)
    diff_y = np.diff(y)

    scale_x = 1 / ( np.abs(np.diff(x) ) ).mean()
    scale_y = 1 / ( np.abs(np.diff(y) ) ).mean()

    for i in range(1,len(x)):
        ret_x[i] = ret_x[i-1] + (diff_x[i-1]*scale_x)
        ret_y[i] = ret_y[i-1] + (diff_y[i-1]*scale_y)


    #return dict(zip(ret_x,ret_y))
    return ret_x, ret_y

def Change_Point(x, y):
    if len(x) != len(y):
        print("ERROR : The length of x and y should be the same")
        return 0

    n_bins = len(x)
    Fv = np.empty(n_bins)
    Fv[:] = np.NAN
    Ev = np.empty(n_bins)
    Ev[:] = np.NAN
    cp = []
    cp.append(0)
    #print x
    i=0
    Fv[0]=0
    while i < n_bins-1:
        j=i+1
        Fv[j] = sqrt( (x[j]-x[i])**2 + (y[j] - y[i] )**2 )
        #print Fv[j]
        while j < n_bins-1:
            j=j+1
            #k=(i+1):(j-1)
            Ev[j] = ( ( np.abs( (y[j]-y[i] )*x[(i+1):j] - (x[j] -x[i])*y[(i+1):j] - (x[i]*y[j]) + (x[j]*y[i]) ) ).sum() / sqrt( (x[j]-x[i])**2 + (y[j] - y[i] )**2 ) )
            #print Ev[j]
            #print x[(i+1):j]
            Fv[j] = sqrt( (x[j]-x[i])**2 + (y[j] - y[i])**2 ) - ( ( np.abs( (y[j]-y[i] )*x[(i+1):j] - (x[j] -x[i])*y[(i+1):j] - (x[i]*y[j]) + (x[j]*y[i]) ) ).sum() / sqrt( (x[j]-x[i])**2 + (y[j] - y[i] )**2 ) )
            #################################################
            #Not Original Code
            if isnan(Fv[j]) or isnan(Fv[j-1]):
                j = j-1
                cp.append(j)
                break
            ####################################################3
            if Fv[j] < Fv[j-1]:
                j = j - 1
                cp.append(j)
                break
        i=j

    cp.append(n_bins-1)

    return cp, Fv, Ev

def Convert_Bin_To_Domain_TMP(n_bins, signal_idx, gap_idx, pvalues=None, pvalue_cut=None):

    bins = dict()

    rmv_idx = np.setdiff1d(np.arange(n_bins),gap_idx)
    proc_region = Which_process_region(rmv_idx, n_bins, min_size=0)
    for key in proc_region:
        bins[proc_region[key]['start']] = {'start': proc_region[key]['start'], 'end'  : (proc_region[key]['end']+1), 'score': 10, 'tag' : 'gap'}


    rmv_idx = np.union1d(signal_idx, gap_idx)
    proc_region = Which_process_region(rmv_idx, n_bins, min_size=0)
    for key in proc_region:
        bins[proc_region[key]['start']] = {'start': proc_region[key]['start'], 'end'  : (proc_region[key]['end']+1), 'score': 10, 'tag' : 'domain'}

    rmv_idx = np.setdiff1d(np.arange(n_bins),signal_idx)
    proc_region = Which_process_region(rmv_idx, n_bins, min_size=1)
    for key in proc_region:
        bins[proc_region[key]['start']] = {'start': proc_region[key]['start'], 'end'  : (proc_region[key]['end']+1), 'score': 10, 'tag' : 'boundary'}




    if pvalues is not None and pvalue_cut is not None:
        for key in bins:
            if bins[key]['tag'] == 'domain':
                start_id = bins[key]['start']
                end_id = bins[key]['end']
                p_value_constr = pvalues[start_id:end_id]
                bins[key]['score'] = p_value_constr.mean()
                p_value_constr = p_value_constr[p_value_constr < pvalue_cut]
                if end_id - start_id == len(p_value_constr):
                    bins[key]['tag'] = "boundary"

    return bins

def scale(y):

    x = y.copy()

    x -= np.mean(x)
    x /= np.std(x, axis = 0, ddof = 1) # WTF different in numpy and R http://stackoverflow.com/questions/6457755/standard-deviation-in-r-seems-to-be-returning-the-wrong-answer-am-i-doing-some

    return x

def Get_Pvalue(data, size, scale):

    n_bins = data.shape[0]
    pvalue = np.ones(n_bins-1)

    for i in range(1,n_bins):
        dia = Get_Diamond_Matrix2(data, i, size=size)
        ups = Get_Upstream_Triangle(data, i, size=size)
        downs = Get_Downstream_Triangle(data, i, size=size)

        wil_test =  mannwhitneyu(x=dia*scale, y=ups+downs, use_continuity=True, alternative='less')
        pvalue[i-1] = wil_test.pvalue


    pvalue[ np.isnan(pvalue) ] = 1

    return(pvalue)

def Get_Diamond_Matrix2(data, i, size):

    n_bins = data.shape[0]
    #new_mat = np.ones_like(data)*np.NaN
    new_mat = np.ones(shape=(size,size))*np.NaN

    for k in range(1,size+1):
        if i-(k-1) >= 1 and i < n_bins:
            lower = min(i+1, n_bins)
            upper = min(i+size, n_bins)
            new_mat[size-(k-1)-1,0:(upper-lower+1)] = data[i-(k-1)-1,lower-1:upper]

    new_mat = new_mat[np.logical_not(np.isnan(new_mat))]

    return ((new_mat.transpose()).flatten()).tolist()

def Get_Upstream_Triangle(data, i, size):

    lower = max(1, i-size)
    tmp_mat = data[lower-1:i,lower-1:i]

    triag = (np.triu(tmp_mat,k=1).flatten())

    return triag[triag!=0].tolist()

def Get_Downstream_Triangle(data, i, size):

    n_bins = data.shape[0]
    if i==n_bins:
        return NaN

    upperbound = min(i+size, n_bins)
    tmp_mat = data[i:upperbound, i:upperbound]

    triag = (np.triu(tmp_mat,k=1).flatten())

    return triag[triag!=0].tolist()


def insulation_score(hic_data, dists, normalize=False, resolution=1,
                     delta=0, silent=False, savedata=None, savedeltas=None):
    """
    Compute insulation score.

    :param hic_dada: HiC_data object already normalized
    :param dists: list of pairs of distances between which to compute the
       insulation score. E.g. 4,5 means that for a given bin B(i), all
       interactions between B(i-4) to B(i-5) and B(i+4) to B(i+5) will be
       summed and used to compute the insulation score.
    :param False normalize: Normalize insulation score by the average in the
       chromosome, and log2 of this ratio.
    :param 1 resolution:
    :param 0 delta: to compute the delta for TAD detection (e.g. at 10kb use 10)
    :param False silent:
    :param None savedata: path to file where to save result
    :param None savedeltas: path to file where to save deltas

    :returns: dictionary with insulation score
    """
    bias = hic_data.bias
    bads = hic_data.bads
    decay = hic_data.expected
    if not decay or not bias:
        raise Exception('ERROR: HiC_data should be normalized by visibility '
                        'and by expected')

    insidx = {}
    deltas = {}
    for dist, end in dists:
        if not silent:
            print(' - computing insulation in band %d-%d' % (dist, end))
        insidx[(dist, end)] = {}
        deltas[(dist, end)] = {}
        for crm in hic_data.chromosomes:
            if crm in decay:
                this_decay = decay[crm]
            else:
                this_decay = decay
            total = 0
            count = 0
            for pos in range(hic_data.section_pos[crm][0] + end,
                             hic_data.section_pos[crm][1] - end):
                val = sum(hic_data[i, j] / bias[i] / bias[j] / this_decay[abs(j-i)]
                          for i in range(pos - end, pos - dist + 1)
                          if not i in bads
                          for j in range(pos + dist, pos + end + 1)
                          if not j in bads)
                total += val
                count += 1
                insidx[(dist, end)][pos] = val
            if normalize:
                try:
                    total /= float(count)
                except ZeroDivisionError:
                    pass
                if total == 0:
                    total = float('nan')
                for pos in range(hic_data.section_pos[crm][0] + end,
                                 hic_data.section_pos[crm][1] - end):
                    try:
                        with np.errstate(divide='ignore'):
                            insidx[(dist, end)][pos] = np.log2(insidx[(dist, end)][pos] / total)
                    except ZeroDivisionError:
                        insidx[(dist, end)][pos] = float('nan')
            if deltas:
                for pos in range(hic_data.section_pos[crm][0] + end,
                                 hic_data.section_pos[crm][1] - end):
                    up_vals = []
                    dw_vals = []
                    for spos in range(delta):
                        try:
                            up_vals.append(insidx[(dist, end)][pos - delta + spos])
                        except KeyError:
                            pass
                        try:
                            dw_vals.append(insidx[(dist, end)][pos + delta - spos])
                        except KeyError:
                            pass
                    with np.errstate(invalid='ignore'):
                        deltas[(dist, end)][pos] = (np.mean(up_vals) - np.mean(dw_vals))

    if savedata:
        out = open(savedata, 'w')
        out.write('# CRM\tCOORD\t' + '\t'.join(['%d-%d' % (d1, d2)
                                                for d1, d2 in dists]) +
                  '\n')
        for crm in hic_data.section_pos:
            for pos in range(*hic_data.section_pos[crm]):
                beg = (pos - hic_data.section_pos[crm][0]) * resolution
                out.write('{}\t{}-{}\t{}\n'.format(
                    crm, beg + 1, beg + resolution,
                    '\t'.join([str(insidx[dist].get(pos, 'NaN'))
                               for dist in dists])))
        out.close()

    if savedeltas:
        out = open(savedeltas, 'w')
        out.write('# CRM\tCOORD\t' + '\t'.join(['%d-%d' % (d1, d2)
                                                for d1, d2 in dists]) +
                  '\n')
        for crm in hic_data.section_pos:
            for pos in range(*hic_data.section_pos[crm]):
                beg = (pos - hic_data.section_pos[crm][0]) * resolution
                out.write('{}\t{}-{}\t{}\n'.format(
                    crm, beg + 1, beg + resolution,
                    '\t'.join([str(deltas[dist].get(pos, 'NaN'))
                               for dist in dists])))
        out.close()

    if delta:
        return insidx, deltas
    return insidx


def insulation_to_borders(ins_score, deltas, min_strength=0.1):
    """
    Best (for human-like genome size) according to https://doi.org/10.1038/nature14450
    is (at 10kb resolution) to use awindow size of 500 kb (use the function
    insulation_score with dist=(1,50)) and a delta of 100 kb (10 bins).


    :returns: the position in bin of each border, and the intensity of the
       border (sigmoid normalized, from 0 to 1)
    """
    borders = []
    for pos in range(max(ins_score)):
        if (ins_score.get(pos, 100) >= ins_score.get(pos + 1, -100)
            or
            ins_score.get(pos, 100) >= ins_score.get(pos - 1, -100)):
            continue
        if not (deltas.get(pos - 1, 100) > 0 and
                deltas.get(pos + 1, -100) < 0):
            continue
        # left
        lo = 1
        prev_lv = deltas.get(pos - 1, 100)
        while pos - 1 - lo > 0:
            lv = deltas.get(pos - 1 - lo, 100)
            if lv < prev_lv:
                break
            prev_lv = lv
            lo += 1
        # right
        ro = 1
        prev_rv = deltas.get(pos + 1, -100)
        while pos + 1 + ro <= len(deltas):
            rv = deltas.get(pos + 1 + ro, -100)
            if rv > prev_rv:
                break
            prev_rv = rv
            ro += 1
        strength = 1. / (1 + np.exp(-(prev_lv - prev_rv))) * 2 - 1
        if strength > min_strength:
            borders.append((pos, strength))
    return borders
