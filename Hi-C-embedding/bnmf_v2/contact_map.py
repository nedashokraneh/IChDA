"""
In support for the BNMF paper.
Author: Xihao Hu <huxihao@gmail.com> at The Chinese University of Hong Kong.
"""

import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class ContactMap:
    """ Class of contact map object """
    def __init__(self, name='DataName', enzyme='Enzyme', sparse=False):
        self.name = name
        self.enzyme = enzyme
        self.chr2idx = {} ## chromosome name map
        self.idx2chr = {} ## name map to chromosome
        self.use_sparse = sparse ## whether use sparse matrix
        ## contact information
        self.inter_chr1 = np.array([],dtype='int32')
        self.inter_loc1 = np.array([],dtype='int64')
        self.inter_chr2 = np.array([],dtype='int32')
        self.inter_loc2 = np.array([],dtype='int64')
        self.inter_freq = np.array([],dtype='int64')
        self.reset_heatmap()
        self.reset_solution()

    def reset_heatmap(self):
        if self.use_sparse:
            self.contact_map = csr_matrix((1,1)) ## V
        else:
            self.contact_map = np.matrix([[]]) ## V

    def reset_solution(self):
        ## dense matrices for H and S
        self.contact_group = np.matrix([[]]) ## H
        self.group_map = np.matrix([[]]) ## S
        if hasattr(self, 'bias_vector'):
            del self.bias_vector

    def get_chr_len(self):
        idx = self.idx2chr.keys()
        a = np.zeros(len(idx))
        for i in xrange(len(idx)):
            a[i] = np.sum(self.frag_chr == idx[i])
        return a

    def is_valide_compose(self, r=None):
        return (self.contact_map.shape[0] == self.contact_group.shape[0]) \
                and (self.group_map.shape[0] == r if not (r is None) else True)

    def sort_bins(self):
        w = np.asarray(self.contact_group*self.group_map)
        mv = np.max(w, axis=1)
        i1 = np.argsort(mv, kind='mergesort')[::-1]
        mc = np.argmax(w, axis=1) 
        i2 = np.argsort(mc[i1], kind='mergesort')
        idx = i1[i2]
        return self.contact_map[idx,:][:,idx]

    def sort_groups(self, order="nature"):
        ''' Sort the group order in contact_group and group_map 

        :param order: default is "diagnal", another choice is "location"
        '''
        sidx = range(self.group_map.shape[0]) ## no change
        if order == "diagnal":
            sidx = np.argsort(np.diagonal(self.group_map))[::-1]
        elif order == "location":
            sidx = np.asarray(np.argsort(self.contact_group.argmax(0)))[0]
        elif order == "nature":
            ref1 = np.array(np.diagonal(self.group_map), dtype='float').reshape(-1)
            ref2 = np.array(self.contact_group.argmax(0), dtype='float').reshape(-1)
            zscore = (ref1 - ref1.mean()) / ref1.std()
            ref2[zscore>3] += ref2.max() + ref1[zscore>3]
            sidx = np.argsort(ref2)
        elif order == "center":
            w = np.matrix(np.arange(self.contact_group.shape[0])+1)
            sidx = np.asarray(np.argsort(w * self.contact_group))[0]
        else: ## using given indeces
            sidx = np.array(order)
        self.contact_group = self.contact_group[:,sidx]
        self.group_map = self.group_map[sidx,:][:,sidx]
        return sidx

    def get_bin_str(self, i, format='nature'):
        if format == 'tab':
            return self.idx2chr[self.frag_chr[i]] + '\t' + `self.frag_sta[i]+1` + '\t' + `self.frag_end[i]`
        elif format == 'nature':
            return self.idx2chr[self.frag_chr[i]] + ':' + `self.frag_sta[i]+1` + '-' + `self.frag_end[i]`
        else:
            raise ValueError("format='tab' or 'nature'")

    def output_groups(self, binary=False):
        ''' Save H matrix and S matrix into files '''
        n,r = self.contact_group.shape
        outfile = open(self.name + '_groups.txt', 'w')
        for i in xrange(n):
            outfile.write(self.get_bin_str(i, 'tab'))
            for j in xrange(r):
                if binary:
                    outfile.write('\t' + `int(self.contact_group[i,j] > self.contact_group[:,j].mean())`)
                else:
                    outfile.write('\t' + str(round(self.contact_group[i,j],5)))
            outfile.write('\n')
        outfile.close()
        return self.name + '_groups.txt'

    def output_weights(self):
        n,r = self.contact_group.shape
        outfile = open(self.name + '_weights.txt', 'w')
        for i in xrange(r):
            for j in xrange(r):
                outfile.write(str(round(self.group_map[i,j],5)) + '\t')
            outfile.write('\n')
        outfile.close()
        return self.name + '_weights.txt'

    def output_bias(self, binary=False):
        n = self.bias_vector.shape[0]
        outfile = open(self.name + '_bias.txt', 'w')
        for i in xrange(n):
            outfile.write(self.get_bin_str(i, 'tab'))
            outfile.write('\t' + str(round(self.bias_vector[i],5)))
            outfile.write('\n')
        outfile.close()
        return self.name + '_groups.txt'

    def save(self, filename=None, savemap=True):
        """ Save the contact map object into several files. 

        :param filename: name of the output files (in .npz/.npy/.mat)
        """
        if filename is None:
            filename = self.name
        np.savez(filename + '.npz', 
                 name=self.name, enzyme=self.enzyme, 
                 chr2idx=self.chr2idx, idx2chr=self.idx2chr,
                 FC=self.frag_chr, FS=self.frag_sta, FE=self.frag_end,
                 US=self.use_sparse, CG=self.contact_group, GM=self.group_map)
        if hasattr(self, 'bias_vector'):
            np.savez(filename + '.npz', BV=self.bias_vector,
                 name=self.name, enzyme=self.enzyme, 
                 chr2idx=self.chr2idx, idx2chr=self.idx2chr,
                 FC=self.frag_chr, FS=self.frag_sta, FE=self.frag_end,
                 US=self.use_sparse, CG=self.contact_group, GM=self.group_map)
        if savemap and self.use_sparse:
            from scipy.io import loadmat, savemat
            savemat(filename + '_map.mat', {'map':self.contact_map}, \
                    format='5', do_compression=True, oned_as='row')
        elif savemap:
            np.save(filename + '_map.npy', self.contact_map)

    def clear(self, files = ['.npz', '_map.mat', '_map.npy', '_nmf.npz']):
        for f in files:
            name = self.name + f
            if os.path.exists(name):
                print '!Remove', name
                os.remove(name)

    def load(self, filename=None, loadmap=True):
        """ Load a contact map object from saved files """
        if filename is None:
            filename = self.name
        if not os.path.exists(filename + '.npz'):
            return False
        npzfile = np.load(filename + '.npz')
        self.contact_group = np.matrix(npzfile['CG'])
        self.group_map = np.matrix(npzfile['GM'])
        if 'BV' in npzfile:
            self.bias_vector = np.array(npzfile['BV'])
        if filename != self.name: ## use the default name
            self.name = npzfile['name'].item()
        self.enzyme = npzfile['enzyme'].item()
        self.chr2idx = npzfile['chr2idx'].item()
        self.idx2chr = npzfile['idx2chr'].item()
        self.frag_chr = npzfile['FC']
        self.frag_sta = npzfile['FS']
        self.frag_end = npzfile['FE']
        self.use_sparse = npzfile['US']
        if not loadmap:
            print 'Import decomposition solution from', filename
            return True
        if self.use_sparse:
            if not os.path.exists(filename + '_map.mat'): return False
            from scipy.io import loadmat, savemat
            self.contact_map = loadmat(filename + '_map.mat')['map'].tocsr()
        else:
            if not os.path.exists(filename + '_map.npy'): return False
            self.contact_map = np.matrix(np.load(filename + '_map.npy'))
        print 'Load contact map of size', self.contact_map.shape
        return True

    def duplicate(self, new_name='duplicated'):
        self.save(new_name)
        map2 = ContactMap()
        map2.load(new_name)
        map2.name = new_name
        return map2

    def fread(self, filename, column=None, start=0, end=-1):
        ## Start with 1 means to skip the header
        ## End with a negative means to read all lines
        cc = -1 ## first line is zero
        if filename.endswith('.txt'):
            infile = open(filename, 'r')
        elif filename.endswith('.RAWobserved'):
            infile = open(filename, 'r', 2<<9)
        elif filename.endswith('.gz'):
            import gzip
            infile = gzip.open(filename, 'r')
        else:
            raise IOError('Unknown file extension! %s'%filename)
        for line in infile:
            cc += 1 ## read a text file line by line
            if cc < start: continue ## skip first few lines
            elif end > 0 and cc == end: break ## skip remaining ones
            if line[0] == '#': continue
            ele = [e.strip() for e in line.strip().split()]
            if column is None: yield ele ## return all members
            elif column < 0: yield cc ## return column index
            else: yield ele[column] ## return the selected column
        infile.close()
        print 'Read file', filename, column, 'column and row [%d,%d)'%(start,cc)

    def hd5fread(self, hdf5file='SRR027956_fragment_dataset.hdf5'):
        from mirnylib import h5dict
        ## chromosomes for each read.
        ## "chrms1": "int8", "chrms2": "int8",
        ## midpoint of a fragment, determined as "(start+end)/2"
        ## "mids1": "int32", "mids2": "int32",
        if not os.path.exists(hdf5file):
            raise IOError('Cannot find file: %s'%hdf5file)
        data = h5dict.h5dict(hdf5file)
        if 'chrms1' in data: ## fragment data
            chr1 = data['chrms1']; loc1 = data['cuts1']
            chr2 = data['chrms2']; loc2 = data['cuts2']
            idx = np.logical_and(chr1>=0, chr2>=0) ## mark mapped
            freq = np.ones(idx.sum())
            return chr1[idx], loc1[idx], chr2[idx], loc2[idx], freq
        elif 'heatmap' in data: ## binned data
            reso = data["resolution"]
            chro_name = data["genomeIdxToLabel"]
            chro_start = data["chromosomeStarts"]
            heatmap = data["heatmap"]
            num = heatmap.shape[0]
            chro_end = [idx for idx in chro_start[1:]]
            chro_end.append(num) ## ending index
            ## get bin ranges
            chros = np.zeros(num, 'int8')
            start = np.zeros(num, 'int32')
            st_idx = 0
            for i in xrange(num):
                if i == chro_end[st_idx]:
                    st_idx += 1
                chros[i] = self.chr2idx.get('chr%s'%chro_name[st_idx], -1)
                start[i] = reso * (i-chro_start[st_idx])
            ## output pairwise
            chr1 = -np.ones(num*num, 'int')
            chr2 = -np.ones(num*num, 'int')
            loc1 = -np.ones(num*num, 'int')
            loc2 = -np.ones(num*num, 'int')
            freq = -np.ones(num*num, 'float')
            for i in xrange(num):
                chr1[i*num : (i+1)*num] = chros[i]
                loc1[i*num : (i+1)*num] = start[i]
                chr2[i*num : (i+1)*num] = chros
                loc2[i*num : (i+1)*num] = start
                freq[i*num : (i+1)*num] = heatmap[i,]
            assert (chr1 == -1).sum() == 0 and (chr2 == -1).sum() == 0
            return chr1, loc1, chr2, loc2, freq
        else:
            raise TypeError('Unknow data format in %s'%hdf5file)

    def genome_info(self, genomefile, i0=0, i1=1, i2=2, i3=3):
        info = self.fread(genomefile, start=1)
        self.chr2idx = {}
        self.idx2chr = {}
        for ele in info:
            self.chr2idx[ele[i3]] = int(ele[i0])
            self.idx2chr[int(ele[i0])] = ele[i3]
        for idx in sorted(self.idx2chr):
            print self.idx2chr[idx], 'with index', idx
        self.update_fragments(genomefile, i0=i0, i1=i1, i2=i2)

    def update_fragments(self, fragfile, i0=0, i1=1, i2=2, pdf=None):
        self.frag_chr = np.array([int(v) for v in self.fread(fragfile,i0,1)])
        self.frag_sta = np.array([int(v) for v in self.fread(fragfile,i1,1)])
        self.frag_end = np.array([int(v) for v in self.fread(fragfile,i2,1)])
        if pdf is None: return
        try: ## check them
            lens = self.frag_end - self.frag_sta
            bins = np.arange(0, lens.max(), lens.max()/50)
            mappable = self.fread(fragfile,3,1)
            self.frag_mappable = np.array([int(v) for v in mappable])
            plt.hist(lens, bins=bins, color='b', histtype='bar', label='Total')
            plt.hist(lens[self.frag_mappable==1], bins=bins, color='g', 
                     histtype='bar', label='Mappable')
            plt.xlabel('Fragment Length')
            plt.ylabel('Counts')
            plt.title('Fragment length distribution for enzyme ' + self.enzyme)
            plt.legend()
            pdf.savefig(); plt.clf();
        except: ## do nothing if error
            pass

    def focus_chromosome(self, ch, st=None, ed=None):
        idx = self.frag_chr == self.chr2idx[ch]
        if st is not None:
            idx = np.logical_and(idx, self.frag_sta >= st)
        if ed is not None:
            idx = np.logical_and(idx, self.frag_end <= ed)
        self.frag_chr = self.frag_chr[idx]
        self.frag_sta = self.frag_sta[idx]
        self.frag_end = self.frag_end[idx]
        self.chr2idx = {ch:self.chr2idx[ch]}
        self.idx2chr = {self.chr2idx[ch]:ch}
        return idx

    def get_locations(self, locfile, st=1, ch=0, po=1, nm=-1, add=-1, skip=True):
        try: ## already mapped
            Chr = [int(v) for v in self.fread(locfile, ch, start=st)]
        except: ## use name map
            Chr = [self.chr2idx.get(v,-1) for v in self.fread(locfile, ch, start=st)]
        Chr = np.array(Chr)
        Pos = np.array([int(v)+add for v in self.fread(locfile, po, start=st)]) ##
        idx = self.choose_map_loc(Chr, Pos)
        val = [v for v in self.fread(locfile, nm, start=st)]
        if skip:
            mapped = [(i,j) for i,j in zip(idx,val) if i >= 0] ## remove unmapped cases
            if len(idx) != len(mapped):
                print '!! There are', len(idx)-len(mapped),'unmapped locations.'
            idx, val = zip(*mapped)
        return list(idx), list(val)
            
    def add_interactions(self, interfile, st=1, ed=-1):
        print 'Add interaction file', interfile
        if interfile.endswith('.txt') or interfile.endswith('.gz'):
            chr1 = np.array([int(v) for v in self.fread(interfile,0,st,ed)])
            loc1 = np.array([int(v)-1 for v in self.fread(interfile,1,st,ed)]) ##
            chr2 = np.array([int(v) for v in self.fread(interfile,2,st,ed)])
            loc2 = np.array([int(v)-1 for v in self.fread(interfile,3,st,ed)]) ##
            freq = np.array([float(v) for v in self.fread(interfile,4,st,ed)])
        elif interfile.endswith('.hdf5'):
            chr1, loc1, chr2, loc2, freq = self.hd5fread(interfile)
        elif interfile.endswith('.RAWobserved'):
            chrs = interfile.split('/')[-1].split('_')[:-1]
            loc1 = np.array([int(v) for v in self.fread(interfile,0,st-1,ed)]) ##
            loc2 = np.array([int(v) for v in self.fread(interfile,1,st-1,ed)]) ##
            freq = np.array([float(v) for v in self.fread(interfile,2,st-1,ed)])
            if len(chrs) == 1:
                chr1 = np.ones_like(loc1)*self.chr2idx[chrs[0]]
                chr2 = np.ones_like(loc2)*self.chr2idx[chrs[0]]
            else:
                chr1 = np.ones_like(loc1)*self.chr2idx[chrs[0]]
                chr2 = np.ones_like(loc2)*self.chr2idx[chrs[1]]
        else:
            raise IOError('Unknown file extension!')
        self.inter_chr1 = np.append(self.inter_chr1, chr1)
        self.inter_loc1 = np.append(self.inter_loc1, loc1)
        self.inter_chr2 = np.append(self.inter_chr2, chr2)
        self.inter_loc2 = np.append(self.inter_loc2, loc2)
        self.inter_freq = np.append(self.inter_freq, freq)
        import gc
        gc.collect()

    def get_sparse_interactions(self):
        A = self.contact_map
        nanidx = np.isnan(A)
        n = A.shape[0]
        np.fill_diagonal(A,0)
        m = (np.count_nonzero(A)-np.isnan(A).sum())/2
        self.inter_chr1 = np.empty(m, dtype='int')
        self.inter_loc1 = np.empty(m, dtype='int')
        self.inter_chr2 = np.empty(m, dtype='int')
        self.inter_loc2 = np.empty(m, dtype='int')
        self.inter_freq = np.empty(m, dtype='float')
        k = 0
        for i in xrange(n):
            for j in xrange(i+1,n):
                if A[i,j] == np.nan or A[i,j] == 0:
                    continue
                self.inter_chr1[k] = self.frag_chr[i]
                self.inter_loc1[k] = self.frag_sta[i]
                self.inter_chr2[k] = self.frag_chr[j]
                self.inter_loc2[k] = self.frag_sta[j]
                self.inter_freq[k] = A[i,j]
                k += 1
        assert k == m
        self.contact_map = 0

    def get_interactions(self):
        A = np.nan_to_num(self.contact_map)
        n = A.shape[0]
        m = n*(n-1)/2
        self.inter_chr1 = np.empty(m, dtype='int')
        self.inter_loc1 = np.empty(m, dtype='int')
        self.inter_chr2 = np.empty(m, dtype='int')
        self.inter_loc2 = np.empty(m, dtype='int')
        self.inter_freq = np.empty(m, dtype='float')
        for i in xrange(1,n):
            k1 = i*(i-1)/2
            k2 = i*(i-1)/2 + i
            self.inter_chr1[k1:k2] = self.frag_chr[i]
            self.inter_loc1[k1:k2] = self.frag_sta[i]
            self.inter_chr2[k1:k2] = self.frag_chr[:i]
            self.inter_loc2[k1:k2] = self.frag_sta[:i]
            self.inter_freq[k1:k2] = A[i,:i]

    def plot_map(self, m=None, size=400, log=True, title=None, cmap='OrRd', vmin=None, vmax=None, colsum=True):
        """ Plot the contact map.

            :param m: contact map. If None, use the default map in the object
            :param size: resolution of plotted matrix
            :param log: whether to do a log-transformation on the plot
            :param title: the title of the plot
            :param cmap: color scheme {'jet', 'hot', 'gray'} + '_r'
        """
        ax1 = plt.gca()
        if m is None: 
            m = self.contact_map
            if self.use_sparse:
                if m.shape[0] > 4000: return
                m = m.todense()
        if title is None:
            plt.title('Heatmap of %s'%(self.name))
        else:
            plt.title(title)
        x, y = m.shape
        r = int(max(1, x/size))
        c = int(max(1, y/size))
        w = m[::r, ::c]
        x, y = w.shape
        if log:
            from matplotlib.colors import LogNorm
            plt.imshow(w, interpolation='none', vmin=vmin, vmax=vmax, 
                       norm=LogNorm(), aspect='auto', cmap=cmap)
            #cbar = plt.colorbar(ticks=[1, 10, 100, 500])
            #cbar.ax.set_yticklabels(['1', '10', '100', '>500'])
        else:
            plt.imshow(w, interpolation='none', vmin=vmin, vmax=vmax,
                       aspect='auto', cmap=cmap)
        plt.colorbar()
        plt.yticks(np.arange(0,x,max(1,x/5-1)), r*np.arange(1,x+1,max(1,x/5-1)), fontsize=size/50)
        plt.xticks(np.arange(0,y,max(1,y/5-1)), c*np.arange(1,y+1,max(1,y/5-1)), fontsize=size/50)
        if hasattr(self, 'frag_chr') and self.frag_chr.shape[0] == m.shape[0]: ## same indeces
            lab = self.frag_chr[::r].tolist()
            lab_pos = []; lab_chr = [];
            for idx in list(set(lab)):
                lab_pos.append((len(lab)+lab.index(idx)-lab[::-1].index(idx))/2)
                lab_chr.append(self.idx2chr[idx])
            plt.yticks(lab_pos, lab_chr, fontsize=2+size/50)
        if colsum:
            p1,p2,p3,p4 = ax1.get_position().bounds
            l1,l2,l3,l4 = ax1.axis()
            plt.axes([p1, p2-0.08, p3, p4/18]) ## left, bottom, width, height
            col_sum = np.array(np.nan_to_num(m))[:,::c].sum(0)
            plt.axis([l1, l2, 0, max(col_sum)]) ## xmin, xmax, ymin, ymax
            plt.plot(col_sum, 'b-')
            plt.plot(np.zeros_like(col_sum), 'k--')
            plt.plot(np.ones_like(col_sum)*col_sum.mean(), 'g--')
            plt.figtext(p1-size*1.7e-4, p2-0.07, 'Col. Sum', fontsize=size/50)
            plt.figtext(p1+p3+0.01, p2-0.07, '%.f'%col_sum.mean())
            plt.axis('off')

    def plot_submap(self, A=None, H=None, S=None, log=True, size=400, vmin=1, vmax=None, title=None):
        """ Plot the contact map and sub-matrices, e.g. m = H * S * H^T

            :param A: Input contact map. If None, use the default one
            :param H: Sub-matrix H. If None, use default in the object
            :param S: Sub-matrix S. If None, use default in the object
            :param size: resolution of plotted matrix
            :param log: whether to do a log-transformation on the plot
            :param title: the title of the plot
        """
        if title is None:
            plt.suptitle('Maps of %s'%self.name)
        else:
            plt.suptitle(title)
        if H is None: H = self.contact_group
        if S is None: S = self.group_map
        plt.subplot(221)
        self.plot_map(H*S*H.T, size=size/2, log=log, vmin=vmin, vmax=vmax, title=r'$R=H*S*H^T$', colsum=False)
        plt.subplot(222)
        self.plot_map(S, size=size/2, log=False, title=r'S', colsum=False)
        plt.subplot(223)
        self.plot_map(H, size=size/2, log=False, vmin=0, vmax=1, title=r'$H$')
        plt.subplot(224)
        self.plot_map(S*H.T, size=size/2, log=False, title=r'$W=S*H^T$')

    def plot_maphist(self, H=None):
        if H is None:
            H = self.contact_group
        for i in xrange(H.shape[1]):
            plt.hist(H[:,i], histtype='step')
#            plt.semilogy()

    def plot_dimcor(self, H=None, title=None, cmap='jet'):
        """ Plot the correlation among each dimensions in H matrix

            :param H: Sub-matrix H. If None, use the available one
        """
        if title is None:
            plt.title('Group correlation of %s'%self.name)
        else:
            plt.title(title)
        if H is None:
            H = self.contact_group
        n = H.shape[1]
        cor = np.corrcoef((H > H.mean(0)).T)
        np.fill_diagonal(cor, np.nan) ## mask diag
        plt.imshow(cor, interpolation='none', aspect='auto', cmap=cmap)
        plt.xticks([0,n-1], ['D1','D'+str(n)])
        plt.yticks(range(n), ['D'+`i+1` for i in range(n)])
        plt.colorbar()

    def plot_mapcor(self, A=None, title=None, cmap='jet'):
        """ Plot the correlation among two heatmap

            :param H: Sub-matrix H. If None, use the available one
        """
        if title is None:
            plt.title('Group correlation of %s'%self.name)
        else:
            plt.title(title)
        if A is None:
            A = self.contact_map
        n = A.shape[1]
        cor = np.corrcoef(A.T)
        np.fill_diagonal(cor, np.nan) ## mask diag
        plt.imshow(cor, interpolation='none', aspect='auto', cmap=cmap)
        plt.colorbar()

    def hash_chr_loc(self, chr, loc):
        assert np.issubdtype(chr.dtype, np.integer)
        assert np.issubdtype(loc.dtype, np.integer)
        ## hash genomic locations to be unique values
        chr_order = self.idx2chr.keys()
        chr_order.sort()
        dist = loc.copy()
        dist = dist.astype('int64')
        chr_len = []
        for ch in chr_order: ## get chromosome lengths
            idx = np.where(ch == self.frag_chr)[0]
            left = self.frag_sta[idx[0]]
            dist[chr==ch] -= left
            length = self.frag_end[idx[-1]] - left
            chr_len.append(length + 100) ## add a small gap
        chr_len = np.array(chr_len, dtype='int64')
        for ci in xrange(len(chr_order)): ## assume they lie together
            idx = np.where(chr_order[ci] == chr)[0]
            if idx.size == 0: continue
            #print chr_order[ci], dist[idx].max(), chr_len[ci]
            #if dist[idx].max() >= chr_len[ci]:
            #    raise ValueError('Loci %s:%s exceeds the maximum length of %s'
            #            %(chr_order[ci], dist[idx].max(), chr_len[ci]))
            ## hash bins on the chromosome
            idx = np.array([i for i in idx if dist[i]>=0 and dist[i] < chr_len[ci]])
            dist[idx] += chr_len[0:ci].sum() + 100 ## gap means modified
        ## mask locations from missing chromosomes
        dist[dist == loc] = -1
        return dist

    def throw_close(self, cut=None):
        if cut is None:
            return
        throw = np.logical_and(self.inter_chr1 == self.inter_chr2,
                    np.abs(self.inter_loc1-self.inter_loc2) < cut)
        if np.any(throw):
            print 'Throw', throw.sum(), 'close interactions'
        remain = np.logical_not(throw)
        self.inter_chr1 = self.inter_chr1[remain]
        self.inter_chr2 = self.inter_chr2[remain]
        self.inter_loc1 = self.inter_loc1[remain]
        self.inter_loc2 = self.inter_loc2[remain]
        self.inter_freq = self.inter_freq[remain]

    def create_contactmap(self, throw=1):
        self.throw_close(cut=throw)
        ## fragment interaction map
        n = self.frag_chr.shape[0]
        f = np.array(self.inter_freq, dtype='float64')
        fr_st = self.hash_chr_loc(self.frag_chr, self.frag_sta)
        fr_ed = self.hash_chr_loc(self.frag_chr, self.frag_end)
        pos1 = self.hash_chr_loc(self.inter_chr1, self.inter_loc1)
        pos2 = self.hash_chr_loc(self.inter_chr2, self.inter_loc2)
        throw1 = np.logical_or(pos1==pos2, np.logical_or(pos1<0, pos2<0))
        ## equal case belongs to the right one: 0->[0,end1) 1->[end1, end2)
        idx1 = np.searchsorted(fr_ed, pos1[np.logical_not(throw1)], side='right')
        idx2 = np.searchsorted(fr_ed, pos2[np.logical_not(throw1)], side='right')
        throw2 = np.logical_or(idx1 == n , idx2 == n)
        if np.any(throw1) or np.any(throw2):
            print 'Throw', throw1.sum()+throw2.sum(), 'interactions'
        #print n, idx1.min(), idx1.max(), idx2.min(), idx2.max()
        from scipy import sparse
        m = sparse.coo_matrix((f[np.logical_not(throw1)][np.logical_not(throw2)],
                              (idx1[np.logical_not(throw2)],
                               idx2[np.logical_not(throw2)])), shape=(n,n), dtype='float')
        m = m.tocsr()
        m.eliminate_zeros()
        if self.use_sparse: 
            self.contact_map = m+m.T
        else: 
            self.contact_map = (m+m.T).todense()
        print 'Create contact map of shape', self.contact_map.shape

    def create_binnedmap(self, binsize=10e3, throw=1, lazy=False):
        ## interaction map for regions of same size
        binsize = int(binsize)
        bin_chr = []; bin_sta = []; bin_end = []
        chr_order = list(set([v for v in self.frag_chr]))
        chr_order.sort()
        for ch in chr_order:
            idx = np.where(ch == self.frag_chr)[0]
            left = self.frag_sta[idx[0]]
            chr_len = self.frag_end[idx[-1]] - left
            num = chr_len // binsize + int(chr_len % binsize > 0) ## ceil
            bin_chr += [ch for i in xrange(num)]
            bin_sta += [left+i*binsize for i in xrange(num)]
            bin_end += [left+i*binsize+binsize for i in xrange(num)]
            bin_end[-1] = left+chr_len ## fix the last bin
        self.frag_chr = np.array(bin_chr) ## update fragments
        self.frag_sta = np.array(bin_sta)
        self.frag_end = np.array(bin_end)
        if lazy: return ## only need to know the bins
        self.create_contactmap(throw=throw) ## reconstruct

    def get_binsize(self):
        bins = self.frag_end - self.frag_sta
        return int(bins.max())

    def create_bestmap(self, throw=1):
        ## automatically choose the best resolution
        binsize = int(round(self.frag_end.max() - self.frag_sta.min(), -3))
        best_bin = 0; best_deg = -1;
        while binsize >= 1000: ## best possible resolution
            self.create_binnedmap(binsize, throw=throw)
            if self.use_sparse:
                degree = self.contact_map.data.shape[0]/float(self.contact_map.shape[0])
            else:
                degree = np.sum(self.contact_map>0)/float(self.contact_map.shape[0])
            if best_deg < degree:
                best_deg = degree
                best_bin = binsize
            else:
                break ## find a peak
            binsize = int(round(binsize*0.8, -3))
        self.create_binnedmap(best_bin, throw=throw)
        return best_bin, best_deg

    def create_densemap(self, files, reso=10e3, throw=1):
        ## directly use dense map to reduce memory cost
        for linkf in files:
            if linkf.endswith('.txt') or linkf.endswith('.gz'):
                step = 4e6 ## suitable for 2G memory
            elif linkf.endswith('.RAWobserved'):
                step = 8e6 ## suitable for 2G memory
            else: 
                step = -1 ## read the whole file
            start = 1; end = start + step
            while True: ## read huge file part by part
                tmp = ContactMap()
                tmp.frag_chr = self.frag_chr
                tmp.frag_sta = self.frag_sta
                tmp.frag_end = self.frag_end
                tmp.chr2idx = self.chr2idx
                tmp.idx2chr = self.idx2chr
                tmp.use_sparse = self.use_sparse
                tmp.add_interactions(linkf, start, end)
                num_get = tmp.inter_chr1.size
                tmp.create_binnedmap(reso, throw=throw)
                if self.contact_map.size == 0:
                    self.contact_map = tmp.contact_map
                else:
                    self.contact_map += tmp.contact_map
                self.frag_chr = tmp.frag_chr
                self.frag_sta = tmp.frag_sta
                self.frag_end = tmp.frag_end
                if step < 0 or  num_get < step: 
                    ## will get no more links
                    break
                start = end
                end += step

    def blur_contactmap(self, sigma=3):
        from scipy import ndimage
        self.contact_map = ndimage.gaussian_filter(self.contact_map, sigma=sigma)

    def randomize_map(self):
        ''' Randomly permute the locations in the contact map '''
        if not self.use_sparse: ## implement for dense matrix
            n = self.contact_map.shape[0]
            i = np.random.permutation(np.arange(n))
            self.contact_map = self.contact_map[i,:][:,i]

    def get_null_map(self, rand_seed=0):
        n = self.contact_map.shape[0]
        f = self.frag_chr[:,np.newaxis]
        v = np.bincount(np.asarray(self.frag_chr, dtype='int')).max()
        A = np.zeros((n,n))
        for k in range(v):
            case1 = np.logical_and(np.eye(n,k= k)==1, f == f.T)
            case2 = np.logical_and(np.eye(n,k=-k)==1, f == f.T)
            A[case1] = v/(k+1.0)
            A[case2] = v/(k+1.0)
        A[f != f.T] =  v/float(n)
        from numpy.random import rand, seed
        seed(rand_seed) ## to avoid randomness in eigen vectors
        return A + 1e-10*rand(n,n)

    def get_expected_map(self):
        M = np.nan_to_num(self.contact_map)
        n = self.contact_map.shape[0]
        f = self.frag_chr[:,np.newaxis]
        A = np.zeros((n,n))
        for k in xrange(n):
            case1 = np.logical_and(np.eye(n,k= k)==1, f == f.T)
            case2 = np.logical_and(np.eye(n,k=-k)==1, f == f.T)
            A[case1] = M[case1].mean()
            A[case2] = M[case2].mean()
        A[f != f.T] = M[f != f.T].mean()
        return A

    def extract_bias(self, H=None, S=None):
        if H is None:
            H = self.contact_group
        if S is None:
            S = self.group_map
        B = np.array(H*S).mean(axis=1) ## bias vector
        B /= B[B>0].mean()
        B[B == 0] = 1
        H /= B[:,np.newaxis]
        self.bias_vector = B
        return np.asmatrix(H)

    def add_bias_back(self):
        if hasattr(self, 'bias_vector'):
            H = np.asarray(self.contact_group)
            H = self.bias_vector[:,np.newaxis] * H
            self.contact_group = np.asmatrix(H)
            del self.bias_vector

    def decompose(self, method='NMF-PoissonManifoldEqual', dim_num=None, par_lam=1, max_iter=3000, stop_thrd=1e-6, A=None, V=None, plot=None):
        if A is None:
            A = self.contact_map
        if dim_num is None:
            dim_num = int(A.shape[0]**0.5)
        n = A.shape[0]
        C = np.array(self.frag_chr, dtype='float')
        obj = 0
        if method == 'NND':
            H, S = NNDSVD(A, r=dim_num)
        elif method == 'EIG':
            H, S = EIG(A, r=dim_num)
        elif method == 'PCA':
            B, X = IterateCorrect(A,1) ## Simple normalization
            C = np.nan_to_num(map_cor(X,X)) ## Pearson correlation
            U = (C-np.mean(C.T,axis=1)).T ## Co-var matrix for PCA
            H, S = EIG(np.cov(U), r=dim_num)
        elif method == 'ICE':
            B, X = IterateCorrect(A)
            #self.fake_cis(X)
            H, S = EIG(X-np.nan_to_num(X).mean(), dim_num)
        elif method == 'Correct':
            B, X = IterateCorrect(A)
            self.contact_map = X
            self.bias_vector = B
            return 0
        else:
            self.add_bias_back()
            H, S, obj = NMF_main(A, C, self.contact_group, self.group_map, J=method, w=1, t=1, r=dim_num, L=par_lam, e=stop_thrd, i=int(max_iter), P=plot)
        if self.use_sparse:
            R = masked_dot(V, H*S, H.T)
        else:
            R = H * S * H.T
        ## with bias vector
        if method == 'ICE':
            R = np.multiply(R, np.dot(B, B.T))
            self.bias_vector = B
        if method.endswith('Equal'):
            H = self.extract_bias(H, S)
        self.contact_group = H
        self.group_map = S
        return obj ## recovered map

    def decompose_auto(self, method='NMF-PoissonManifoldEqual', dim_num=None, par_lam=1, max_iter=3000, stop_thrd=1e-6, beta=5, update=False, plot=None):
        savename = self.name+'_nmf.npz'
        matrix_pool = {}
        if os.path.exists(savename):
            archive = np.load(savename)
            for name in archive:
                #print 'Load', name, 'from', savename
                matrix_pool[name] = archive[name]
            archive.close()
        ## decompose the contact map
        A = self.contact_map
        n = A.shape[0]
        paras = []
        if dim_num is None:
            r = int((beta*n)**0.5) ## largest value
        elif type(dim_num) == type([]):
            r = dim_num[0]
        else:
            r = dim_num
        import gc
        while True:
            print 'dim_num =', r
            if '%s-%s-H-%s'%(method,r,n) not in matrix_pool or \
               '%s-%s-S-%s'%(method,r,n) not in matrix_pool or update:
                self.reset_solution()
                try:
                    self.decompose(method='NND', dim_num=r)
                except:
                    print '!! SVD failed and so use random initalization'
                    pass
            else:
                self.contact_group = np.matrix(matrix_pool['%s-%s-H-%s'%(method,r,n)])
                self.group_map = np.matrix(matrix_pool['%s-%s-S-%s'%(method,r,n)])
            ## decompose
            H, S, obj = NMF_main(A = A,
                                 C = np.array(self.frag_chr, dtype='float'),
                                 H = self.contact_group,
                                 S = self.group_map,
                                 J = method, w=1, t=1,
                                 r = r, e=stop_thrd,
                                 i = int(max_iter),
                                 L = par_lam)
            matrix_pool['%s-%s-H-%s'%(method,r,n)] = H
            matrix_pool['%s-%s-S-%s'%(method,r,n)] = S
            np.savez_compressed(savename, **matrix_pool)
            gc.collect() ## reduce memory cost
            ## calculate metric
            now_obj = gini_impurity(np.diag(S))
            max_obj = gini_impurity(np.ones(S.shape[1]))
            paras.append((r, now_obj))
            if dim_num is None:
                alpha = now_obj/max_obj
                ref_dim = int((alpha*beta*n)**0.5)
                if ref_dim in zip(*paras)[0]:
                    break
                r = ref_dim
            elif type(dim_num) == type([]):
                if len(paras) == len(dim_num):
                    break
                else:
                    r = dim_num[len(paras)]
            else:
                break
        if not plot is None:
            X, Y = zip(*paras)
            X = np.array(X)
            X1 = np.arange(X.min()-1, X.max()+2, 1)
            Y = np.array(Y)
            plt.plot(X, Y, 'ro')
            if len(X) > 1:
                plt.quiver(X[:-1], Y[:-1], X[1:]-X[:-1], Y[1:]-Y[:-1],
                           scale_units='xy', angles='xy', scale=1)
            plt.plot(X1, 1-1.0/X1, 'k.--')
            plt.xlabel('Number of clusters')
            plt.ylabel('Gini impurity')
            plt.title('Decomposed by various number of clusters')
            if plot != plt:
                plot.savefig(); plt.clf();
        ## use specified
        H = np.matrix(matrix_pool['%s-%s-H-%s'%(method,r,n)])
        S = np.matrix(matrix_pool['%s-%s-S-%s'%(method,r,n)])
        if self.use_sparse:
            R = masked_dot(A, H*S, H.T)
        else:
            R = H * S * H.T
        if method.endswith('Equal'): ## with bias vector
            H = self.extract_bias(H, S)
        self.contact_group = H
        self.group_map = S
        if plot == plt: return
        return paras

    def plot_bias(self):
        if not hasattr(self, 'bias_vector'):
            raise ValueError('Please use a decomposation method with bias correction.')
        plt.plot(self.bias_vector, '-')
        plt.xlim([0, self.bias_vector.shape[0]])
        plt.title('The bias vector from %s'%self.name)

    def random_split(self, ratio=0.5, A=None):
        if A is None:
            A = self.contact_map
        if not self.use_sparse:
            a = np.array(A)
            m = np.random.choice([1,0], size=a.shape, p=[ratio, 1-ratio])
            t = np.triu(m, 1)
            m = t + t.T + np.diag(m)
            print 'Split matrix by', m.sum()/float(m.shape[0]*m.shape[1])
            a1 = a.copy()
            a1[m == 0] = np.nan
            a1 = np.matrix(a1)
            a2 = a.copy()
            a2[m == 1] = np.nan
            a2 = np.matrix(a2)
        else:
            a = A.tocoo()
            m = np.random.choice([1,0], size=a.data.shape, p=[ratio, 1-ratio])
            ## TODO: need to make it symmetric
            print 'Split matrix by', m.sum()/float(m.shape[0])
            a1 = a.copy()
            a1.data *= m
            a1 = a1.tocsr()
            a1.eliminate_zeros()
            a2 = a.copy()
            a2.data *= 1-m
            a2 = a2.tocsr()
            a2.eliminate_zeros()
        return a1, a2

    def choose_map_loc(self, chr, loc=None):
        if loc is None:
            loc = [int(p.split(':')[1]) for p in chr]
            chr = [self.chr2idx[p.split(':')[0]] for p in chr]
        n = np.array(chr).shape[0]
        frag = self.hash_chr_loc(self.frag_chr, self.frag_end)
        pos = self.hash_chr_loc(np.array(chr), np.array(loc))
        idx = np.searchsorted(frag, pos, side='right')
        idx[np.logical_or(pos < 0, idx == len(frag))] = -1
        return idx.tolist()

    def shift_index(self, idx, sh=0):
        ridx = []
        chrs = np.array(self.frag_chr, dtype='int')
        from random import choice
        for ch in np.unique(chrs):
            for i in idx:
                if chrs[i] == ch:
                    ri = i + choice(range(-sh, sh+1))
                    if 0 <= ri and ri < len(chrs) and chrs[ri] == ch:
                        ridx.append(ri)
        return ridx

    def test_enrichment_dims(self, dims, idx, method='AvgCCD', plot=None, title='Test', pages=3):
        W = []; V = []; P = []
        for dim in dims:
            self.decompose_auto(dim_num=dim)
            srt, val, pval = self.test_enrichment(idx, method=method, N=1000, correct=True, normp=True, plot=None, title=title, pages=1)
            W.append(np.asarray(self.contact_group*self.group_map/self.group_map.sum()))
            V.append(val)
            P.append(pval)
        w = np.hstack(W)
        val = np.concatenate(V)
        pval = np.concatenate(P)
        if method == 'AvgCCD':
            srt = pval.argsort()
        else:
            srt = val.argsort()[::-1]
        sign = sum([pval[i] < 0.01 for i in srt])
        if plot is not None:
            self.plot_map(w[:, srt[:max(10,sign)]], title='%s significant clusters from %s decompositions'%(sign, len(dims)), log=False)
            plot.savefig(); plt.clf();
        return srt, val, pval

    def test_enrichment(self, idx, method='AvgCCD', N=1000, correct=True, normp=False, plot=None, title='Test', pages=3):
        ''' Sort the dimensions according to some measurement:
            {'AvgRnd', 'AUC', 'AvgCCD'}
        '''
        group = np.array(self.contact_group*self.group_map/self.group_map.sum())
        n,r = group.shape
        if not normp:
            N *= r
        if correct: ## correct for multiple test
            correct = r
        else:
            correct = 1
        if method.startswith('Tau'): ## compare the two vectors
            ## REF: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.kendalltau.html
            from scipy.stats import kendalltau 
            val = np.ones(r)
            pval = np.ones(r)
            for i in xrange(r):
                choose = np.arange(n)
                if method == 'Tau-pos':
                    choose = np.array(group[:,i]) > 1/float(n*r)
                tau, p_value = kendalltau(group[choose,i], idx[choose])
                val[i] = tau
                pval[i] = p_value * correct
            srt = val.argsort(kind='mergesort')[::-1] ## stable sort
            return srt, val, pval
        elif method.startswith('PCC'): ## compare the two vectors
            idx = np.array(idx).reshape(-1)
            assert group.shape[0] == idx.shape[0]
            ## REF: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
            from scipy.stats import pearsonr
            val = np.ones(r)
            pval = np.ones(r)
            for i in xrange(r):
                choose = np.arange(n)
                if method == 'PCC-pos':
                    choose = np.array(group[:,i]) > 0.1/float(r)
                pcc, p_value = pearsonr(group[choose,i], idx[choose])
                val[i] = pcc
                pval[i] = p_value * correct
            srt = val.argsort(kind='mergesort')[::-1] ## stable sort
            if (not plot is None) and pages >= 1 and len(srt)>10:
                plt.bar(np.arange(1,11)-0.4, val[srt[:10]], color='w')
                plt.xticks(range(1,11), ['C%s'%(i+1) for i in srt[:10]])
                plt.xlabel('Cluster Index')
                plt.ylabel('Enrichment of cluster memberships')
                plt.title(title+' by '+method)
                plot.savefig(); plt.clf()
            if (not plot is None) and pages >= 2 and len(srt) > pages:
                for i in srt[:pages]:
                    plt.plot(idx, group[:,i], '.', label='C%d(%.3G)'%(i+1, pval[i]))
                plt.legend()
                plt.title('Test %s by %s'%(title, method))
                plot.savefig(); plt.clf()
            return srt, val, pval
        elif method.startswith('SPC'): ## compare the two vectors
            idx = np.array(idx).reshape(-1)
            assert group.shape[0] == idx.shape[0]
            ## REF: http://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
            from scipy.stats import spearmanr
            val = np.ones(r)
            pval = np.ones(r)
            for i in xrange(r):
                choose = np.arange(n)
                if method == 'SPC-pos':
                    choose = np.array(group[:,i]) > 1/float(r*n)
                elif method == 'SPC-msk':
                    coverage = np.array(np.nan_to_num(self.contact_map)).sum(0)
                    choose = coverage > 0
                rho, p_value = spearmanr(group[choose,i], idx[choose])
                val[i] = rho
                pval[i] = p_value * correct
            srt = val.argsort(kind='mergesort')[::-1] ## stable sort
            if method == 'SPC-abs':
                srt = np.abs(val).argsort(kind='mergesort')[::-1]
            return srt, val, pval
        idx = np.arange(n)[idx]
        lab_pos = []; lab_chr = []; ## used to plot genome
        for i in xrange(0, n, n/5):
            lab_pos.append(i)
            lab_chr.append(self.idx2chr[self.frag_chr[i]])
        if (not plot is None) and pages >= 4:
            plt.hist(idx, n)
            plt.xlim([0, n])
            plt.xticks(lab_pos, lab_chr, fontsize=12)
            plt.ylabel('Number of test loci on each bin')
            plt.title(title+' by '+method)
            plot.savefig(); plt.clf()
        if method.startswith('AUC'):
            val = np.zeros(r)
            pval = np.zeros(r)
            from scipy.stats import mannwhitneyu
            coverage = np.array(np.nan_to_num(self.contact_map)).sum(0)
            for i in xrange(r):
                pos = []; neg = []
                for j in xrange(n):
                    if method == 'AUC-pos':
                        if group[j,i] <= 0:
                            continue
                    elif method == 'AUC-msk':
                        if coverage[j] == 0:
                            continue
                    if j in idx:
                        pos.append(group[j,i])
                    else:
                        neg.append(group[j,i])
                ## REF: http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.mannwhitneyu.html
                U, p_value = mannwhitneyu(pos, neg)
                val[i] = 1-U/float(len(pos)*len(neg))
                pval[i] = 2 * p_value * correct
                if len(pos) == 0 or len(neg) == 0:
                    val[i] = 0.5
                    pval[i] = correct
            srt = pval.argsort()
            if method == 'AUC-abs':
                srt = np.abs(val-0.5).argsort()[::-1]
            if (not plot is None) and pages >= 1 and len(srt) > 10:
                plt.bar(np.arange(1,11)-0.4, val[srt[:10]], color='w')
                plt.ylim([0.5,val[srt[0]]+0.05])
                plt.xticks(range(1,11), ['C%s'%(i+1) for i in srt[:10]])
                plt.xlabel('Cluster Index')
                plt.ylabel('Enrichment by AUC')
                plt.title(title+' by '+method)
                plot.savefig(); plt.clf()
        elif method == 'GSEA':
            idx = np.unique(idx)
            rank = np.argsort(group,0)[::-1,:]
            es = -np.ones_like(group)
            for j in xrange(r): 
                es[idx,j] = 1
                es[es[:,j]>0,j] /= es[es[:,j]>0,j].sum()
                es[es[:,j]<0,j] /= abs(es[es[:,j]<0,j].sum())
                es[:,j] = es[rank[:,j],j]
            val = np.cumsum(es, 0).max(0) ## inqury set
            if (not plot is None) and pages >= 2:
                plt.plot(np.cumsum(es, 0))
                plt.ylabel('Sum of Enrichment Scores')
                plt.xlim([0,n])
                plt.title(title+' by '+method)
                plot.savefig(); plt.clf()
            rval = np.zeros(r)
            aval = np.zeros((r, N-1))
            for i in xrange(N-1): ## sampling N-1 times
                rnd = np.random.permutation(range(n))
                aval[:,i] = np.cumsum(es[rnd,:], 0).max(0) ## inqury set
                rval += (aval[:,i] >= val).reshape(-1)
            if normp:
                from scipy.stats import norm
                dists = norm(aval.mean(1), aval.std(1))
                pval = (1 - dists.cdf(val)) * correct
            else:
                pval = correct*(rval+1)/(N+1.0)
            srt = pval.argsort(kind='mergesort')
            if (not plot is None) and pages >= 1:
                plt.boxplot(aval[srt[:10],:].T)
                plt.plot(range(1,11), val[srt[:10]], 'ro')
                plt.xticks(range(1,11), ['C%s'%(i+1) for i in srt[:10]])
                plt.xlabel('Cluster Index')
                plt.ylabel('Enrichment of cluster memberships')
                plt.title(title+' by '+method)
                plot.savefig(); plt.clf()
        elif method.startswith('Avg'):
            background = method[3:]
            C = np.array(self.frag_chr, dtype='int')
            val = group[idx,:].mean(0) ## inqury set
            ## generate
            rval = np.zeros(r)
            aval = np.zeros((r, N-1))
            if background == 'Rnd':
                for i in xrange(N-1): ## sampling N-1 times
                    ridx = np.random.choice(range(n), len(idx), replace=True)
                    aval[:,i] = group[ridx,:].mean(0)
                    rval += (aval[:,i] >= val).reshape(-1)
            elif background == 'CCD':
                ccd_data = sample_ccd_init(C, idx)
                for i in xrange(N-1): ## sampling N-1 times
                    ridx = sample_ccd_run(ccd_data, cut=-1)
                    aval[:,i] = group[ridx,:].mean(0)
                    rval += (aval[:,i] >= val).reshape(-1)
            elif background == 'CCD20':
                ccd_data = sample_ccd_init(C, idx)
                for i in xrange(N-1): ## sampling N-1 times
                    ridx = sample_ccd_run(ccd_data, cut=20)
                    aval[:,i] = group[ridx,:].mean(0)
                    rval += (aval[:,i] >= val).reshape(-1)
            elif background == 'Uni':
                for i in xrange(N-1): ## sampling N-1 times
                    ridx = sample_uni(C, idx)
                    aval[:,i] = group[ridx,:].mean(0)
                    rval += (aval[:,i] >= val).reshape(-1)
            elif background == 'Sft':
                for i in xrange(N-1): ## sampling N-1 times
                    ridx = sample_sft(C, idx)
                    aval[:,i] = group[ridx,:].mean(0)
                    rval += (aval[:,i] >= val).reshape(-1)
            else:
                raise ValueError('Unknown background: %s'%background)
            if normp:
                from scipy.stats import norm
                dists = norm(aval.mean(1), aval.std(1))
                pval = (1 - dists.cdf(val)) * correct
            else:
                pval = correct*(rval+1)/(N+1.0)
            srt = pval.argsort(kind='mergesort')
            if (not plot is None) and pages >= 1 and len(srt) > 10:
                plt.boxplot(aval[srt[:10],:].T)
                plt.plot(range(1,11), val[srt[:10]], 'ro')
                plt.xticks(range(1,11), ['C%s'%(i+1) for i in srt[:10]])
                plt.xlabel('Cluster Index')
                plt.ylabel('Enrichment of cluster memberships')
                plt.title(title+' by '+method)
                plot.savefig(); plt.clf()
            val /= group.mean(0) ## calculate enrichment
        else:
            raise ValueError('Unknown method: %s'%method)
        if (not plot is None) and pages >= 2 and len(srt) > 3:
            #plt.plot(np.ones(n), linestyle='--')
            for i in srt[:pages]:
                good = group[idx,i] >= group[:,i].mean()
                plt.plot(idx[good], group[idx,i][good]/group[:,i].mean(), 
                         linestyle='None', marker='o', markersize=5, 
                         label='C%d(%.3G) %d/%d'%(i+1, pval[i], good.sum(), len(idx)))
            plt.legend()
            plt.xlim([0,n])
            plt.xticks(lab_pos, lab_chr, fontsize=12)
            plt.gca().set_yscale('log')
            plt.ylabel('Cluster weight enrichment')
            plt.title('Test %s by %s'%(title, method))
            plot.savefig(); plt.clf()
        if (not plot is None) and pages >= 3:
            for i in srt[:pages]:
                plt.plot(group[:,i], 'b-')
                plt.plot(idx, group[idx,i], 'r.')
                plt.plot([0, n], [group[:,i].mean()]*2, 'k--')
                plt.xlim([0, n])
                plt.xticks(lab_pos, lab_chr, fontsize=12)
                plt.ylabel('Cluster membership weight')
                plt.title('In C%s value is %.3G by %s'% (i+1, val[i], method))
                plot.savefig(); plt.clf()
        return srt, val, pval

    def label_bins(self, idx, method='AUC', cutoff=0.1, title='', plot=None):
        H = np.asarray(self.contact_group)
        n,r = H.shape
        full = np.arange(n)[idx].tolist()
        bins = full
        labels = {}
        while bins != []:
            srt, val, pval = self.test_enrichment(bins, method=method, plot=plot, title=title, pages=2)
            chosen = []
            remain = []
            j = -1
            for j in srt:
                if j not in labels: ## not used
                    break
            if j<0 or pval[j] > 0.1:
                break
            pb = H[:,j]*self.group_map[j,j]
            for i in bins: 
                if pb[i] > cutoff: ## enriched
                    chosen.append(i)
                else:
                    remain.append(i)
            if len(chosen) == 0:
                break
            plt.hist(pb)
            plt.title(self.name+' C%s'%(j+1))
            plot.savefig(); plt.clf()
            labels[j] = [i for i in full if pb[i] > cutoff]
            bins = remain
        labels['unknown'] = bins
        return labels

    def get_gc_content(self, ccfile):
        st, A = self.get_locations(ccfile, st=1, ch=0, po=1, nm=3, add=0)
        ed, C = self.get_locations(ccfile, st=1, ch=0, po=2, nm=4, add=-1)
        ed, G = self.get_locations(ccfile, st=1, ch=0, po=2, nm=5, add=-1)
        ed, T = self.get_locations(ccfile, st=1, ch=0, po=2, nm=6, add=-1)
        m = min(len(st), len(ed))
        n = self.contact_map.shape[0]
        gc = np.zeros(n, dtype='float')
        at = np.zeros(n, dtype='float')
        for i in xrange(m):
            gc[(st[i]+ed[i])/2] += int(G[i]) + int(C[i])
            at[(st[i]+ed[i])/2] += int(A[i]) + int(T[i])
        acgt = gc + at
        acgt[acgt < 1.0] = 1.0
        return gc/acgt

    def min_rmsd(self, map2, dims=False):
        w1 = np.asarray(self.contact_group * self.group_map / self.group_map.sum())
        w2 = np.asarray(map2.contact_group * map2.group_map / self.group_map.sum())
        N,I = self.contact_group.shape
        N,J = map2.contact_group.shape
        C = np.zeros((I,J))
        for i in xrange(I):
            for j in xrange(J):
                C[i,j] = np.power(w1[:,i]-w2[:,j],2).sum()
                C[i,j] /= np.power(np.power(w1[:,i],2).sum(), 0.5)
                C[i,j] /= np.power(np.power(w2[:,j],2).sum(), 0.5)
        m1 = np.argmin(C,axis=1)
        m2 = np.argmin(C,axis=0)
        s1 = set([(i,m1[i]) for i in xrange(I)])
        s2 = set([(m2[j],j) for j in xrange(J)])
        mm = list(s1 & s2)
        if dims: return mm
        i,j = zip(*mm)
        return np.power(np.power(w1[:,i]-w2[:,j],2).mean(),0.5)

    def best_cor(self, map2):
        w1 = np.asarray(self.contact_group * self.group_map / self.group_map.sum())
        w2 = np.asarray(map2.contact_group * map2.group_map / self.group_map.sum())
        N,I = self.contact_group.shape
        N,J = map2.contact_group.shape
        C = np.zeros((I,J))
        for i in xrange(I):
            for j in xrange(J):
                C[i,j] = np.dot(w1[:,i], w2[:,j])
                C[i,j] /= np.power(np.dot(w1[:,i], w1[:,i]), 0.5)
                C[i,j] /= np.power(np.dot(w2[:,j], w2[:,j]), 0.5)
        plt.imshow(C, interpolation='none', aspect='auto', cmap='jet')
        plt.xlabel(self.name)
        plt.ylabel(map2.name)
        plt.colorbar()
        m1 = np.argmax(C,axis=1)
        m2 = np.argmax(C,axis=0)
        return w1,w2

    def compare(self, map2=None, raw=True, metric='RMSD'):
        if raw:
            R1 = np.array(self.contact_map).copy()
            if map2 is not None:
                R2 = np.array(map2.contact_map).copy()
            else:
                R2 = np.asarray(self.contact_group * S1 * self.contact_group.T)
        else:
            S1 = self.group_map
            S2 = map2.group_map
            R1 = np.asarray(self.contact_group * S1 * self.contact_group.T)
            R2 = np.asarray(map2.contact_group * S2 * map2.contact_group.T)
        if metric == 'RMSD':
            return metric_l2(R1,R2)
        elif metric == 'PCC':
            return metric_cor(R1,R2,rank=False)
        elif metric == 'SPC':
            return metric_cor(R1,R2,rank=True)
        elif metric == 'KL_D':
            return metric_kl_div(R1,R2)
        elif metric == 'JS_D':
            return metric_js_div(R1,R2)
        else:
            raise ValueError('Unknow metric function '+metric)

    def label_groups(self, method='PCC', plot=None):
        chn = [[self.chr2idx[na],na] for na in self.chr2idx]
        chn.sort() ## default order
        n = len(chn) ## number of chr
        m = self.contact_group.shape[1] ## number of dim
        k = self.contact_group.shape[0] ## number of bin
        info = np.zeros((n,m), dtype='float')
        refs = []
        for i in xrange(n):
            idx = [chn[i][0] == int(self.frag_chr[j]) for j in xrange(k)]
            srt, val, p_val = self.test_enrichment(idx, method=method)
            info[i,:] = val 
            srt, val, p_val = self.test_enrichment(self.contact_group[:,i], method=method)
            refs += [val[j] for j in xrange(n) if i>j]
        arm = info.argmax(axis=0)
        cutoff = max(refs)
        print 'The cutoff is set to be', cutoff
        grp = {}
        for i in xrange(m):
            if info[arm[i],i] > cutoff: ## significant
                idx = grp.get(chn[arm[i]][1], [])
                idx.append(i)
                grp[chn[arm[i]][1]] = idx
            else:
                idx = grp.get('global', [])
                idx.append(i)
                grp['global'] = idx
        srt = np.argsort(arm)
        if plot is not None:
            plt.imshow(info[:,srt], interpolation='none', aspect='auto')
            plt.xticks(np.arange(0,m,5), srt[::5])
            plt.yticks(np.arange(n), [na for ch,na in chn])
            plt.xlabel('Clusters')
            plt.colorbar()
            plot.savefig(); plt.clf();
        return grp

    def get_bias(self, biasfile):
        """ Get the three bias terms from Hi-C experiments: 
                fragment length, GC content, and mappability.
            :param biasfile: file from the tool in Hu et al. 2012 Bioinfor.
        """
        '''
        Definition of the bias file:
        Col 0 - Restriction Fragment ID
        Col 1 - Direction of Fragment
        Col 2 - Chromosome
        Col 3 - Position of the Cut site
        Col 4 - Restriction Fragment Length
        Col 5 - GC Content
        Col 6 - Mappability X (No of Reads mapped)
        Col 7 - Total Reads (55)
        Col 8 - Mappability Score - X/55.
        '''
        chro = [self.chr2idx.get(v,-1) for v in self.fread(biasfile, 2, 0)]
        site = [int(v) for v in self.fread(biasfile, column=3, start=0)]
        idx = self.choose_map_loc(np.array(chro), np.array(site))
        leng = [int(v) for v in self.fread(biasfile, column=4, start=0)]
        gc = [float(v) for v in self.fread(biasfile, column=5, start=0)]
        mp = [float(v) for v in self.fread(biasfile, column=8, start=0)]
        b_len = np.zeros(self.frag_chr.shape[0], dtype='float')
        b_gcc = np.zeros(self.frag_chr.shape[0], dtype='float')
        b_map = np.zeros(self.frag_chr.shape[0], dtype='float')
        count = np.zeros(self.frag_chr.shape[0], dtype='float')
        for i in xrange(len(idx)):
            if idx[i] < 0: continue ## unused bins
            b_len[idx[i]] += np.log(leng[i])
            b_gcc[idx[i]] += gc[i]
            b_map[idx[i]] += mp[i]
            count[idx[i]] += 1
        count[count > 0] = 1.0 / count[count > 0]
        return b_len*count, b_gcc*count, b_map*count

    def gene_count(self, gfile):
        chro = [self.chr2idx.get(v,-1) for v in self.fread(gfile, 0, 0)]
        gene = [int(v=='transcript') for v in self.fread(gfile, column=2, start=0)]
        site = [int(v) for v in self.fread(gfile, column=3, start=0)]
        idx = self.choose_map_loc(np.array(chro), np.array(site))
        count = np.zeros(self.frag_chr.shape[0], dtype='float')
        for i in xrange(len(idx)):
            if idx[i] >= 0: ## mapped
                count[idx[i]] += gene[i]
        return count

    def gene_exp(self, gfile, efile):
        ## read gene locations
        chro = [self.chr2idx.get(v,-1) for v in self.fread(gfile, 0, 0)]
        site = [int(v) for v in self.fread(gfile, column=1, start=0)]
        gene = [v for v in self.fread(gfile, column=4, start=0)]
        idx = self.choose_map_loc(np.array(chro), np.array(site))
        ## read expression profile
        exps = []
        import math
        with open(efile, 'r') as infile:
            infile.readline() ## read header
            for line in infile:
                exps.append(sum([math.log(float(e)+0.03,2) for e in line.split(',')]))
        sums = np.zeros(self.frag_chr.shape[0], dtype='float')
        print len(idx), len(exps)
        for i in xrange(len(idx)):
            if idx[i] >= 0: ## mapped
                sums[idx[i]] += exps[i]
        return sums

    def trace_sum(self, a):
        n = a.shape[0]
        s = np.zeros(n)
        for i in xrange(n):
            s[i] = np.trace(a,i)
        return s/s.sum()


    def fake_cis(self, data=None):
        f = self.frag_chr
        N = data.shape[0]
        for i in range(N):
            for j in range(i,N):
                if f[i] == f[j]:
                    while True:
                        s = np.random.randint(N)
                        if np.random.randint(1):
                            if f[i] != f[s]:
                                data[i,j] = data[i,s]
                                data[j,i] = data[i,s]
                                break
                        else:
                            if f[j] != f[s]:
                               data[i,j] = data[j,s]
                               data[j,i] = data[j,s]
                               break

    def mask_cis(self, A=None, mask=np.nan):
        f = self.frag_chr[:,np.newaxis]
        if A is None: A = self.contact_map
        A[f == f.T] = mask

    def mask_trans(self, A=None, mask=np.nan):
        f = self.frag_chr[:,np.newaxis]
        if A is None: A = self.contact_map
        A[f != f.T] = mask

    def mask_diag(self, k=0):
        'Masking the diagnal values in the contact map'
        if self.use_sparse:
            self.contact_map.setdiag(np.nan, k=k)
        else:
            a = np.eye(self.contact_map.shape[0], k=k)
            self.contact_map[a==1] = np.nan

    def mask_nearby(self, k=0):
        'Masking the bins close to the diagnal in the contact map'
        trans = self.contact_map.copy()
        self.mask_cis(trans)
        self.mask_trans()
        for i in xrange(-k, k+1):
            self.mask_diag(i)
        self.contact_map[np.isnan(trans)==False] = 0
        self.contact_map += np.nan_to_num(trans)
        print 'Masking interactions of bins <=', k*self.get_binsize()

    def mask_short(self, ratio=0.5):
        'Masking the bins corresponding a short region in chromosomes'
        binlen = self.frag_end - self.frag_sta
        mask = binlen <= ratio * binlen.max()
        print 'Masking', mask.sum(), 'bins with short length'
        self.contact_map[mask,:] = np.nan
        self.contact_map[:,mask] = np.nan

    def mask_low(self, p=1):
        coverage = np.asarray(np.nan_to_num(self.contact_map)).sum(0)
        cutoff = np.percentile(coverage[coverage>0], p)
        mask = coverage <= cutoff 
        print 'Masking', mask.sum(), 'bins with links lower than', cutoff
        self.contact_map[mask,:] = np.nan
        self.contact_map[:,mask] = np.nan

    def mask_poor(self, k=2):
        coverage = np.asarray(np.nan_to_num(self.contact_map)).sum(0)
        cutoff = np.mean(coverage[coverage>0]) - k*np.std(coverage[coverage>0])
        mask = coverage <= cutoff 
        print 'Masking', mask.sum(), 'bins with links lower than', cutoff
        self.contact_map[mask,:] = np.nan
        self.contact_map[:,mask] = np.nan

    def trim_high(self, p=0.05, cis=False):
        'Trim high interchromosomal interactions'
        valid = np.logical_not(np.isnan(self.contact_map))
        the_map = np.asarray(np.nan_to_num(self.contact_map))
        f = self.frag_chr[:,np.newaxis]
        intra = np.equal(f, f.T)
        inter = np.logical_not(intra)
        v1 = np.logical_and(valid, intra)
        v2 = np.logical_and(valid, inter)
        c1 = np.percentile(the_map[v1], 100-p)
        c2 = np.percentile(the_map[v2], 100-p)
        if cis:
            mask1 = np.logical_and(v1, the_map > c1)
            print 'Triming', mask1.sum(), 'intra interactions to', c1
            self.contact_map[mask1] = c1
        mask2 = np.logical_and(v2, the_map > c2)
        print 'Triming', mask2.sum(), 'inter interactions to', c2
        self.contact_map[mask2] = c2

    def remove_extreme(self, q=0.01):
        v = self.inter_freq > 0 ## non-zero
        i = np.equal(self.inter_chr1, self.inter_chr2) ## intra
        j = np.logical_not(i) ## inter-chromosomal interactions
        vi = np.logical_and(v, i)
        vj = np.logical_and(v, j)
        s1 = np.percentile(self.inter_freq[vi], q*100)
        m1 = np.percentile(self.inter_freq[vi], (1-q)*100)
        s2 = np.percentile(self.inter_freq[vj], q*100)
        m2 = np.percentile(self.inter_freq[vj], (1-q)*100)
        print 'Seting intra:', s1, m1, 'inter:', s2, m2
        self.inter_freq[np.logical_and(vi, self.inter_freq<s1)] = 0
        self.inter_freq[np.logical_and(vi, self.inter_freq>m1)] = m1
        self.inter_freq[np.logical_and(vj, self.inter_freq<s2)] = 0
        self.inter_freq[np.logical_and(vj, self.inter_freq>m2)] = m2

    def remove_small(self, m=4):
        idx = self.inter_freq>=m 
        print 'Removing', '%.4f'%(1-idx.sum()/float(len(idx))), 'of links below', m
        self.inter_chr1 = self.inter_chr1[idx]
        self.inter_loc1 = self.inter_loc1[idx]
        self.inter_chr2 = self.inter_chr2[idx]
        self.inter_loc2 = self.inter_loc2[idx]
        self.inter_freq = self.inter_freq[idx]

    def plot_loglog(self, m=None):
        plt.title('Distribution of interactions along 1D')
        if m is None: m = self.contact_map
        if self.use_sparse:
            if m.shape[0] > 4000: return
            m = m.todense()
        for ch in list(set(self.frag_chr.tolist())):
            idx = self.frag_chr == ch
            v = m[idx,:][:,idx]
            bins = [(i+1.0)/v.shape[0] for i in range(v.shape[0])]
            vals = self.trace_sum(v)
            plt.loglog(bins, vals, linestyle='-')
        plt.xlabel('Ratio of linked locations to the total length')
        plt.ylabel('Number of observed links')


###############################################################################
## Weighted Balanced Update for Symmetric Matrix by Ding et. al., 2005
def nmf_j1(X,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import array, dot, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[np.isnan(X)] = 0
    for iter in xrange(maxiter):
        R = dot(dot(H,S), H.T)
        obj.append(np.power((I*(X-R)).reshape(-1,),2).sum())
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %G; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        H *= dot(I*X, dot(H,S)) / (dot(I*R, dot(H,S))+eps)
        hm = H.sum(axis=0) ## normalize H and update S
        H /= hm[np.newaxis,:]
        S *= dot(hm.T, hm)
        R = dot(dot(H,S), H.T)
        S *= dot(dot(H.T, I*X), H) / dot(dot(H.T, I*R), H)
        print '\r'*len(strobj),
    print ''
    return H,S,obj


###############################################################################
## Modified Update Rules by Lin 2007
def nmf_j1b(X,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    sigma = delta = eps
    from numpy import array, dot, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[np.isnan(X)] = 0
    for iter in xrange(maxiter):
        R = dot(dot(H,S), H.T)
        obj.append(np.power((I*(X-R)).reshape(-1,),2).sum())
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %G; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        gradH = dot(I*(R-X), dot(H,S))
        Ha = np.zeros_like(H)
        Ha[gradH < 0] = sigma
        Hb = np.maximum(H, Ha)
        H -= Hb / (dot(I*R, dot(H,S)) + delta) * gradH
        hm = H.sum(axis=0) ## normalize H and update S
        H /= hm[np.newaxis,:]
        S *= dot(hm.T, hm)
        R = dot(dot(H,S), H.T)
        S *= dot(dot(H.T, I*X), H) / dot(dot(H.T, I*R), H)
        print '\r'*len(strobj),
    print ''
    return H,S,obj

def nmf_j2(X,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import array, dot, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[np.isnan(X)] = 0
    const = (X*np.log(X+eps)).sum() - X.sum() 
    for iter in xrange(maxiter):
        R = dot(dot(H,S), H.T)
        obj.append(const + (I*(R - X*np.log(R))).sum())
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %G; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        H *= dot(X/R, dot(H,S)) / (dot(I, dot(H,S))+eps)
        hm = H.sum(axis=0) ## normalize H and update S
        H /= hm[np.newaxis,:]
        S *= dot(hm.T, hm)
        R = dot(dot(H,S), H.T)
        S *= dot(dot(H.T, I*X/R), H) / dot(dot(H.T, I), H)
        print '\r'*len(strobj),
    print ''
    return H,S,obj

def nmf_j2e(X,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import array, dot, outer, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[I == 0] = 0
    G = H.copy() ## no bias
    B = dot(G,S).mean(axis=1)
    B /= B.sum()/(B>0).sum()
    B[B==0] = 1
    for iter in xrange(maxiter):
        R = dot(dot(G,S), G.T); R[R == 0] = eps;
        obj.append((I*(R - X*np.log(R))).sum())
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %s; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        G *= dot(I*X/R, dot(G,S)) / (dot(I, dot(G,S)) + eps)
        ## normalized row of H and update bias
        B += dot(G,S).mean(axis=1) ## bias vector
        B /= B.sum()/(B>0).sum() ## normalize
        B[B==0] = 1
        H = G / B[:,np.newaxis]
        ## normalize column of H and update S
        h = H.mean(axis=0)
        S *= outer(h, h)
        H /= h[np.newaxis,:]
        ## add the bias back to H for reconstruction
        G = H * B[:,np.newaxis]
        R = dot(dot(G,S), G.T) + eps
        S *= dot(dot(G.T, I*X/R), G) / dot(dot(G.T, I), G)
        print '\r'*len(strobj),
    print ''
    return G,S,obj

def nmf_j2le(X,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import array, dot, outer, ones, exp
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[I == 0] = 0
    G = H.copy() ## no bias
    B = dot(G,S).mean(axis=1)
    B /= B.sum()/(B>0).sum()
    B[B==0] = 1
    for iter in xrange(maxiter):
        R = dot(dot(G,S), G.T); R[R > 10] = 1e30;
        obj.append((I*(exp(R) - X*R)).sum())
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %s; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        G *= dot(I*X, dot(G,S)) / (dot(I*R, dot(G,S)) + eps)
        ## normalized row of H and update bias
        B += dot(G,S).mean(axis=1) ## bias vector
        B /= B.max() ## normalize
        B[B==0] = 1
        H = G / B[:,np.newaxis]
        ## normalize column of H and update S
        h = H.mean(axis=0)
        S *= outer(h, h)
        H /= h[np.newaxis,:]
        ## add the bias back to H for reconstruction
        G = H * B[:,np.newaxis]
        R = dot(dot(G,S), G.T) + eps
        S *= dot(dot(G.T, I*X), G) / dot(dot(G.T, I*R), G)
        print '\r'*len(strobj),
    print ''
    return G,S,obj

###############################################################################
## Multiplicative Rule for manifold case by Cai et al. 2008
def mark2network(m,near=2):
    ## transform landmark info into neareast-neighbor network
    from numpy import diagflat
    n = m.shape[0]
    E = 0
    for k in xrange(near):
        E += diagflat(np.ones(n-k), -k) + diagflat(np.ones(n-k), k)
    idx = m[:,np.newaxis]
    E[(idx == idx.T) == False] = 0 ## mask regions not close,
        ## 1 == 1 gives True -> False be zero
        ## np.nan == 1 gives False -> True be zero
        ## np.nan == np.nan gives False -> True be zero
    D = diagflat(E.sum(0), 0) ## Diagnal matrix
    v = np.logical_not(np.isnan(m))
    return E[v,:][:,v], D[v,:][:,v]

def nmf_j3(X,C,lm,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import array, dot, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[I == 0] = 0
    E,D = mark2network(C)
    for iter in xrange(maxiter):
        R = dot(dot(H,S), H.T)
        obj.append(np.power((I*(X-R)).reshape(-1,),2).sum() + \
                   lm * np.trace(dot(dot(H.T,(D-E)), H)))
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %G; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        H *= (dot(I*X, dot(H,S)) + lm*dot(E,H)) / \
             (dot(I*R, dot(H,S)) + lm*dot(D,H) + eps)
        hm = H.sum(axis=0) ## normalize H and update S
        H /= hm[np.newaxis,:]
        S *= dot(hm.T, hm)
        R = dot(dot(H,S),H.T)
        S *= dot(dot(H.T, I*X), H) / dot(dot(H.T, I*R), H)
        print '\r'*len(strobj),
    print ''
    return H,S,obj

def nmf_j3a(X,C,lm,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    ## Normalized both raw and column in H
    from numpy import array, dot, outer, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[np.isnan(X)] = 0
    E,D = mark2network(C)
    G = H.copy() ## no bias
    B = dot(G,S).mean(axis=1)
    B /= B.sum()/(B>0).sum()
    B[B==0] = 1
    for iter in xrange(maxiter):
        R = dot(dot(G,S), G.T)
        obj.append(np.power((I*(X-R)).reshape(-1,),2).sum() + \
                   lm * np.trace(dot(dot(G.T,(D-E)), G)))
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %G; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        G *= (dot(I*X, dot(G,S)) + lm*dot(E,G)) / \
             (dot(I*R, dot(G,S)) + lm*dot(D,G) + eps)
        ## normalized row of H and update bias
        B += dot(G,S).mean(axis=1) ## bias vector
        B /= B.sum()/(B>0).sum() ## normalize
        B[B==0] = 1
        H = G / B[:,np.newaxis]
        ## normalize column of H and update S
        h = H.mean(axis=0) 
        S *= outer(h, h)
        H /= h[np.newaxis,:]
        ## add the bias back to H for reconstruction
        G = H * B[:,np.newaxis]
        R = dot(dot(G,S), G.T)
        S *= dot(dot(G.T, I*X), G) / (dot(dot(G.T, I*R), G) + eps)
        print '\r'*len(strobj),
    print ''
    return G,S,obj

def nmf_j4(X,C,lm,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import array, dot, outer, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[I == 0] = 0
    E,D = mark2network(C)
    const = (X*np.log(X+eps)).sum() - X.sum()
    for iter in xrange(maxiter):
        R = dot(dot(H,S), H.T) + eps
        obj.append(const + (I*(R - X*np.log(R))).sum() + \
                   lm * np.trace(dot(dot(H.T,(D-E)), H)))
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %s; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        H *= (dot(I*X/R, dot(H,S)) + 2*lm*dot(E,H)) / \
             (dot(I, dot(H,S)) + 2*lm*dot(D,H))
        h = H.mean(axis=0) ## normalize H and update S
        H /= h[np.newaxis,:]
        S *= outer(h, h)
        R = dot(dot(H,S), H.T) + eps
        S *= dot(dot(H.T, I*X/R), H) / dot(dot(H.T, I), H)
        print '\r'*len(strobj),
    print ''
    return H,S,obj

def nmf_j4a(X,C,lm,H,S,minimp,maxiter,eps):
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import dot, outer, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[I == 0] = 0
    E,D = mark2network(C)
    G = H.copy() ## no bias
    B = dot(G,S).mean(axis=1)
    B /= B.sum()/(B>0).sum()
    B[B==0] = 1
    H = G / B[:,np.newaxis]
    const = (X*np.log(X+eps)).sum() - X.sum()
    for iter in xrange(maxiter):
        R = dot(dot(G,S), G.T) + eps;
        obj.append(const + (I*(R - X*np.log(R))).sum() + \
                   lm * np.trace(dot(dot(H.T,(D-E)), H)))
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %s; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        G *= (dot(X/R, dot(G,S)) + 2*lm*dot(E,H)) / \
             (dot(I, dot(G,S)) + 2*lm*dot(D,H) + eps)
        if True:
            ## normalized row of H and update bias
            B = dot(G,S).mean(axis=1)
            B /= B.sum()/(B>0).sum()
            B[B==0] = 1
        H = G / B[:,np.newaxis]
        ## normalize column of H and update S
        h = H.mean(axis=0)
        H /= h[np.newaxis,:]
        S *= outer(h, h)
        ## add the bias back to H for reconstruction
        G = H * B[:,np.newaxis]
        R = dot(dot(G,S), G.T) + eps
        S *= dot(dot(G.T, X/R), G) / (dot(dot(G.T, I), G)+eps)
        print '\r'*len(strobj),
    print ''
    return G,S,obj

def nmf_j4a_numexpr(X,C,lm,H,S,minimp,maxiter,eps):
    from numexpr import evaluate
    if type(X) == type(''): X = np.load(X)
    else: X = np.array(X)
    from numpy import dot, outer, ones
    obj = [float("infinity")]
    I = ones(X.shape)
    I[np.isnan(X)] = 0 ## missing data
    X[I == 0] = 0
    E,D = mark2network(C)
    L = D-E
    G = H.copy() ## no bias
    B = dot(G,S).mean(axis=1)
    B /= B.sum()/(B>0).sum()
    B[B==0] = 1
    H = G / B[:,np.newaxis]
    const = (X*np.log(X+eps)).sum() - X.sum()
    for iter in xrange(maxiter):
        R = dot(dot(G,S), G.T) + eps
        temp1 = evaluate('sum(I*(R - X*log(R)))')
        if lm == 0:
            obj.append(const + temp1)
        else:
            obj.append(const + temp1 + lm * np.trace(dot(dot(H.T,L), H)))
        if iter == 0: print 'Initial objective is', obj[-1]
        strobj = 'Current objective is %s; '%obj[-1]
        print strobj,
        if abs(obj[-2]-obj[-1]) <= abs(minimp*obj[-1]): break
        GS = dot(G,S)
        temp1 = dot(X/R, GS)
        temp3 = dot(I, GS)
        if lm == 0:
            temp2 = 0
            temp4 = 0
        else:
            temp2 = 2*lm*dot(E,H)
            temp4 = 2*lm*H*(np.diag(D)[:,np.newaxis])
        evaluate('G * (temp1 + temp2) / (temp3 + temp4 + eps)', out=G)
        if True:
            ## normalized row of H and update bias
            B = dot(G,S).mean(axis=1)
            B /= B.sum()/(B>0).sum()
            B[B==0] = 1
        H = G / B[:,np.newaxis]
        h = H.mean(axis=0)
        evaluate('H / h', out=H)
        temp1 = outer(h, h)
        evaluate('S * temp1', out=S)
        ## add the bias back to H for reconstruction
        G = H * B[:,np.newaxis]
        R = dot(dot(G,S), G.T) + eps
        XdR = evaluate('X/R')
        temp1 = dot(dot(G.T, XdR), G)
        temp2 = dot(dot(G.T, I), G)
        evaluate('S * temp1 / (temp2 + eps)', out=S)
        print '\r'*len(strobj),
    print ''
    return G,S,obj

def fun_map(p,f):
    ## Fake function for passing multiple arguments to Pool.map()
    return f(*p)

def NMF_main(A,C=None,H=None,S=None,J='NMF-Poisson',w=1,t=1,r=50,e=1e-6,i=3000,E=1e-30,L=1,P=None):
    """ Non-negative matrix factorization on a square matrix.
        ``i.e.: A = H * S * H.T``

        :param A: input matrix
        :param C: cluster indication vector
        :param H: initial H matrix. If None, create random ones.
        :param S: initial S matrix. If None, create random ones.
        :param w: number of workers avaliable (depending on #CPU-Cores)
        :param t: number of tasks to process (each with random initialization)
        :param r: number of groups/clusters/dimensions to factorize
        :param e: minimum improve on objective to stop the iteration
        :param i: maximum number of iteration for the algorithms
        :param E: epsilon a small contant to avoid dividing by zero
        :param L: lambda for objective function
        :param P: pdf objective to plot the trends of the objective during iteration
        :return (H,S): H is a cluster membership matrix, and S is a cluster size matrix
    """
    print '> Solve NMF for size', A.shape, type(A)
    tf = 'TEMP_MAP_%s_%s'%A.shape
    if isinstance(A, np.matrix) or isinstance(A, np.ndarray): ## dense
        coverage = np.array(np.nan_to_num(A)).sum(0)
        density = coverage.sum() / float(A.shape[0])**2
        if C is None:
            C = np.ones(A.shape[0]) ## a string
        C[coverage == 0] = np.nan
        M = np.array(A[coverage>0,:][:,coverage>0])
        if w != 1:
            tf += '.npy'
            np.save(tf, M) ## pass huge data by file, not parameter
        fun1 = {'NMF-Gaussian':nmf_j1,
                'NMF-Poisson':nmf_j2, 
                'NMF-PoissonEqual':nmf_j2e,
                'NMF-PoissonLogEqual':nmf_j2le}
        fun2 = {'NMF-GaussianManifold':nmf_j3, 
                'NMF-GaussianManifoldEqual':nmf_j3a, 
                'NMF-PoissonManifold':nmf_j4, 
                'NMF-PoissonManifoldEqual':nmf_j4a}
        try:
            import numexpr
            fun2['NMF-PoissonManifoldEqual'] = nmf_j4a_numexpr
        except:
            print 'Recommend to install the `numexpr` package'
        fun = fun1.copy(); fun.update(fun2)
    else: ## assume sparse matrix in csr format
        M = A.tocsr() ## the same copy
        density = (M.data>0).sum() / float(M.shape[0])**2
        if w != 1:
            tf += '.mat'
            from scipy.io import loadmat, savemat
            savemat(tf, {'A':M}, format='5', do_compression=True, oned_as='row')
        raise ValueError('BNMF on sparse matrices is not implemented')
    print 'Matrix density is', density, 'and mask', np.isnan(M).sum()
    ## Set Parameters and Run Optimization algorithms
    lm = L ## lambda for J3 and J4
    para = []
    from numpy.random import rand
    for tt in xrange(t):
        if H is None or H.shape != (A.shape[0],r): ## wrong init, so create new
            print 'Initialize a random solution for H!'
            init_H = rand(M.shape[0], r)+E
        else: ## copy from avaliable ones
            print 'Optimize available solution for H!'
            init_H = np.abs(np.array(H[coverage>0,:], copy=True))+E
        if S is None or S.shape != (r,r): ## wrong init, so create new
            print 'Initialize a random solution for S!'
            init_S = np.eye(r) + E
        else: ## copy from avaliable ones
            print 'Optimize available solution for S!'
            init_S = np.array(np.nan_to_num(S), copy=True)
        if J in fun1:
            para.append((tf, init_H, init_S, e, int(i)+1, E))
        elif J in fun2:
            para.append((tf, C, lm, init_H, init_S, e, int(i)+1, E))
            print 'Lambda for', J, 'is set to', lm
        else:
            raise ValueError('Unknown objective %s'%J)
    if w == 1: ## for checking bugs
        out = [fun[J](M, *p[1:]) for p in para]
    else: ## map to multiple threads
        pl = Pool(w)
        out = pl.map(partial(fun_map, f=fun[J]), para)
        pl.close()
        pl.join()
    if w != 1:
        os.remove(tf) ## clean
    out.sort(key=lambda tup:tup[2][-1]) ## sort by the last objective
    out_H = np.zeros((A.shape[0],r))
    out_H[coverage>0,:] = np.matrix(out[0][0])
    out_S = np.matrix(out[0][1])
    print 'Density of H is %.3f;'%(np.sum(out_H>E) / float(out_H.shape[0]*out_H.shape[1])),
    print 'Density of S is %.3f;'%(np.sum(out_S>E) / float(out_S.shape[0]*out_S.shape[1]))
    if not P is None: ## plot objectives to a graph
        plt.plot(range(2,len(out[0][2])), out[0][2][2:])
        plt.xlabel('Number of iteration')
        plt.ylabel('Objective')
        plt.title('Objective values when solving %s (r=%s)'%(J,r))
        P.savefig(); plt.clf();
    print('The best %s objective for NMF is %s with r=%s after %s iterations.'%
            (J, out[0][2][-1], r, len(out[0][2])-2))
    if len(out[0][2])-2 == i:
        print 'Warning: may need more iterations to converage!'
    return out_H, out_S, out[0][2][-1]

def shrink_by(a, r, c):
    'Return a low resolution matrix by taking the sum of squares'
    return a.reshape(a.shape[0]/r, r, a.shape[1]/c, c).sum(axis=1).sum(axis=2)

def expand_by(a, r):
    'Return a expanded matrix by repeating the elements'
    return np.repeat(a, r, axis=0).copy()

def gini_coeff(x):
    'REF: http://www.ellipsix.net/blog/2012/11/the-gini-coefficient-for-distribution-inequality.html'
    # requires all values in x to be zero or positive numbers,
    # otherwise results are undefined
    n = len(x)
    s = x.sum()
    r = np.argsort(np.argsort(-x)) # calculates zero-based ranks
    return 1 - (2.0 * (r*x).sum() + s)/(n*s)

def gini_coeff2(x):
    n = len(x)
    g1 = gini_coeff(x[:n/2])
    g2 = gini_coeff(x[n/2:])
    return g2 - g1

def gini_impurity(x):
    'REF: http://en.wikipedia.org/wiki/Decision_tree_learning'
    if len(x.shape) == 1:
        s = x.sum()
        p = x/s
        return 1 - (p*p).sum()
    if len(x.shape) == 2:
        s = x.sum(1)
        p = x[s>0]/s[s>0,np.newaxis]
        return (1 - (p*p).sum(1)).mean()

def entropy(x):
    'REF: http://en.wikipedia.org/wiki/Entropy_%28information_theory%29'
    p = x/x.sum()
    return  - (p*np.log2(p)).sum()

def entropy_norm(x):
    p = x/x.sum()
    return  - (p*np.log2(p)).sum() / np.log2(x.shape)[0]

def map_cor(A, B):
    """ Return the pearson correlation of two matrix
    """
    A = np.array(np.nan_to_num(A))
    B = np.array(np.nan_to_num(B))
    m = A.shape[1]
    n = B.shape[1]
    C = np.zeros((m,n))
    from scipy.stats import pearsonr
    for i in xrange(m):
        for j in xrange(n):
            r = pearsonr(A[:,i], B[:,j])[0]
            if r == np.nan: continue
            C[i,j] = r
    return C

def SVD(A, r=3, sparse=True, N=1):
    A = np.asarray(np.nan_to_num(A))
    u = 0; s = 0; v = 0
    from scipy.sparse.linalg import svds
    for i in xrange(N):
        u1,s1,v1 = svds(A, r)
        alatent = np.argsort(s1)[::-1]
        s1 = s1[alatent]
        u1 = u1[:, alatent]
        v1 = v1[alatent, :]
        u += u1; s += s1; v += v1;
    u /= N; s /= N; v /= N;
    print "Singular values are:", s[:3], '...'
    U = np.matrix(u[:,:r])
    S = np.matrix(np.diag(s[:r]))
    V = np.matrix(v[:r,:])
    return U,S,V

def EIG(A, r=5, sparse=True):
    """ Eigen value decomposation on a square matrix.
        ``i.e.: A = Q * M * Q.T``

        :param A: the input square matrix
        :param r: number of leading eigen values to use
        :return (Q,M): Q is a matrix of eigen vectors, and M is 
                       a diagonal matrix of eigen values.
    """
    if sparse:
        from scipy.sparse.linalg import eigsh
        if np.isnan(A).any():
            latent, coeff = eigsh(np.nan_to_num(A), r)
        else:
            latent, coeff = eigsh(A, r)
    else:
        from numpy.linalg import eigh
        if np.isnan(A).any():
            latent, coeff = eigh(np.nan_to_num(A))
        else:
            latent, coeff = eigh(A)
    alatent = np.argsort(np.abs(latent))[::-1]
    latent = latent[alatent]
    pos = coeff.copy()
    pos[coeff < 0] = 0
    sign = np.asarray(np.power(pos,2)).sum(0)<np.asarray(np.power(pos-coeff,2)).sum(0)
    coeff[:,sign] *= -1
    coeff = coeff[:,alatent]
    print "Eigenvalues are:", latent[:3], '...'
    Q = np.matrix(coeff[:,:r])
    M = np.matrix(np.diag(latent[:r]))
    return Q, M

def NNDSVD(A, r=5, add=1e-5):
    H, S, _H = SVD(A+add, r=r)
    PosH = np.clip(H, 0, H.max())
    NegH = PosH - H
    PosNorm = np.power(np.power(PosH,2).sum(0),0.5)
    NegNorm = np.power(np.power(NegH,2).sum(0),0.5)
    for i in xrange(r):
        if PosNorm[0,i] >= NegNorm[0,i]:
            H[:,i] = PosH[:,i]/PosNorm[0,i]
            S[i,i] *= PosNorm[0,i]*PosNorm[0,i]
        else:
            H[:,i] = NegH[:,i]/NegNorm[0,i]
            S[i,i] *= NegNorm[0,i]*NegNorm[0,i]
    H += add
    S = np.abs(S) + add
    return H, S

def IterateCorrect(A, max_iter=200):
    X = np.array(np.nan_to_num(A), copy=True)
    B = np.ones(A.shape[0])
    for i in xrange(max_iter):
        b = X.sum(1)
        b /= np.mean(b[b != 0])
        b[b == 0] = 1
        B *= b
        X /= np.outer(b, b)
    print 'Bias are', B.min(), B.mean(), B.max()
    #load = X.sum()/np.logical_not(np.isnan(A)).sum()
    X[np.isnan(A)] = np.nan
    return B, X

def metric_l2(A, B):
    return np.power(np.power(A-B, 2).mean(), 0.5)

def metric_cor(A, B, rank=False):
    a = []; b = [];
    from scipy.sparse import issparse
    if issparse(A):
        I,J = A.nonzero()
    else:
        I,J = np.indices(A.shape)
        I = I.reshape(-1)
        J = J.reshape(-1)
    for i,j in zip(I.tolist(), J.tolist()):
        if np.isnan(A[i,j]) or np.isnan(B[i,j]):
            continue
        if A[i,j] == 0 or B[i,j] == 0:
            continue
        if i < j:
            a.append(A[i,j])
            b.append(B[i,j])
    from scipy.stats import pearsonr, spearmanr
    if rank:
        return spearmanr(a,b)[0]
    else:
        return pearsonr(a,b)[0]

def metric_avgcor(A, B):
    cor = []
    for i in xrange(A.shape[0]):
        cor.append(float(np.corrcoef(np.array(A)[i,], np.array(B)[i,])[0,1]))
        if str(cor[-1]) == 'nan':
            del cor[-1]
    return mean_std(cor)[0]

def metric_kl_div(A, B):
    A = np.nan_to_num(np.array(A))
    B = np.nan_to_num(np.array(B))
    A /= A.sum()
    B /= B.sum()
    logA = np.log(A+1e-100)
    logB = np.log(B+1e-100)
    return (A*(logA-logB)).sum()/2 + (B*(logB-logA)).sum()/2

def metric_js_div(A, B):
    A = np.nan_to_num(np.array(A))
    B = np.nan_to_num(np.array(B))
    A /= A.sum()
    B /= B.sum()
    logA = np.log(A+1e-100)
    logB = np.log(B+1e-100)
    logC = np.log(A/2+B/2+1e-100)
    return (A*(logA-logC)).sum()/2 + (B*(logB-logC)).sum()/2

def sample_ccd(chrs, bins):
    ''' Conserved Consecutive Distances (CCD) Method
        from Paulsen et al. NAR 2013
    '''
    from random import sample
    n = chrs.shape[0]
    rbins = []
    for ch in np.unique(chrs):
        chr_idx = np.arange(n)[ch == chrs].tolist()
        bin_idx = np.arange(n)[bins].tolist()
        bin_dist = [] ## distance among bins
        last = chr_idx[0]
        for i in bin_idx:
            if i in chr_idx:
                bin_dist.append(i - last)
                last = i
        if len(bin_dist) == 0:
            continue
        bin_dist = bin_dist[1:] ## remove index of the first bin
        last = sample(range(len(chr_idx)), 1)[0] ## first bin is random
        for dist in sample(bin_dist, len(bin_dist)): ## order is random
            rbins.append(chr_idx[last])
            last += dist
            if last >= len(chr_idx): ## loop back
                last -= len(chr_idx)
        rbins.append(chr_idx[last])
    return rbins

def sample_ccd_init(chrs, bins):
    ''' Conserved Consecutive Distances (CCD) Method
        from Paulsen et al. NAR 2013
    '''
    n = chrs.shape[0]
    data = []
    ## For example:
    ## chrs is [1,1,3,2,2,2,4]
    ## bins is [4,2,4,6,5]
    bin_idx = np.arange(n)[bins].tolist()
    bin_idx.sort()
    ## make bin_idx to be [2,4,4,5,6]
    for ch in np.unique(chrs):
        chr_idx = np.arange(n)[ch == chrs].tolist()
        ## if ch == 2, we have chr_idx = [3,4,5]
        bin_dist = [] ## distance among bins
        last = chr_idx[0]
        for i in bin_idx:
            if i in chr_idx:
                bin_dist.append(i - last)
                last = i
        ## now bin_dist is [1,0,1]
        if len(bin_dist) == 0:
            continue
        data.append((ch, chr_idx, np.array(bin_dist[1:])))
        ## return (1, [3,4,5], [0,1])
    return data

def sample_ccd_run(data, cut=-1):
    rbins = []
    for ch, chr_idx, bin_dist in data:
        last = np.random.randint(0, len(chr_idx)) ## first bin is random
        ## if ch == 2, and say last = 2
        rbins.append(chr_idx[last])
        ## rbins add 5
        np.random.shuffle(bin_dist) ## order is random
        ## say the new bin_dist is [1,0] 
        for dist in bin_dist:
            if cut >= 0 and dist >= cut:
                dist = np.random.randint(cut, len(chr_idx))
            last = (last + dist) % len(chr_idx)
            rbins.append(chr_idx[last])
            ## when dist = 1, rbins add 3
            ## when dist = 0, rbins add 3
        ## the random bins are [5,3,3], compared with input [4,4,5]
    return rbins

def sample_uni(chrs, bins):
    rbins = []
    for ch in np.unique(chrs):
        rbins += np.random.choice(np.arange(chrs.shape[0])[ch == chrs], 
                    (chrs[bins] == ch).sum(), replace=True).tolist()
    return rbins

def sample_sft(chrs, bins):
    rbins = []
    for ch in np.unique(chrs):
        all_idx = np.arange(chrs.shape[0])[ch == chrs]
        bin_idx = np.arange(chrs.shape[0])[chrs[bins] == ch]
        left = all_idx[0]
        righ = all_idx[-1]
        rand = np.random.randint(1, righ-left+1)
        rbins += (left + (bin_idx-left+rand)%(righ-left+1)).tolist()
    return rbins

def demo_create():
    os.chdir('../work')
    pdf = PdfPages('demo-plot.pdf')
    ## initalization
    map1 = ContactMap('demo')
    map1.clear()
    ## read chromosome sizes
    map1.genome_info('../data/yeast_chr_len.txt')
    datafiles = ['../data/Duan2010N/interactions_HindIII_fdr0.01_inter.txt',
                 '../data/Duan2010N/interactions_HindIII_fdr0.01_intra.txt'] 
    for datafile in datafiles:
        map1.add_interactions(datafile)

    ## create a binned heatmap
    map1.create_binnedmap(binsize=20e3)
    map1.mask_diag()
    map1.mask_short()
    map1.mask_low()
    map1.plot_map(vmin=1, vmax=1000, title='$X$')
    pdf.savefig(); plt.clf();

    ## using NMF
    map1.decompose_auto(plot=pdf)
    map1.sort_groups()
#    map1.plot_submap(title='Factorizing Yeast Hi-C Contact Map')
#    pdf.savefig(); plt.clf();

    ## test clustering enrichment
    idx, names = map1.get_locations('../data/Duan2010N/origins_nonCDR_early.txt', st=0, ch=0, po=1, nm=0)
    srt, val, pval = map1.test_enrichment(idx, method='AUC', title='Early Replicate', plot=pdf, pages=2)
    print pval[srt[0]]

    map1.save('demo')
    pdf.close()

def demo_run():
    map1 = ContactMap()
    map1.load('demo')
    map1.plot_submap()

if __name__ == "__main__": 
    demo_create()
#    demo_run()
