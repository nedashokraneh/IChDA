import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


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
