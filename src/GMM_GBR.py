import os
import numpy as np
import pandas as pd
from mpgraph import (MPGraphNode, MPGraph)
from sklearn import mixture
from scipy.special import logsumexp
import struct
import subprocess
import shutil

class GMM_MP():

    def __init__(self, args):

        self.signals_path = args['i']
        self.edges_path = args['e']
        self.bins_path = args['g']
        self.resolution = args['r']
        self.num_states = args['n']
        self.mp_num_iters = args['ml']
        self.mp_dir_path = args['o']
        self.mp_weight = args['w']
        self.mp_exe = args['mp']


        if os.path.exists(self.mp_dir_path):
            shutil.rmtree(self.mp_dir_path)
        os.mkdir(self.mp_dir_path)
        self.signals = pd.read_csv(self.signals_path, sep = "\t", header = None)
        self.signals = np.arcsinh(self.signals)
        self.bins = pd.read_csv(self.bins_path, sep = "\t", header = None)
        self.num_bins = self.bins.shape[0]
        self.iters_labels = []



    def write_mp_graph(self):

        self.hic_df = []
        mp_graph_filepath = os.path.join(self.mp_dir_path, 'mp_graph.bin')
        #if os.path.exists(mp_graph_filepath):
        #    print("mp graph file exists...")
        #else:
        neighbors = [[] for i in range(self.num_bins)]
        with open(self.edges_path) as edges_file:
            for line in edges_file:
                node1, node2, value = line.split()
                node1 = int(node1)
                node2 = int(node2)
                self.hic_df.append({"node1": node1, "node2": node2, "value": value})
                neighbors[node1].append((node2,1))
                neighbors[node2].append((node1,1))

        nodes = []
        for n in range(self.num_bins):
            nodes.append(MPGraphNode().init(n,neighbors[n]))
        graph = MPGraph().init(nodes)
        print("writing mp graph file...")
        with open(mp_graph_filepath, "wb") as mp_graph_file:
            graph.write_to(mp_graph_file)

        mp_trans_filepath = os.path.join(self.mp_dir_path, 'mp_trans.bin')
        #if os.path.exists(mp_trans_filepath):
        #    print("mp transduction file exists...")
        #else:
        print("writing mp transduction file...")
        label_fmt = "i"
        with open(mp_trans_filepath, "wb") as mp_trans_file:
            for _ in range(self.num_bins):
                mp_trans_file.write(struct.pack(label_fmt, 1))
        self.hic_df = pd.DataFrame(self.hic_df)



    def write_mp_post(self, egpr_iter_index):

        mp_dirname = "mp_iter{}".format(egpr_iter_index)
        mp_dirpath = os.path.join(self.mp_dir_path, mp_dirname)
        #if not os.path.exists(mp_dirpath):
        os.mkdir(mp_dirpath)
        mp_label_filepath = os.path.join(mp_dirpath, "mp.label")
        mp_label_fmt = "%sf" % self.num_states
        mp_label_file = open(mp_label_filepath, "wb")
        for post in self.posterior_probs:
            mp_label_file.write(struct.pack(mp_label_fmt, *post))
        mp_label_file.close()

    def read_mp_post(self, egpr_iter_index):

        mp_dirname = "mp_iter{}".format(egpr_iter_index)
        mp_dirpath = os.path.join(self.mp_dir_path, mp_dirname)
        mp_post_filepath = os.path.join(mp_dirpath, "mp.post")
        mp_post_file = open(mp_post_filepath, "rb")
        num_nodes, num_states = struct.unpack(
            "IH", mp_post_file.read(struct.calcsize("IH")))
        assert num_states == self.num_states
        assert num_nodes == self.num_bins
        node_fmt = "I%sf" % self.num_states

        posterior_probs = []
        for i in range(num_nodes):
            post = struct.unpack(node_fmt, mp_post_file.read(struct.calcsize(node_fmt)))
            posterior_probs.append(post[1:])

        self.virtual_evidence = posterior_probs


    def run_measure_prop(self, egpr_iter_index):

        mp_graph_filepath = os.path.join(self.mp_dir_path, 'mp_graph.bin')
        mp_trans_filepath = os.path.join(self.mp_dir_path, 'mp_trans.bin')
        mp_dirname = "mp_iter{}".format(egpr_iter_index)
        mp_dirpath = os.path.join(self.mp_dir_path, mp_dirname)
        mp_label_filepath = os.path.join(mp_dirpath, "mp.label")
        mp_post_filepath = os.path.join(mp_dirpath, "mp.post")
        mp_obj_filepath = os.path.join(mp_dirpath, "mp.obj")
        cmd = [self.mp_exe,
               "-inputGraphName", str(mp_graph_filepath),
               "-transductionFile", str(mp_trans_filepath),
               "-labelFile", str(mp_label_filepath),
               "-numThreads", "1",
               "-outPosteriorFile", str(mp_post_filepath),
               "-numClasses", str(self.num_states),
               "-mu", str(1),
               "-nu", str(0),
               "-selfWeight", str(1),
               "-nWinSize", "1",
               "-printAccuracy", "false",
               "-measureLabels", "true",
               "-maxIters", str(self.mp_num_iters),
               "-outObjFile", str(mp_obj_filepath),
               "-useSQL", "false"
               ]

        subprocess.Popen(cmd).wait()


    def GMM_inference(self, egpr_iter_index):

        if egpr_iter_index == 0:
            self.posterior_probs = self.gmm_model.predict_proba(self.signals)
        else:
            log_prob = self.gmm_model._estimate_log_prob(self.signals)
            self.virtual_evidence = np.power(self.virtual_evidence, self.mp_weight /
                         (1.0 + self.mp_weight)) + 1e-250
            self.virtual_evidence = self.virtual_evidence / np.sum(self.virtual_evidence)
            log_VE = np.log(self.virtual_evidence)
            weighted_log_prob = log_prob + log_VE
            log_prob_norm = logsumexp(weighted_log_prob, axis=1)
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
            self.posterior_probs = np.exp(log_resp)
        self.labels = np.argmax(self.posterior_probs, axis = 1)
        self.iters_labels.append(self.labels)



    def GMM_train(self):

        self.gmm_model = mixture.GaussianMixture(n_components=self.num_states, covariance_type='full')
        self.gmm_model.fit(self.signals)

    def write_posterior(self):

        posterior_path = os.path.join(self.mp_dir_path, "final_posterior.txt")
        with open(posterior_path, "w") as post_file:
            for ind, post in enumerate(self.posterior_probs):
                chr_name, start, end = self.bins.iloc[ind,0:3]
                line = '{}\t{}\t{}'.format(chr_name, start, end)
                for p in post:
                    line = line + '\t{}'.format(p)
                line = line + '\n'
                post_file.write(line)



def parse_arg():
    p = argparse.ArgumentParser()

    p.add_argument('-i', help="Path of input signals (node features)")
    p.add_argument('-e', help="Path of interactions file (edges)")
    p.add_argument('-g', help="Path of genome bin file (indices and corresponding genomic regions)")
    p.add_argument('-r', help="Resolution", type=int, default=1)
    p.add_argument('-n', help="Number of states.", type=int, default=5)
    p.add_argument('--il', help="Number of inference loops", default=5)
    p.add_argument('--ml', help="Number of measure propagation loops", default=5)
    #p.add_argument('-t', help="Threshold of p-value to chose edges of mp graph", type=float)
    p.add_argument('-w', help="measure prop weight", type=float, default=1)
    # measure_prop_weight = 0 doesn't consider mp posterior as GMM prior while
    # greater measure propagation weight assigns more difference between probability values
    p.add_argument('-o', help="Path of a directory to save mp graph and transduction files, and mp label files per iteration.")
    p.add_argument('--mp', help="Path of measure propagation exe file")
    #p.add_argument('-o', help="Output dir.")
    #p.add_argument("--prev", help="reload existing model.")
    #p.add_argument("--save", help="save model.", action='store_true')

def main():

    args = parse_arg()
    GMM_MP.__init__(args)
    GMM_MP.GMM_train()
    for l in range(args.il):
        if l != 0:
            GMM_MP.write_mp_label(l)
            GMM_MP.run_measure_prop(l)
            GMM_MP.read_mp_post(l)
        GMM_MP.GMM_inference(l)
    GMM_MP.write_posterior()



if __name__ == "__main__":
    main()
