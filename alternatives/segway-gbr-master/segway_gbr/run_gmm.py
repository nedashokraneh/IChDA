import os
import pandas as pd
from .mpgraph import (MPGraphNode, MPGraph)

class GMM_MP():

    def __init__(self, obs_path, interaction_path, num_labels, mp_dir_path):

        self.obs_path = obs_path
        self.interaction_path = interaction_path
        self.num_labels num_labels
        self.mp_dir_path = mp_dir_path

    def set_num_nodes(self):

        obs = pd.read_csv(self.obs_path, sep = "\t", header = None)
        self.num_nodes = obs.shape[0]

    def write_mp_graph(self):

        mp_graph_filepath = os.path.join(self.mp_dir_path, 'mp_graph.bin')
        if os.path.exists(mp_graph_filepath):
            print("mp graph file exists...")
        else:
            neighbors = [[] for i in range(self.num_nodes)]
            with open(self.interaction_path) as interaction_file:
                for line in interaction_file:
                    node1, node2, weight = line.split()
                    if node1 != node2:
                        neighbors[node1].append(node2,weight)
                        neighbors[node2].append(node1,weight)

            nodes = []
            for n in range(self.num_nodes):
                nodes.append(MPGraphNode.init(n,neighbors[n]))
            graph = MPGraph.init(nodes)
            print("writing mp graph file...")
            with open(mp_graph_filepath, "wb") as mp_graph_file:
                graph.write_to(mp_graph_file)

        mp_trans_filepath = os.path.join(self.mp_dir_path, 'mp_trans.bin')
        if os.path.exists(mp_trans_filepath):
            print("mp transduction file exists...")
        else:
            print("writing mp transduction file...")
            label_fmt = "i"
            with open(mp_trans_filepath, "wb") as mp_trans_file:
                for _ in range(self.num_nodes):
                    mp_trans_file.write(struct.pack(label_fmt, 1))

    def write_mp_label(self, egpr_iter_index):


    def init_gmm(self):

    
