import struct

class MPGraphNode:
    def __init__(self):
        pass

    def init(self, index, neighbors):
        self.index = index
        self.neighbors = neighbors
        self.num_neighbors = len(neighbors)
        return self

    node_fmt = "IH"
    neighbor_fmt = "If"

    def write_to(self, f):
        f.write(struct.pack(MPGraphNode.node_fmt,
                            self.index, self.num_neighbors))
        for i in range(self.num_neighbors):
            f.write(struct.pack(MPGraphNode.neighbor_fmt,
                                *self.neighbors[i]))
    def read_from(self, f):
        self.index, self.num_neighbors = \
            struct.unpack(MPGraphNode.node_fmt,
                          f.read(struct.calcsize(MPGraphNode.node_fmt)))
        self.neighbors = [None for i in range(self.num_neighbors)]
        for i in range(self.num_neighbors):
            self.neighbors[i] = \
                struct.unpack(MPGraphNode.neighbor_fmt,
                              f.read(struct.calcsize(MPGraphNode.neighbor_fmt)))
            index, weight = self.neighbors[i]
            try:
                assert weight > 0
            except:
                print(self.neighbors[i])
                exit()
        return self

class MPGraph:
    def __init__(self):
        pass

    def init(self, nodes):
        self.nodes = nodes
        self.num_nodes = len(nodes)
        return self

    num_nodes_fmt = "I"

    def write_to(self, f):
        f.write(struct.pack(MPGraph.num_nodes_fmt, self.num_nodes))
        for i in range(self.num_nodes):
            self.nodes[i].write_to(f)

    def read_from(self, f):
        self.num_nodes, = struct.unpack(MPGraph.num_nodes_fmt,
                                       f.read(struct.calcsize(MPGraph.num_nodes_fmt)))

        self.nodes = [None for i in range(self.num_nodes)]
        for i in range(self.num_nodes):
            self.nodes[i] = MPGraphNode().read_from(f)
        return self

class MPTransduction:
    def __init__(self):
        pass

    def init(self, labels):
        self.labels = labels
        return self

    label_fmt = "i"
    def write_to(self, f):
        for i in range(len(self.labels)):
            f.write(struct.pack(MPTransduction.label_fmt, self.labels[i]))

    def read_from(self, f, num_nodes):
        self.labels = [None for i in range(num_nodes)]
        for i in range(num_nodes):
            self.labels[i], = \
                struct.unpack(MPTransduction.label_fmt,
                              f.read(struct.calcsize(MPTransduction.label_fmt)))
        return self

class MPLabels:
    def __init__(self):
        pass

    def init(self, labels):
        self.labels = labels
        return self

    label_fmt = "i"
    def write_to(self, f):
        for i in range(len(self.labels)):
            f.write(struct.pack(MPLabels.label_fmt, self.labels[i]))

    def read_from(self, f, num_nodes):
        self.labels = [None for i in range(num_nodes)]
        for i in range(num_nodes):
            self.labels[i], = \
                struct.unpack(MPLabels.label_fmt,
                              f.read(struct.calcsize(MPLabels.label_fmt)))
        return self

class MPMeasureLabels:
    def __init__(self):
        pass

    def init(self, measure_labels):
        self.measure_labels = measure_labels
        return self

    measure_label_fmt = "f"
    def write_to(self, f):
        for i in range(len(self.measure_labels)):
            for c in range(len(self.measure_labels[i])):
                f.write(struct.pack(MPMeasureLabels.measure_label_fmt, 
                                    self.measure_labels[i][c]))

    def read_from(self, f, num_nodes, num_classes):
        self.measure_labels = [[None for j in range(num_classes)] for i in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_classes):
                self.measure_labels[i][j], = \
                    struct.unpack(MPMeasureLabels.measure_label_fmt,
                                  f.read(struct.calcsize(MPMeasureLabels.measure_label_fmt)))
        return self

class MPPosterior:
    def __init__(self):
        pass

    def init(self, posterior):
        self.posterior = posterior
        return self

    header_fmt = "IH"
    line_header_fmt = "I"
    posterior_fmt = "f"
    def write_to(self, f):
        f.write(struct.pack(MPPosterior.header_fmt, (len(posterior), len(posterior[0]))))
        for i in range(len(self.posterior)):
            f.write(struct.pack(MPPosterior.line_header_fmt, i))
            for c in range(len(self.posterior[i])):
                f.write(struct.pack(MPPosterior.posterior_fmt, posterior[i][c]))

    def read_from(self, f):
        num_nodes, num_classes = \
            struct.unpack(MPPosterior.header_fmt, 
                          f.read(struct.calcsize(MPPosterior.header_fmt)))
        self.posterior = [[None for j in range(num_classes)] for i in range(num_nodes)]
        for i in range(num_nodes):
            node_index, = \
                struct.unpack(MPPosterior.line_header_fmt,
                    f.read(struct.calcsize(MPPosterior.line_header_fmt)))
            assert node_index == i
            for j in range(num_classes):
                self.posterior[i][j] = \
                    struct.unpack(MPPosterior.posterior_fmt,
                                  f.read(struct.calcsize(MPPosterior.posterior_fmt)))[0]
        return self