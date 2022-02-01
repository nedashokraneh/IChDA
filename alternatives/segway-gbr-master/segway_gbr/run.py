#!/usr/bin/env python

"""
run: main Segway-GBR implementation
"""

import sys
import os
import struct
import subprocess
from pathlib import Path
from argparse import ArgumentParser
from distutils.spawn import find_executable
from numpy import array, float64, power, log, sum as npsum
from optplus import str2slice_or_int
from version import __version__
from segway import run as segway_runner
from ._egpr_utils import (coords_file, maybe_gzip_open,
                         create_interval_tree, get_frame_index, round_sig, ceildiv)
from .mpgraph import (MPGraphNode, MPGraph)


RESOLUTION = 1
NUM_SEGS = 2

FLOAT_NUM_SIG_FIGS = 6

VIRTUAL_EVIDENCE_SUBDIRNAME = "virtual_evidence"
VIRTUAL_EVIDENCE_OBS_FILENAME = "obs_{window_index}.list"
VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME = "window_list_{window_index}.list"

POSTERIOR_TMPL_FILENAME = "posterior{label_index}.{window_index}.bed"

MEASURE_PROP_SUBDIRNAME = "measure_prop"
MEASURE_PROP_GRAPH_FILENAME = "mp_graph"
MEASURE_PROP_TRANS_FILENAME = "mp_trans"
MEASURE_PROP_WORKDIRNAME_TMPL = "mp.{mp_round_index}"
MEASURE_PROP_POST_FILENAME = "mp_post"
MEASURE_PROP_OBJ_FILENAME = "objective.tab"
MEASURE_PROP_LABEL_FILENAME = "mp_label"


class MeasurePropRunner():
    """
    1. Interfaces with Segway to train a model
    2. Create a MP graph and trans containing fithic interactions
    3. EGPR iterations
        3.1. Convert Segway posterior to MP label file
        3.2. Run Measure Propagation
        3.3. Convert Measure Propagation posterior to Segway VE
        3.4. Run posterior-run
    4. Write the final annotation
    """

    options_to_attrs = {
        "num_labels": "num_segs",
        "resolution": "resolution",
        "measure_prop_exe": "measure_prop_exe",
        "measure_prop_mu": "measure_prop_mu",
        "measure_prop_nu": "measure_prop_nu",
        "measure_prop_weight": "measure_prop_weight",
        "measure_prop_self_weight": "measure_prop_alpha",
        "measure_prop_num_iters": "measure_prop_num_iters",
        "measure_prop_am_num_iters": "measure_prop_am_num_iters",
        "measure_prop_reuse_evidence": "measure_prop_reuse_evidence",
        "measure_prop_init_uniform": "measure_prop_init_uniform",
        "measure_prop_interaction_type": "interaction_type",
        "measure_prop_interactions_sorted": "interactions_sorted",
        "measure_prop_interaction_threshold": "threshold"}

    def __init__(self, **kwargs):
        # Segway attrs
        self.resolution = None
        self.num_segs = None
        self.work_dirpath = None
        self.genomedata_names = None
        self.segway_options = None
        self.windows_filepath = None
        self.windows = None
        self.num_frames = None

        # Measure prop attrs
        self.hic_path = None
        self.post_dirname = None
        self.measure_prop_mu = None
        self.measure_prop_nu = None
        self.measure_prop_alpha = None
        self.measure_prop_exe = None
        self.measure_prop_weight = None
        self.measure_prop_num_iters = None
        self.interaction_type = None
        self.interactions_sorted = None
        self.threshold = None

        self.__dict__.update(kwargs)

    def make_post_dir(self):
        res = Path(self.post_dirname)
        if not os.path.isdir(res):
            os.makedirs(res)
        return res

    def make_ve_subdir(self):
        res = Path(self.make_post_dir() / VIRTUAL_EVIDENCE_SUBDIRNAME)
        if not os.path.isdir(res):
            os.makedirs(res)
        return res

    def make_ve_filename(self):
        return self.make_ve_subdir() / MEASURE_PROP_GRAPH_FILENAME

    def make_mp_subdir(self):
        res = Path(self.make_post_dir() / MEASURE_PROP_SUBDIRNAME)
        if not os.path.isdir(res):
            os.makedirs(res)
        return res

    def make_mp_workdir(self, egpr_iter):
        res = Path(self.make_post_dir() /
                   MEASURE_PROP_SUBDIRNAME /
                   MEASURE_PROP_WORKDIRNAME_TMPL.format(mp_round_index=egpr_iter))
        if not os.path.isdir(res):
            os.makedirs(res)
        return res

    def make_mp_graph_filename(self):
        return (self.make_mp_subdir() /
                MEASURE_PROP_GRAPH_FILENAME)

    def make_mp_trans_filename(self):
        return (self.make_mp_subdir() /
                MEASURE_PROP_TRANS_FILENAME)

    def make_mp_label_filename(self, egpr_iter):
        return (self.make_mp_workdir(egpr_iter) /
                MEASURE_PROP_LABEL_FILENAME)

    def make_mp_post_filename(self, egpr_iter):
        return (self.make_mp_workdir(egpr_iter) /
                MEASURE_PROP_POST_FILENAME)

    def make_mp_obj_filename(self, egpr_iter):
        return (self.make_mp_workdir(egpr_iter) /
                MEASURE_PROP_OBJ_FILENAME)

    def make_virtual_evidence_obs_filename(self, window_index):
        return (self.make_ve_subdir() /
                VIRTUAL_EVIDENCE_OBS_FILENAME.format(window_index=window_index))

    def make_virtual_evidence_window_list_filename(self, window_index):
        return (self.make_ve_subdir() /
                VIRTUAL_EVIDENCE_WINDOW_LIST_FILENAME.format(window_index=window_index))

    def get_post_filename(self, window_index, label_index):
        return (self.make_post_dir() /
                "posterior" /
                POSTERIOR_TMPL_FILENAME.format(label_index=label_index, window_index=window_index))

    def write_mp_graph(self):
        windows_map = create_interval_tree(self.windows)

        neighbours = [[] for i in range(self.num_frames)]
        num_inters = 0
        num_skipped = 0

        with maybe_gzip_open(self.hic_path) as fithic_file:
            for line_index, line in enumerate(fithic_file):
                if (line_index % 1000000) == 0:
                    print(line_index, num_inters)
                # if num_inters == max_inters: break
                if ("contactCount" in line) or ("fragmentMid" in line):  # skip header line
                    print("skipping: ", line.strip())
                    continue
                line = line.split()
                if len(line) > 5:
                    chrom1, pos1, chrom2, pos2, _, pvalue, qvalue, *_ = line
                    pos1 = int(pos1)
                    pos2 = int(pos2)
                    pvalue = float(pvalue)
                    qvalue = float(qvalue)
                else:
                    chrom1, pos1, chrom2, pos2, _, *_ = line
                    pvalue, qvalue = 1.0, 1.0

                if (self.interaction_type == "intra") and (chrom1 != chrom2):
                    num_skipped += 1
                    continue
                if (self.interaction_type == "inter") and (chrom1 == chrom2):
                    num_skipped += 1
                    continue

                # if (max_dist != -1) and (abs(pos1-pos2) > max_dist):
                #     num_skipped += 1
                #     continue

                pvalue = max(pvalue, 1e-100)
                if pvalue > self.threshold:
                    if self.interactions_sorted.lower() == "true":
                        print("Stopping after finding an interaction with "
                              "pvalue=%s >= threshold=%s."
                              "If intreractions aren't sorted, do not specify --sorted"
                              % (pvalue, self.threshold))
                        print("Line was:", line)
                        break
                    else:
                        num_skipped += 1
                        continue

                _, frame1 = get_frame_index(
                    windows_map, self.resolution, chrom1, pos1)
                _, frame2 = get_frame_index(
                    windows_map, self.resolution, chrom2, pos2)

                if ((frame1 != -1) and (frame2 != -1)):
                    try:
                        if pvalue and self.threshold == 1:
                            weight = 1e-5
                        else:
                            weight = round_sig(-log(self.threshold *
                                                    pvalue), FLOAT_NUM_SIG_FIGS)
                    except ValueError:
                        print("pvalue and threshold should be between 0.0-1.0")

                    neighbours[frame1].append((frame2, weight))
                    neighbours[frame2].append((frame1, weight))
                    num_inters += 1
                else:
                    pass

        print("Total number of interactions written:", num_inters)
        print("Total number of interactions skipped:", num_skipped)

        mp_graph_filepath = self.make_mp_graph_filename()
        mp_trans_filepath = self.make_mp_trans_filename()

        nodes = []
        for frame_index in range(self.num_frames):
            nodes.append(MPGraphNode().init(
                frame_index, neighbours[frame_index]))

        graph = MPGraph().init(nodes)
        with open(mp_graph_filepath, "wb") as mp_graph_file:
            graph.write_to(mp_graph_file)

        print("Writing measure prop trans...")
        label_fmt = "i"
        with open(mp_trans_filepath, "wb") as mp_trans_file:
            for _ in range(self.num_frames):
                mp_trans_file.write(struct.pack(label_fmt, 1))

    def train_task(self):
        cmd = ["train-run", self.genomedata_names, self.work_dirpath]
        segway_runner.main(cmd)

        cmd = ["train-finish",
               self.genomedata_names, self.work_dirpath]
        segway_runner.main(cmd)

    def posterior_init_task(self):
        cmd = ["posterior-init",
               "--include-coords={}".format(self.windows_filepath),
               self.genomedata_names, self.work_dirpath, self.post_dirname]
        segway_runner.main(cmd)

    def write_mp_label(self, egpr_iter_index):
        print('Writing mp label...')
        mp_label_filepath = self.make_mp_label_filename(egpr_iter_index)
        with open(mp_label_filepath, "wb") as mp_label_file:
            # read posterior files from self.posterior_tmpls
            for window_index, (window_chrom, window_start, window_end, _, _) \
                    in enumerate(self.windows):
                # read segway posterior for this window
                window_num_frames = ceildiv(
                    window_end - window_start, self.resolution)
                posteriors = [
                    [10 for i in range(self.num_segs)] for j in range(window_num_frames)]
                for label_index in range(self.num_segs):
                    post_fname = self.get_post_filename(
                        window_index=window_index, label_index=label_index)
                    with maybe_gzip_open(post_fname, "r") as post:
                        for line in post:
                            row = line.split()
                            chrom, start, end, prob = row[:4]
                            start = int(start)
                            end = int(end)
                            prob = float(prob) / 100
                            # The bed entry should be within the window
                            assert chrom == window_chrom
                            assert start >= window_start
                            assert end <= window_end
                            # segway's posteriors should line up with the resolution
                            assert ((end - start) % self.resolution) == 0
                            assert ((start - window_start) %
                                    self.resolution) == 0
                            num_obs = ceildiv(end - start, self.resolution)
                            first_obs_index = int(
                                (start - window_start) / self.resolution)
                            for obs_index in range(first_obs_index, first_obs_index+num_obs):
                                posteriors[obs_index][label_index] = prob

                # add pseudocounts to avoid breaking measure prop
                for frame_index in range(window_num_frames):
                    posteriors[frame_index] = [((posteriors[frame_index][i] + 1e-20) /
                                                (1 + 1e-20*self.num_segs))
                                               for i in range(len(posteriors[frame_index]))]
                # assert distribution constraint
                assert len(posteriors[frame_index]) == self.num_segs
                for frame_index in range(window_num_frames):
                    # This should be 0.01
                    assert abs(sum(posteriors[frame_index]) - 1) < 0.03
                measure_label_fmt = "%sf" % self.num_segs
                for i in range(len(posteriors)):
                    mp_label_file.write(struct.pack(
                        measure_label_fmt, *posteriors[i]))

    def run_measure_prop(self, egpr_iter_index):
        print('Writing mp post...')
        mp_exe = self.measure_prop_exe if self.measure_prop_exe \
            else find_executable("MP_large_scale")
        if mp_exe is None:
            raise Exception("Could not find measure propogation executable!")
        mp_graph_filepath = self.make_mp_graph_filename()
        mp_trans_filepath = self.make_mp_trans_filename()
        mp_label_filepath = self.make_mp_label_filename(egpr_iter_index)
        mp_post_filepath = self.make_mp_post_filename(egpr_iter_index)
        mp_obj_filepath = self.make_mp_obj_filename(egpr_iter_index)

        # create a variable for useSQL
        cmd = [mp_exe,
               "-inputGraphName", str(mp_graph_filepath),
               "-transductionFile", str(mp_trans_filepath),
               "-labelFile", str(mp_label_filepath),
               "-numThreads", "1",
               "-outPosteriorFile", str(mp_post_filepath),
               "-numClasses", str(self.num_segs),
               "-mu", str(self.measure_prop_mu),
               "-nu", str(self.measure_prop_nu),
               "-selfWeight", str(self.measure_prop_alpha),
               "-nWinSize", "1",
               "-printAccuracy", "false",
               "-measureLabels", "true",
               "-maxIters", str(self.measure_prop_num_iters),
               "-outObjFile", str(mp_obj_filepath),
               "-useSQL", "false"
               ]

        print(" ".join(cmd))
        subprocess.Popen(cmd).wait()

    def write_virtual_evidence(self, egpr_iter_index):
        print('Writing virtual evidence...')
        header_fmt = "IH"
        frame_fmt = "%sf" % self.num_segs
        mp_post_filepath = self.make_mp_post_filename(egpr_iter_index)
        virtual_evidence_filepath = self.make_ve_filename()

        if egpr_iter_index > 0:
            mp_post_file = open(mp_post_filepath, "rb")
            num_nodes, num_classes = struct.unpack(
                header_fmt, mp_post_file.read(struct.calcsize(header_fmt)))
            assert num_classes == self.num_segs
            assert num_nodes == self.num_frames
            node_fmt = "I%sf" % self.num_segs

        for window_index, (_, window_start, window_end, _, _) in enumerate(self.windows):
            window_posts = []
            window_num_frames = ceildiv(
                window_end-window_start, self.resolution)
            for _ in range(window_num_frames):
                if egpr_iter_index == 0:
                    post = array(
                        [1.0/self.num_segs for _ in range(self.num_segs)])
                else:
                    # Read MP posteriors from f
                    line = struct.unpack(
                        node_fmt, mp_post_file.read(struct.calcsize(node_fmt)))
                    post = array(line[1:], dtype=float64)
                    # Transform posterior with mp_weight parameter, normalize and take log
                    post = power(post, self.measure_prop_weight /
                                 (1.0 + self.measure_prop_weight)) + 1e-250
                    post = log(post / npsum(post))
                window_posts.append(post)

            # write observations
            obs_filenames = [str(self.make_virtual_evidence_obs_filename(window_index))
                             for window_index in range(len(self.windows))]
            obs_filename = obs_filenames[window_index]
            if not os.path.isfile(obs_filename):
                with open(obs_filename, "wb") as obs_file:
                    for _, obs in enumerate(window_posts):
                        assert len(obs) == self.num_segs
                        obs_file.write(struct.pack(frame_fmt, *obs))
        if egpr_iter_index != 0:
            mp_post_file.close()
        # Write other virtual evidence files
        # full_list_filename = ve_dirname / "ve_full.list"
        window_list_filenames = [str(self.make_virtual_evidence_window_list_filename(window_index))
                                 for window_index in range(len(self.windows))]
        # write full list
        with open(virtual_evidence_filepath, "w") as list_file:
            list_file.write("\n".join(obs_filenames))
        # write window lists
        for window_index in range(len(self.windows)):
            window_list_filename = window_list_filenames[window_index]
            obs_filename = obs_filenames[window_index]
            with open(window_list_filename, "w") as window_list_file:
                window_list_file.write(obs_filename)

    def posterior_run_task(self, egpr_iter_index):
        virtual_evidence_filepath = self.make_ve_filename()

        if egpr_iter_index > 0:
            cmd = ["posterior-run", "--virtual-evidence={}".format(virtual_evidence_filepath),
                   self.genomedata_names, self.work_dirpath, self.post_dirname]
            segway_runner.main(cmd)
        else:
            cmd = ["posterior-run",
                   self.genomedata_names, self.work_dirpath, self.post_dirname]
            segway_runner.main(cmd)

    def write_annotation(self):
        cmd = ["posterior-finish",
               self.genomedata_names, self.work_dirpath, self.post_dirname]
        segway_runner.main(cmd)

    def set_option(self, name, value):
        if value or value == 0 or value is False or value == []:
            setattr(self, name, value)

    @classmethod
    def fromargs(cls, archives, traindir, fithic, postdir, segway_args):
        """Parses the arguments (not options) that were given to segway"""
        res = cls()

        res.work_dirpath = traindir
        res.post_dirname = postdir
        res.hic_path = fithic
        res.genomedata_names = archives

        segway_options = []
        for opt in segway_args:
            if "=" in opt:
                segway_options.append(opt)
        res.segway_options = segway_options
        return res

    @classmethod
    def fromoptions(cls, archives, traindir, fithic, postdir, segway_args, mp_options):
        """This is the usual way a Runner is created.
        Calls Runner.fromargs() first.
        """
        res = cls.fromargs(archives, traindir, fithic, postdir, segway_args)

        # Convert the options namespace into a dictionary
        options_dict = vars(mp_options)

        for option in options_dict.keys():
            if not (option == "archives" or option == "traindir" or
                    option == "fithic" or option == "postdir"):
                # Convert labels string into potential slice or an int
                # If num labels was specified
                if (option == "num_labels"
                        and options_dict[option]):
                    mp_options.num_labels = str2slice_or_int(
                        options_dict[option])
                    res.segway_options.append(
                        "--num-labels={}".format(options_dict[option]))
                if (option == "resolution"
                        and options_dict[option]):
                    res.segway_options.append(
                        "--resolution={}".format(options_dict[option]))
                # bulk copy options that need no further processing
                dst = cls.options_to_attrs[option]

                res.set_option(dst, options_dict[option])

        segway_runner.main(["train-init", *res.segway_options,
                            res.genomedata_names, res.work_dirpath])

        res.windows_filepath = Path(res.work_dirpath) / "window.bed"
        res.windows, res.num_frames = coords_file(
            res.windows_filepath, res.resolution)

        return res


def parse_options(args):
    description = """
    Segmentation and automated genome annotation."""
    usage = "segway-gbr [OPTION]... GENOMEDATA TRAINDIR FITHIC POSTERIORDIR"
    version = __version__
    parser = ArgumentParser(description=description, usage=usage)

    parser.add_argument("--version", action="version", version=version)

    parser.add_argument("archives", nargs=1)
    parser.add_argument("traindir", nargs=1)
    parser.add_argument("fithic", nargs=1)
    parser.add_argument("postdir", nargs=1)

    parser.add_argument("--measure-prop-exe", metavar="FILE",
                        help="path to measure prop executable")

    parser.add_argument("--measure-prop-mu", metavar="FILE", default=1, type=float,
                        help="mu hyperparameter for measure prop")

    parser.add_argument("--measure-prop-nu", metavar="FILE", default=0, type=float,
                        help="nu hyperparameter for measure prop")

    parser.add_argument("--measure-prop-weight", metavar="FILE", default=1.0, type=float,
                        help="weight hyperparameter for EGBR")

    parser.add_argument("--measure-prop-self-weight", metavar="FILE", default=1.0, type=float,
                        help="self-weight (alpha) hyperparameter for measure prop")

    parser.add_argument("--measure-prop-num-iters", metavar="FILE", default=5, type=int,
                        help="number of iterations to run posterior/measure prop")

    parser.add_argument("--measure-prop-interaction-type", metavar="intra/inter/total",
                        default="total",
                        help="Type of interactions to store in measure prop graph.")

    parser.add_argument("--measure-prop-interactions-sorted", metavar="true/false",
                        default="false",
                        help="The fithic interactions are sorted by pvalue.")

    parser.add_argument("--measure-prop-interaction-threshold", metavar="NUM",
                        default=1.0, type=float,
                        help="The pvalue cut off for fithic interaction.")

    parser.add_argument("-N", "--num-labels", metavar="SLICE",
                        default=str(NUM_SEGS), type=str,
                        help="make SLICE segment labels"
                        " (default %d)" % NUM_SEGS)  # will use str2slice_or_int

    parser.add_argument("--resolution", metavar="RES",
                        default=RESOLUTION, type=int,
                        help="downsample to every RES bp (default %d)" %
                        RESOLUTION)

    excluded_options = ["--minibatch-fraction", "--clobber"]

    mp_args, segway_args = parser.parse_known_args(args)

    # Add [0] to unlist directory names
    genomedata_archive = mp_args.archives[0]
    traindir = mp_args.traindir[0]
    hic_path = mp_args.fithic[0]
    postdir = mp_args.postdir[0]

    for excluded_option in excluded_options:
        if any(excluded_option in s for s in segway_args):
            raise NotImplementedError(
                "{} option is not compatible with Segway-GBR,"
                "or has not been implemented.".format(excluded_option))

    return segway_args, mp_args, genomedata_archive, traindir, hic_path, postdir


def run(mp_runner):
    mp_runner.posterior_init_task()
    mp_runner.train_task()
    mp_runner.write_mp_graph()

    for i in range(mp_runner.measure_prop_num_iters):
        print('Iteration', i)
        if i != 0:
            mp_runner.write_mp_label(i)
            mp_runner.run_measure_prop(i)
        mp_runner.write_virtual_evidence(i)
        mp_runner.posterior_run_task(i)

    mp_runner.write_annotation()


def main(argv=sys.argv[1:]):
    segway_args, mp_options, genomedata_archive, traindir, hic_path, postdir = parse_options(
        argv)
    mp_runner = MeasurePropRunner.fromoptions(
        genomedata_archive, traindir, hic_path, postdir, segway_args, mp_options)

    run(mp_runner)


if __name__ == "__main__":
    sys.exit(main())
