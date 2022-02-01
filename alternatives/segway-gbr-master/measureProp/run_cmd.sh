#!/bin/bash
set -x

./MP_large_scale -inputGraphName data/reformated_graph.bin -transductionFile data/transduction_list.1 -labelFileList data/phonetically_labeled_files_mapped_nonZero -sigma 249.713 -nWinSize 7 -maxIters 300 -reOrderGraph f -numThreads 3 -mu 1.0 -nu 1.0 -numClasses 54 $*

