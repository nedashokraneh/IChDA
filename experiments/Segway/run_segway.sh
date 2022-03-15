#for k in 4 5 6 7 8 9 10; do
#    for res in 10000 25000 100000; do
#        genomedata_path="../../data/GM12878/hg19/res${res}_datasets/structural_signals.genomedata"
#        traindir="traindir_res${res}_k${k}"
#        annotdir="annotdir_res${res}_k${k}"
#        segway train ${genomedata_path} ${traindir} --resolution 1 --num-labels ${k} --distribution asinh_norm --ruler-scale 1 --seg-table ../../data/supp/hard_constraint.txt
#        segway annotate ${genomedata_path} ${traindir} ${annotdir}
#    done
#done

segway train ../../data/GM12878/hg19/res100000_datasets/structural_signals.genomedata res100000/structural/train_K8 --resolution 1 --num-labels 8 --distribution asinh_norm --ruler-scale 1 --seg-table ../../data/supp/hard_constraint.txt
segway annotate ../../data/GM12878/hg19/res100000_datasets/structural_signals.genomedata res100000/structural/train_K8 res100000/structural/annot_K8


segway train ../../data/GM12878/hg19/res100000_datasets/structural_signals.genomedata ../../data/GM12878/hg19/res100000_datasets/functional_signals.genomedata res100000/combined/train_K8 --resolution 1 --num-labels 8 --distribution asinh_norm --ruler-scale 1 --seg-table ../../data/supp/hard_constraint.txt
segway annotate ../../data/GM12878/hg19/res100000_datasets/structural_signals.genomedata ../../data/GM12878/hg19/res100000_datasets/functional_signals.genomedata res100000/combined/train_K8 res100000/combined/annot_K8
