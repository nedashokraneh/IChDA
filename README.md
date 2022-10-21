# Integrative Chromatin Domain Annotation through Graph Embedding of Hi-C data 

This pipeline annotates genomic regions in a domain-scale based on both functional and structural features. Structural features are embedded in
high-throughput chromosome conformation capture (Hi-C) data that captures 3D interactions between genomic regions, represented as a 2D matrix. This problem requires a framework that jointly clusters the genomic regions based on their functional features (node attributes) and physical interactions (neighborhood information) between them. We first 
embed the Hi-C graph information into structural features, and then jointly cluster functional and structural features together. 

## Requirements

* numpy 
* pandas 
* hmmlearn 
* scipy 
* GSL should be installed and added to $LD_LIBRARY_PATH.


## Data generation
```python src/HC_data_generation.py -c test/HC_data_generation_config.json```

```src/HC_data_generation.py``` generates processed data given raw input signals (.bigwig format) and Hi-C data (.hic format). The parameters are:

* cell_type: The cell type corresponding to data.
* signals_dir: The path of directory including [signal_name].bedgraph files. The .bigwig files from Roadmap can be converted to .bedgraph files using ```bigWigToBedGraph``` tool in utilities.
* signals_names: The list of input signals names. Note that the files in signals_dir should be in format [signal_name].bedgraph .
* hic_file: The Hi-C data file path in .hic format. 
* temp_dir: The path of directory to save dump files such as binned genomic signals and extracted Hi-C interaction matrices files.
* chr_size_path: The path of genome chromosomes sizes file. Available in ```data/supp/hg19.chrom.sizes``` .
* valid_chroms: The list of chromosomes included in the experiment.
* valid_bins: The path of file including valid bins for a specific cell type in a specific resolution. Available in ```data/valid_bins``` . Those genomic bins having node features and neighbors are considered valid.
* juicer_path: The path of juicer tool to extract Hi-C interaction matrices. Available in ```utilities/juicer_tools_1.22.01.jar``` .
* resolution: The resolution of genomic regions.
* processed_dir: The path of directory to save processed files.



, which are passed in .json file. Example: ```test/HC_data_generation_config.json```. This script makes genomic bin file (each row has 4 columns, chromosome name, start, end, and index corresponding to valid bins. Indices are between 0 and N-1, where N is total number of valid bins.), signals file (i-th row signals values for i-th bin), and interactions file (each row has 3 columns, index of source bin, index of target bin, and the interaction weight between two bins) in processed_dir.

## Clustering

```python src/HC.py -c test/HC_config.json```

```src/HC.py``` learns structural embeddings and combinatorial domain annotation given generated data using ```src/HC_data_generation.py``` . The parameters are:

* resolution: The resolution of genomic regions.
* num_labels: The number of domain types of a domain annotation.
* line_path: The path to line exe file. Available in ```utilities/LINE/linux/line``` .
* chr_size_path: The path of genome chromosomes sizes file. Available in ```data/supp/hg19.chrom.sizes``` .
* valid_chroms: The list of chromosomes included in the experiment.
* bins_file: The path of generated genomic bin file.
* signals_file: The path of generated signals file.
* interactions_file: The path of generated interactions file.
* processed_dir: The path of directory to save embeddings and annotations.

, which are passed in .json file. Example: ```test/HC_config.json```. This script learns LINE embeddings and save it in processed_dir/embeddings, and learn combinatorial domain annotation and save it in processed_dir/annotations .

## Annotations

Our annotations are available in ```data/processed_data/annotations``` .

